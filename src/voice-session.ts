import { Readable } from "node:stream";
import { OpusEncoder } from "@discordjs/opus";
import {
  AudioPlayer,
  AudioPlayerStatus,
  EndBehaviorType,
  StreamType,
  VoiceConnection,
  createAudioPlayer,
  createAudioResource,
} from "@discordjs/voice";
import { PermissionFlagsBits, type Client, type GuildMember } from "discord.js";
import { buildVoiceActivityStats, voiceCaptureLooksLikeSpeech, voicedSampleFloor } from "./audio-level.js";
import { config } from "./config.js";
import { createVoiceCue, type VoiceCueType } from "./audio-cues.js";
import {
  detectHotword,
  interruptGeneration,
  processTurn,
  startFollowupCountdown,
  type ImagePrompt,
} from "./core-client.js";
import type { VoiceCoreEvent } from "./core-events.js";
import { logError, logInfo, logWarn } from "./logger.js";
import { microphoneIsNeeded } from "./microphone-state.js";
import {
  captureAllowedForUser,
  hotwordActionForCapture,
  userHasActiveTurn,
} from "./voice-turn-state.js";

const PCM_BYTES_PER_SECOND = 48_000 * 2 * 2;
const MIN_UTTERANCE_BYTES = Math.floor(PCM_BYTES_PER_SECOND * 0.35);
const VOICE_STATE_SETTLE_MS = 120;
const MUTE_GRACE_MS = 350;

export class VoiceSession {
  readonly player: AudioPlayer;
  private readonly capturingUserIds = new Set<string>();
  private readonly playbackQueue: Array<{
    pcm: Buffer;
    kind: "cue" | "status" | "reply";
    userId?: string;
  }> = [];
  private readonly activeFollowupUsers = new Set<string>();
  private readonly pendingWakeUsers = new Set<string>();
  private readonly pendingImagePrompts = new Map<string, ImagePrompt>();
  private readonly processingCaptures = new Map<number, string>();
  private readonly processingAbortControllers = new Map<number, AbortController>();
  private awaitingContentUserId: string | null = null;
  private awaitingContentConfirmed = false;
  private currentPlaybackKind: "cue" | "status" | "reply" | null = null;
  private currentPlaybackUserId: string | null = null;
  private readonly pendingFollowupCountdownUsers = new Set<string>();
  private selfMuted: boolean;
  private playbackStartTimer: ReturnType<typeof setTimeout> | null = null;
  private muteTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingMuteReason = "idle";
  private nextCaptureId = 1;
  private stopRevision = 0;
  private readonly hotwordBackoffUntilByUser = new Map<string, number>();
  private closed = false;

  constructor(
    private readonly client: Client,
    readonly connection: VoiceConnection,
    readonly channelId: string,
  ) {
    this.selfMuted = connection.joinConfig.selfMute;
    this.player = createAudioPlayer();
    this.connection.subscribe(this.player);

    this.player.on("stateChange", (oldState, newState) => {
      if (newState.status === AudioPlayerStatus.Playing) {
        logInfo("voice.playback.started", this.context({ previousState: oldState.status }));
        this.updateMicrophoneState("playback_started");
      } else if (
        newState.status === AudioPlayerStatus.Idle &&
        oldState.status !== AudioPlayerStatus.Idle
      ) {
        const finishedKind = this.currentPlaybackKind;
        const finishedUserId = this.currentPlaybackUserId;
        logInfo("voice.playback.finished", this.context({ previousState: oldState.status }));
        this.currentPlaybackKind = null;
        this.currentPlaybackUserId = null;
        if (finishedKind === "reply" && finishedUserId) {
          this.pendingFollowupCountdownUsers.add(finishedUserId);
        }
        this.playNextReply();
        this.flushFollowupCountdowns();
        this.updateMicrophoneState("playback_finished");
      }
    });
    this.player.on("error", (error) => {
      logError("voice.playback.failed", error, this.context());
    });

    this.connection.receiver.speaking.on("start", (userId) => {
      void this.captureUser(userId);
    });
  }

  close(reason = "requested"): void {
    if (this.closed) return;
    logInfo("voice.session.closing", this.context({ reason }));
    this.closed = true;
    this.activeFollowupUsers.clear();
    this.pendingWakeUsers.clear();
    this.pendingImagePrompts.clear();
    this.processingCaptures.clear();
    for (const controller of this.processingAbortControllers.values()) controller.abort();
    this.processingAbortControllers.clear();
    this.awaitingContentUserId = null;
    this.awaitingContentConfirmed = false;
    this.pendingFollowupCountdownUsers.clear();
    this.hotwordBackoffUntilByUser.clear();
    this.playbackQueue.length = 0;
    if (this.playbackStartTimer) clearTimeout(this.playbackStartTimer);
    this.playbackStartTimer = null;
    if (this.muteTimer) clearTimeout(this.muteTimer);
    this.muteTimer = null;
    this.player.stop(true);
    this.connection.destroy();
  }

  playCue(type: VoiceCueType): void {
    if (this.closed) return;
    const item = { pcm: createVoiceCue(type, config.audioCueVolume), kind: "cue" as const };
    const firstReply = this.playbackQueue.findIndex((queued) => queued.kind === "reply");
    if (firstReply === -1) this.playbackQueue.push(item);
    else this.playbackQueue.splice(firstReply, 0, item);
    logInfo("voice.cue.queued", this.context({ type, queued: this.playbackQueue.length }));
    this.startQueuedPlayback();
  }

  armImagePrompt(userId: string, image: ImagePrompt): void {
    if (this.closed) throw new Error("Voice session is closed");
    this.pendingImagePrompts.set(userId, image);
    this.interruptForWake();
    logInfo(
      "voice.image_prompt.armed",
      this.context({
        userId,
        fileName: image.name,
        contentType: image.contentType,
        bytes: image.data.length,
      }),
    );
    this.playCue("image_prompt_ready");
    this.updateMicrophoneState("image_prompt_armed");
    void interruptGeneration(this.connection.joinConfig.guildId).catch((error) => {
      logError("voice.generation.interrupt_failed", error, this.context({ userId }));
    });
  }

  handleCoreEvent(event: VoiceCoreEvent): void {
    switch (event.type) {
      case "request_recognized": {
        this.interruptForWake();
        const earlyCuePlayed = this.pendingWakeUsers.delete(event.userId);
        this.activeFollowupUsers.add(event.userId);
        if (event.awaitingContent) {
          this.awaitingContentUserId = event.userId;
          this.awaitingContentConfirmed = true;
          this.pendingFollowupCountdownUsers.add(event.userId);
        } else {
          this.clearAwaitingContent(event.userId, "request_has_content");
        }
        if (!earlyCuePlayed) {
          this.playCue("request_recognized");
        }
        this.flushFollowupCountdowns();
        this.updateMicrophoneState("followup_started");
        break;
      }
      case "generation_started":
        this.playCue("generation_started");
        break;
      case "followup_recognized":
        this.activeFollowupUsers.add(event.userId);
        this.clearAwaitingContent(event.userId, "followup_recognized");
        this.playCue("followup_recognized");
        break;
      case "followup_expired":
        this.activeFollowupUsers.delete(event.userId);
        this.clearAwaitingContent(event.userId, "followup_expired");
        this.playCue("followup_expired");
        this.updateMicrophoneState("followup_expired");
        break;
      case "followup_stopped":
        this.stopRevision += 1;
        for (const controller of this.processingAbortControllers.values()) controller.abort();
        this.processingAbortControllers.clear();
        this.processingCaptures.clear();
        this.interruptForStop();
        this.activeFollowupUsers.clear();
        this.pendingWakeUsers.clear();
        this.pendingImagePrompts.clear();
        this.pendingFollowupCountdownUsers.clear();
        this.awaitingContentUserId = null;
        this.awaitingContentConfirmed = false;
        if (event.audioBase64) {
          try {
            const pcm = Buffer.from(event.audioBase64, "base64");
            if (pcm.length > 0) {
              this.enqueueStatusSpeech(pcm, event.userId);
            }
          } catch (error) {
            logError("voice.farewell.decode_failed", error, this.context({ userId: event.userId }));
          }
        }
        this.playCue("followup_stopped");
        this.updateMicrophoneState("followup_stopped");
        break;
      case "followup_reopened":
        this.activeFollowupUsers.add(event.userId);
        this.updateMicrophoneState("followup_reopened");
        break;
      case "status_speech": {
        if (!event.audioBase64) break;
        try {
          const pcm = Buffer.from(event.audioBase64, "base64");
          if (pcm.length > 0) {
            this.enqueueStatusSpeech(pcm, event.userId);
          }
        } catch (error) {
          logError("voice.status_speech.decode_failed", error, this.context({ userId: event.userId }));
        }
        break;
      }
    }
  }

  private captureUser(userId: string): void {
    const awaitingContentUserId = this.awaitingContentUserId;
    if (!captureAllowedForUser(awaitingContentUserId, userId)) {
      logInfo(
        "voice.capture.ignored_while_awaiting_owner",
        this.context({ userId, awaitingUserId: awaitingContentUserId ?? "unknown" }),
      );
      return;
    }
    if (
      this.closed ||
      this.capturingUserIds.has(userId) ||
      userId === this.client.user?.id
    ) {
      return;
    }

    this.capturingUserIds.add(userId);
    const captureId = this.nextCaptureId++;
    const captureRevision = this.stopRevision;
    const guild = this.client.guilds.cache.get(this.connection.joinConfig.guildId);
    const member = guild?.members.cache.get(userId) as GuildMember | undefined;
    const displayName = member?.displayName ?? "Участник Discord";
    const userIsAdmin = member?.permissions.has(PermissionFlagsBits.Administrator) ?? false;
    const followupAtCaptureStart =
      this.userHasActiveTurn(userId);
    const replyPlayingAtCaptureStart = this.currentPlaybackKind === "reply";
    const imageAtCaptureStart = this.pendingImagePrompts.get(userId);
    logInfo(
      "voice.capture.started",
      this.context({
        userId,
        displayName,
        followup: followupAtCaptureStart,
        imagePrompt: Boolean(imageAtCaptureStart),
      }),
    );
    const opus = this.connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: config.silenceMs },
    });
    const decoder = new OpusEncoder(48_000, 2);
    const chunks: Buffer[] = [];
    const maxBytes = PCM_BYTES_PER_SECOND * config.maxUtteranceSeconds;
    let byteLength = 0;
    let truncated = false;
    let invalidPackets = 0;
    let energySum = 0;
    let energySamples = 0;
    let voicedSamples = 0;
    let peakAmplitude = 0;
    let finished = false;
    let earlyHotwordDetected = false;
    let activeHotwordInterrupted = false;
    let hotwordTimedOut = false;
    let hotwordCheckPromise: Promise<void> | null = null;
    const captureStartedAt = performance.now();
    const sampleVoiceFloor = voicedSampleFloor(config.minVoicePeak);

    const voiceGateOptions = {
      minimumRms: config.minVoiceRms,
      minimumPeak: config.minVoicePeak,
      minimumVoicedMs: config.minVoicedMs,
      minimumVoicedRatio: config.minVoicedRatio,
    };

    const currentVoiceStats = (durationSeconds: number) =>
      buildVoiceActivityStats(
        energySum,
        energySamples,
        peakAmplitude,
        voicedSamples,
        durationSeconds,
      );

    const activateHotword = (transcript: string, detection: "early" | "final"): void => {
      if (
        this.closed ||
        earlyHotwordDetected ||
        this.userHasActiveTurn(userId) ||
        !captureAllowedForUser(this.awaitingContentUserId, userId)
      ) {
        return;
      }
      earlyHotwordDetected = true;
      this.pendingWakeUsers.add(userId);
      this.awaitingContentUserId = userId;
      this.awaitingContentConfirmed = false;
      logInfo(
        detection === "early"
          ? "voice.hotword.detected_early"
          : "voice.hotword.detected_final",
        this.context({ userId, displayName, transcript }),
      );
      this.interruptForWake();
      this.playCue("request_recognized");
      this.updateMicrophoneState(`hotword_detected_${detection}`);
      void interruptGeneration(this.connection.joinConfig.guildId).catch((error) => {
        logError(
          "voice.generation.interrupt_failed",
          error,
          this.context({ userId, displayName }),
        );
      });
    };

    const interruptActiveTurnForHotword = (transcript: string): void => {
      if (this.closed || activeHotwordInterrupted) return;
      activeHotwordInterrupted = true;
      logInfo(
        "voice.hotword.detected_during_active_turn",
        this.context({ userId, displayName, transcript }),
      );
      this.interruptForWake();
      void interruptGeneration(this.connection.joinConfig.guildId).catch((error) => {
        logError(
          "voice.generation.interrupt_failed",
          error,
          this.context({ userId, displayName }),
        );
      });
    };

    const checkHotword = (): Promise<void> | null => {
      if (
        finished ||
        (followupAtCaptureStart && !replyPlayingAtCaptureStart) ||
        imageAtCaptureStart ||
        earlyHotwordDetected ||
        activeHotwordInterrupted ||
        hotwordTimedOut ||
        hotwordCheckPromise ||
        performance.now() < (this.hotwordBackoffUntilByUser.get(userId) ?? 0) ||
        byteLength < (PCM_BYTES_PER_SECOND * config.hotwordMinAudioMs) / 1000
      ) {
        return hotwordCheckPromise;
      }
      const currentStats = currentVoiceStats(byteLength / PCM_BYTES_PER_SECOND);
      if (!voiceCaptureLooksLikeSpeech(currentStats, voiceGateOptions)) {
        return null;
      }
      const snapshot = Buffer.concat(chunks, byteLength);
      const check = detectHotword(snapshot)
        .then((result) => {
          if (
            result.busy ||
            !result.detected ||
            this.closed ||
            earlyHotwordDetected ||
            activeHotwordInterrupted
          ) {
            return;
          }
          if (
            hotwordActionForCapture(
              followupAtCaptureStart,
              this.userHasActiveTurn(userId),
            ) === "interrupt"
          ) {
            interruptActiveTurnForHotword(result.transcript);
            return;
          }
          activateHotword(result.transcript, "early");
        })
        .catch((error) => {
          if (error instanceof DOMException && error.name === "TimeoutError") {
            hotwordTimedOut = true;
            this.hotwordBackoffUntilByUser.set(
              userId,
              performance.now() + config.hotwordTimeoutMs,
            );
            logWarn(
              "voice.hotword.detection_timed_out",
              this.context({ userId, displayName, timeoutMs: config.hotwordTimeoutMs }),
            );
            return;
          }
          logError("voice.hotword.detection_failed", error, this.context({ userId, displayName }));
        })
        .finally(() => {
          if (hotwordCheckPromise === check) hotwordCheckPromise = null;
        });
      hotwordCheckPromise = check;
      return check;
    };

    const hotwordTimer = setInterval(() => {
      void checkHotword();
    }, config.hotwordCheckIntervalMs);
    hotwordTimer.unref();

    opus.on("data", (packet: Buffer) => {
      try {
        const pcm = decoder.decode(packet);
        const remainingBytes = maxBytes - byteLength;
        if (remainingBytes <= 0) {
          truncated = true;
          opus.destroy();
          return;
        }

        const acceptedPcm = pcm.length > remainingBytes ? pcm.subarray(0, remainingBytes) : pcm;
        for (let offset = 0; offset + 1 < acceptedPcm.length; offset += 8) {
          const sample = acceptedPcm.readInt16LE(offset);
          const amplitude = Math.abs(sample);
          energySum += sample * sample;
          energySamples += 1;
          if (amplitude >= sampleVoiceFloor) {
            voicedSamples += 1;
          }
          peakAmplitude = Math.max(peakAmplitude, amplitude);
        }
        chunks.push(acceptedPcm);
        byteLength += acceptedPcm.length;
        if (acceptedPcm.length < pcm.length || byteLength >= maxBytes) {
          truncated = true;
          opus.destroy();
        }
      } catch {
        invalidPackets += 1;
      }
    });

    const finish = async () => {
      if (finished) return;
      finished = true;
      clearInterval(hotwordTimer);
      this.capturingUserIds.delete(userId);
      const durationSeconds = byteLength / PCM_BYTES_PER_SECOND;
      const voiceStats = currentVoiceStats(durationSeconds);
      if (invalidPackets > 0) {
        logWarn(
          "voice.capture.invalid_packets",
          this.context({ userId, displayName, invalidPackets }),
        );
      }
      if (this.closed) {
        logInfo("voice.capture.cancelled", this.context({ userId, displayName }));
        return;
      }
      if (captureRevision !== this.stopRevision) {
        logInfo(
          "voice.capture.cancelled_after_stop",
          this.context({ userId, displayName }),
        );
        return;
      }
      if (byteLength < MIN_UTTERANCE_BYTES) {
        this.releasePendingWake(userId, earlyHotwordDetected, "capture_too_short");
        logInfo(
          "voice.capture.discarded",
          this.context({ userId, displayName, durationSeconds: durationSeconds.toFixed(2) }),
        );
        return;
      }
      if (!voiceCaptureLooksLikeSpeech(voiceStats, voiceGateOptions)) {
        this.releasePendingWake(userId, earlyHotwordDetected, "capture_not_speech");
        logInfo(
          "voice.capture.discarded_not_speech",
          this.context({
            userId,
            displayName,
            durationSeconds: durationSeconds.toFixed(2),
            rmsAmplitude: Math.round(voiceStats.rmsAmplitude),
            peakAmplitude: voiceStats.peakAmplitude,
            voicedMs: Math.round(voiceStats.voicedMs),
            voicedRatio: Number(voiceStats.voicedRatio.toFixed(3)),
            minimumRms: config.minVoiceRms,
            minimumPeak: config.minVoicePeak,
            minimumVoicedMs: config.minVoicedMs,
            minimumVoicedRatio: config.minVoicedRatio,
          }),
        );
        return;
      }

      if (!captureAllowedForUser(this.awaitingContentUserId, userId)) {
        logInfo(
          "voice.capture.discarded_while_awaiting_owner",
          this.context({
            userId,
            displayName,
            awaitingUserId: this.awaitingContentUserId ?? "unknown",
          }),
        );
        return;
      }

      const followupEligible = followupAtCaptureStart || this.userHasActiveTurn(userId);
      if (!followupEligible && !imageAtCaptureStart) {
        if (hotwordCheckPromise) await hotwordCheckPromise;
        if (!earlyHotwordDetected) {
          try {
            const result = await detectHotword(Buffer.concat(chunks, byteLength), true);
            logInfo(
              "voice.hotword.final_check_completed",
              this.context({
                userId,
                displayName,
                detected: result.detected,
                transcript: result.transcript,
                previousTimeout: hotwordTimedOut,
              }),
            );
            if (result.detected) activateHotword(result.transcript, "final");
          } catch (error) {
            logError(
              "voice.hotword.final_check_failed",
              error,
              this.context({ userId, displayName }),
            );
          }
        }
        if (!earlyHotwordDetected) {
          logInfo(
            "voice.capture.discarded_without_hotword",
            this.context({
              userId,
              displayName,
              durationSeconds: durationSeconds.toFixed(2),
              hotwordTimedOut,
            }),
          );
          return;
        }
      }

      this.processingCaptures.set(captureId, userId);
      const processingController = new AbortController();
      this.processingAbortControllers.set(captureId, processingController);
      this.updateMicrophoneState("turn_processing_started");

      if (
        imageAtCaptureStart &&
        this.pendingImagePrompts.get(userId) === imageAtCaptureStart
      ) {
        this.pendingImagePrompts.delete(userId);
      }

      logInfo(
        "voice.capture.completed",
        this.context({
          userId,
          displayName,
          durationSeconds: durationSeconds.toFixed(2),
          pcmBytes: byteLength,
          rmsAmplitude: Math.round(voiceStats.rmsAmplitude),
          peakAmplitude: voiceStats.peakAmplitude,
          voicedMs: Math.round(voiceStats.voicedMs),
          voicedRatio: Number(voiceStats.voicedRatio.toFixed(3)),
          truncated,
        }),
      );

      const processingStartedAt = performance.now();
      logInfo("voice.processing.started", this.context({ userId, displayName }));
      try {
        const replyPcm = await processTurn(
          Buffer.concat(chunks, byteLength),
          {
            guildId: this.connection.joinConfig.guildId,
            channelId: this.channelId,
            userId,
            displayName,
            userIsAdmin,
            audioAgeMs: performance.now() - captureStartedAt,
            earlyHotwordDetected,
            image: imageAtCaptureStart,
          },
          processingController.signal,
        );

        const processingMs = Math.round(performance.now() - processingStartedAt);
        if (truncated) {
          logWarn("voice.capture.truncated", this.context({ userId, displayName }));
        }
        if (replyPcm && !this.closed && captureRevision === this.stopRevision) {
          logInfo(
            "voice.processing.completed",
            this.context({ userId, displayName, processingMs, replyPcmBytes: replyPcm.length }),
          );
          this.enqueueReply(replyPcm, userId);
        } else {
          logInfo(
            "voice.processing.no_reply",
            this.context({ userId, displayName, processingMs }),
          );
          this.queueFollowupCountdown(userId, "no_reply");
        }
      } catch (error) {
        if (processingController.signal.aborted) {
          logInfo("voice.processing.cancelled_after_stop", this.context({ userId, displayName }));
          return;
        }
        if (imageAtCaptureStart && !this.pendingImagePrompts.has(userId)) {
          this.pendingImagePrompts.set(userId, imageAtCaptureStart);
          this.updateMicrophoneState("image_prompt_restored");
        }
        logError("voice.processing.failed", error, this.context({ userId, displayName }));
        this.queueFollowupCountdown(userId, "processing_failed");
      } finally {
        this.processingAbortControllers.delete(captureId);
        this.processingCaptures.delete(captureId);
        this.pendingWakeUsers.delete(userId);
        if (
          this.awaitingContentUserId === userId &&
          !this.awaitingContentConfirmed
        ) {
          this.clearAwaitingContent(userId, "turn_processing_finished_without_confirmation");
        }
        this.updateMicrophoneState("turn_processing_finished");
      }
    };

    opus.once("end", () => void finish());
    opus.once("close", () => void finish());
    opus.once("error", (error) => {
      logError("voice.capture.stream_failed", error, this.context({ userId, displayName }));
      void finish();
    });
  }

  private enqueueReply(replyPcm: Buffer, userId: string): void {
    this.playbackQueue.push({ pcm: replyPcm, kind: "reply", userId });
    logInfo("voice.playback.queued", this.context({ queued: this.playbackQueue.length }));
    this.startQueuedPlayback();
  }

  private enqueueStatusSpeech(pcm: Buffer, userId: string): void {
    if (this.closed) return;
    const item = { pcm, kind: "status" as const, userId };
    const firstReply = this.playbackQueue.findIndex((queued) => queued.kind === "reply");
    if (firstReply === -1) this.playbackQueue.push(item);
    else this.playbackQueue.splice(firstReply, 0, item);
    logInfo("voice.status_speech.queued", this.context({ userId, queued: this.playbackQueue.length }));
    this.startQueuedPlayback();
  }

  private startQueuedPlayback(): void {
    this.playNextReply();
  }

  private playNextReply(): void {
    if (
      this.closed ||
      this.player.state.status !== AudioPlayerStatus.Idle
    ) {
      return;
    }

    if (this.playbackQueue.length === 0) {
      this.updateMicrophoneState("playback_queue_empty");
      return;
    }

    if (this.selfMuted) {
      this.setSelfMuted(false, "playback_pending");
      if (!this.playbackStartTimer) {
        this.playbackStartTimer = setTimeout(() => {
          this.playbackStartTimer = null;
          this.playNextReply();
        }, VOICE_STATE_SETTLE_MS);
        this.playbackStartTimer.unref();
      }
      return;
    }

    const item = this.playbackQueue.shift();
    if (!item) return;
    const resource = createAudioResource(Readable.from(item.pcm), {
      inputType: StreamType.Raw,
    });
    this.currentPlaybackKind = item.kind;
    this.currentPlaybackUserId = item.userId ?? null;
    logInfo(
      "voice.playback.dequeued",
      this.context({ kind: item.kind, userId: item.userId ?? "" }),
    );
    this.player.play(resource);
  }

  private queueFollowupCountdown(userId: string, reason: string): void {
    if (!this.activeFollowupUsers.has(userId) || this.closed) return;
    this.pendingFollowupCountdownUsers.add(userId);
    logInfo("voice.followup.countdown_queued", this.context({ userId, reason }));
    this.flushFollowupCountdowns();
  }

  private flushFollowupCountdowns(): void {
    if (
      this.closed ||
      this.player.state.status !== AudioPlayerStatus.Idle ||
      this.playbackQueue.length > 0 ||
      this.playbackStartTimer
    ) {
      return;
    }
    const userIds = [...this.pendingFollowupCountdownUsers];
    this.pendingFollowupCountdownUsers.clear();
    for (const userId of userIds) {
      if (!this.activeFollowupUsers.has(userId)) continue;
      logInfo("voice.followup.countdown_starting", this.context({ userId }));
      void startFollowupCountdown(
        this.connection.joinConfig.guildId,
        this.channelId,
        userId,
      ).catch((error) => {
        if (this.closed) return;
        this.pendingFollowupCountdownUsers.add(userId);
        logError(
          "voice.followup.countdown_failed",
          error,
          this.context({ userId }),
        );
        const retry = setTimeout(() => this.flushFollowupCountdowns(), 1_000);
        retry.unref();
      });
    }
  }

  private interruptForWake(): void {
    let removedReplies = 0;
    for (let index = this.playbackQueue.length - 1; index >= 0; index -= 1) {
      if (this.playbackQueue[index]?.kind !== "reply") continue;
      const removedUserId = this.playbackQueue[index]?.userId;
      if (removedUserId) this.pendingFollowupCountdownUsers.add(removedUserId);
      this.playbackQueue.splice(index, 1);
      removedReplies += 1;
    }
    const interruptedPlayback = this.currentPlaybackKind === "reply";
    if (interruptedPlayback && this.currentPlaybackUserId) {
      this.pendingFollowupCountdownUsers.add(this.currentPlaybackUserId);
    }
    if (interruptedPlayback) this.player.stop(true);
    logInfo(
      "voice.playback.interrupted_for_wake",
      this.context({ interruptedPlayback, removedReplies }),
    );
  }

  private interruptForStop(): void {
    const removedPlayback = this.playbackQueue.length;
    this.playbackQueue.length = 0;
    this.pendingFollowupCountdownUsers.clear();
    if (this.playbackStartTimer) clearTimeout(this.playbackStartTimer);
    this.playbackStartTimer = null;
    const interruptedPlayback = this.player.state.status !== AudioPlayerStatus.Idle;
    if (interruptedPlayback) this.player.stop(true);
    else {
      this.currentPlaybackKind = null;
      this.currentPlaybackUserId = null;
    }
    logInfo(
      "voice.playback.interrupted_for_stop",
      this.context({ interruptedPlayback, removedPlayback }),
    );
  }

  private updateMicrophoneState(reason: string): void {
    if (this.closed) return;
    if (this.microphoneNeeded()) {
      if (this.muteTimer) clearTimeout(this.muteTimer);
      this.muteTimer = null;
      this.setSelfMuted(false, reason);
      return;
    }

    this.pendingMuteReason = reason;
    if (this.selfMuted || this.muteTimer) return;
    this.muteTimer = setTimeout(() => {
      this.muteTimer = null;
      if (this.closed || this.microphoneNeeded()) return;
      this.setSelfMuted(true, this.pendingMuteReason);
    }, MUTE_GRACE_MS);
    this.muteTimer.unref();
  }

  private microphoneNeeded(): boolean {
    return microphoneIsNeeded({
      activeFollowups: this.activeFollowupUsers.size,
      pendingWakes: this.pendingWakeUsers.size,
      pendingImages: this.pendingImagePrompts.size,
      processingTurns: this.processingCaptures.size,
      queuedPlayback: this.playbackQueue.length,
      playbackActive:
        this.currentPlaybackKind !== null ||
        this.player.state.status !== AudioPlayerStatus.Idle,
    });
  }

  private releasePendingWake(userId: string, detected: boolean, reason: string): void {
    if (!detected || !this.pendingWakeUsers.delete(userId)) return;
    if (!this.awaitingContentConfirmed) this.clearAwaitingContent(userId, reason);
    this.updateMicrophoneState(reason);
  }

  private userHasActiveTurn(userId: string): boolean {
    return userHasActiveTurn(
      userId,
      this.activeFollowupUsers,
      this.pendingWakeUsers,
      this.processingCaptures.values(),
    );
  }

  private clearAwaitingContent(userId: string, reason: string): void {
    if (this.awaitingContentUserId !== userId) return;
    this.awaitingContentUserId = null;
    this.awaitingContentConfirmed = false;
    logInfo("voice.followup.awaiting_content_cleared", this.context({ userId, reason }));
  }

  private setSelfMuted(muted: boolean, reason: string): void {
    if (this.closed || this.selfMuted === muted) return;
    if (!muted && this.muteTimer) {
      clearTimeout(this.muteTimer);
      this.muteTimer = null;
    }
    const sent = this.connection.rejoin({
      channelId: this.channelId,
      selfDeaf: false,
      selfMute: muted,
    });
    if (!sent) {
      logWarn("voice.microphone.state_update_failed", this.context({ muted, reason }));
      return;
    }
    this.selfMuted = muted;
    logInfo("voice.microphone.state_updated", this.context({ muted, reason }));
  }

  private context(extra: Record<string, string | number | boolean> = {}) {
    return {
      guildId: this.connection.joinConfig.guildId,
      channelId: this.channelId,
      ...extra,
    };
  }
}
