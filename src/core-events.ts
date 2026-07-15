import { config } from "./config.js";
import { logError, logInfo, logWarn } from "./logger.js";
import type { VoiceCueType } from "./audio-cues.js";

type VoiceCoreEventType = VoiceCueType | "followup_reopened" | "chat_message" | "status_speech";

export type VoiceCoreEvent = {
  type: VoiceCoreEventType;
  guildId: string;
  channelId: string;
  userId: string;
  followupMs?: number | null;
  awaitingContent?: boolean | null;
  content?: string | null;
  audioBase64?: string | null;
};

const EVENT_TYPES = new Set<VoiceCoreEventType>([
  "request_recognized",
  "generation_started",
  "followup_recognized",
  "followup_stopped",
  "followup_expired",
  "followup_reopened",
  "chat_message",
  "status_speech",
]);

export class CoreEventClient {
  private socket: WebSocket | null = null;
  private stopped = true;

  constructor(private readonly onEvent: (event: VoiceCoreEvent) => void) {}

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.socket?.close();
    this.socket = null;
  }

  private connect(): void {
    if (this.stopped) return;
    const url = new URL("/v1/events", config.voiceCoreUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.addEventListener("open", () => {
      logInfo("voice.events.connected", { url: url.toString() });
    });
    socket.addEventListener("message", (message) => {
      try {
        const event = JSON.parse(String(message.data)) as Partial<VoiceCoreEvent>;
        if (
          typeof event.type !== "string" ||
          !EVENT_TYPES.has(event.type as VoiceCoreEventType) ||
          typeof event.guildId !== "string" ||
          typeof event.channelId !== "string" ||
          typeof event.userId !== "string"
        ) {
          logWarn("voice.events.invalid", { payload: String(message.data) });
          return;
        }
        if (
          event.followupMs !== undefined &&
          event.followupMs !== null &&
          typeof event.followupMs !== "number"
        ) {
          logWarn("voice.events.invalid", { payload: String(message.data) });
          return;
        }
        if (
          event.awaitingContent !== undefined &&
          event.awaitingContent !== null &&
          typeof event.awaitingContent !== "boolean"
        ) {
          logWarn("voice.events.invalid", { payload: String(message.data) });
          return;
        }
        if (
          event.type === "chat_message" &&
          (typeof event.content !== "string" || event.content.trim().length === 0)
        ) {
          logWarn("voice.events.invalid", { payload: String(message.data) });
          return;
        }
        if (
          event.type === "status_speech" &&
          (typeof event.audioBase64 !== "string" || event.audioBase64.trim().length === 0)
        ) {
          logWarn("voice.events.invalid", { payload: String(message.data) });
          return;
        }
        this.onEvent(event as VoiceCoreEvent);
      } catch (error) {
        logError("voice.events.parse_failed", error);
      }
    });
    socket.addEventListener("error", () => {
      logWarn("voice.events.socket_error", { url: url.toString() });
    });
    socket.addEventListener("close", () => {
      if (this.socket === socket) this.socket = null;
      if (this.stopped) return;
      logWarn("voice.events.disconnected", { retryMs: 10_000 });
      const retry = setTimeout(() => this.connect(), 1_000);
      retry.unref();
    });
  }
}
