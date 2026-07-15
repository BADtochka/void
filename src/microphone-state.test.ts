import { describe, expect, test } from "bun:test";
import { microphoneIsNeeded, type MicrophoneActivity } from "./microphone-state.js";

const idle: MicrophoneActivity = {
  activeFollowups: 0,
  pendingWakes: 0,
  pendingImages: 0,
  processingTurns: 0,
  queuedPlayback: 0,
  playbackActive: false,
};

describe("microphone activity", () => {
  test("pending hotword keeps the bot unmuted after its cue ends", () => {
    expect(microphoneIsNeeded({ ...idle, pendingWakes: 1 })).toBeTrue();
  });

  test("handoff from pending hotword to follow-up has no idle gap", () => {
    expect(microphoneIsNeeded({ ...idle, pendingWakes: 1 })).toBeTrue();
    expect(microphoneIsNeeded({ ...idle, activeFollowups: 1 })).toBeTrue();
  });

  test("late processing keeps the bot unmuted after follow-up expires", () => {
    expect(microphoneIsNeeded({ ...idle, processingTurns: 1 })).toBeTrue();
  });

  test("handoff from processing to dequeued playback has no idle gap", () => {
    expect(microphoneIsNeeded({ ...idle, processingTurns: 1 })).toBeTrue();
    expect(microphoneIsNeeded({ ...idle, playbackActive: true })).toBeTrue();
  });

  test("bot can mute only when every activity source is idle", () => {
    expect(microphoneIsNeeded(idle)).toBeFalse();
  });
});
