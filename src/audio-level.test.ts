import { describe, expect, test } from "bun:test";
import {
  buildVoiceActivityStats,
  voiceCaptureLooksLikeSpeech,
  voiceLevelPassesThresholds,
  voicedSampleFloor,
} from "./audio-level.js";

describe("voice level gate", () => {
  test("requires both RMS and peak thresholds", () => {
    expect(voiceLevelPassesThresholds(120, 600, 120, 600)).toBeTrue();
    expect(voiceLevelPassesThresholds(119, 1_000, 120, 600)).toBeFalse();
    expect(voiceLevelPassesThresholds(500, 599, 120, 600)).toBeFalse();
    expect(voiceLevelPassesThresholds(0, 0, 120, 600)).toBeFalse();
  });

  test("rejects captures without enough voiced activity", () => {
    const quietSpike = buildVoiceActivityStats(
      120 * 120 * 100,
      100,
      2_000,
      5,
      1.0,
    );
    expect(
      voiceCaptureLooksLikeSpeech(quietSpike, {
        minimumRms: 120,
        minimumPeak: 600,
        minimumVoicedMs: 180,
        minimumVoicedRatio: 0.12,
      }),
    ).toBeFalse();

    const spoken = buildVoiceActivityStats(
      200 * 200 * 100,
      100,
      2_000,
      40,
      1.0,
    );
    expect(
      voiceCaptureLooksLikeSpeech(spoken, {
        minimumRms: 120,
        minimumPeak: 600,
        minimumVoicedMs: 180,
        minimumVoicedRatio: 0.12,
      }),
    ).toBeTrue();
  });

  test("voiced sample floor stays below peak threshold", () => {
    expect(voicedSampleFloor(600)).toBe(240);
    expect(voicedSampleFloor(100)).toBe(180);
  });
});
