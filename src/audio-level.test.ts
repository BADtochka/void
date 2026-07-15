import { describe, expect, test } from "bun:test";
import { voiceLevelPassesThresholds } from "./audio-level.js";

describe("voice level gate", () => {
  test("requires both RMS and peak thresholds", () => {
    expect(voiceLevelPassesThresholds(120, 600, 120, 600)).toBeTrue();
    expect(voiceLevelPassesThresholds(119, 1_000, 120, 600)).toBeFalse();
    expect(voiceLevelPassesThresholds(500, 599, 120, 600)).toBeFalse();
    expect(voiceLevelPassesThresholds(0, 0, 120, 600)).toBeFalse();
  });
});
