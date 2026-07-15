export type VoiceActivityStats = {
  rmsAmplitude: number;
  peakAmplitude: number;
  voicedSamples: number;
  totalSamples: number;
  durationSeconds: number;
  voicedMs: number;
  voicedRatio: number;
};

export function voiceLevelPassesThresholds(
  rmsAmplitude: number,
  peakAmplitude: number,
  minimumRms: number,
  minimumPeak: number,
): boolean {
  return rmsAmplitude >= minimumRms && peakAmplitude >= minimumPeak;
}

export function voiceCaptureLooksLikeSpeech(
  stats: VoiceActivityStats,
  options: {
    minimumRms: number;
    minimumPeak: number;
    minimumVoicedMs: number;
    minimumVoicedRatio: number;
  },
): boolean {
  if (
    !voiceLevelPassesThresholds(
      stats.rmsAmplitude,
      stats.peakAmplitude,
      options.minimumRms,
      options.minimumPeak,
    )
  ) {
    return false;
  }
  return (
    stats.voicedMs >= options.minimumVoicedMs &&
    stats.voicedRatio >= options.minimumVoicedRatio
  );
}

export function buildVoiceActivityStats(
  energySum: number,
  energySamples: number,
  peakAmplitude: number,
  voicedSamples: number,
  durationSeconds: number,
): VoiceActivityStats {
  const totalSamples = Math.max(0, energySamples);
  const voiced = Math.max(0, Math.min(voicedSamples, totalSamples));
  const voicedRatio = totalSamples > 0 ? voiced / totalSamples : 0;
  return {
    rmsAmplitude: totalSamples > 0 ? Math.sqrt(energySum / totalSamples) : 0,
    peakAmplitude,
    voicedSamples: voiced,
    totalSamples,
    durationSeconds,
    voicedMs: voicedRatio * Math.max(0, durationSeconds) * 1_000,
    voicedRatio,
  };
}

/** Per-sample amplitude floor used to count "voiced" activity during capture. */
export function voicedSampleFloor(minimumPeak: number): number {
  return Math.max(180, Math.floor(minimumPeak * 0.4));
}
