export function voiceLevelPassesThresholds(
  rmsAmplitude: number,
  peakAmplitude: number,
  minimumRms: number,
  minimumPeak: number,
): boolean {
  return rmsAmplitude >= minimumRms && peakAmplitude >= minimumPeak;
}
