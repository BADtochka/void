export type MicrophoneActivity = {
  activeFollowups: number;
  pendingWakes: number;
  pendingImages: number;
  processingTurns: number;
  queuedPlayback: number;
  playbackActive: boolean;
};

export function microphoneIsNeeded(activity: MicrophoneActivity): boolean {
  return (
    activity.activeFollowups > 0 ||
    activity.pendingWakes > 0 ||
    activity.pendingImages > 0 ||
    activity.processingTurns > 0 ||
    activity.queuedPlayback > 0 ||
    activity.playbackActive
  );
}
