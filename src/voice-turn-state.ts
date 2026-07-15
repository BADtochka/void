export function captureAllowedForUser(
  awaitingContentUserId: string | null,
  userId: string,
): boolean {
  return awaitingContentUserId === null || awaitingContentUserId === userId;
}

export function userHasActiveTurn(
  userId: string,
  activeFollowupUsers: ReadonlySet<string>,
  pendingWakeUsers: ReadonlySet<string>,
  processingUsers: Iterable<string>,
): boolean {
  if (activeFollowupUsers.has(userId) || pendingWakeUsers.has(userId)) return true;
  for (const processingUserId of processingUsers) {
    if (processingUserId === userId) return true;
  }
  return false;
}

export function hotwordActionForCapture(
  wasActiveAtCaptureStart: boolean,
  isActiveNow: boolean,
): "activate" | "interrupt" {
  return wasActiveAtCaptureStart || isActiveNow ? "interrupt" : "activate";
}

export function captureBlockedByCooldown(
  captureStartedAt: number,
  globalCooldownUntil: number,
  userCooldownUntil: number,
): boolean {
  return captureStartedAt <= Math.max(globalCooldownUntil, userCooldownUntil);
}
