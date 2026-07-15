import { describe, expect, test } from "bun:test";
import {
  captureAllowedForUser,
  captureBlockedByCooldown,
  hotwordActionForCapture,
  userHasActiveTurn,
} from "./voice-turn-state.js";

describe("voice turn ownership", () => {
  test("only the owner is captured while hotword waits for content", () => {
    expect(captureAllowedForUser("owner", "owner")).toBe(true);
    expect(captureAllowedForUser("owner", "other")).toBe(false);
    expect(captureAllowedForUser(null, "other")).toBe(true);
  });

  test("active, pending, and processing users do not start another hotword", () => {
    expect(userHasActiveTurn("active", new Set(["active"]), new Set(), [])).toBe(true);
    expect(userHasActiveTurn("pending", new Set(), new Set(["pending"]), [])).toBe(true);
    expect(userHasActiveTurn("processing", new Set(), new Set(), ["processing"])).toBe(true);
    expect(userHasActiveTurn("idle", new Set(), new Set(), [])).toBe(false);
  });

  test("hotword interrupts an active turn without activating another one", () => {
    expect(hotwordActionForCapture(true, true)).toBe("interrupt");
    expect(hotwordActionForCapture(true, false)).toBe("interrupt");
    expect(hotwordActionForCapture(false, true)).toBe("interrupt");
    expect(hotwordActionForCapture(false, false)).toBe("activate");
  });

  test("captures queued before or during dialogue cooldown are blocked", () => {
    expect(captureBlockedByCooldown(99, 100, 0)).toBe(true);
    expect(captureBlockedByCooldown(101, 100, 0)).toBe(false);
    expect(captureBlockedByCooldown(49, 0, 50)).toBe(true);
    expect(captureBlockedByCooldown(51, 0, 50)).toBe(false);
  });
});
