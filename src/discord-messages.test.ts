import { describe, expect, test } from "bun:test";
import { splitDiscordMessage } from "./discord-messages.js";

describe("Discord chat messages", () => {
  test("keeps a short message intact", () => {
    expect(splitDiscordMessage("Короткий ответ.")).toEqual(["Короткий ответ."]);
  });

  test("splits long messages at word boundaries", () => {
    const parts = splitDiscordMessage("один два три четыре пять", 12);

    expect(parts).toEqual(["один два три", "четыре пять"]);
    expect(parts.every((part) => part.length <= 12)).toBeTrue();
  });
});
