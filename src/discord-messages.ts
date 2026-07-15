export function splitDiscordMessage(content: string, limit = 2_000): string[] {
  const parts: string[] = [];
  let remaining = content.trim();
  while (remaining.length > limit) {
    const candidate = remaining.slice(0, limit + 1);
    const newline = candidate.lastIndexOf("\n");
    const space = candidate.lastIndexOf(" ");
    const boundary = Math.max(newline, space);
    const splitAt = boundary >= Math.floor(limit / 2) ? boundary : limit;
    parts.push(remaining.slice(0, splitAt).trimEnd());
    remaining = remaining.slice(splitAt).trimStart();
  }
  if (remaining) parts.push(remaining);
  return parts;
}
