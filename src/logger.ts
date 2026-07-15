type LogFields = Record<string, string | number | boolean | null | undefined>;

function formatFields(fields: LogFields): string {
  return Object.entries(fields)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
}

function line(level: string, event: string, fields: LogFields): string {
  const suffix = formatFields(fields);
  return `${new Date().toISOString()} ${level} ${event}${suffix ? ` ${suffix}` : ""}`;
}

export function logInfo(event: string, fields: LogFields = {}): void {
  console.log(line("INFO", event, fields));
}

export function logWarn(event: string, fields: LogFields = {}): void {
  console.warn(line("WARN", event, fields));
}

export function logError(event: string, error: unknown, fields: LogFields = {}): void {
  console.error(line("ERROR", event, fields), error);
}
