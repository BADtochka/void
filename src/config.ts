function required(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function positiveInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const value = Number.parseInt(raw, 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function nonNegativeInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const value = Number.parseInt(raw, 10);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

function boundedFloat(name: string, fallback: number, minimum: number, maximum: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be between ${minimum} and ${maximum}`);
  }
  return value;
}

export const config = {
  discordToken: required("DISCORD_TOKEN"),
  discordGuildId: process.env.DISCORD_GUILD_ID?.trim(),
  voiceCoreUrl: (process.env.VOICE_CORE_URL ?? "http://127.0.0.1:8765").replace(/\/$/, ""),
  silenceMs: positiveInteger("DISCORD_SILENCE_MS", 850),
  minVoiceRms: nonNegativeInteger("DISCORD_MIN_VOICE_RMS", 120),
  minVoicePeak: nonNegativeInteger("DISCORD_MIN_VOICE_PEAK", 600),
  audioCueVolume: boundedFloat("AUDIO_CUE_VOLUME", 0.045, 0, 0.2),
  hotwordCheckIntervalMs: positiveInteger("HOTWORD_CHECK_INTERVAL_MS", 650),
  hotwordMinAudioMs: positiveInteger("HOTWORD_MIN_AUDIO_MS", 600),
  hotwordTimeoutMs: positiveInteger("HOTWORD_TIMEOUT_MS", 5_000),
  maxUtteranceSeconds: positiveInteger("MAX_UTTERANCE_SECONDS", 30),
  coreTimeoutMs: positiveInteger("VOICE_CORE_TIMEOUT_MS", 600_000),
};
