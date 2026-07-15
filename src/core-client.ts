import { config } from "./config.js";

type TurnMetadata = {
  guildId: string;
  channelId: string;
  userId: string;
  displayName: string;
  userIsAdmin: boolean;
  audioAgeMs: number;
  earlyHotwordDetected: boolean;
  image?: ImagePrompt;
};

export type ImagePrompt = {
  data: Buffer;
  contentType: string;
  name: string;
};

export type HotwordResult = { detected: boolean; transcript: string; busy?: boolean };
export type UserDirectoryEntry = { userId: string; displayName: string };
export type TtsVoice = { id: string; label: string; engine: string };
export type TtsEffect = { id: string; label: string };
export type WebSearchAccessUser = { userId: string; displayName: string | null };

export async function detectHotword(
  pcm: Buffer,
  final = false,
): Promise<HotwordResult> {
  const url = new URL(`${config.voiceCoreUrl}/v1/hotword`);
  if (final) url.searchParams.set("final", "true");
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: new Uint8Array(pcm),
    signal: AbortSignal.timeout(
      final ? config.coreTimeoutMs : config.hotwordTimeoutMs,
    ),
  });
  if (!response.ok) {
    throw new Error(`voice-core hotword detector returned ${response.status}`);
  }
  return (await response.json()) as HotwordResult;
}

export async function interruptGeneration(guildId: string): Promise<void> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/generations/${encodeURIComponent(guildId)}/interrupt`,
    {
      method: "POST",
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core interrupt returned ${response.status}`);
  }
}

export async function startFollowupCountdown(
  guildId: string,
  channelId: string,
  userId: string,
): Promise<void> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/users/${encodeURIComponent(userId)}/followup/playback-finished`,
    {
      method: "POST",
      headers: { "x-channel-id": channelId },
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core follow-up countdown returned ${response.status}`);
  }
}

export async function syncUserDirectory(
  guildId: string,
  users: UserDirectoryEntry[],
): Promise<void> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/users`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(users),
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core user directory sync returned ${response.status}`);
  }
}

export async function processTurn(
  pcm: Buffer,
  metadata: TurnMetadata,
  signal?: AbortSignal,
): Promise<Buffer | null> {
  const body = metadata.image ? Buffer.concat([pcm, metadata.image.data]) : pcm;
  const headers: Record<string, string> = {
    "content-type": "application/octet-stream",
    "x-guild-id": metadata.guildId,
    "x-channel-id": metadata.channelId,
    "x-user-id": metadata.userId,
    "x-display-name": encodeURIComponent(metadata.displayName),
    "x-user-is-admin": String(metadata.userIsAdmin),
    "x-audio-age-ms": String(Math.round(metadata.audioAgeMs)),
    "x-early-hotword-detected": String(metadata.earlyHotwordDetected),
  };
  if (metadata.image) {
    headers["x-audio-byte-length"] = String(pcm.length);
    headers["x-image-content-type"] = metadata.image.contentType;
  }
  const response = await fetch(`${config.voiceCoreUrl}/v1/turn`, {
    method: "POST",
    headers,
    body: new Uint8Array(body),
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(config.coreTimeoutMs)])
      : AbortSignal.timeout(config.coreTimeoutMs),
  });

  if (response.status === 204) return null;
  if (!response.ok) {
    throw new Error(`voice-core returned ${response.status}: ${await response.text()}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

export async function resetConversation(guildId: string): Promise<void> {
  const response = await fetch(`${config.voiceCoreUrl}/v1/conversations/${guildId}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(5_000),
  });
  if (!response.ok) throw new Error(`voice-core returned ${response.status}`);
}

export async function getCoreHealth(): Promise<string> {
  const response = await fetch(`${config.voiceCoreUrl}/health`, {
    signal: AbortSignal.timeout(5_000),
  });
  if (!response.ok) throw new Error(`voice-core returned ${response.status}`);
  const data = (await response.json()) as { status: string };
  return data.status;
}

export async function getTtsVoices(
  guildId: string,
): Promise<{
  selected: string;
  voices: TtsVoice[];
  selectedEffect: string;
  effects: TtsEffect[];
}> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/tts`,
    { signal: AbortSignal.timeout(5_000) },
  );
  if (!response.ok) throw new Error(`voice-core TTS list returned ${response.status}`);
  return (await response.json()) as {
    selected: string;
    voices: TtsVoice[];
    selectedEffect: string;
    effects: TtsEffect[];
  };
}

export async function setTtsVoice(guildId: string, voiceId: string): Promise<TtsVoice> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/tts`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ voiceId }),
      signal: AbortSignal.timeout(config.coreTimeoutMs),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core TTS selection returned ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as TtsVoice;
}

export async function setTtsEffect(guildId: string, effectId: string): Promise<TtsEffect> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/tts/effect`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ effectId }),
      signal: AbortSignal.timeout(config.coreTimeoutMs),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core TTS effect returned ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as TtsEffect;
}

export async function listWebSearchAccess(guildId: string): Promise<WebSearchAccessUser[]> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/web-search-access`,
    {
      headers: { "x-requester-is-admin": "true" },
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core web access list returned ${response.status}: ${await response.text()}`);
  }
  const result = (await response.json()) as { users: WebSearchAccessUser[] };
  return result.users;
}

export async function grantWebSearchAccess(
  guildId: string,
  userId: string,
  displayName: string,
): Promise<void> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/web-search-access/${encodeURIComponent(userId)}`,
    {
      method: "PUT",
      headers: {
        "content-type": "application/json",
        "x-requester-is-admin": "true",
      },
      body: JSON.stringify({ displayName }),
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core web access grant returned ${response.status}: ${await response.text()}`);
  }
}

export async function revokeWebSearchAccess(guildId: string, userId: string): Promise<void> {
  const response = await fetch(
    `${config.voiceCoreUrl}/v1/guilds/${encodeURIComponent(guildId)}/web-search-access/${encodeURIComponent(userId)}`,
    {
      method: "DELETE",
      headers: { "x-requester-is-admin": "true" },
      signal: AbortSignal.timeout(5_000),
    },
  );
  if (!response.ok) {
    throw new Error(`voice-core web access revoke returned ${response.status}: ${await response.text()}`);
  }
}
