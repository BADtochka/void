export type VoiceCueType =
  | "request_recognized"
  | "image_prompt_ready"
  | "generation_started"
  | "followup_recognized"
  | "followup_stopped"
  | "followup_expired";

type Tone = { frequency: number; durationMs: number };

const SAMPLE_RATE = 48_000;
const CHANNELS = 2;

const CUES: Record<VoiceCueType, Tone[]> = {
  request_recognized: [
    { frequency: 620, durationMs: 65 },
    { frequency: 0, durationMs: 25 },
    { frequency: 820, durationMs: 75 },
  ],
  image_prompt_ready: [
    { frequency: 520, durationMs: 90 },
    { frequency: 0, durationMs: 35 },
    { frequency: 680, durationMs: 140 },
  ],
  generation_started: [{ frequency: 440, durationMs: 300 }],
  followup_recognized: [{ frequency: 720, durationMs: 220 }],
  followup_stopped: [
    { frequency: 680, durationMs: 60 },
    { frequency: 0, durationMs: 25 },
    { frequency: 440, durationMs: 90 },
  ],
  followup_expired: [
    { frequency: 560, durationMs: 55 },
    { frequency: 0, durationMs: 20 },
    { frequency: 360, durationMs: 80 },
  ],
};

export function createVoiceCue(type: VoiceCueType, volume: number): Buffer {
  const amplitude = Math.round(32767 * Math.min(Math.max(volume, 0), 0.2));
  const parts = CUES[type].map((tone) => createTone(tone, amplitude));
  return Buffer.concat(parts);
}

function createTone(tone: Tone, amplitude: number): Buffer {
  const frames = Math.round((SAMPLE_RATE * tone.durationMs) / 1000);
  const output = Buffer.alloc(frames * CHANNELS * 2);
  const fadeFrames = Math.min(Math.round(SAMPLE_RATE * 0.008), Math.floor(frames / 2));

  for (let frame = 0; frame < frames; frame += 1) {
    let envelope = 1;
    if (fadeFrames > 0 && frame < fadeFrames) envelope = frame / fadeFrames;
    if (fadeFrames > 0 && frame >= frames - fadeFrames) {
      envelope = Math.min(envelope, (frames - frame - 1) / fadeFrames);
    }
    const sample =
      tone.frequency === 0
        ? 0
        : Math.round(
            Math.sin((2 * Math.PI * tone.frequency * frame) / SAMPLE_RATE) *
              amplitude *
              envelope,
          );
    output.writeInt16LE(sample, frame * 4);
    output.writeInt16LE(sample, frame * 4 + 2);
  }
  return output;
}
