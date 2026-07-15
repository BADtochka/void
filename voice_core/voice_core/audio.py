from __future__ import annotations

import numpy as np

DISCORD_SAMPLE_RATE = 48_000
DISCORD_CHANNELS = 2
STT_SAMPLE_RATE = 16_000


def limit_discord_pcm(data: bytes, max_seconds: int) -> tuple[bytes, bool]:
    max_bytes = DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * 2 * max_seconds
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


def discord_pcm_to_whisper(data: bytes) -> np.ndarray:
    """Convert signed 16-bit 48 kHz stereo PCM to float32 16 kHz mono."""
    samples = np.frombuffer(data, dtype="<i2")
    complete_frames = samples.size - (samples.size % DISCORD_CHANNELS)
    if complete_frames == 0:
        return np.empty(0, dtype=np.float32)

    stereo = samples[:complete_frames].reshape(-1, DISCORD_CHANNELS).astype(np.float32)
    mono_48k = stereo.mean(axis=1)

    # 48 kHz is an exact multiple of 16 kHz. Averaging each three-sample window
    # is sufficient for the speech-only prototype and avoids another native dependency.
    complete_windows = mono_48k.size - (mono_48k.size % 3)
    mono_16k = mono_48k[:complete_windows].reshape(-1, 3).mean(axis=1)
    return np.ascontiguousarray(mono_16k / 32768.0, dtype=np.float32)


def float_mono_to_discord_pcm(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert floating point mono audio to 48 kHz stereo signed 16-bit PCM."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return b""

    if sample_rate != DISCORD_SAMPLE_RATE:
        duration = mono.size / sample_rate
        source_x = np.linspace(0.0, duration, mono.size, endpoint=False)
        target_size = max(1, round(duration * DISCORD_SAMPLE_RATE))
        target_x = np.linspace(0.0, duration, target_size, endpoint=False)
        mono = np.interp(target_x, source_x, mono).astype(np.float32)

    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    return np.repeat(pcm[:, None], DISCORD_CHANNELS, axis=1).tobytes()
