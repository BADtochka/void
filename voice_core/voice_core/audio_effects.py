from __future__ import annotations

import numpy as np


def apply_robotic_voice_effect(
    audio: np.ndarray,
    sample_rate: int,
    *,
    pitch_semitones: float,
    harmony_volume: float,
    modulation_hz: float,
    modulation_depth: float,
    reverb_amount: float,
) -> np.ndarray:
    if audio.size == 0:
        return np.asarray(audio, dtype=np.float32)

    from pedalboard import (
        Chorus,
        Compressor,
        HighpassFilter,
        Limiter,
        Pedalboard,
        PitchShift,
        Reverb,
    )

    source = np.nan_to_num(np.asarray(audio, dtype=np.float32)).reshape(-1)
    main = Pedalboard(
        [
            PitchShift(semitones=pitch_semitones),
            HighpassFilter(cutoff_frequency_hz=85.0),
            Compressor(
                threshold_db=-20.0,
                ratio=3.0,
                attack_ms=4.0,
                release_ms=80.0,
            ),
            Chorus(
                rate_hz=0.7,
                depth=0.12,
                centre_delay_ms=7.0,
                feedback=0.04,
                mix=0.08,
            ),
        ]
    )(source, sample_rate)

    if harmony_volume > 0:
        harmony = Pedalboard([PitchShift(semitones=7.0)])(source, sample_rate)
        main = main + harmony * harmony_volume

    if modulation_depth > 0:
        time = np.arange(main.size, dtype=np.float32) / float(sample_rate)
        carrier = np.sin(2.0 * np.pi * modulation_hz * time)
        main = main * (1.0 + carrier * modulation_depth)

    processed = Pedalboard(
        [
            Reverb(
                room_size=0.10,
                damping=0.78,
                wet_level=reverb_amount,
                dry_level=1.0 - reverb_amount,
                width=0.20,
            ),
            Limiter(threshold_db=-1.5, release_ms=80.0),
        ]
    )(np.asarray(main, dtype=np.float32), sample_rate)
    return np.clip(np.nan_to_num(processed), -1.0, 1.0).astype(np.float32)
