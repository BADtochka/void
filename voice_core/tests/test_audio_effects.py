import unittest

import numpy as np

from voice_core.audio_effects import apply_robotic_voice_effect


class RoboticVoiceEffectTests(unittest.TestCase):
    def test_empty_audio_remains_empty(self) -> None:
        result = apply_robotic_voice_effect(
            np.empty(0, dtype=np.float32),
            48_000,
            pitch_semitones=-1.5,
            harmony_volume=0.1,
            modulation_hz=35.0,
            modulation_depth=0.07,
            reverb_amount=0.06,
        )

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.size, 0)

    def test_effect_preserves_shape_and_pcm_range(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        source = (0.25 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)

        result = apply_robotic_voice_effect(
            source,
            sample_rate,
            pitch_semitones=-1.5,
            harmony_volume=0.1,
            modulation_hz=35.0,
            modulation_depth=0.07,
            reverb_amount=0.06,
        )

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(np.max(np.abs(result))), 1.0)
        self.assertGreater(float(np.max(np.abs(result - source))), 0.01)


if __name__ == "__main__":
    unittest.main()
