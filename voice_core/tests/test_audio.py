import unittest

import numpy as np

from voice_core.audio import discord_pcm_to_whisper, float_mono_to_discord_pcm, limit_discord_pcm


class AudioConversionTests(unittest.TestCase):
    def test_discord_pcm_is_downmixed_and_downsampled(self) -> None:
        frames = np.array([[3276, 3276]] * 480, dtype="<i2")
        audio = discord_pcm_to_whisper(frames.tobytes())

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (160,))
        self.assertAlmostEqual(float(audio.mean()), 3276 / 32768, places=4)

    def test_float_audio_is_returned_as_stereo_pcm(self) -> None:
        pcm = float_mono_to_discord_pcm(np.array([0.0, 0.5, -0.5], dtype=np.float32), 48_000)
        frames = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)

        self.assertEqual(frames.shape, (3, 2))
        np.testing.assert_array_equal(frames[:, 0], frames[:, 1])

    def test_long_discord_pcm_is_truncated_to_exact_limit(self) -> None:
        bytes_per_second = 48_000 * 2 * 2
        audio, truncated = limit_discord_pcm(b"x" * (bytes_per_second + 100), max_seconds=1)

        self.assertTrue(truncated)
        self.assertEqual(len(audio), bytes_per_second)


if __name__ == "__main__":
    unittest.main()
