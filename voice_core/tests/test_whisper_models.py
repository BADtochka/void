import unittest
from types import SimpleNamespace

from voice_core.whisper_models import _download_size, _human_size


class WhisperModelTests(unittest.TestCase):
    def test_download_size_only_counts_runtime_files(self) -> None:
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename="model.bin", size=1_000),
                SimpleNamespace(rfilename="config.json", size=200),
                SimpleNamespace(rfilename="README.md", size=500),
            ]
        )
        self.assertEqual(_download_size(info), 1_200)

    def test_human_size(self) -> None:
        self.assertEqual(_human_size(75_500_000), "75.5 MB")
        self.assertEqual(_human_size(1_520_000_000), "1.5 GB")


if __name__ == "__main__":
    unittest.main()
