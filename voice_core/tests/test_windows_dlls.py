import tempfile
import unittest
from pathlib import Path

from voice_core.windows_dlls import discover_nvidia_dll_directories


class WindowsDllDiscoveryTests(unittest.TestCase):
    def test_discovers_nvidia_package_bins_with_dlls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site_packages = Path(directory)
            cublas = site_packages / "nvidia" / "cublas" / "bin"
            runtime = site_packages / "nvidia" / "cuda_runtime" / "bin"
            empty = site_packages / "nvidia" / "empty" / "bin"
            for path in (cublas, runtime, empty):
                path.mkdir(parents=True)
            (cublas / "cublas64_12.dll").touch()
            (runtime / "cudart64_12.dll").touch()

            discovered = discover_nvidia_dll_directories([site_packages])

        self.assertEqual(
            discovered,
            [cublas.resolve(), runtime.resolve()],
        )

    def test_ignores_missing_site_packages(self) -> None:
        self.assertEqual(discover_nvidia_dll_directories(["/missing/path"]), [])


if __name__ == "__main__":
    unittest.main()
