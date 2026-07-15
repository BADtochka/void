from __future__ import annotations

import logging
import os
import site
from pathlib import Path

logger = logging.getLogger(__name__)

_DLL_DIRECTORY_HANDLES: list[object] = []
_configured = False


def discover_nvidia_dll_directories(
    site_package_roots: list[str | Path],
) -> list[Path]:
    directories: list[Path] = []
    for root in site_package_roots:
        site_packages = Path(root)
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for package in sorted(nvidia_root.iterdir()):
                binary_directory = package / "bin"
                if binary_directory.is_dir() and any(binary_directory.glob("*.dll")):
                    directories.append(binary_directory.resolve())

    return list(dict.fromkeys(directories))


def configure_windows_cuda_dlls() -> list[Path]:
    global _configured
    if os.name != "nt" or _configured:
        return []

    directories = discover_nvidia_dll_directories(site.getsitepackages())
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for directory in directories:
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))

    if directories:
        current_path = os.environ.get("PATH", "")
        configured_path = os.pathsep.join(str(directory) for directory in directories)
        os.environ["PATH"] = (
            f"{configured_path}{os.pathsep}{current_path}"
            if current_path
            else configured_path
        )
        logger.info(
            "Registered Windows CUDA DLL directories count=%s paths=%s",
            len(directories),
            ", ".join(str(directory) for directory in directories),
        )
    else:
        logger.warning(
            "No NVIDIA CUDA DLL directories found in the active Python environment"
        )

    _configured = True
    return directories
