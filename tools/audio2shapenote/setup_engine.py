#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Create the Python environment MuseScore's audio import uses.

    python3 setup_engine.py            # .venv next to this script
    python3 setup_engine.py --user     # .venv in the MuseScore user data folder
                                       # (use this when the app folder is read-only)

MuseScore looks for <engine folder>/.venv and <user data>/audio2shapenote/.venv
automatically; no settings change is needed afterwards.
"""
import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def user_data_dir():
    home = Path.home()
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        return base / "MuseScore" / "MuseScore5" / "audio2shapenote"
    if system == "Darwin":
        return home / "Library" / "Application Support" / "MuseScore" / "MuseScore5" / "audio2shapenote"
    base = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    return base / "MuseScore" / "MuseScore5" / "audio2shapenote"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--user", action="store_true", help="install into the user data folder")
    a = ap.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required")

    target = (user_data_dir() if a.user else HERE) / ".venv"
    print(f"creating {target}")
    venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(target)
    py = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def pip(*args):
        subprocess.check_call([str(py), "-m", "pip", "install", "--disable-pip-version-check", *args])

    pip("setuptools", "wheel")
    pip("-r", str(HERE / "requirements.txt"))
    # basic-pitch pins TensorFlow and old numpy; we run its bundled ONNX model instead.
    pip("--no-deps", "basic-pitch")
    subprocess.check_call([str(py), "-c",
                           "import basic_pitch, music21, librosa, onnxruntime; print('engine ready')"])
    print(f"done. Python for MuseScore: {py}")


if __name__ == "__main__":
    main()
