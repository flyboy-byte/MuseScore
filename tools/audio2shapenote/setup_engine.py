#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Create the Python environment MuseScore's audio import uses.

    python3 setup_engine.py            # .venv next to this script
    python3 setup_engine.py --user     # .venv in ~/.local/share/audio2shapenote (Linux),
                                       # %LOCALAPPDATA%\\audio2shapenote, ~/Library/Application Support/audio2shapenote;
                                       # use this for read-only installs such as the AppImage

Pass --upgrade to refresh dependencies in the selected environment.
MuseScore looks in both places automatically; no settings change is needed afterwards.
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
    """Same folder MuseScore checks: QStandardPaths::GenericDataLocation/audio2shapenote."""
    home = Path.home()
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    elif system == "Darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "audio2shapenote"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--user", action="store_true", help="install into the user data folder")
    ap.add_argument("--upgrade", action="store_true", help="upgrade dependencies in the selected environment")
    a = ap.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required")

    target = (user_data_dir() if a.user else HERE) / ".venv"
    py = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (target / "pyvenv.cfg").is_file() and py.is_file():
        print(f"reusing {target}")
    else:
        print(f"creating {target}")
        venv.EnvBuilder(with_pip=True).create(target)

    def pip(*args):
        subprocess.check_call([str(py), "-m", "pip", "install", "--disable-pip-version-check", *args])

    upgrade = ("--upgrade",) if a.upgrade else ()
    pip(*upgrade, "setuptools", "wheel")
    pip(*upgrade, "-r", str(HERE / "requirements.txt"))
    # basic-pitch pins TensorFlow and old numpy; we run its bundled ONNX model instead.
    pip(*upgrade, "--no-deps", "basic-pitch")
    subprocess.check_call([str(py), "-c",
                           "import basic_pitch, music21, librosa, onnxruntime; print('engine ready')"])
    # vocal-isolation model (~65 MB) used to strip instruments before transcribing
    subprocess.check_call([str(py), "-c",
                           "import sys; sys.path.insert(0, %r); import separate; "
                           "print('vocal model:', separate.ensure_model(prefer_user=%r))" % (str(HERE), a.user)])
    print(f"done. Python for MuseScore: {py}")


if __name__ == "__main__":
    main()
