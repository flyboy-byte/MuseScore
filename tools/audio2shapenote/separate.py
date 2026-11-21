#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Vocal isolation with an MDX-Net ONNX model (UVR "Voc_FT"), numpy + onnxruntime only.

Removes guitar, banjo, mandolin, bass, etc. so the transcriber only hears singing.

    python separate.py song.mp3 vocals.wav

The model (about 65 MB) is downloaded once to the engine's model folder by
setup_engine.py, or on first use.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://github.com/TRvlvr/model_repo/releases/download/"
             "all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx")
MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"

# Parameters UVR uses for this model
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATE = 1.021
SR = 44100


def _user_models():
    import os
    import platform
    home = Path.home()
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    elif platform.system() == "Darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "audio2shapenote" / "models"


def _candidates():
    # next to this script (dev checkout / writable install), then the per-user folder
    # (read-only installs such as the AppImage; same folder setup_engine.py --user uses)
    return [Path(__file__).resolve().parent / "models", _user_models()]


def ensure_model(verbose=True, prefer_user=False):
    dirs = _candidates()
    if prefer_user:
        dirs = dirs[::-1]
    for d in dirs:
        if (d / MODEL_NAME).exists():
            return d / MODEL_NAME
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            path = d / MODEL_NAME
            if verbose:
                print(f"downloading vocal model to {path}", file=sys.stderr)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.rename(path)
            return path
        except OSError:
            continue
    raise OSError("no writable folder for the vocal model")


def _stft(x):
    import librosa
    # x: (channels, samples) -> (channels, bins, frames), like torch.stft(center=True, reflect)
    return np.stack([librosa.stft(c, n_fft=N_FFT, hop_length=HOP, window="hann", center=True,
                                  pad_mode="reflect") for c in x])


def _istft(S, length):
    import librosa
    return np.stack([librosa.istft(c, hop_length=HOP, window="hann", center=True, length=length) for c in S])


def separate_vocals(audio_path, threads=0):
    """Return (vocals, sr) as a stereo float32 array (2, n) at 44.1 kHz."""
    import librosa
    import onnxruntime as ort

    mix, _ = librosa.load(str(audio_path), sr=SR, mono=False)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    mix = mix[:2].astype(np.float32)
    n = mix.shape[1]

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(ensure_model()), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    trim = N_FFT // 2
    chunk = HOP * (DIM_T - 1)
    gen = chunk - 2 * trim
    pad = gen - n % gen
    mixp = np.concatenate([np.zeros((2, trim), np.float32), mix,
                           np.zeros((2, pad), np.float32), np.zeros((2, trim), np.float32)], axis=1)

    out = []
    for i in range(0, n + pad, gen):
        wave = mixp[:, i:i + chunk]
        S = _stft(wave)[:, :DIM_F, :DIM_T]                # (2, F, T) complex
        x = np.stack([S[0].real, S[0].imag, S[1].real, S[1].imag])[None].astype(np.float32)
        y = sess.run(None, {inp: x})[0][0]                # (4, F, T)
        V = np.zeros((2, N_FFT // 2 + 1, DIM_T), np.complex64)
        V[0, :DIM_F] = y[0] + 1j * y[1]
        V[1, :DIM_F] = y[2] + 1j * y[3]
        w = _istft(V, chunk)
        out.append(w[:, trim:-trim])
    voc = np.concatenate(out, axis=1)[:, :n] * COMPENSATE
    return voc, SR


def main(argv):
    import soundfile as sf
    voc, sr = separate_vocals(argv[1])
    sf.write(argv[2], voc.T, sr)
    print(f"wrote {argv[2]}")


if __name__ == "__main__":
    main(sys.argv)
