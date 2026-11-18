#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Synthesize voice-like test clips (original tunes) plus a ground-truth JSON.

  python make_test_clip.py melody  out.wav   # D major, 4/4, 90 bpm, 1 voice
  python make_test_clip.py chorale out.wav   # G major, 3/4, 80 bpm, 4 voices, 1-beat pickup
"""
import json
import sys

import numpy as np
import soundfile as sf

SR = 22050

MELODY = {
    "bpm": 90, "key_sharps": 2, "time": "4/4", "pickup": 0,
    "parts": {"Melody": [(62, 1), (66, 1), (69, 1), (71, 1), (69, 1), (66, 1), (62, 1), (64, 1),
                         (67, 1), (71, 1), (69, 2), (62, 4)]},
}

# Original 4-part tune in G major, 3/4, one-beat pickup. (midi, beats)
CHORALE = {
    "bpm": 80, "key_sharps": 1, "time": "3/4", "pickup": 1,
    "parts": {
        "Treble": [(71, 1), (74, 2), (72, 1), (71, 2), (69, 1), (67, 2), (69, 1), (71, 3)],
        "Alto":   [(67, 1), (67, 2), (66, 1), (67, 2), (64, 1), (62, 2), (66, 1), (67, 3)],
        "Tenor":  [(62, 1), (62, 2), (57, 1), (59, 2), (60, 1), (59, 2), (60, 1), (62, 3)],
        "Bass":   [(55, 1), (55, 2), (50, 1), (52, 2), (48, 1), (50, 2), (50, 1), (43, 3)],
    },
}


def voice(f0, dur, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    rate = 5.0 + rng.uniform(0, 1.2)
    vib = 1 + 0.004 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)) * np.minimum(1, t / 0.25)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
    y = sum(np.sin(h * ph) / h ** 1.3 for h in range(1, 9))
    env = np.minimum(1, t / 0.04) * np.minimum(1, np.maximum(0, dur - t) / 0.07)
    return y * env


def render(spec):
    sec = 60 / spec["bpm"]
    lead = int(SR * 0.3)
    tracks = []
    for vi, notes in enumerate(spec["parts"].values()):
        out = [np.zeros(lead)]
        for i, (midi, beats) in enumerate(notes):
            f0 = 440 * 2 ** ((midi - 69) / 12)
            d = beats * sec
            out.append(voice(f0, d * 0.95, vi * 100 + i))
            out.append(np.zeros(int(SR * d) - int(SR * d * 0.95)))
        tracks.append(np.concatenate(out))
    n = max(len(t) for t in tracks)
    y = sum(np.pad(t, (0, n - len(t))) for t in tracks)
    return 0.5 * y / np.abs(y).max()


def main(kind, path):
    spec = {"melody": MELODY, "chorale": CHORALE}[kind]
    sf.write(path, render(spec), SR)
    with open(path.rsplit(".", 1)[0] + ".truth.json", "w") as f:
        json.dump(spec, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "melody",
         sys.argv[2] if len(sys.argv) > 2 else "test.wav")
