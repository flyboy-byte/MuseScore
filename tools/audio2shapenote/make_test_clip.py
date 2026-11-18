#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Synthesize a short voice-like test melody (original tune, D major, 4/4, 90 bpm)."""
import sys
import numpy as np
import soundfile as sf

SR = 22050
BPM = 90
# (midi, beats) original melody: D4 F#4 A4 | B4 A4 F#4 D4 | E4 G4 B4 A4 | D4(held)
MELODY = [(62, 1), (66, 1), (69, 1), (71, 1), (69, 1), (66, 1), (62, 1), (64, 1),
          (67, 1), (71, 1), (69, 2), (62, 4)]


def voice(f0, dur):
    t = np.arange(int(SR * dur)) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.2)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
    y = sum(np.sin(h * ph) / h ** 1.2 for h in range(1, 9))
    env = np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.06)
    return y * env


def main(path):
    sec = 60 / BPM
    out = [np.zeros(int(SR * 0.3))]
    for midi, beats in MELODY:
        f0 = 440 * 2 ** ((midi - 69) / 12)
        d = beats * sec
        out.append(voice(f0, d * 0.96))
        out.append(np.zeros(int(SR * d * 0.04)))
    y = np.concatenate(out)
    sf.write(path, 0.5 * y / np.abs(y).max(), SR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_melody.wav")
