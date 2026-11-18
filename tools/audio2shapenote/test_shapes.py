#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Offline checks of the shape-note logic (no audio/model needed). Run: python test_shapes.py"""
from music21 import pitch

from audio2shapenote import SYLLABLES_4, SYLLABLES_7, shape_degree, spell


def syl(midi, sharps, table=SYLLABLES_7):
    return table[shape_degree(spell(midi, sharps), sharps)]


# D major (2 sharps): D=do, F#=mi, B=la, C#=ti
assert syl(62, 2) == "do" and syl(66, 2) == "mi" and syl(71, 2) == "la" and syl(61, 2) == "ti"
# A minor (0 sharps): shapes follow the relative major, so A=la, C=do
assert syl(69, 0) == "la" and syl(72, 0) == "do" and syl(64, 0) == "mi"
# F major (-1): Bb must be spelled flat and be fa
assert spell(70, -1).name == "B-" and syl(70, -1) == "fa"
# 4-shape: do/fa -> fa, re/sol -> sol, mi/la -> la, ti -> mi
assert [syl(m, 0, SYLLABLES_4) for m in (60, 62, 64, 65, 67, 69, 71)] == \
    ["fa", "so", "la", "fa", "so", "la", "mi"]
print("ok")
