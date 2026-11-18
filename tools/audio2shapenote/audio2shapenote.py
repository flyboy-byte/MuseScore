#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""
Proof of concept: audio -> shape-note MusicXML (opens in MuseScore).

Pipeline: audio -> Basic Pitch (ONNX) note events -> melody cleanup -> tempo/key
estimate -> grid quantisation -> key-aware spelling -> MusicXML whose <notehead>
values are the shape-note syllables MuseScore already understands.

Scope of this PoC: single melodic line (e.g. one sung part), fixed time signature.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SYLLABLES_7 = ["do", "re", "mi", "fa", "so", "la", "ti"]
# 4-shape: fa sol la fa sol la mi (degrees 0..6)
SYLLABLES_4 = ["fa", "so", "la", "fa", "so", "la", "mi"]
LETTERS = "CDEFGAB"


def transcribe(audio_path, onset=0.5, frame=0.3, min_ms=120.0):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    onnx = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
    model = Model(onnx if onnx.exists() else ICASSP_2022_MODEL_PATH)
    _, _, events = predict(
        str(audio_path), model,
        onset_threshold=onset, frame_threshold=frame,
        minimum_note_length=min_ms, melodia_trick=True,
    )
    # (start_s, end_s, midi, amplitude, bends)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]


def to_melody(events, tol=0.04):
    """Greedy monophonic reduction: strongest notes win, overlaps are dropped."""
    accepted = []
    for s, e, p, a in sorted(events, key=lambda n: -n[3]):
        if all(e <= s2 + tol or s >= e2 - tol for s2, e2, _, _ in accepted):
            accepted.append((s, e, p, a))
    accepted.sort()
    # trim tiny overlaps
    out = []
    for i, (s, e, p, a) in enumerate(accepted):
        if i + 1 < len(accepted):
            e = min(e, accepted[i + 1][0])
        if e > s:
            out.append((s, e, p, a))
    return out


def estimate_bpm(audio_path, lo=60, hi=140):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    while bpm > hi:
        bpm /= 2
    while bpm < lo:
        bpm *= 2
    return int(round(bpm))


def quantise(events, bpm, subdiv=2):
    """Return [(start_units, dur_units, midi)] on a grid of 1/subdiv beat."""
    unit_s = 60.0 / bpm / subdiv
    t0 = events[0][0]  # assume first note starts the first bar
    notes = []
    cursor = 0
    for s, e, p, _ in events:
        su = max(cursor, int(round((s - t0) / unit_s)))
        eu = max(su + 1, int(round((e - t0) / unit_s)))
        notes.append([su, eu - su, p])
        cursor = eu
    return notes


def pick_key(midis_with_dur):
    """Krumhansl-Schmuckler via music21 on duration-weighted pitches."""
    from music21 import analysis, note, stream

    s = stream.Stream()
    for midi, dur in midis_with_dur:
        n = note.Note(midi)
        n.quarterLength = max(0.25, dur)
        s.append(n)
    k = analysis.discrete.KrumhanslSchmuckler().getSolution(s)
    return k


def spell(midi, sharps):
    from music21 import pitch

    p = pitch.Pitch(midi=midi)
    if sharps < 0 and p.accidental is not None and p.accidental.alter > 0:
        p = p.getEnharmonic()
    return p


def shape_degree(p, sharps):
    """Degree of the note's letter relative to the MAJOR tonic of the key signature.
    (Same convention as MuseScore: la = tonic of the relative minor.)"""
    from music21 import key as m21key

    major = m21key.KeySignature(sharps).asKey("major")
    return (LETTERS.index(p.step) - LETTERS.index(major.tonic.step)) % 7


def build_score(notes, bpm, time_sig, key_obj, system, title, grid=2):
    from music21 import clef, metadata, meter, note, stream, tempo
    from music21 import key as m21key

    sharps = key_obj.sharps
    sylls = SYLLABLES_4 if system == "4shape" else SYLLABLES_7
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    part.insert(0, m21key.KeySignature(sharps))
    ts = meter.TimeSignature(time_sig)
    part.insert(0, ts)
    part.insert(0, tempo.MetronomeMark(number=bpm))

    cursor = 0
    report = []
    for su, du, midi in notes:
        if su > cursor:
            r = note.Rest()
            r.quarterLength = (su - cursor) / grid
            part.insert(cursor / grid, r)
        p = spell(midi, sharps)
        n = note.Note(p)
        n.quarterLength = du / grid
        deg = shape_degree(p, sharps)
        n.notehead = sylls[deg]
        part.insert(su / grid, n)
        report.append((su / grid, p.nameWithOctave, du / grid, sylls[deg]))
        cursor = su + du

    part.makeMeasures(inPlace=True)
    sc = stream.Score()
    sc.metadata = metadata.Metadata(title=title)
    sc.insert(0, part)
    return sc, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", help="output .musicxml (default: alongside input)")
    ap.add_argument("--system", choices=["aikin", "funk", "walker", "4shape"], default="aikin")
    ap.add_argument("--bpm", type=int, help="override tempo estimate")
    ap.add_argument("--time-sig", default="4/4")
    ap.add_argument("--grid", type=int, default=2, choices=[1, 2, 4],
                    help="rhythm grid: notes per beat (2 = eighth notes)")
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--frame", type=float, default=0.3)
    ap.add_argument("--min-ms", type=float, default=120.0)
    ap.add_argument("--title")
    a = ap.parse_args(argv)

    audio = Path(a.audio)
    out = Path(a.output) if a.output else audio.with_suffix(".musicxml")

    ev = transcribe(audio, a.onset, a.frame, a.min_ms)
    mel = to_melody(ev)
    if not mel:
        sys.exit("no notes detected")
    bpm = a.bpm or estimate_bpm(audio)
    notes = quantise(mel, bpm, a.grid)
    key_obj = pick_key([(m, d / a.grid) for _, d, m in notes])
    sc, report = build_score(notes, bpm, a.time_sig, key_obj, a.system,
                             a.title or audio.stem, a.grid)
    sc.write("musicxml", fp=str(out))

    print(f"events={len(ev)} melody={len(mel)} bpm={bpm} key={key_obj.tonic.name} {key_obj.mode} "
          f"sharps={key_obj.sharps} system={a.system}")
    for off, name, dur, syl in report:
        print(f"  beat {off:6.2f}  {name:4s} dur {dur:4.2f}  {syl}")
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
