#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end regression test: synthesize clips with known answers, transcribe,
and check accuracy. Needs the engine environment (run setup_engine.py first).

    python test_engine.py            # clean clips must be exact
"""
from collections import Counter
import json
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audio2shapenote as engine  # noqa: E402
import make_test_clip  # noqa: E402


def note_counts(truth, report):
    from music21 import pitch

    expected = Counter()
    for name, notes in truth["parts"].items():
        onset = 0
        for midi, duration in notes:
            expected[name, midi, onset, duration] += 1
            onset += duration

    actual = Counter(
        (n["voice"], pitch.Pitch(n["pitch"]).midi, n["beat"], n["beats"])
        for n in report["notes"]
    )
    return (expected & actual).total(), expected.total(), actual.total()


def check(kind, voices):
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / f"{kind}.wav"
        make_test_clip.main(kind, str(wav))
        truth = json.loads(wav.with_suffix(".truth.json").read_text())
        rep = Path(d) / "r.json"
        code = engine.main([str(wav), "-o", str(Path(d) / "o.musicxml"), "--report", str(rep), "-q"])
        assert code == 0, f"{kind}: engine exit {code}"
        r = json.loads(rep.read_text())
        matched, expected, actual = note_counts(truth, r)
        print(f"{kind}: voices={len(r['voices'])} time={r['time_signature']} pickup={r['pickup_beats']} "
              f"bpm={r['bpm']} sharps={r['sharps']} matched={matched}/{expected} emitted={actual}")
        assert len(r["voices"]) == voices
        assert r["time_signature"] == truth["time"]
        assert r["pickup_beats"] == truth["pickup"]
        assert r["sharps"] == truth["key_sharps"]
        assert abs(r["bpm"] - truth["bpm"]) <= 2
        assert matched == expected == actual
        xml = ElementTree.parse(Path(d) / "o.musicxml")
        for part in xml.findall("part"):
            first_measure = part.find("measure")
            if truth["pickup"]:
                assert first_measure.get("number") == "0"
                assert first_measure.get("implicit") == "yes"
                divisions = int(first_measure.findtext("attributes/divisions"))
                duration = sum(int(n.text) for n in first_measure.findall("note/duration"))
                assert duration == truth["pickup"] * divisions
            assert first_measure.findtext("attributes/key/fifths") == str(truth["key_sharps"])
        assert len(xml.findall(".//notehead")) == expected


def check_fixed_tempo():
    events = [(onset, onset + 0.2, 60, 1) for onset in (0, 0.35, 0.7, 1.05)]
    notes = engine.quantise(events, bpm=120, grid=2, t0=0, follow=0)
    assert [n[0] for n in notes] == [0, 1, 3, 4]


def check_pickup_with_late_entrance():
    parts = engine.ENSEMBLES["mixed"][2]
    score, _ = engine.build_score(
        [[(0, 1, 66)], [(2, 7, 50)]], parts, 90, 4, 1, 2, 2, "aikin", "Pickup"
    )
    with tempfile.TemporaryDirectory() as d:
        xml = ElementTree.parse(score.write("musicxml", fp=str(Path(d) / "pickup.musicxml")))

    for part in xml.findall("part"):
        measures = part.findall("measure")
        assert [measure.get("number") for measure in measures] == ["0", "1"]
        assert measures[0].get("implicit") == "yes"
        divisions = int(measures[0].findtext("attributes/divisions"))
        assert sum(int(n.text) for n in measures[0].findall("note/duration")) == divisions / 2
    assert xml.find("part[2]/measure/note/rest") is not None


if __name__ == "__main__":
    check_fixed_tempo()
    check_pickup_with_late_entrance()
    check("melody", 1)
    check("chorale", 4)
    print("engine ok")
