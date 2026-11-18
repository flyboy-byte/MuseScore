#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end regression test: synthesize clips with known answers, transcribe,
and check accuracy. Needs the engine environment (run setup_engine.py first).

    python test_engine.py            # clean clips must be exact
"""
import difflib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audio2shapenote as engine  # noqa: E402
import make_test_clip  # noqa: E402


def accuracy(truth, report):
    from music21 import pitch
    tot = ok = 0
    for name, notes in truth["parts"].items():
        got = [(pitch.Pitch(n["pitch"]).midi, n["beats"]) for n in report["notes"] if n["voice"] == name]
        want = [(m, float(b)) for m, b in notes]
        sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
        ok += sum(b.size for b in sm.get_matching_blocks())
        tot += len(want)
    return ok / tot


def check(kind, voices):
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / f"{kind}.wav"
        make_test_clip.main(kind, str(wav))
        truth = json.loads(wav.with_suffix(".truth.json").read_text())
        rep = Path(d) / "r.json"
        code = engine.main([str(wav), "-o", str(Path(d) / "o.musicxml"), "--report", str(rep), "-q"])
        assert code == 0, f"{kind}: engine exit {code}"
        r = json.loads(rep.read_text())
        acc = accuracy(truth, r)
        print(f"{kind}: voices={len(r['voices'])} time={r['time_signature']} pickup={r['pickup_beats']} "
              f"bpm={r['bpm']} sharps={r['sharps']} accuracy={acc:.0%}")
        assert len(r["voices"]) == voices
        assert r["time_signature"] == truth["time"]
        assert r["pickup_beats"] == truth["pickup"]
        assert r["sharps"] == truth["key_sharps"]
        assert abs(r["bpm"] - truth["bpm"]) <= 2
        assert acc == 1.0, f"{kind}: accuracy {acc:.0%}"
        xml = (Path(d) / "o.musicxml").read_text()
        assert xml.count("<notehead") == sum(len(n) for n in truth["parts"].values())


if __name__ == "__main__":
    check("melody", 1)
    check("chorale", 4)
    print("engine ok")
