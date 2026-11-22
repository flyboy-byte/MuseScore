#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end regression test: synthesize clips with known answers, transcribe,
and check accuracy. Needs the engine environment (run setup_engine.py first).

    python test_engine.py            # clean clips must be exact
"""
import argparse
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


def check_timing_grid():
    events = [(onset, onset + 0.2, 60, 1) for onset in (0, 0.35, 0.7, 1.05)]
    notes = engine.quantise(events, bpm=120, grid=2, t0=0)
    assert [n[0] for n in notes] == [0, 1, 3, 4]
    events = [(i * .35, i * .35 + .2, 60, 1) for i in range(250)]
    notes = engine.quantise(events, bpm=120, grid=2, t0=0)
    assert notes[-1][0] == round(events[-1][0] * 4)
    assert notes[-1][0] + notes[-1][1] == max(notes[-1][0] + 1, round(events[-1][1] * 4))


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


def check_score_review():
    from music21 import key, note, stream
    from score_review import apply_patches, compare, read_score

    def write(path, pitches):
        score = stream.Score()
        part = stream.Part()
        part.partName = "Lead"
        part.append(key.KeySignature(0))
        for midi in pitches:
            event = note.Note(midi)
            event.notehead = engine.SYLLABLES_7[engine.shape_degree(event.pitch, 0)]
            part.append(event)
        score.append(part)
        score.write("musicxml", fp=str(path))

    with tempfile.TemporaryDirectory() as d:
        reference, transcription, output, review = [Path(d) / f for f in (
            "reference.musicxml", "transcription.musicxml", "patched.musicxml", "review.json"
        )]
        write(reference, [60, 62, 64, 65, 67, 69, 71, 72])
        write(transcription, [60, 62, 66, 65, 67, 69, 71, 72])
        payload = compare(reference, transcription)
        assert len(payload["pitch_patches"]) == 1
        patch = payload["pitch_patches"][0]
        assert patch["reference_beat"] == "2" and patch["from"] == "E4" and patch["to"] == "F#4"
        review.write_text(json.dumps(payload), encoding="utf-8")
        apply_patches(reference, review, [patch["id"]], output)
        actual = read_score(output).parts[0].recurse().notes
        assert [n.pitch.midi for n in actual] == [60, 62, 66, 65, 67, 69, 71, 72]
        assert actual[2].notehead == "fa"
        assert read_score(reference).parts[0].recurse().notes[2].pitch.midi == 64


def check_recording(path, tolerance):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "result.musicxml"
        args = engine.parse_args([str(path), "--separate", "off", "-o", str(out)])
        report, code = engine.run(args)
        assert code == 0, f"transcription failed with exit code {code}"
        score_span = max(n["beat"] + n["beats"] for n in report["notes"]) * 60 / report["bpm"]
        audio_span = report["audio_event_span_seconds"]
        drift = score_span - audio_span
        print(f"{path.name}: audio events={audio_span:.2f}s score={score_span:.2f}s drift={drift:+.2f}s")
        assert abs(drift) <= tolerance, f"duration difference exceeded {tolerance:g}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="check a local recording for end-to-end timing drift")
    parser.add_argument("--tolerance", type=float, default=2.0, help="allowed duration difference in seconds")
    args = parser.parse_args()
    if args.audio:
        check_recording(args.audio, args.tolerance)
        sys.exit(0)
    check_timing_grid()
    check_score_review()
    check_pickup_with_late_entrance()
    check("melody", 1)
    check("chorale", 4)
    print("engine ok")
