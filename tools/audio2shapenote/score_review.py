#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Align a transcribed MusicXML score to a reference and review pitch patches."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from music21 import converter, note


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_score(path):
    score = converter.parse(str(path), format="musicxml")
    if not score.parts:
        raise ValueError(f"No parts found in {path}")
    return score


def pitch_profile(score, grid):
    frames = max(1, int(np.ceil(float(score.highestTime) * grid)))
    profile = np.zeros((12, frames))
    for part in score.parts:
        for event in part.recurse().notes:
            start = round(float(event.getOffsetInHierarchy(part)) * grid)
            stop = max(start + 1, round((float(event.getOffsetInHierarchy(part)) + float(event.quarterLength)) * grid))
            for pitch in event.pitches:
                profile[pitch.pitchClass, start:stop] = 1
    return profile


def align_scores(reference, transcription, grid=2):
    from librosa.sequence import dtw

    ref = pitch_profile(reference, grid)
    got = pitch_profile(transcription, grid)
    # An explicit rest channel keeps silence from resembling any harmony.
    ref = np.vstack((ref, ~ref.any(axis=0)))
    got = np.vstack((got, ~got.any(axis=0)))
    _, path = dtw(X=ref, Y=got, metric="cosine", global_constraints=True, band_rad=0.15,
                  weights_add=np.array([0, 0.15, 0.15]))
    matches = {}
    for ref_frame, got_frame in path[::-1]:
        matches.setdefault(int(ref_frame), []).append(int(got_frame))
    return {frame: positions[0] for frame, positions in matches.items() if len(positions) == 1}


def editable_notes(part):
    by_beat = {}
    for event in part.recurse().notes:
        beat = Fraction(event.getOffsetInHierarchy(part))
        by_beat.setdefault(beat, []).append(event)
    return {
        beat: events[0] for beat, events in by_beat.items()
        if len(events) == 1 and isinstance(events[0], note.Note) and not events[0].tie
        and not events[0].duration.isGrace
    }


def compare(reference_path, transcription_path):
    reference = read_score(reference_path)
    transcription = read_score(transcription_path)
    matches = align_scores(reference, transcription)
    patches = []
    def part_names(score):
        parts = {}
        for part in score.parts:
            if part.partName:
                parts.setdefault(part.partName.casefold(), []).append(part)
        return parts

    ref_names = part_names(reference)
    got_names = part_names(transcription)
    for index, part in enumerate(reference.parts):
        name = (part.partName or "").casefold()
        other = got_names.get(name, [])
        if len(other) != 1 or len(ref_names[name]) != 1:
            continue
        instruments = [p.getInstrument(returnDefault=False) for p in (part, other[0])]
        if any(inst and inst.transposition for inst in instruments):
            continue
        candidates = editable_notes(other[0])
        for beat, original in editable_notes(part).items():
            position = beat * 2
            if position.denominator != 1:
                continue
            frame = matches.get(int(position))
            if (frame is None or matches.get(int(position) - 1) == frame
                    or matches.get(int(position) + 1) == frame):
                continue
            candidate = candidates.get(Fraction(frame, 2))
            if candidate is None or candidate.quarterLength != original.quarterLength:
                continue
            if candidate.pitch.midi != original.pitch.midi:
                patches.append({
                    "id": len(patches) + 1,
                    "part": index + 1,
                    "part_name": part.partName,
                    "measure": original.measureNumber,
                    "reference_beat": str(beat),
                    "transcription_beat": frame / 2,
                    "from": original.pitch.nameWithOctave,
                    "to": candidate.pitch.nameWithOctave,
                })
    return {
        "reference_sha256": digest(reference_path),
        "transcription_sha256": digest(transcription_path),
        "alignment": [{"reference_beat": a / 2, "transcription_beat": b / 2} for a, b in matches.items()],
        "pitch_patches": patches,
    }


def apply_patches(reference_path, review_path, ids, output):
    from music21.key import KeySignature
    from music21.musicxml.m21ToXml import GeneralObjectExporter
    from music21.pitch import Pitch

    from audio2shapenote import SYLLABLES_4, SYLLABLES_7, shape_degree

    if output.resolve() == reference_path.resolve() or output.resolve() == review_path.resolve():
        raise ValueError("Choose a new output path")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if digest(reference_path) != review["reference_sha256"]:
        raise ValueError("The reference has changed since this review")
    reference = read_score(reference_path)
    patches = {p["id"]: p for p in review["pitch_patches"]}
    head_shapes = set(SYLLABLES_7)
    tables = [SYLLABLES_4, SYLLABLES_7]
    for event in reference.recurse().getElementsByClass(note.Note):
        if event.notehead in head_shapes:
            key = event.getContextByClass(KeySignature)
            degree = shape_degree(event.pitch, key.sharps if key else 0)
            tables = [table for table in tables if table[degree] == event.notehead]

    for patch_id in dict.fromkeys(ids):
        if patch_id not in patches:
            raise ValueError(f"Unknown patch ID: {patch_id}")
        patch = patches[patch_id]
        part = reference.parts[patch["part"] - 1]
        original = editable_notes(part).get(Fraction(patch["reference_beat"]))
        if original is None or original.pitch.nameWithOctave != patch["from"]:
            raise ValueError(f"Patch {patch_id} no longer identifies the original note")
        replacement = Pitch(patch["to"])
        if original.notehead in head_shapes:
            key = original.getContextByClass(KeySignature)
            degree = shape_degree(replacement, key.sharps if key else 0)
            shapes = {table[degree] for table in tables}
            if len(shapes) != 1:
                raise ValueError(f"Patch {patch_id} has an ambiguous shape-note scheme")
            original.notehead = shapes.pop()
        original.pitch = replacement
    contents = GeneralObjectExporter().parse(reference)
    with output.open("xb") as f:
        f.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("compare", help="write an alignment and tentative pitch changes")
    match.add_argument("reference", type=Path)
    match.add_argument("transcription", type=Path)
    match.add_argument("-o", "--output", type=Path, required=True)
    apply = commands.add_parser("apply", help="write a copy with the selected pitch changes")
    apply.add_argument("reference", type=Path)
    apply.add_argument("review", type=Path)
    apply.add_argument("--select", type=int, nargs="+", required=True)
    apply.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            result = compare(args.reference, args.transcription)
            with args.output.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            print(f"{len(result['pitch_patches'])} tentative pitch changes")
        else:
            apply_patches(args.reference, args.review, args.select, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
