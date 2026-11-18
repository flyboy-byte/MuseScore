#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""
audio2shapenote: turn a recording into shape-note sheet music (MusicXML).

Pipeline
  audio -> Basic Pitch (ONNX) note events
        -> harmonic/ghost-note cleanup
        -> tempo estimate (librosa prior, refined by grid fit) -> quantise to a beat grid
        -> voice assignment (1-4 voices: melody, or Treble/Alto/Tenor/Bass)
        -> key estimate + key-aware spelling
        -> meter + pickup estimate
        -> MusicXML; every note carries a <notehead> shape syllable
           (do re mi fa so la ti), the same values MuseScore writes for its
           shape-note schemes.

MuseScore's audio import module runs this script and then switches the staves to
the chosen dynamic notehead scheme (7-shape Aikin by default).

Exit codes: 0 ok, 2 bad arguments, 3 no notes found, 4 missing dependency, 5 other error.
"""
import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np

SYLLABLES_7 = ["do", "re", "mi", "fa", "so", "la", "ti"]
# 4-shape (Sacred Harp): fa so la fa so la mi
SYLLABLES_4 = ["fa", "so", "la", "fa", "so", "la", "mi"]
LETTERS = "CDEFGAB"

# Shape-note books order parts Treble, Alto, Tenor (melody), Bass.
VOICE_SETS = {
    1: [("Melody", (55, 79), "treble")],
    2: [("Treble", (57, 81), "treble"), ("Bass", (40, 64), "bass")],
    3: [("Treble", (57, 81), "treble"), ("Tenor", (48, 69), "treble8vb"), ("Bass", (40, 64), "bass")],
    4: [("Treble", (59, 81), "treble"), ("Alto", (53, 74), "treble"),
        ("Tenor", (48, 69), "treble8vb"), ("Bass", (40, 62), "bass")],
}


# ----------------------------------------------------------------------------- transcription

def transcribe(audio_path, onset=0.5, frame=0.3, min_ms=110.0, polyphonic=False):
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict
    except ImportError as e:  # pragma: no cover
        raise MissingDependency(f"basic-pitch is not installed ({e})")

    onnx = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
    model = Model(onnx if onnx.exists() else ICASSP_2022_MODEL_PATH)
    _, _, events = predict(
        str(audio_path), model,
        onset_threshold=onset, frame_threshold=frame,
        minimum_note_length=min_ms,
        # melodia trick helps single lines but deletes inner voices of chords
        melodia_trick=not polyphonic,
        minimum_frequency=60.0, maximum_frequency=1200.0,
    )
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]


def remove_harmonics(events, min_lead=0.12):
    """Drop overtone ghosts: a note an octave, twelfth or two octaves above a
    louder note that started earlier and is still sounding. Real doublings in
    sung parts start together with the lower note, so they are kept."""
    keep = []
    for i, (s, e, p, a) in enumerate(events):
        ghost = any(
            p - p2 in (12, 19, 24) and s2 <= s - min_lead and e2 > s + 0.05 and a < 0.9 * a2
            for j, (s2, e2, p2, a2) in enumerate(events) if j != i
        )
        if not ghost:
            keep.append((s, e, p, a))
    return keep


def thin_chord(items, n):
    """Reduce a chord to n notes: drop overtone-looking notes first, then the quietest."""
    items = dict(items)
    while len(items) > n:
        cands = [p for p in items if any(p - q in (12, 19, 24) and items[q][1] >= items[p][1] for q in items)]
        pool = cands or list(items)
        drop = min(pool, key=lambda p: items[p][1])
        del items[drop]
    return list(items.items())


def detect_onsets(audio_path):
    """Audio onset times (seconds) from spectral flux; used to tell real
    re-attacks from vibrato/reverb fragmentation."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr)
        return librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time", backtrack=False, delta=0.12)
    except Exception:
        return np.array([])


def merge_fragments(events, onsets, gap=0.16, tol=0.07, min_len=0.1):
    """Join consecutive same-pitch pieces unless the second piece starts at a
    real onset, then drop leftover slivers.

    A time counts as a real onset when the audio onset detector fires there, or
    when another voice changes to a NEW pitch there (parts in shape-note music
    move together, while vibrato/reverb fragments only re-trigger the same pitch).
    """
    onsets = np.asarray(onsets)
    events = sorted(events)

    def continues(ev):  # same pitch was sounding right before -> not a pitch change
        s, _, p, _ = ev
        return any(p2 == p and s - gap <= e2 <= s + tol and s2 < s - 0.02 for s2, e2, p2, _ in events)

    # pitch changes that last: short pieces are usually fragments themselves
    changes = [(ev[0], ev[2]) for ev in events if not continues(ev) and ev[1] - ev[0] >= 0.25]

    def at_onset(t, pitch):
        if len(onsets) and np.min(np.abs(onsets - t)) <= tol:
            return True
        return any(abs(c - t) <= tol and p != pitch for c, p in changes)

    by_pitch = {}
    for ev in events:
        by_pitch.setdefault(ev[2], []).append(list(ev))
    merged = []
    for p, evs in by_pitch.items():
        cur = evs[0]
        for nxt in evs[1:]:
            if nxt[0] - cur[1] <= gap and not at_onset(nxt[0], p):
                cur[1] = max(cur[1], nxt[1])
                cur[3] = max(cur[3], nxt[3])
            else:
                merged.append(tuple(cur))
                cur = nxt
        merged.append(tuple(cur))
    return sorted(e for e in merged if e[1] - e[0] >= min_len or at_onset(e[0], e[2]))


# ----------------------------------------------------------------------------- rhythm

def estimate_bpm(audio_path, events, grid, lo=50, hi=150):
    """librosa beat tracker as a prior, refined so note onsets sit on the grid."""
    prior = None
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        prior = float(np.atleast_1d(tempo)[0])
        while prior > hi:
            prior /= 2
        while prior < lo and prior > 0:
            prior *= 2
    except Exception:
        prior = None

    onsets = np.array(sorted({round(s, 3) for s, *_ in events}))
    if len(onsets) < 3:
        return int(round(prior or 90))
    onsets = onsets - onsets[0]

    def misfit(bpm):
        unit = 60.0 / bpm / grid
        x = onsets / unit
        return float(np.mean(np.abs(x - np.round(x))))  # 0 .. 0.5 grid units

    candidates = np.arange(lo, hi + 0.01, 0.5)
    scores = []
    for b in candidates:
        m = misfit(b)
        # finer grids always fit better: penalise fast tempi, pull toward the prior
        pen = 0.002 * (b - lo) / 10
        if prior:
            pen += 0.15 * abs(np.log2(b / prior))
        scores.append(m + pen)
    best = float(candidates[int(np.argmin(scores))])
    return int(round(best))


def cluster_times(times, tol=0.05):
    out = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= tol:
            out[-1].append(t)
        else:
            out.append([t])
    return [float(np.mean(c)) for c in out]


def quantise(events, bpm, grid, t0, follow=0.25):
    """Map note times to grid units, letting the tempo drift slowly (a simple
    phase-locked loop on chord onsets), so rubato/drift does not pile up."""
    unit0 = 60.0 / bpm / grid
    unit = unit0
    starts = cluster_times([s for s, *_ in events])
    pos_of = {}
    t_prev, q_prev = t0, 0
    unit_at = {}
    for t in starts:
        raw = q_prev + (t - t_prev) / unit
        q = max(q_prev + (1 if t > t_prev + 0.5 * unit else 0), int(round(raw)))
        span = q - q_prev
        if span > 0:
            err = raw - q
            unit *= 1 + follow * err / span
            unit = min(max(unit, unit0 * 0.8), unit0 * 1.25)
        pos_of[t] = q
        unit_at[t] = unit
        t_prev, q_prev = t, q

    keys = np.array(sorted(pos_of))

    def nearest(t):
        i = int(np.argmin(np.abs(keys - t)))
        return keys[i]

    out = []
    for s, e, p, a in events:
        k = nearest(s)
        su = pos_of[k]
        # end: snap to the next chord onset when the note runs into it
        later = keys[keys > k]
        ke = later[np.argmin(np.abs(later - e))] if len(later) else None
        if ke is not None and abs(ke - e) < 0.12:
            eu = pos_of[ke]
        else:
            eu = su + int(round((e - s) / unit_at[k]))
        out.append([su, max(1, eu - su), p, a])
    return out


# ----------------------------------------------------------------------------- voices

def monophonic(notes):
    """Strongest note wins at each onset; each note ends where the next begins."""
    by_start = {}
    for su, du, p, a in notes:
        if su not in by_start or a > by_start[su][3]:
            by_start[su] = [su, du, p, a]
    seq = [by_start[k] for k in sorted(by_start)]
    for i in range(len(seq) - 1):
        seq[i][1] = max(1, min(seq[i][1], seq[i + 1][0] - seq[i][0]))
    return [(su, du, p) for su, du, p, _ in seq]


def assign_voices(notes, n_voices):
    """Assign quantised notes to n monophonic voices (top voice first).

    Notes starting together form a chord. Each chord's pitches are matched to an
    order-preserving subset of voices, minimising voice-leading distance plus a
    penalty for leaving a voice's normal range. Extra notes (k > n) keep the
    loudest n. A voice that gets no new note keeps sounding if its last note is
    still ringing, otherwise rests.
    """
    if n_voices == 1:
        return [monophonic(notes)]
    ranges = [r for _, r, _ in VOICE_SETS[n_voices]]
    centers = [(lo + hi) / 2 for lo, hi in ranges]
    last = list(centers)
    voices = [[] for _ in range(n_voices)]

    chords = {}
    for su, du, p, a in notes:
        chords.setdefault(su, {})
        if p not in chords[su] or a > chords[su][p][1]:
            chords[su][p] = (du, a)

    def range_pen(v, p):
        lo, hi = ranges[v]
        return 3.0 * max(0, lo - p, p - hi)

    for su in sorted(chords):
        items = thin_chord(chords[su].items(), n_voices)
        items.sort(key=lambda kv: -kv[0])  # high to low
        k = len(items)
        best, best_cost = None, None
        for combo in itertools.combinations(range(n_voices), k):
            cost = 0.0
            for (p, _), v in zip(items, combo):
                cost += abs(p - last[v]) + range_pen(v, p)
            # unused voices that would have to be skipped over cost a little
            cost += 1.5 * (n_voices - k)
            if best_cost is None or cost < best_cost:
                best, best_cost = combo, cost
        for (p, (du, _)), v in zip(items, best):
            voices[v].append([su, du, p])
            last[v] = p

    # make each voice monophonic: a note ends when the voice's next note starts
    for v in voices:
        v.sort()
        for i in range(len(v) - 1):
            v[i][1] = max(1, min(v[i][1], v[i + 1][0] - v[i][0]))
    return [[tuple(n) for n in v] for v in voices]


def auto_voice_count(notes):
    counts = {}
    for su, _, _, _ in notes:
        counts[su] = counts.get(su, 0) + 1
    if not counts:
        return 1
    c = np.array(list(counts.values()))
    return int(min(4, max(1, np.percentile(c, 75))))


# ----------------------------------------------------------------------------- key, meter

def pick_key(pitch_weights):
    from music21 import analysis, note, stream
    s = stream.Stream()
    for midi, dur in pitch_weights:
        n = note.Note(midi)
        n.quarterLength = max(0.25, dur)
        s.append(n)
    return analysis.discrete.KrumhanslSchmuckler().getSolution(s)


def estimate_meter(voices, grid):
    """Pick beats-per-bar (2, 3, 4) and the pickup length (grid units) so that long
    notes and chord onsets fall on downbeats."""
    weight = {}
    for v in voices:
        for su, du, _ in v:
            weight[su] = weight.get(su, 0.0) + du
    if not weight:
        return 4, 0
    total = sum(weight.values())
    best = None
    for beats in (4, 3, 2):
        bar = beats * grid
        for off in range(0, bar, grid):  # pickups start on a beat
            down = sum(w for t, w in weight.items() if (t + off) % bar == 0)
            mid = sum(w for t, w in weight.items()
                      if beats == 4 and (t + off) % bar == 2 * grid)
            score = (down + 0.6 * mid) / total
            score *= {4: 1.0, 3: 1.04, 2: 0.72}[beats]  # 2/4 fits anything; damp it
            if best is None or score > best[0] + 1e-9:
                best = (score, beats, off)
    _, beats, off = best
    # off = shift that puts bar lines on the grid; the pickup is what is left of the bar
    pickup = (beats * grid - off) % (beats * grid)
    return beats, pickup


# ----------------------------------------------------------------------------- spelling, shapes

def spell(midi, sharps):
    from music21 import pitch
    p = pitch.Pitch(midi=midi)
    if sharps < 0 and p.accidental is not None and p.accidental.alter > 0:
        p = p.getEnharmonic()
    elif sharps > 0 and p.accidental is not None and p.accidental.alter < 0:
        p = p.getEnharmonic()
    return p


def shape_degree(p, sharps):
    """Scale degree of the note's letter relative to the MAJOR tonic of the key
    signature (MuseScore's convention: in minor, the tonic is 'la')."""
    from music21 import key as m21key
    major = m21key.KeySignature(sharps).asKey("major")
    return (LETTERS.index(p.step) - LETTERS.index(major.tonic.step)) % 7


# ----------------------------------------------------------------------------- score

def make_clef(name):
    from music21 import clef
    return {"treble": clef.TrebleClef, "bass": clef.BassClef,
            "treble8vb": clef.Treble8vbClef}[name]()


def build_score(voices, voice_set, bpm, beats, pickup, grid, sharps, system, title):
    from music21 import instrument, key as m21key, metadata, meter, note, stream, tempo

    sylls = SYLLABLES_4 if system == "4shape" else SYLLABLES_7
    shift = (beats * grid - pickup) % (beats * grid) if pickup else 0
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    report = []

    for vi, ((name, _, clef_name), vnotes) in enumerate(zip(voice_set, voices)):
        part = stream.Part(id=f"P{vi + 1}")
        inst = instrument.Vocalist()
        inst.partName = name
        inst.partAbbreviation = name[0]
        inst.instrumentName = name
        part.insert(0, inst)
        part.insert(0, make_clef(clef_name))
        part.insert(0, m21key.KeySignature(sharps))
        part.insert(0, meter.TimeSignature(f"{beats}/4"))
        if vi == 0:
            part.insert(0, tempo.MetronomeMark(number=bpm))
        for su, du, midi in vnotes:
            p = spell(midi, sharps)
            n = note.Note(p)
            n.quarterLength = du / grid
            syl = sylls[shape_degree(p, sharps)]
            n.notehead = syl
            part.insert((su + shift) / grid, n)
            report.append({"voice": name, "beat": su / grid, "pitch": p.nameWithOctave,
                           "beats": du / grid, "shape": syl})
        end = max((su + du for su, du, _ in vnotes), default=0) + shift
        bar = beats * grid
        total = max(bar, -(-end // bar) * bar)  # pad last bar
        part.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=False,
                       refStreamOrTimeRange=[0, total / grid])
        part.makeNotation(inPlace=True)
        if shift:
            first = part.getElementsByClass(stream.Measure).first()
            if first is not None:
                lead = first.notesAndRests.first()
                if lead is not None and lead.isRest and lead.offset == 0 and lead.quarterLength == shift / grid:
                    first.remove(lead)
                    first.paddingLeft = shift / grid
                    for el in list(first.elements):
                        if el.offset >= shift / grid:
                            first.setElementOffset(el, el.offset - shift / grid)
                    for i, m in enumerate(part.getElementsByClass(stream.Measure)):
                        m.number = i  # pickup is bar 0
        score.insert(0, part)
    return score, report


# ----------------------------------------------------------------------------- main

class MissingDependency(Exception):
    pass


def run(a):
    audio = Path(a.audio)
    if not audio.exists():
        raise FileNotFoundError(audio)
    out = Path(a.output) if a.output else audio.with_suffix(".musicxml")

    voices_req = a.voices
    poly = voices_req != 1
    ev = transcribe(audio, a.onset, a.frame, a.min_ms, polyphonic=poly)
    ev = merge_fragments(ev, detect_onsets(audio))
    if poly:
        ev = remove_harmonics(ev)
    if not ev:
        return None, 3
    ev.sort()
    t0 = ev[0][0]
    bpm = a.bpm or estimate_bpm(audio, ev, a.grid)
    q = quantise(ev, bpm, a.grid, t0)
    n_voices = auto_voice_count(q) if voices_req == 0 else voices_req
    voice_set = VOICE_SETS[n_voices]
    voices = assign_voices(q, n_voices)

    weights = [(p, du / a.grid) for v in voices for _, du, p in v]
    key_obj = pick_key(weights)
    sharps = key_obj.sharps if a.key is None else a.key

    if a.time_sig == "auto":
        beats, pickup = estimate_meter(voices, a.grid)
    else:
        beats, pickup = int(a.time_sig.split("/")[0]), a.pickup * a.grid

    score, report = build_score(voices, voice_set, bpm, beats, pickup, a.grid,
                                sharps, a.system, a.title or audio.stem)
    score.write("musicxml", fp=str(out))
    if pickup:
        # mark the pickup bar as implicit so notation programs do not count it
        xml = out.read_text(encoding="utf-8")
        out.write_text(xml.replace('<measure implicit="no" number="0"', '<measure implicit="yes" number="0"'),
                       encoding="utf-8")
    summary = {
        "output": str(out), "events": len(ev), "voices": [name for name, *_ in voice_set],
        "bpm": bpm, "key": f"{key_obj.tonic.name} {key_obj.mode}", "sharps": sharps,
        "time_signature": f"{beats}/4", "pickup_beats": pickup / a.grid, "system": a.system,
        "notes": report,
    }
    return summary, 0


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Audio to shape-note MusicXML")
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", help="output .musicxml (default: next to the input)")
    ap.add_argument("--system", choices=["aikin", "funk", "walker", "4shape"], default="aikin")
    ap.add_argument("--voices", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="number of parts; 0 = detect (default)")
    ap.add_argument("--bpm", type=int, help="tempo (default: estimate)")
    ap.add_argument("--time-sig", default="auto", help="e.g. 4/4, 3/4, or auto (default)")
    ap.add_argument("--pickup", type=float, default=0, help="pickup length in beats (with fixed --time-sig)")
    ap.add_argument("--key", type=int, help="key signature as sharps (+) / flats (-); default: estimate")
    ap.add_argument("--grid", type=int, default=2, choices=[1, 2, 4], help="notes per beat on the rhythm grid")
    ap.add_argument("--onset", type=float, default=0.3)
    ap.add_argument("--frame", type=float, default=0.2)
    ap.add_argument("--min-ms", type=float, default=110.0)
    ap.add_argument("--title")
    ap.add_argument("--report", help="write a JSON summary here")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        summary, code = run(a)
    except MissingDependency as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        traceback.print_exc()
        print(f"error: {e}", file=sys.stderr)
        return 5
    if code == 3:
        print("error: no notes detected", file=sys.stderr)
        return 3
    if a.report:
        Path(a.report).write_text(json.dumps(summary, indent=2))
    if not a.quiet:
        print(f"voices={summary['voices']} bpm={summary['bpm']} key={summary['key']} "
              f"time={summary['time_signature']} pickup={summary['pickup_beats']} system={a.system}")
        for n in summary["notes"]:
            print(f"  {n['voice']:7s} beat {n['beat']:6.2f}  {n['pitch']:4s} {n['beats']:4.2f}  {n['shape']}")
        print(f"wrote {summary['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
