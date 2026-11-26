# audio2shapenote

Turns a recording (one melody, or up to four singing parts) into **shape-note
sheet music**. It is used two ways:

- **Inside MuseScore** (module `src/importexport/audioimport`): *File ▸ Open* a
  `.wav/.mp3/.flac/.ogg/.m4a/.aif` file. MuseScore runs this engine, imports the
  result and sets every staff to a dynamic shape-note scheme (7-shape Aikin by
  default), so shapes follow the pitches when you edit.
- **On the command line**: audio in, MusicXML out.

## Install or update the engine

Requires Python 3.10+.

```
python3 setup_engine.py          # creates .venv next to this script
python3 setup_engine.py --user   # or in the MuseScore user data folder (read-only app installs)
```

After downloading a newer build, use its setup script with the same install
location as before. For the AppImage, run:

```sh
python3 audio2shapenote/setup_engine.py --user --upgrade
```

Setup reuses the existing virtual environment and cached vocal model. Use
`--upgrade` to refresh Python dependencies. The new AppImage contains the new
engine scripts, so launch that AppImage after updating.

MuseScore finds that `.venv` automatically. Otherwise it uses `python3`/`python`
on PATH, or the `MUSESCORE_AUDIO_PYTHON` environment variable.

## Settings in MuseScore (Preferences ▸ Advanced)

| Key | Default | Meaning |
|---|---|---|
| `io/audioImport/noteheadScheme` | `shape-7-aikin` | also `shape-7-funk`, `shape-7-walker`, `shape-4`, `normal` |
| `io/audioImport/voices` | `0` | parts to write, 1–4; 0 = detect |
| `io/audioImport/timeSignature` | `auto` | or e.g. `3/4` |
| `io/audioImport/tempo` | `0` | starting BPM, whole numbers only here (the command line takes fractions); 0 = detect |
| `io/audioImport/pythonExecutable` | empty | Python to use; empty = auto |
| `io/audioImport/enginePath` | empty | this script; empty = bundled copy (or `MUSESCORE_AUDIO2SHAPENOTE`) |
| `io/audioImport/timeoutSeconds` | `1800` | give up after this long |

The command line also works: `mscore -o song.mscz song.wav`.

## Command line

```
.venv/bin/python audio2shapenote.py song.wav                    # -> song.musicxml
.venv/bin/python audio2shapenote.py song.wav --voices 4 --system funk --time-sig 3/4 --bpm 80
```

Options: `--system aikin|funk|walker|4shape`, `--voices 0-4`, `--ensemble auto|mixed|male`,
`--separate auto|on|off`, `--legato BEATS`, `--time-sig auto|N/4`,
`--pickup BEATS` (with a fixed time signature), `--key SHARPS` (negative = flats),
`--bpm` (starting tempo, may be fractional), `--grid 1|2|4` (notes per beat),
`--onset/--frame` (model sensitivity), `--report out.json` (summary with every note
and its shape).

The report also says where the music is in the recording: `beat_seconds[b]` is the
time in the audio file of beat `b` (the beats in `notes`, where beat 0 is the first
note), and `tempo_model` says whether one steady tempo was used or a tempo that
follows the performance.

## Match a transcription to an existing score

Export your existing score as MusicXML from MuseScore. Generate a second MusicXML
file from the audio, then align the two:

```sh
.venv/bin/python audio2shapenote.py recording.mp3 -o transcription.musicxml
.venv/bin/python score_review.py compare existing.musicxml transcription.musicxml -o review.json
.venv/bin/python score_review.py apply existing.musicxml review.json --select 1 3 -o patched.musicxml
```

The review lists corresponding beats and tentative pitch changes for uniquely
named parts. Each suggestion includes the staff, measure, and before/after pitches.
Choose the patch IDs you have checked against the recording. Apply writes a new
MusicXML copy and checks that the reference still matches the reviewed file.
This compares transcribed pitches, so a disagreement can come from either score
or from an alignment error. It does not assign singers or patch durations, ties,
chords, or transposing parts. Use same-key scores in performance order with
matching part names.

## How it works

0. **Vocal isolation** (`separate.py`): an MDX-Net model (UVR "Voc_FT", ONNX) strips guitar,
   banjo, bass etc. Used automatically when the recording has instruments; skipped for
   pure a cappella. `setup_engine.py` downloads the model (~65 MB) from GitHub.
1. **Notes**: Spotify Basic Pitch (bundled ONNX model, no TensorFlow).
2. **Clean-up**: same-pitch fragments caused by vibrato or reverb are joined unless
   a real onset is there (audio onset detector, or another part moving to a new
   pitch); overtone ghosts are dropped, and so are notes where the original recording
   is silent (vocal isolation leaves a faint noise floor there).
3. **Rhythm**: the opening tempo comes from librosa, refined by fitting the first
   20 seconds of onsets to the grid. A fine search over the whole piece then finds
   the best steady tempo (any value, not only whole BPM). If the performance speeds
   up or slows down so that a steady grid fits poorly, a tempo that follows the
   onsets is used instead. Every onset is rounded against that one clock, so rounding
   cannot accumulate. The score shows the tempo as a whole number.
4. **Parts**: the group is detected as **mixed** (Treble/Alto/Tenor/Bass, shape-note
   order) or **male** (Tenor/Lead/Baritone/Bass, gospel/bluegrass quartet) from its
   pitch range. Notes starting together are matched to parts by voice-leading distance
   plus each part's normal range; short rests inside a phrase are closed by holding the
   previous note (`--legato`). Overtones (octave/12th/15th ghosts) and notes far outside
   the group's range are dropped.
5. **Key, meter, pickup**: Krumhansl–Schmuckler key; meter and pickup chosen so
   long notes and chord changes land on downbeats.
6. **Output**: MusicXML where every note has a `<notehead>` syllable
   (`do re mi fa so la ti`), the same values MuseScore writes for shape notes.
   Shapes are relative to the major tonic of the key signature (minor tonic = `la`).

## Tests

```
.venv/bin/python test_shapes.py   # shape logic, no audio
.venv/bin/python test_engine.py   # synthesizes clips with known answers, must be exact (~2 min)
```

`make_test_clip.py melody|chorale out.wav` writes the test audio and a `.truth.json`.
`test_engine.py` also renders two 150-second melodies, one at a steady 78.4 BPM and
one drifting from 76 to 80 BPM; every note must land on the right beat.

To check a recording you already have:

```sh
.venv/bin/python test_engine.py --audio recording.mp3 --tolerance 2
```

This runs transcription on the mix and checks the span from the first to the last
detected note against the duration of the score at its reported tempo. Both come
from the same tempo, so this only catches rounding that accumulates; it cannot tell
whether the tempo or the bar lines are right, and it measures nothing about pitch or
parts. The audio is read locally and does not need to be checked into the repository.

## Accuracy so far

| Clip | Notes right | Notes + rhythm | Meter / pickup / key |
|---|---|---|---|
| Synthetic melody (vibrato) | 100% | 100% | all right |
| Synthetic 4-part chorale, 3/4 with pickup | 100% | 100% | all right |
| Same chorale + reverb, noise, ±3% tempo drift, low-pass | 59% | 34% | meter right, pickup missed |
| Synthetic 150 s melodies at 63.7, 78.4, 91.25 BPM; 78.4 with 25 ms timing jitter; 76→80 BPM drift (2026-11-26) | 100% | 100% | all right |
| Same melody at a steady 117.3 BPM | — | 7% | written in half time (58.65 BPM); see Known limits |

Before 2026-11-26 the tempo search used 0.5 BPM steps rounded to whole BPM, and only
the whole-number tempo in the rows above came out right (2–14% of notes otherwise).

| Doyle Lawson & Quicksilver, "Passed All Sorrow" (real men's quartet, 2:38) | not measurable (no reference score) | readable quartet layout; before: mixed-choir staves full of 16ths |

First real recording: earlier versions read overtones as extra singers and assumed
mixed-choir ranges, which produced unreadable output. Fixed by ensemble detection,
simultaneous-overtone removal, range limits, legato fill and vocal isolation. Expect real Sacred Harp
recordings (big rooms, many singers per part, octave doubling of treble and
tenor) to land between the last two rows and need hand correction.

### Reference score check on a real choir

For a reproducible check, use the 69-second *Tebe Poem* full-choir mix and the
published MEI score.
[Recording](https://helenacuesta.github.io/MultipleF0EstimationVocalEnsembles/files/orig/DCS_TP_FullChoir_Take03_mix.wav)
· [Reference score and recording notes](https://www.audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet/data/DCS_TP_FullChoir_Take03)

```sh
.venv/bin/python audio2shapenote.py DCS_TP_FullChoir_Take03_mix.wav \
  --voices 4 --ensemble mixed --separate off --report original.json -o original.musicxml
```

Repeat with `--separate on` for the isolated mix. For the guided pass, retain
`--separate off` and add `--time-sig 4/4 --pickup 1 --key -2 --bpm 78 --legato 0`.
The guided tempo is an approximate average for this performance.

| Pass | Score measures | Notes emitted | Reference pitches matched in order |
|---|---:|---:|---:|
| Published score | 23 | 231 | 231 |
| Original mix, rhythm estimated | 43 | 431 | 177 |
| Isolated vocals, rhythm estimated | 28 | 374 | 163 |
| Original mix, rhythm guided | 24 | 385 | 144 |

The last column compares MIDI pitch sequences independently per staff using
longest common subsequence, including written repeated notes. It ignores timing
and can match a pitch in the wrong measure, so it is an upper bound on exact
score agreement under this alignment. The two estimated passes found 4/4 and
two flats but guessed a three-beat pickup where the score has one.

For real recordings, review the draft alongside the audio. Start with a solo
recording or isolated part tracks when available. If a score already exists,
use it for pitches, part assignments, and notation, and align it to the audio.
For mixed choir recordings with no reference, check entrances, unisons,
inner-voice assignments, and bar lines before relying on the shape-note draft.

## Known limits

- Four parts at most; parts are assigned by range and voice leading, so crossing
  parts can be swapped.
- No lyrics, dynamics, repeats, written tempo or meter changes, triplets, or compound meters (6/8).
- Gradual tempo changes are followed, but not sudden ones (a fermata or a held
  ending adds beats instead of stretching one).
- Half time and double time can be confused: a steady 117 BPM melody came out at
  58.65 BPM. The notes are then written at half their length (quarters as eighths);
  set `--bpm` when that happens.
- Long recordings are slow on CPU (Basic Pitch runs at about real time or faster).
