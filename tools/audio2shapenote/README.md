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
| `io/audioImport/tempo` | `0` | BPM; 0 = detect |
| `io/audioImport/pythonExecutable` | empty | Python to use; empty = auto |
| `io/audioImport/enginePath` | empty | this script; empty = bundled copy (or `MUSESCORE_AUDIO2SHAPENOTE`) |
| `io/audioImport/timeoutSeconds` | `600` | give up after this long |

The command line also works: `mscore -o song.mscz song.wav`.

## Command line

```
.venv/bin/python audio2shapenote.py song.wav                    # -> song.musicxml
.venv/bin/python audio2shapenote.py song.wav --voices 4 --system funk --time-sig 3/4 --bpm 80
```

Options: `--system aikin|funk|walker|4shape`, `--voices 0-4`, `--ensemble auto|mixed|male`,
`--separate auto|on|off`, `--legato BEATS`, `--time-sig auto|N/4`,
`--pickup BEATS` (with a fixed time signature), `--key SHARPS` (negative = flats),
`--bpm`, `--grid 1|2|4` (notes per beat), `--onset/--frame` (model sensitivity),
`--report out.json` (summary with every note and its shape).

## How it works

0. **Vocal isolation** (`separate.py`): an MDX-Net model (UVR "Voc_FT", ONNX) strips guitar,
   banjo, bass etc. Used automatically when the recording has instruments; skipped for
   pure a cappella. `setup_engine.py` downloads the model (~65 MB) from GitHub.
1. **Notes**: Spotify Basic Pitch (bundled ONNX model, no TensorFlow).
2. **Clean-up**: same-pitch fragments caused by vibrato or reverb are joined unless
   a real onset is there (audio onset detector, or another part moving to a new
   pitch); overtone ghosts are dropped.
3. **Rhythm**: tempo from librosa refined by fitting onsets to the grid, then a
   tempo-following quantiser so slow drift does not accumulate.
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
.venv/bin/python test_engine.py   # synthesizes clips with known answers, must be exact
```

`make_test_clip.py melody|chorale out.wav` writes the test audio and a `.truth.json`.

## Accuracy so far

| Clip | Notes right | Notes + rhythm | Meter / pickup / key |
|---|---|---|---|
| Synthetic melody (vibrato) | 100% | 100% | all right |
| Synthetic 4-part chorale, 3/4 with pickup | 100% | 100% | all right |
| Same chorale + reverb, noise, ±3% tempo drift, low-pass | 59% | 34% | meter right, pickup missed |

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
- No lyrics, dynamics, repeats, tempo or meter changes, triplets, or compound meters (6/8).
- Long recordings are slow on CPU (Basic Pitch runs at about real time or faster).
