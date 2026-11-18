# audio2shapenote

Turns a recording (one melody, or up to four singing parts) into **shape-note
sheet music**. It is used two ways:

- **Inside MuseScore** (module `src/importexport/audioimport`): *File ▸ Open* a
  `.wav/.mp3/.flac/.ogg/.m4a/.aif` file. MuseScore runs this engine, imports the
  result and sets every staff to a dynamic shape-note scheme (7-shape Aikin by
  default), so shapes follow the pitches when you edit.
- **On the command line**: audio in, MusicXML out.

## Setup (once)

Requires Python 3.10+.

```
python3 setup_engine.py          # creates .venv next to this script
python3 setup_engine.py --user   # or in the MuseScore user data folder (read-only app installs)
```

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

Options: `--system aikin|funk|walker|4shape`, `--voices 0-4`, `--time-sig auto|N/4`,
`--pickup BEATS` (with a fixed time signature), `--key SHARPS` (negative = flats),
`--bpm`, `--grid 1|2|4` (notes per beat), `--onset/--frame` (model sensitivity),
`--report out.json` (summary with every note and its shape).

## How it works

1. **Notes**: Spotify Basic Pitch (bundled ONNX model, no TensorFlow).
2. **Clean-up**: same-pitch fragments caused by vibrato or reverb are joined unless
   a real onset is there (audio onset detector, or another part moving to a new
   pitch); overtone ghosts are dropped.
3. **Rhythm**: tempo from librosa refined by fitting onsets to the grid, then a
   tempo-following quantiser so slow drift does not accumulate.
4. **Parts**: notes starting together are matched to Treble/Alto/Tenor/Bass by
   voice-leading distance plus each part's normal range (shape-note order:
   Treble, Alto, Tenor, Bass; tenor on a treble-8vb clef).
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

No real choir recordings have been tested yet. Expect real Sacred Harp
recordings (big rooms, many singers per part, octave doubling of treble and
tenor) to land between the last two rows and need hand correction.

## Known limits

- Four parts at most; parts are assigned by range and voice leading, so crossing
  parts can be swapped.
- No lyrics, dynamics, repeats, tempo or meter changes, triplets, or compound meters (6/8).
- Long recordings are slow on CPU (Basic Pitch runs at about real time or faster).
