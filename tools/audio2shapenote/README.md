# audio2shapenote (proof of concept)

Turns a recording of a single melodic line (for example one sung part) into a
MusicXML file with **shape-note noteheads**, ready to open in MuseScore Studio.

```
python -m venv .venv && . .venv/bin/activate
pip install --no-deps basic-pitch && pip install -r requirements.txt
python make_test_clip.py test_melody.wav            # synthetic D-major tune
python audio2shapenote.py test_melody.wav --bpm 90  # -> test_melody.musicxml
python test_shapes.py                               # offline shape-logic checks
```

Options: `--system aikin|funk|walker|4shape`, `--time-sig 3/4`, `--grid 1|2|4`
(notes per beat), `--bpm` (otherwise estimated), `--onset/--frame` (model thresholds).

## How it works
1. Basic Pitch (ONNX) -> note events
2. Greedy monophonic reduction (strongest note wins)
3. Tempo estimate (librosa) or `--bpm`; quantise to a beat grid
4. Key estimate (Krumhansl-Schmuckler, music21); key-aware spelling
5. Each note gets `<notehead>do|re|mi|fa|so|la|ti</notehead>`, the same values
   MuseScore's own MusicXML export writes for its shape-note schemes
   (`exportmusicxml.cpp`). Shapes are relative to the MAJOR tonic of the key
   signature, so in a minor key `la` is the tonic (MuseScore's convention).

## Known limits
- One voice only; no SATB split, lyrics, pickup bars, tempo changes or ties across barlines yet.
- Rhythm assumes the first note starts bar 1 and tempo is steady.
- Not yet checked inside a built MuseScore: after opening, use
  Staff/Part properties > Notehead scheme to switch to a dynamic shape scheme
  (e.g. "7 Shape (Aikin)") if the imported heads need to follow later edits.
