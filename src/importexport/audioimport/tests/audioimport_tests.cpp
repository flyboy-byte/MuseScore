/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * MuseScore Studio
 * Music Composition & Notation
 *
 * Copyright (C) 2026 MuseScore Limited and others
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <gtest/gtest.h>

#include <set>

#include "engraving/dom/chord.h"
#include "engraving/dom/masterscore.h"
#include "engraving/dom/note.h"
#include "engraving/dom/segment.h"
#include "engraving/dom/staff.h"
#include "engraving/dom/stafftype.h"
#include "engraving/compat/scoreaccess.h"

#include "importexport/musicxml/internal/musicxmlreader.h"
#include "importexport/audioimport/internal/audioimportoptions.h"
#include "importexport/audioimport/internal/notationaudioreader.h"

using namespace mu::engraving;
using namespace mu::iex::audioimport;

static const muse::String DATA_DIR = muse::String::fromUtf8(iex_audioimport_tests_DATA_ROOT) + u"/data/";

TEST(AudioImport_Options, SchemeFromString)
{
    EXPECT_EQ(schemeFromString("shape-7-aikin"), NoteHeadScheme::HEAD_SHAPE_NOTE_7_AIKIN);
    EXPECT_EQ(schemeFromString("shape-7-funk"), NoteHeadScheme::HEAD_SHAPE_NOTE_7_FUNK);
    EXPECT_EQ(schemeFromString("shape-7-walker"), NoteHeadScheme::HEAD_SHAPE_NOTE_7_WALKER);
    EXPECT_EQ(schemeFromString("shape-4"), NoteHeadScheme::HEAD_SHAPE_NOTE_4);
    EXPECT_EQ(schemeFromString("normal"), NoteHeadScheme::HEAD_NORMAL);
    EXPECT_EQ(schemeFromString("nonsense"), NoteHeadScheme::HEAD_SHAPE_NOTE_7_AIKIN);
}

TEST(AudioImport_Options, EngineArguments)
{
    AudioImportOptions o;
    o.scheme = "shape-4";
    o.voices = 4;
    o.timeSignature = "3/4";
    o.tempo = 80;
    std::vector<std::string> a = engineArguments("e.py", "in.wav", "out.musicxml", "r.json", o);
    std::vector<std::string> expected { "e.py", "in.wav", "--output", "out.musicxml", "--report", "r.json",
                                        "--system", "4shape", "--quiet", "--voices", "4", "--time-sig", "3/4",
                                        "--bpm", "80" };
    EXPECT_EQ(a, expected);

    AudioImportOptions d; // defaults: detect everything, Aikin
    a = engineArguments("e.py", "in.wav", "o", "r", d);
    EXPECT_EQ(a.size(), 9u);
    EXPECT_EQ(a[7], "aikin");
}

TEST(AudioImport_Reader, Suffixes)
{
    std::vector<std::string> s = NotationAudioReader::supportedSuffixes();
    EXPECT_NE(std::find(s.begin(), s.end(), "wav"), s.end());
    EXPECT_NE(std::find(s.begin(), s.end(), "mp3"), s.end());
}

// The engine's MusicXML imports as a 4-part score; applying the scheme makes
// every staff use dynamic 7-shape heads instead of fixed per-note heads.
TEST(AudioImport_Reader, ApplySchemeToEngineOutput)
{
    MasterScore* score = compat::ScoreAccess::createMasterScoreWithBaseStyle(nullptr);
    mu::iex::musicxml::MusicXmlReader reader;
    muse::Ret ret = reader.read(score, muse::io::path_t(DATA_DIR + u"chorale.musicxml"));
    ASSERT_TRUE(ret);
    ASSERT_EQ(score->nstaves(), 4u);

    NotationAudioReader::applyNoteheadScheme(score, NoteHeadScheme::HEAD_SHAPE_NOTE_7_AIKIN);
    score->doLayout();

    for (Staff* st : score->staves()) {
        EXPECT_EQ(st->staffType(Fraction(0, 1))->noteHeadScheme(), NoteHeadScheme::HEAD_SHAPE_NOTE_7_AIKIN);
    }

    std::set<SymId> shapes;
    int notes = 0;
    for (Segment* s = score->firstSegment(SegmentType::ChordRest); s; s = s->next1(SegmentType::ChordRest)) {
        for (EngravingItem* e : s->elist()) {
            if (!e || !e->isChord()) {
                continue;
            }
            for (Note* n : toChord(e)->notes()) {
                EXPECT_EQ(n->headGroup(), NoteHeadGroup::HEAD_NORMAL);
                shapes.insert(n->noteHead());
                ++notes;
            }
        }
    }
    EXPECT_EQ(notes, 32);
    // G major chorale uses do re mi fa so la ti: many distinct shapes, not just round heads
    EXPECT_GE(shapes.size(), 5u);

    delete score;
}
