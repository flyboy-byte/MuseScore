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
#include "audioimportoptions.h"

using namespace mu::engraving;

namespace mu::iex::audioimport {
NoteHeadScheme schemeFromString(const std::string& value)
{
    if (value == "shape-4") {
        return NoteHeadScheme::HEAD_SHAPE_NOTE_4;
    }
    if (value == "shape-7-funk") {
        return NoteHeadScheme::HEAD_SHAPE_NOTE_7_FUNK;
    }
    if (value == "shape-7-walker") {
        return NoteHeadScheme::HEAD_SHAPE_NOTE_7_WALKER;
    }
    if (value == "normal") {
        return NoteHeadScheme::HEAD_NORMAL;
    }
    return NoteHeadScheme::HEAD_SHAPE_NOTE_7_AIKIN;
}

std::string engineSystemFor(NoteHeadScheme scheme)
{
    switch (scheme) {
    case NoteHeadScheme::HEAD_SHAPE_NOTE_4: return "4shape";
    case NoteHeadScheme::HEAD_SHAPE_NOTE_7_FUNK: return "funk";
    case NoteHeadScheme::HEAD_SHAPE_NOTE_7_WALKER: return "walker";
    default: return "aikin";
    }
}

std::vector<std::string> engineArguments(const std::string& enginePath, const std::string& audioPath, const std::string& outputPath,
                                         const std::string& reportPath, const AudioImportOptions& options)
{
    std::vector<std::string> args {
        enginePath, audioPath,
        "--output", outputPath,
        "--report", reportPath,
        "--system", engineSystemFor(schemeFromString(options.scheme)),
        "--quiet",
    };
    if (options.voices >= 1 && options.voices <= 4) {
        args.push_back("--voices");
        args.push_back(std::to_string(options.voices));
    }
    if (!options.timeSignature.empty() && options.timeSignature != "auto") {
        args.push_back("--time-sig");
        args.push_back(options.timeSignature);
    }
    if (options.tempo > 0) {
        args.push_back("--bpm");
        args.push_back(std::to_string(options.tempo));
    }
    return args;
}

std::string engineErrorMessage(int exitCode)
{
    switch (exitCode) {
    case 2: return "The transcription engine rejected its arguments (check the audio import settings).";
    case 3: return "No notes were detected in this recording.";
    case 4: return "The transcription engine is missing Python packages. Run setup_engine.py from the audio2shapenote folder.";
    default: return "The transcription engine failed (exit code " + std::to_string(exitCode) + ").";
    }
}
}
