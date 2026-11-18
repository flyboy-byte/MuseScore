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
#pragma once

#include <string>
#include <vector>

#include "engraving/types/types.h"

namespace mu::iex::audioimport {
struct AudioImportOptions {
    std::string scheme = "shape-7-aikin"; // as stored in settings / .mscx
    int voices = 0;                        // 0 = detect
    std::string timeSignature = "auto";
    int tempo = 0;                         // 0 = detect
};

//! Notehead scheme for a settings value; unknown values fall back to 7-shape Aikin.
engraving::NoteHeadScheme schemeFromString(const std::string& value);

//! The engine's --system argument for a scheme ("aikin", "funk", "walker", "4shape").
std::string engineSystemFor(engraving::NoteHeadScheme scheme);

//! Command-line arguments for audio2shapenote.py.
std::vector<std::string> engineArguments(const std::string& enginePath, const std::string& audioPath, const std::string& outputPath,
                                         const std::string& reportPath, const AudioImportOptions& options);

//! Human-readable explanation of an engine exit code.
std::string engineErrorMessage(int exitCode);
}
