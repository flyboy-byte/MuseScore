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

#include "modularity/imoduleinterface.h"
#include "io/path.h"

namespace mu::iex::audioimport {
class IAudioImportConfiguration : MODULE_GLOBAL_INTERFACE
{
    INTERFACE_ID(IAudioImportConfiguration)

public:
    virtual ~IAudioImportConfiguration() = default;

    //! Python interpreter used to run the engine. Empty = auto-detect.
    virtual std::string pythonExecutable() const = 0;
    virtual void setPythonExecutable(const std::string& path) = 0;

    //! Path to audio2shapenote.py. Empty = the copy bundled with the app.
    virtual muse::io::path_t enginePath() const = 0;
    virtual void setEnginePath(const muse::io::path_t& path) = 0;

    //! Notehead scheme applied to the imported staves, as in .mscx files:
    //! "shape-7-aikin" (default), "shape-7-funk", "shape-7-walker", "shape-4", "normal".
    virtual std::string noteheadScheme() const = 0;
    virtual void setNoteheadScheme(const std::string& scheme) = 0;

    //! Number of vocal parts to write (1-4), 0 = detect.
    virtual int voiceCount() const = 0;
    virtual void setVoiceCount(int voices) = 0;

    //! "auto" or an explicit time signature such as "3/4".
    virtual std::string timeSignature() const = 0;
    virtual void setTimeSignature(const std::string& timeSig) = 0;

    //! Tempo in BPM, 0 = estimate.
    virtual int tempo() const = 0;
    virtual void setTempo(int bpm) = 0;

    virtual int timeoutSeconds() const = 0;
};
}
