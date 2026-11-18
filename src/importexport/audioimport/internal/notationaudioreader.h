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

#include "project/inotationreader.h"
#include "modularity/ioc.h"
#include "iglobalconfiguration.h"
#include "project/inotationreadersregister.h"

#include "../iaudioimportconfiguration.h"

#include "engraving/types/types.h"

namespace mu::engraving {
class MasterScore;
}

namespace mu::iex::audioimport {
//! Reads an audio file by running the audio2shapenote engine (Python) to get
//! MusicXML, importing that with the MusicXML reader, and switching every
//! standard staff to the configured shape-note notehead scheme.
class NotationAudioReader : public project::INotationReader
{
    muse::GlobalInject<IAudioImportConfiguration> configuration;
    muse::GlobalInject<muse::IGlobalConfiguration> globalConfiguration;
    muse::GlobalInject<project::INotationReadersRegister> readers;

public:
    static std::vector<std::string> supportedSuffixes();

    muse::Ret read(mu::engraving::MasterScore* score, const muse::io::path_t& path, const Options& options = Options()) override;

    //! Make shapes follow pitch: set the staff scheme and clear fixed per-note heads.
    static void applyNoteheadScheme(mu::engraving::MasterScore* score, mu::engraving::NoteHeadScheme scheme);

private:
    muse::io::path_t resolveEnginePath() const;
    std::string resolvePython(const muse::io::path_t& enginePath) const;
};
}
