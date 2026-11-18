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

#include "../iaudioimportconfiguration.h"

#include "async/asyncable.h"

namespace mu::iex::audioimport {
class AudioImportConfiguration : public IAudioImportConfiguration, public muse::async::Asyncable
{
public:
    void init();

    std::string pythonExecutable() const override;
    void setPythonExecutable(const std::string& path) override;

    muse::io::path_t enginePath() const override;
    void setEnginePath(const muse::io::path_t& path) override;

    std::string noteheadScheme() const override;
    void setNoteheadScheme(const std::string& scheme) override;

    int voiceCount() const override;
    void setVoiceCount(int voices) override;

    std::string timeSignature() const override;
    void setTimeSignature(const std::string& timeSig) override;

    int tempo() const override;
    void setTempo(int bpm) override;

    int timeoutSeconds() const override;
};
}
