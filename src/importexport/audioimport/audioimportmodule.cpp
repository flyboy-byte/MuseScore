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
#include "audioimportmodule.h"

#include "modularity/ioc.h"

#include "project/inotationreadersregister.h"

#include "internal/audioimportconfiguration.h"
#include "internal/notationaudioreader.h"

using namespace muse::modularity;
using namespace mu::iex::audioimport;
using namespace mu::project;

std::string AudioImportModule::moduleName() const
{
    return "iex_audioimport";
}

void AudioImportModule::registerExports()
{
    m_configuration = std::make_shared<AudioImportConfiguration>();

    globalIoc()->registerExport<IAudioImportConfiguration>(moduleName(), m_configuration);
}

void AudioImportModule::resolveImports()
{
    auto readers = globalIoc()->resolve<INotationReadersRegister>(moduleName());
    if (readers) {
        readers->reg(NotationAudioReader::supportedSuffixes(), std::make_shared<NotationAudioReader>());
    }
}

void AudioImportModule::onInit(const muse::IApplication::RunMode&)
{
    m_configuration->init();
}
