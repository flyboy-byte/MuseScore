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
#include "audioimportconfiguration.h"

#include "settings.h"

using namespace muse;
using namespace mu::iex::audioimport;

static const std::string MODULE("iex_audioimport");

static const Settings::Key PYTHON_KEY(MODULE, "io/audioImport/pythonExecutable");
static const Settings::Key ENGINE_KEY(MODULE, "io/audioImport/enginePath");
static const Settings::Key SCHEME_KEY(MODULE, "io/audioImport/noteheadScheme");
static const Settings::Key VOICES_KEY(MODULE, "io/audioImport/voices");
static const Settings::Key TIMESIG_KEY(MODULE, "io/audioImport/timeSignature");
static const Settings::Key TEMPO_KEY(MODULE, "io/audioImport/tempo");
static const Settings::Key TIMEOUT_KEY(MODULE, "io/audioImport/timeoutSeconds");

void AudioImportConfiguration::init()
{
    // All keys are editable in Preferences > Advanced.
    settings()->setDefaultValue(PYTHON_KEY, Val(std::string()));
    settings()->setDescription(PYTHON_KEY, "Audio import: Python interpreter for the transcription engine (empty = auto-detect)");
    settings()->setCanBeManuallyEdited(PYTHON_KEY, true);

    settings()->setDefaultValue(ENGINE_KEY, Val(std::string()));
    settings()->setDescription(ENGINE_KEY, "Audio import: path to audio2shapenote.py (empty = bundled copy)");
    settings()->setCanBeManuallyEdited(ENGINE_KEY, true);

    settings()->setDefaultValue(SCHEME_KEY, Val(std::string("shape-7-aikin")));
    settings()->setDescription(SCHEME_KEY,
                               "Audio import: notehead scheme (shape-7-aikin, shape-7-funk, shape-7-walker, shape-4, normal)");
    settings()->setCanBeManuallyEdited(SCHEME_KEY, true);

    settings()->setDefaultValue(VOICES_KEY, Val(0));
    settings()->setDescription(VOICES_KEY, "Audio import: number of vocal parts, 1-4 (0 = detect)");
    settings()->setCanBeManuallyEdited(VOICES_KEY, true, Val(0), Val(4));

    settings()->setDefaultValue(TIMESIG_KEY, Val(std::string("auto")));
    settings()->setDescription(TIMESIG_KEY, "Audio import: time signature, e.g. 3/4 (auto = detect)");
    settings()->setCanBeManuallyEdited(TIMESIG_KEY, true);

    settings()->setDefaultValue(TEMPO_KEY, Val(0));
    settings()->setDescription(TEMPO_KEY, "Audio import: tempo in BPM (0 = detect)");
    settings()->setCanBeManuallyEdited(TEMPO_KEY, true, Val(0), Val(300));

    settings()->setDefaultValue(TIMEOUT_KEY, Val(600));
    settings()->setDescription(TIMEOUT_KEY, "Audio import: maximum transcription time in seconds");
    settings()->setCanBeManuallyEdited(TIMEOUT_KEY, true, Val(10), Val(7200));
}

std::string AudioImportConfiguration::pythonExecutable() const
{
    return settings()->value(PYTHON_KEY).toString();
}

void AudioImportConfiguration::setPythonExecutable(const std::string& path)
{
    settings()->setSharedValue(PYTHON_KEY, Val(path));
}

io::path_t AudioImportConfiguration::enginePath() const
{
    return io::path_t(settings()->value(ENGINE_KEY).toString());
}

void AudioImportConfiguration::setEnginePath(const io::path_t& path)
{
    settings()->setSharedValue(ENGINE_KEY, Val(path.toStdString()));
}

std::string AudioImportConfiguration::noteheadScheme() const
{
    return settings()->value(SCHEME_KEY).toString();
}

void AudioImportConfiguration::setNoteheadScheme(const std::string& scheme)
{
    settings()->setSharedValue(SCHEME_KEY, Val(scheme));
}

int AudioImportConfiguration::voiceCount() const
{
    return settings()->value(VOICES_KEY).toInt();
}

void AudioImportConfiguration::setVoiceCount(int voices)
{
    settings()->setSharedValue(VOICES_KEY, Val(voices));
}

std::string AudioImportConfiguration::timeSignature() const
{
    return settings()->value(TIMESIG_KEY).toString();
}

void AudioImportConfiguration::setTimeSignature(const std::string& timeSig)
{
    settings()->setSharedValue(TIMESIG_KEY, Val(timeSig));
}

int AudioImportConfiguration::tempo() const
{
    return settings()->value(TEMPO_KEY).toInt();
}

void AudioImportConfiguration::setTempo(int bpm)
{
    settings()->setSharedValue(TEMPO_KEY, Val(bpm));
}

int AudioImportConfiguration::timeoutSeconds() const
{
    return settings()->value(TIMEOUT_KEY).toInt();
}
