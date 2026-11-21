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
#include "notationaudioreader.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>

#include "engraving/dom/chord.h"
#include "engraving/dom/masterscore.h"
#include "engraving/dom/note.h"
#include "engraving/dom/segment.h"
#include "engraving/dom/staff.h"
#include "engraving/dom/stafftype.h"
#include "engraving/engravingerrors.h"

#include "audioimportoptions.h"

#include "log.h"

using namespace muse;
using namespace mu::engraving;
using namespace mu::iex::audioimport;

static const char* ENGINE_FILE = "audio2shapenote.py";

std::vector<std::string> NotationAudioReader::supportedSuffixes()
{
    return { "wav", "mp3", "flac", "ogg", "m4a", "aif", "aiff" };
}

io::path_t NotationAudioReader::resolveEnginePath() const
{
    io::path_t configured = configuration()->enginePath();
    if (!configured.empty()) {
        return configured;
    }

    const QByteArray env = qgetenv("MUSESCORE_AUDIO2SHAPENOTE");
    if (!env.isEmpty()) {
        return io::path_t(QString::fromLocal8Bit(env));
    }

    // Installed with the app resources (see CMakeLists.txt).
    return globalConfiguration()->appDataPath() + "/audio2shapenote/" + ENGINE_FILE;
}

std::string NotationAudioReader::resolvePython(const io::path_t& enginePath) const
{
    std::string configured = configuration()->pythonExecutable();
    if (!configured.empty()) {
        return configured;
    }

    const QByteArray env = qgetenv("MUSESCORE_AUDIO_PYTHON");
    if (!env.isEmpty()) {
        return env.toStdString();
    }

    // A virtual environment created by setup_engine.py: next to the engine, or
    // (`setup_engine.py --user`, needed for read-only installs such as AppImage)
    // in the per-user generic data folder: ~/.local/share, %LOCALAPPDATA%,
    // ~/Library/Application Support.
    const QStringList venvDirs {
        QFileInfo(enginePath.toQString()).absolutePath() + "/.venv",
        QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/audio2shapenote/.venv",
        globalConfiguration()->userAppDataPath().toQString() + "/audio2shapenote/.venv",
    };
    for (const QString& dir : venvDirs) {
#ifdef Q_OS_WIN
        const QString candidate = dir + "/Scripts/python.exe";
#else
        const QString candidate = dir + "/bin/python";
#endif
        if (QFileInfo::exists(candidate)) {
            return candidate.toStdString();
        }
    }

    for (const char* name : { "python3", "python" }) {
        const QString found = QStandardPaths::findExecutable(name);
        if (!found.isEmpty()) {
            return found.toStdString();
        }
    }
    return "python3";
}

muse::Ret NotationAudioReader::read(MasterScore* score, const io::path_t& path, const Options& options)
{
    const io::path_t enginePath = resolveEnginePath();
    if (!QFileInfo::exists(enginePath.toQString())) {
        return make_ret(Ret::Code::NotSupported,
                        "Audio import engine not found at " + enginePath.toStdString()
                        + ". Set io/audioImport/enginePath in Preferences > Advanced.");
    }

    QTemporaryDir tmp;
    if (!tmp.isValid()) {
        return make_ret(Err::FileOpenError, path);
    }
    const QString xmlPath = tmp.filePath("transcription.musicxml");
    const QString reportPath = tmp.filePath("report.json");

    AudioImportOptions opts;
    opts.scheme = configuration()->noteheadScheme();
    opts.voices = configuration()->voiceCount();
    opts.timeSignature = configuration()->timeSignature();
    opts.tempo = configuration()->tempo();

    const std::string python = resolvePython(enginePath);
    const std::vector<std::string> args = engineArguments(enginePath.toStdString(), path.toStdString(),
                                                          xmlPath.toStdString(), reportPath.toStdString(), opts);
    QStringList qargs;
    for (const std::string& a : args) {
        qargs << QString::fromStdString(a);
    }

    LOGI() << "audio import: " << python << " " << qargs.join(' ');

    QProcess process;
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start(QString::fromStdString(python), qargs);
    if (!process.waitForStarted(15000)) {
        return make_ret(Ret::Code::NotSupported,
                        "Could not start Python (" + python + "). Install Python 3.10+ and run setup_engine.py, "
                                                              "or set io/audioImport/pythonExecutable in Preferences > Advanced.");
    }

    const int timeoutMs = std::max(10, configuration()->timeoutSeconds()) * 1000;
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished(3000);
        return make_ret(Ret::Code::UnknownError, std::string("Audio transcription timed out."));
    }

    const QString stderrText = QString::fromUtf8(process.readAllStandardError()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        LOGE() << "audio import engine failed: " << stderrText;
        std::string msg = engineErrorMessage(process.exitStatus() == QProcess::NormalExit ? process.exitCode() : -1);
        const QString lastLine = stderrText.section('\n', -1);
        if (!lastLine.isEmpty()) {
            msg += "\n" + lastLine.toStdString();
        }
        return make_ret(Ret::Code::UnknownError, msg);
    }

    if (!QFileInfo::exists(xmlPath)) {
        return make_ret(Ret::Code::UnknownError, std::string("The transcription engine produced no score."));
    }

    project::INotationReaderPtr xmlReader = readers() ? readers()->reader("musicxml") : nullptr;
    if (!xmlReader) {
        return make_ret(Ret::Code::NotSupported, std::string("MusicXML import is not available in this build."));
    }

    Ret ret = xmlReader->read(score, io::path_t(xmlPath), options);
    if (!ret) {
        return ret;
    }

    applyNoteheadScheme(score, schemeFromString(opts.scheme));
    return make_ok();
}

void NotationAudioReader::applyNoteheadScheme(MasterScore* score, NoteHeadScheme scheme)
{
    if (!score) {
        return;
    }

    for (Staff* staff : score->staves()) {
        StaffType* st = staff->staffType(Fraction(0, 1));
        if (st && st->group() == StaffGroup::STANDARD) {
            st->setNoteHeadScheme(scheme);
        }
    }

    // MusicXML import turns <notehead>do</notehead> into fixed heads; reset them
    // so the staff scheme decides the shape, and shapes follow later edits.
    for (Segment* s = score->firstSegment(SegmentType::ChordRest); s; s = s->next1(SegmentType::ChordRest)) {
        for (EngravingItem* e : s->elist()) {
            if (!e || !e->isChord()) {
                continue;
            }
            for (Note* n : toChord(e)->notes()) {
                n->setHeadGroup(NoteHeadGroup::HEAD_NORMAL);
                n->setHeadScheme(NoteHeadScheme::HEAD_AUTO);
            }
        }
    }
}
