# SPDX-License-Identifier: GPL-3.0-only
#
# MuseScore Studio
# Music Composition & Notation
#
# Copyright (C) 2026 MuseScore Limited and others
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


muse_create_module(iex_audioimport)

target_sources(iex_audioimport PRIVATE
    audioimportmodule.cpp
    audioimportmodule.h
    iaudioimportconfiguration.h
    internal/audioimportconfiguration.cpp
    internal/audioimportconfiguration.h
    internal/audioimportoptions.cpp
    internal/audioimportoptions.h
    internal/notationaudioreader.cpp
    internal/notationaudioreader.h
)

target_link_libraries(iex_audioimport PRIVATE engraving)

# Ship the transcription engine (Python) with the app resources:
#   <resources>/audio2shapenote/audio2shapenote.py
install(FILES
    ${PROJECT_SOURCE_DIR}/tools/audio2shapenote/audio2shapenote.py
    ${PROJECT_SOURCE_DIR}/tools/audio2shapenote/requirements.txt
    ${PROJECT_SOURCE_DIR}/tools/audio2shapenote/setup_engine.py
    ${PROJECT_SOURCE_DIR}/tools/audio2shapenote/separate.py
    DESTINATION ${MUSE_APP_INSTALL_RESOURCES_LOCATION}/audio2shapenote
)

if (MUE_BUILD_IMPORTEXPORT_TESTS)
    add_subdirectory(tests)
endif()
