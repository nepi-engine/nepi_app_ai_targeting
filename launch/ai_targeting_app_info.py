#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

APP_NAME = 'AI_Targeting' # Use in display menus
DESCRIPTION = 'Application for advanced targeting of AI detected objects'
PKG_NAME = 'nepi_app_ai_targeting'
APP_FILE = 'ai_targeting_app_node.py'
NODE_NAME = 'app_ai_targeting'
RUI_FILES = ['NepiAppAiTargeting.js','NepiAppAiTargetingControls.js']
RUI_MAIN_FILE = "NepiAppAiTargeting.js"
RUI_MAIN_CLASS = "AiTargetingApp"
RUI_MENU_NAME = "AI Targeting"

