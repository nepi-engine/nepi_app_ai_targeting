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
FILE_TYPE = 'APP'
APP_DICT = dict(
    description = 'Application for advanced targeting of AI detected objects',
    pkg_name = 'nepi_app_ai_targeting',
    group_name = 'AI',
    config_file = 'app_ai_targeting.yaml',
    app_file = 'ai_targeting_app_node.py',
    node_name = 'app_ai_targeting'
)
RUI_DICT = dict(
    rui_menu_name = "AI Targeting", # RUI menu name or "None" if no rui support
    rui_files = ['NepiAppAiTargeting.js','NepiAppAiTargetingControls.js'],
    rui_main_file = "NepiAppAiTargeting.js",
    rui_main_class = "AiTargetingApp"
)




