# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os
from pathlib import Path

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

_env_root_override = os.environ.get("RL_SIM_ENV_ROOT_DIR")
RL_SIM_ENV_ROOT_DIR = (
    Path(_env_root_override).expanduser().resolve()
    if _env_root_override
    else Path(__file__).resolve().parents[4]
)
