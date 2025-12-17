# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

# 1. 注册标准训练环境 (V2d3)
gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V2d3-v0",
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V2d3AmpVaeEnvCfg",
        # === 修改处：同时保留旧键，并添加新键 rsl_rl_cfg_entry_point ===
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg", # <--- 新增这一行
    },
)
# 2. 注册 Play 环境 (用于推理/演示)
gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V2d3-Play-v0",
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V2d3AmpVaeEnvCfg_PLAY",
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg", # <--- 新增
    },
)

# 3. 注册 Replay 环境 (如果有需要的话)
gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V2d3-ReplayAmpData-v0",  # <--- 关键修改
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        # <--- 关键修改：指向 V2d3 的 Replay 配置类
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V2d3AmpVaeEnvCfg_REPLAY_AMPDATA",
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
    },
)