## 安装
```bash
python -m pip install -e rl_sim_env-amp_vae_vit/source/rl_sim_env
```

## 训练

```bash
conda activate isaaclab

# 可选：清理残留
fuser -k -9 /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3
```

### 单卡

```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py     --task Rl-Sim-Env-AmpVae-Grq20-V2d3-v0     --run_name v2d3_arm_load_test     --headless     --num_envs 4096
```
### 多卡

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
    --task Rl-Sim-Env-AmpVae-Grq20-V2d3-v0 \
    --run_name v2d3_arm_load_test_dist \
    --distributed \
    --headless \
    --num_envs 2500
```

说明：
- 使用 `torchrun` 启动，`--nproc_per_node` 与可用 GPU 数一致；多节点时再加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。
- `--distributed` 开启多卡分布式，脚本会根据 `local_rank` 自动设置 `env_cfg.sim.device` 与 `agent_cfg.device`。
- `num_envs` 表示每张卡上创建的环境数；多卡总环境数约等于 `num_envs * nproc_per_node`。根据显存和负载自行调整。

## Play 可视化

```bash

conda activate isaaclab

python rl_sim_env-amp_vae_vit/scripts/rsl_rl/play_amp_vae.py \
 --task Rl-Sim-Env-AmpVae-Grq20-V2d3-Play-v0 \
 --num_envs 25 \
--checkpoint logs/rsl_rl/grq20_v2d3_amp_vae/

```

## 近期改进（AMP‑VAE + System CoM）
**Change Summary**
- VAE 输出拆分为 `vel/mass/com/latent`，`code_com` 不再污染 latent；decoder 使用完整拼接向量进行重建。
- 新增系统辨识：`critic_obs.random_com`/`random_mass` 改为系统级 `system_com` 与 `system_mass_delta`（全机器人+负载、base frame）。
- Actor/VAE 观测加入 `joint_torques` 并归一化为 `scale=0.02`，抑制 `vae_decode_loss` 的量级爆炸。
- 引入 `vae_com_scale=10.0` 并在 `act`/`update` 一致使用；`loss_recon_com` 权重下调为 `0.1`。
- 去除 `critic_obs` 固定索引假设：训练时通过 ObservationManager 动态解析切片，缺失会直接报错。
- 推理默认使用纯 VAE（`p_boot_mean=1.0`），需要上限对比可改为 `0.0`。
- 替换 `quat_rotate_inverse` 为 `quat_apply_inverse`，移除 deprecation warning。
- rl_sim_env-amp_vae_vit/scripts/rsl_rl/play_amp_vae.py 路径下 p_boot_mean 为纯 VAE 开关。
- 观测层面屏蔽机械臂/负载关节：actor/critic/VAE 仅使用腿部关节位置、速度与力矩，避免 arm 扭矩污染 VAE 解码与策略输入。

**Rationale**
- 系统 COM/总质量差异与负载强相关，能让 VAE 学到有效的负载表征。
- 扭矩/COM 量级需要与其他重建项同阶，避免梯度被压制或放大。
- 动态切片能避免观测顺序变动导致的“静默错标签”。
- 机械臂锁死时仍可能产生大扭矩，若进入 VAE/actor 会破坏解码与 AMP 稳定性。

**Config Notes**
- grq20_v2d3：`num_critic_obs=243`，`num_vae_out=23`（vel3 + mass1 + com3 + latent16）。
- grq20_v2d3（锁臂版本）：`num_actor_obs=57`、`num_vae_obs=57`、`num_critic_obs=239`（腿部 12 关节）。
- 负载随机化：`arm_load_link` 质量范围改为 `0–6kg`（`operation="abs"`）。

**Compatibility**
- 旧 checkpoint 与当前 VAE 输出维度/Actor 输入维度不兼容；需使用匹配配置重新训练或加载旧配置。

**Testing**
- 未运行完整训练，仅做静态一致性与运行时警告修复。

**Arm/Load 易错点**
- 机械臂“锁死”只代表动作不控制，但物理引擎仍可能输出很大的关节力矩（高阻尼/重载时更明显）。
- 若这些扭矩被误加入 `actor_obs` 或 `vae_decode_target`，会造成 `vae_decode_loss` 爆炸、AMP 判别器失效。
- 新增观测项时务必确认 `asset_cfg` 只覆盖腿部关节，避免 arm/负载扭矩回流到算法输入。

## 改进方向：
1. 增加 派生动作奖励，让机器人学会落足点定位在足端半径范围内方差比较小的地方。
2.  用VAE去尝试学习机器人背上的负载状态。
3.  

## idea
1. 能不能通过约束让机器人理解约束，最后达到约束几乎不被违反。
