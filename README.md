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

## 近期稳定性修复（VAE + CoM）
**Summary**
- Actor/VAE 观测加入 `joint_torques` 且统一 `scale=0.05`，避免 `vae_decode_loss` 被扭矩量级拉爆。
- AMP-VAE 与 AMP-VAE-ViT 中引入 `vae_com_scale=10.0`，并在 `act` 与 `update` 中同步放大 CoM（Teacher Forcing 一致）。
- VIT/Perception 分支同步更新观测维度（`num_actor_obs`/`num_vae_obs` +12 for v1d6）。

**Rationale**
- 扭矩未归一化会导致 VAE 解码重建误差占据主导，掩盖 Mass/CoM 梯度。
- CoM 量级过小会出现梯度消失，放大后能与其它重建项同阶。

**Impact**
- 新模型与旧 checkpoint 在 VAE 解码维度上不兼容（需用匹配配置加载）。

**Testing**
- 未运行完整训练，仅做静态一致性修改。

## 改进方向：
1. 增加 派生动作奖励，让机器人学会落足点定位在足端半径范围内方差比较小的地方。
2.  用VAE去尝试学习机器人背上的负载状态。
3.  

## idea
1. 能不能通过约束让机器人理解约束，最后达到约束几乎不被违反。
