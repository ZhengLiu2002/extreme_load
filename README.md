### 单卡

```bash
python -m pip install -e rl_sim_env-amp_vae_vit/source/rl_sim_env
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
    --num_envs 4096
```

说明：
- 使用 `torchrun` 启动，`--nproc_per_node` 与可用 GPU 数一致；多节点时再加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。
- `--distributed` 开启多卡分布式，脚本会根据 `local_rank` 自动设置 `env_cfg.sim.device` 与 `agent_cfg.device`。
- `num_envs` 填总环境数，内部会按 rank 平分；可根据显存适当增减。*** End Patch" ***!
