# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foothold_terrain_flatness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    variance_scale: float | None = None,
    sample_stride: int = 1,
    variance_kernel_size: int = 3,
    decay_k: float = 1.0,
    decay_sigma: float = 1.0,
    min_sigma: float = 1.0e-3,
    max_decay_exp: float = 5.0,
    use_derived_action: bool = True,
) -> torch.Tensor:
    """Penalize footholds on uneven terrain using local height variance.

    If derived-action output is available (env.derived_action), compute the expected
    terrain variance under the predicted foothold distribution (Eq. 4 in DAO).
    Otherwise, fall back to local variance at the actual foot positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None or len(asset_cfg.body_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]

    sensor = env.scene.sensors.get(sensor_cfg.name, None) if sensor_cfg is not None else None
    if isinstance(sensor, RayCaster):
        pattern_cfg = getattr(sensor.cfg, "pattern_cfg", None)
        resolution = getattr(pattern_cfg, "resolution", None)
        size = getattr(pattern_cfg, "size", None)
        if resolution is None or size is None or resolution <= 0:
            return torch.zeros(env.scene.num_envs, device=env.device)

        size_x = float(size[0])
        size_y = float(size[1])
        num_x = int(round(size_x / resolution)) + 1
        num_y = int(round(size_y / resolution)) + 1
        stride = max(int(sample_stride), 1)

        ray_z = sensor.data.ray_hits_w[..., 2]
        ray_z = torch.where(torch.isfinite(ray_z), ray_z, torch.zeros_like(ray_z))
        if ray_z.shape[1] != num_x * num_y:
            return torch.zeros(env.scene.num_envs, device=env.device)

        derived_action = getattr(env, "derived_action", None) if use_derived_action else None
        if derived_action is not None:
            if derived_action.ndim == 2:
                if derived_action.shape[1] % 3 != 0:
                    return torch.zeros(env.scene.num_envs, device=env.device)
                derived_action = derived_action.view(derived_action.shape[0], -1, 3)
            if derived_action.ndim != 3 or derived_action.shape[2] < 3:
                return torch.zeros(env.scene.num_envs, device=env.device)
            derived_action = derived_action.to(foot_pos_w.device)

            ordering = getattr(pattern_cfg, "ordering", "xy")
            if ordering == "yx":
                ray_z_2d = ray_z.view(ray_z.shape[0], num_x, num_y).transpose(1, 2)
            else:
                ray_z_2d = ray_z.view(ray_z.shape[0], num_y, num_x)

            k = max(int(variance_kernel_size), 1)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            kernel = torch.ones((1, 1, k, k), device=ray_z_2d.device, dtype=ray_z_2d.dtype) / float(k * k)
            ray_z_2d = ray_z_2d.unsqueeze(1)
            ray_z_pad = F.pad(ray_z_2d, (pad, pad, pad, pad), mode="replicate")
            mean = F.conv2d(ray_z_pad, kernel)
            mean_sq = F.conv2d(ray_z_pad.pow(2), kernel)
            var_map = (mean_sq - mean.pow(2)).squeeze(1)

            if stride > 1:
                var_map = var_map[:, ::stride, ::stride]

            x_coords = torch.linspace(
                -0.5 * size_x, 0.5 * size_x, num_x, device=var_map.device, dtype=var_map.dtype
            )
            y_coords = torch.linspace(
                -0.5 * size_y, 0.5 * size_y, num_y, device=var_map.device, dtype=var_map.dtype
            )
            if stride > 1:
                x_coords = x_coords[::stride]
                y_coords = y_coords[::stride]

            num_x_eff = x_coords.numel()
            num_y_eff = y_coords.numel()
            grid_x = x_coords.repeat(num_y_eff)
            grid_y = y_coords.repeat_interleave(num_x_eff)

            var_flat = var_map.reshape(var_map.shape[0], -1)
            if variance_scale is None:
                variance_scale = 1.0 / max(float(resolution), 1.0e-6) ** 2
            var_flat = var_flat * float(variance_scale)

            mu_b = derived_action[..., :2]
            sigma = torch.clamp(derived_action[..., 2], min=min_sigma)
            mu_b3 = torch.cat((mu_b, torch.zeros_like(mu_b[..., :1])), dim=-1)
            base_quat = sensor.data.quat_w.unsqueeze(1).expand(-1, mu_b3.shape[1], -1)
            base_pos = sensor.data.pos_w.unsqueeze(1)
            mu_w = quat_apply(base_quat, mu_b3) + base_pos
            rel = mu_w - base_pos
            alignment = getattr(sensor.cfg, "ray_alignment", "yaw")
            if alignment == "world":
                rel_local = rel
            elif alignment == "base":
                rel_local = quat_apply_inverse(base_quat, rel)
            else:
                yaw_q = yaw_quat(sensor.data.quat_w).unsqueeze(1).expand(-1, mu_b3.shape[1], -1)
                rel_local = quat_apply_inverse(yaw_q, rel)
            offset = mu_b3.new_tensor(sensor.cfg.offset.pos)
            rel_local = rel_local - offset
            mu_local = rel_local[..., :2]

            dx = grid_x[None, None, :] - mu_local[..., 0].unsqueeze(-1)
            dy = grid_y[None, None, :] - mu_local[..., 1].unsqueeze(-1)
            dist_sq = dx.pow(2) + dy.pow(2)
            sigma_sq = torch.clamp(sigma.unsqueeze(-1).pow(2), min=min_sigma**2)
            norm = 2.0 * math.pi * sigma_sq
            p = torch.exp(-0.5 * dist_sq / sigma_sq) / norm
            cell_area = (float(resolution) * stride) ** 2
            expected = (p * var_flat.unsqueeze(1)).sum(dim=-1) * cell_area
            decay_denom = torch.clamp(decay_sigma * sigma, min=min_sigma)
            decay = torch.exp(torch.clamp(decay_k / decay_denom, max=max_decay_exp))
            return torch.sum(expected * decay, dim=1)

        rel = foot_pos_w - sensor.data.pos_w.unsqueeze(1)
        alignment = getattr(sensor.cfg, "ray_alignment", "yaw")
        if alignment == "world":
            rel_local = rel
        elif alignment == "base":
            quat = sensor.data.quat_w.unsqueeze(1).repeat(1, rel.shape[1], 1)
            rel_local = quat_apply_inverse(quat, rel)
        else:
            quat = yaw_quat(sensor.data.quat_w).unsqueeze(1).repeat(1, rel.shape[1], 1)
            rel_local = quat_apply_inverse(quat, rel)

        offset = foot_pos_w.new_tensor(sensor.cfg.offset.pos)
        rel_local = rel_local - offset

        x_min = -0.5 * size_x
        y_min = -0.5 * size_y
        ix = torch.round((rel_local[..., 0] - x_min) / resolution).long()
        iy = torch.round((rel_local[..., 1] - y_min) / resolution).long()
        ix = torch.clamp(ix, 0, num_x - 1)
        iy = torch.clamp(iy, 0, num_y - 1)

        ix_plus = torch.clamp(ix + stride, 0, num_x - 1)
        ix_minus = torch.clamp(ix - stride, 0, num_x - 1)
        iy_plus = torch.clamp(iy + stride, 0, num_y - 1)
        iy_minus = torch.clamp(iy - stride, 0, num_y - 1)

        ordering = getattr(pattern_cfg, "ordering", "xy")
        if ordering == "yx":
            idx = ix * num_y + iy
            idx_xp = ix_plus * num_y + iy
            idx_xm = ix_minus * num_y + iy
            idx_yp = ix * num_y + iy_plus
            idx_ym = ix * num_y + iy_minus
        else:
            idx = iy * num_x + ix
            idx_xp = iy * num_x + ix_plus
            idx_xm = iy * num_x + ix_minus
            idx_yp = iy_plus * num_x + ix
            idx_ym = iy_minus * num_x + ix

        h0 = ray_z.gather(1, idx)
        hx_plus = ray_z.gather(1, idx_xp)
        hx_minus = ray_z.gather(1, idx_xm)
        hy_plus = ray_z.gather(1, idx_yp)
        hy_minus = ray_z.gather(1, idx_ym)
        heights = torch.stack((h0, hx_plus, hx_minus, hy_plus, hy_minus), dim=-1)
        variances = torch.var(heights, dim=-1, unbiased=False)
        if variance_scale is None:
            variance_scale = 1.0 / max(float(resolution), 1.0e-6) ** 2
        variances = variances * float(variance_scale)
        return torch.sum(variances, dim=1)

    terrain = getattr(env.scene, "terrain", None)
    height_field = getattr(terrain, "height_field", None) if terrain is not None else None
    if height_field is None:
        return torch.zeros(env.scene.num_envs, device=env.device)

    horizontal_scale = getattr(terrain, "horizontal_scale", None)
    vertical_scale = getattr(terrain, "vertical_scale", None)
    if horizontal_scale is None or vertical_scale is None:
        cfg = getattr(terrain, "cfg", None)
        gen_cfg = getattr(cfg, "terrain_generator", None) if cfg is not None else None
        horizontal_scale = getattr(gen_cfg, "horizontal_scale", horizontal_scale)
        vertical_scale = getattr(gen_cfg, "vertical_scale", vertical_scale)
    if horizontal_scale is None or vertical_scale is None:
        return torch.zeros(env.scene.num_envs, device=env.device)

    hf = torch.as_tensor(height_field, device=env.device, dtype=torch.float32) * float(vertical_scale)
    num_rows, num_cols = hf.shape
    width_m = (num_rows - 1) * horizontal_scale
    length_m = (num_cols - 1) * horizontal_scale

    foot_pos_xy = foot_pos_w[..., :2]

    r = ((foot_pos_xy[..., 0] + 0.5 * width_m) / horizontal_scale).long()
    c = ((foot_pos_xy[..., 1] + 0.5 * length_m) / horizontal_scale).long()
    r = torch.clamp(r, 0, num_rows - 1)
    c = torch.clamp(c, 0, num_cols - 1)

    r_plus = torch.clamp(r + 1, 0, num_rows - 1)
    r_minus = torch.clamp(r - 1, 0, num_rows - 1)
    c_plus = torch.clamp(c + 1, 0, num_cols - 1)
    c_minus = torch.clamp(c - 1, 0, num_cols - 1)

    h0 = hf[r, c]
    hx_plus = hf[r_plus, c]
    hx_minus = hf[r_minus, c]
    hy_plus = hf[r, c_plus]
    hy_minus = hf[r, c_minus]
    heights = torch.stack((h0, hx_plus, hx_minus, hy_plus, hy_minus), dim=-1)
    variances = torch.var(heights, dim=-1, unbiased=False)
    if variance_scale is None:
        variance_scale = 1.0 / max(float(horizontal_scale), 1.0e-6) ** 2
    variances = variances * float(variance_scale)
    return torch.sum(variances, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power distribution."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.var(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def amp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward AMP."""
    # extract the used quantities (to enable type-hinting)
    reward = torch.clamp(1 - (1 / 4) * torch.square(env.amp_out - 1), min=0)
    return reward.squeeze()


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward action smoothness."""
    # extract the used quantities (to enable type-hinting)
    actions = env.actions_history
    diff = torch.square(actions.get_data_vec([0]) - 2 * actions.get_data_vec([1]) + actions.get_data_vec([2]))
    return torch.sum(diff, dim=1)


def stand_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    return reward


def base_height_l2_fix(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        # 检查sensor数据是否包含inf或nan
        ray_hits = sensor.data.ray_hits_w[..., 2]
        ray_hits = torch.where(torch.isinf(ray_hits), 0.0, ray_hits)
        ray_hits = torch.where(torch.isnan(ray_hits), 0.0, ray_hits)
        adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
