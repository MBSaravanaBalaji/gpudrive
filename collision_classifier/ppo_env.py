"""
CrashVecEnv: multi-world GPUDrive VecEnv for crash NPC training.

Runs num_worlds parallel Waymo scenes, each with one controlled NPC (agent 0).
Compatible with IPPO from gpudrive.integrations.sb3.ppo and FeedForwardPolicy.

Key choices vs single-env DQN:
- num_worlds parallelism → more diverse scene geometry per rollout
- PPO (on-policy) → stable with sparse terminal rewards + dense shaping
- Auto-reset done worlds immediately → standard SB3 VecEnv contract
- Same 9-float custom obs and crash classifier reward as env_wrapper.py
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv

from collision_classifier.classifier import check_and_classify
from collision_classifier.env_wrapper import (
    CRASH_TYPES,
    _CRASH_TYPE_MAP,
    _TARGET_DIRS,
    _PADDING_DIST,
    R_MATCH,
    R_WRONG,
    W_SHAPING,
    PROXIMITY_SCALE,
)

_VEHICLE_SCALE = None


def _vscale() -> float:
    global _VEHICLE_SCALE
    if _VEHICLE_SCALE is None:
        from madrona_gpudrive import vehicleScale
        _VEHICLE_SCALE = vehicleScale
    return _VEHICLE_SCALE


class CrashVecEnv(VecEnv):
    """
    num_worlds parallel GPUDrive scenes.
    Each world has one controlled NPC (agent 0) and log-replay background agents.
    """

    OBS_DIM = 9
    DATA_DIR = "data/processed/examples"

    def __init__(
        self,
        crash_type: str,
        num_worlds: int = 64,
        data_dir: str = DATA_DIR,
        dataset_size: int = 50,
        seed: int = 42,
        device: str = _DEFAULT_DEVICE,
    ):
        assert crash_type in CRASH_TYPES, f"crash_type must be one of {CRASH_TYPES}"
        self.crash_type = crash_type
        self.target_labels = _CRASH_TYPE_MAP[crash_type]
        self.target_dir = np.array(_TARGET_DIRS[crash_type], dtype=np.float32)
        self.target_dir_t = torch.tensor(_TARGET_DIRS[crash_type], dtype=torch.float32)
        self.num_worlds = num_worlds
        self.device = device

        config = EnvConfig(
            dynamics_model="classic",
            collision_behavior="stop",
            max_controlled_agents=1,
            num_worlds=num_worlds,
            ego_state=True,
            road_map_obs=False,
            partner_obs=False,
            norm_obs=False,
        )

        data_loader = SceneDataLoader(
            root=data_dir,
            batch_size=num_worlds,
            dataset_size=dataset_size,
            sample_with_replacement=True,
            shuffle=True,
            seed=seed,
            file_prefix="tfrecord",
        )

        self._env = GPUDriveTorchEnv(
            config=config,
            data_loader=data_loader,
            max_cont_agents=1,
            device=device,
            action_type="discrete",
            render_config=RenderConfig(),
        )

        self.max_agent_count = self._env.max_agent_count
        self.episode_len = self._env.episode_len

        observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS_DIM,), np.float32)
        action_space = gym.spaces.Discrete(self._env.action_space.n)
        super().__init__(num_worlds, observation_space, action_space)

        self.controlled_agent_mask = torch.zeros(
            num_worlds, self.max_agent_count, dtype=torch.bool, device=device
        )
        self.controlled_agent_mask[:, 0] = True
        self.dead_agent_mask = torch.zeros(
            num_worlds, self.max_agent_count, dtype=torch.bool, device=device
        )

        self.exp_config = SimpleNamespace(resample_scenes=False)

        # Per-world bookkeeping — kept on device for vectorized indexing
        self._step_counts = torch.zeros(num_worlds, dtype=torch.int32, device=device)
        self._ego_idxs = torch.ones(num_worlds, dtype=torch.long, device=device)
        self._arange = torch.arange(num_worlds, dtype=torch.long, device=device)

        # Move target direction tensor to device after super().__init__
        self.target_dir_t = self.target_dir_t.to(device)

        # Stats for external logging
        self.crash_hits: List[int] = []
        self.crash_labels_buf: List[Optional[str]] = []

    # ── VecEnv interface ──────────────────────────────────────────────────────

    def reset(self, world_idx=None, seed=None, **kwargs):
        if world_idx is None:
            self._env.reset()
            self._step_counts.zero_()
            self._update_all_ego_targets()
        else:
            worlds = world_idx.tolist() if isinstance(world_idx, torch.Tensor) else list(world_idx)
            self._env.sim.reset(worlds)
            self._step_counts[world_idx] = 0
            self._update_ego_targets_for(worlds)
        return self._get_obs()

    def swap_scenes(self):
        """Load a new batch of Waymo scenes from the data loader and full-reset."""
        self._env.swap_data_batch()
        return self.reset()

    def step(self, actions) -> VecEnvStepReturn:
        full_actions = torch.zeros(
            self.num_worlds, self.max_agent_count, dtype=torch.long, device=self.device
        )
        full_actions[:, 0] = actions.long()

        self._env.step_dynamics(full_actions)
        self._step_counts += 1

        # Single state copy for the whole step
        state  = self._global_state()   # (W, A, 14) on device
        speeds = self._local_speeds()   # (W, A)     on device

        # info_tensor layout: [off_road, collidedWithVehicle, collidedWithNonVehicle, goal_achieved, agent_type]
        # get_infos().collided sums columns 1+2 (vehicle + non-vehicle) — we want vehicle-only.
        info_t       = self._env.sim.info_tensor().to_torch().clone()  # (W, A, 5)
        collided     = info_t[:, 0, 1].bool()   # collidedWithVehicle for NPC
        goal_reached = info_t[:, 0, 3].bool()
        timeout      = self._step_counts >= self.episode_len

        shaping  = self._compute_shaping(state)
        terminal, crash_labels = self._compute_terminal(collided, state)
        rewards  = shaping + terminal  # on device

        dones = collided | goal_reached | timeout

        # Collect stats before reset
        for w in range(self.num_worlds):
            if dones[w]:
                label = crash_labels[w]
                if collided[w].item() and label is not None:
                    self.crash_labels_buf.append(label)
                    self.crash_hits.append(1 if label in self.target_labels else 0)
                else:
                    self.crash_hits.append(0)

        obs = self._build_obs(state, speeds)
        done_worlds = torch.where(dones)[0]
        if len(done_worlds) > 0:
            self.reset(done_worlds)
            fresh_state  = self._global_state()
            fresh_speeds = self._local_speeds()
            obs[done_worlds] = self._build_obs(fresh_state, fresh_speeds)[done_worlds]

        info_list = [
            {
                "crash_label": crash_labels[w],
                "collided": bool(collided[w].item()),
                "goal_reached": bool(goal_reached[w].item()),
            }
            for w in range(self.num_worlds)
        ]

        return obs, rewards, dones.float(), info_list

    def close(self) -> None:
        pass

    def seed(self, seed=None):
        return [seed] * self.num_worlds

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * self.num_worlds
        raise NotImplementedError(f"get_attr({attr_name})")

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> torch.Tensor:
        return self._build_obs(self._global_state(), self._local_speeds())

    def _build_obs(self, state: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
        """Vectorized 9-float obs, stays on device — no Python loop."""
        npc_xy  = state[:, 0, :2]                                     # (W, 2)
        npc_h   = state[:, 0, 7]                                      # (W,)
        npc_spd = speeds[:, 0]                                        # (W,)

        ego_xy  = state[self._arange, self._ego_idxs, :2]            # (W, 2)
        ego_h   = state[self._arange, self._ego_idxs, 7]             # (W,)
        ego_spd = speeds[self._arange, self._ego_idxs]               # (W,)

        delta = ego_xy - npc_xy                                       # (W, 2)
        cos_nh = torch.cos(-npc_h)
        sin_nh = torch.sin(-npc_h)
        rel_x = cos_nh * delta[:, 0] - sin_nh * delta[:, 1]
        rel_y = sin_nh * delta[:, 0] + cos_nh * delta[:, 1]

        rel_h = ((ego_h - npc_h + math.pi) % (2 * math.pi)) - math.pi
        dist  = torch.linalg.norm(delta, dim=1)

        return torch.stack([
            npc_spd,
            torch.cos(npc_h), torch.sin(npc_h),
            rel_x, rel_y,
            torch.cos(rel_h), torch.sin(rel_h),
            ego_spd, dist,
        ], dim=1)  # (W, 9) on device

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_shaping(self, state: torch.Tensor) -> torch.Tensor:
        """Vectorized per-step alignment shaping, stays on device."""
        npc_xy = state[:, 0, :2]                                      # (W, 2)
        ego_xy = state[self._arange, self._ego_idxs, :2]             # (W, 2)
        ego_h  = state[self._arange, self._ego_idxs, 7]              # (W,)

        delta = npc_xy - ego_xy                                       # (W, 2)
        d = torch.linalg.norm(delta, dim=1)                          # (W,)

        cos_h = torch.cos(ego_h)
        sin_h = torch.sin(ego_h)
        lx =  cos_h * delta[:, 0] + sin_h * delta[:, 1]
        ly = -sin_h * delta[:, 0] + cos_h * delta[:, 1]

        safe_d = d.clamp(min=1e-6)
        alignment = lx / safe_d * self.target_dir_t[0] + ly / safe_d * self.target_dir_t[1]
        proximity  = 1.0 / (1.0 + d / PROXIMITY_SCALE)

        shaping = W_SHAPING * alignment * proximity
        shaping = shaping * (d >= 1e-6).float()
        return shaping

    def _compute_terminal(
        self, collided: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Optional[str]]]:
        terminal = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        labels: List[Optional[str]] = [None] * self.num_worlds

        collided_ws = torch.where(collided)[0]
        if len(collided_ws) == 0:
            return terminal, labels

        vs = _vscale()
        # Only pay the CPU transfer when there are actual collisions
        state_np = state.cpu().numpy()

        for w in collided_ws.tolist():
            npc = state_np[w, 0]
            npc_pos = npc[:2]
            for i in range(1, state_np.shape[1]):
                other = state_np[w, i]
                if np.linalg.norm(other[:2] - npc_pos) > _PADDING_DIST:
                    continue
                result = check_and_classify(
                    ego_pos_x=float(other[0]), ego_pos_y=float(other[1]),
                    ego_heading=float(other[7]),
                    ego_length=float(other[10]) * vs, ego_width=float(other[11]) * vs,
                    npc_pos_x=float(npc[0]), npc_pos_y=float(npc[1]),
                    npc_heading=float(npc[7]),
                    npc_length=float(npc[10]) * vs, npc_width=float(npc[11]) * vs,
                )
                if result is not None:
                    label = result.collision_type
                    labels[w] = label
                    terminal[w] = R_MATCH if label in self.target_labels else -R_WRONG
                    break

        return terminal, labels

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _global_state(self) -> torch.Tensor:
        """(W, max_agents, 14) on self.device."""
        return (
            self._env.sim.absolute_self_observation_tensor()
            .to_torch().clone()
        )

    def _local_speeds(self) -> torch.Tensor:
        """(W, max_agents) on self.device."""
        return (
            self._env.sim.self_observation_tensor()
            .to_torch().clone()[:, :, 0]
        )

    def _update_all_ego_targets(self):
        state_np = self._global_state().cpu().numpy()
        for w in range(self.num_worlds):
            self._ego_idxs[w] = self._find_target_ego(state_np[w])

    def _update_ego_targets_for(self, worlds: List[int]):
        state_np = self._global_state().cpu().numpy()
        for w in worlds:
            self._ego_idxs[w] = self._find_target_ego(state_np[w])

    def _find_target_ego(self, world_state_np: np.ndarray) -> int:
        npc_xy = world_state_np[0, :2]
        best_idx, best_dist = 1, float("inf")
        for i in range(1, world_state_np.shape[0]):
            xy = world_state_np[i, :2]
            d = float(np.linalg.norm(xy - npc_xy))
            if d < _PADDING_DIST and d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def crash_rate(self, window: int = 200) -> float:
        hits = self.crash_hits[-window:]
        return sum(hits) / max(len(hits), 1)
