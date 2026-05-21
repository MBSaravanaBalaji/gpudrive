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

from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
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
        num_worlds: int = 10,
        data_dir: str = DATA_DIR,
        dataset_size: int = 50,
        seed: int = 42,
        device: str = "cpu",
    ):
        assert crash_type in CRASH_TYPES, f"crash_type must be one of {CRASH_TYPES}"
        self.crash_type = crash_type
        self.target_labels = _CRASH_TYPE_MAP[crash_type]
        self.target_dir = _TARGET_DIRS[crash_type]
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
            sample_with_replacement=True,  # allows num_worlds > unique scene count
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

        # VecEnv: 1 NPC per world → num_envs = num_worlds
        observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS_DIM,), np.float32)
        action_space = gym.spaces.Discrete(self._env.action_space.n)
        super().__init__(num_worlds, observation_space, action_space)

        # IPPO compatibility — agent 0 in each world is the controlled NPC
        self.controlled_agent_mask = torch.zeros(
            num_worlds, self.max_agent_count, dtype=torch.bool, device=device
        )
        self.controlled_agent_mask[:, 0] = True
        # dead_agent_mask: stays False (we auto-reset, so NPC is always alive)
        self.dead_agent_mask = torch.zeros(
            num_worlds, self.max_agent_count, dtype=torch.bool, device=device
        )

        # IPPO checks env.exp_config.resample_scenes in collect_rollouts
        self.exp_config = SimpleNamespace(resample_scenes=False)

        # Per-world bookkeeping
        self._step_counts = torch.zeros(num_worlds, dtype=torch.int32)
        self._ego_idxs = torch.ones(num_worlds, dtype=torch.long)

        # Stats for external logging
        self.crash_hits: List[int] = []
        self.crash_labels_buf: List[Optional[str]] = []

    # ── VecEnv interface ──────────────────────────────────────────────────────

    def reset(self, world_idx=None, seed=None, **kwargs):
        if world_idx is None:
            self._env.reset()
            self._step_counts[:] = 0
            self._update_all_ego_targets()
        else:
            worlds = world_idx.tolist() if isinstance(world_idx, torch.Tensor) else list(world_idx)
            self._env.sim.reset(worlds)
            self._step_counts[world_idx] = 0
            self._update_ego_targets_for(worlds)
        return self._get_obs()

    def step(self, actions) -> VecEnvStepReturn:
        full_actions = torch.zeros(
            self.num_worlds, self.max_agent_count, dtype=torch.long, device=self.device
        )
        full_actions[:, 0] = actions.long()

        self._env.step_dynamics(full_actions)
        self._step_counts += 1

        infos_raw = self._env.get_infos()
        collided = infos_raw.collided[:, 0].bool()
        goal_reached = infos_raw.goal_achieved[:, 0].bool()
        timeout = self._step_counts >= self.episode_len

        shaping = self._compute_shaping_all()
        terminal, crash_labels = self._compute_terminal_all(collided)
        rewards = (shaping + terminal).to(self.device)

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

        # Get terminal obs, then auto-reset done worlds
        obs = self._get_obs()
        done_worlds = torch.where(dones)[0]
        if len(done_worlds) > 0:
            self.reset(done_worlds)
            fresh = self._get_obs()
            obs[done_worlds] = fresh[done_worlds]

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
        # SB3 VecEnv base calls this for render_mode on init
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
        state = self._global_state()   # (num_worlds, max_agents, 14)
        speeds = self._local_speeds()  # (num_worlds, max_agents)
        obs = torch.zeros(self.num_worlds, self.OBS_DIM, dtype=torch.float32)

        for w in range(self.num_worlds):
            ei = int(self._ego_idxs[w])
            npc = state[w, 0]
            ego = state[w, ei]

            npc_pos = npc[:2].numpy()
            npc_h = float(npc[7])
            npc_spd = float(speeds[w, 0])

            ego_pos = ego[:2].numpy()
            ego_h = float(ego[7])
            ego_spd = float(speeds[w, ei])

            dx, dy = ego_pos - npc_pos
            ch, sh = np.cos(-npc_h), np.sin(-npc_h)
            rel_x = ch * dx - sh * dy
            rel_y = sh * dx + ch * dy

            rel_h = (ego_h - npc_h + np.pi) % (2 * np.pi) - np.pi
            dist = float(np.linalg.norm([dx, dy]))

            obs[w] = torch.tensor([
                npc_spd,
                np.cos(npc_h), np.sin(npc_h),
                rel_x, rel_y,
                np.cos(rel_h), np.sin(rel_h),
                ego_spd, dist,
            ], dtype=torch.float32)

        return obs.to(self.device)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_shaping_all(self) -> torch.Tensor:
        state = self._global_state()
        shaping = torch.zeros(self.num_worlds, dtype=torch.float32)

        for w in range(self.num_worlds):
            ei = int(self._ego_idxs[w])
            npc = state[w, 0].numpy()
            ego = state[w, ei].numpy()

            dx = float(npc[0] - ego[0])
            dy = float(npc[1] - ego[1])
            d = float(np.sqrt(dx * dx + dy * dy))
            if d < 1e-6:
                continue

            ego_h = float(ego[7])
            ch, sh = np.cos(ego_h), np.sin(ego_h)
            lx = ch * dx + sh * dy
            ly = -sh * dx + ch * dy

            unit = np.array([lx / d, ly / d])
            alignment = float(np.dot(unit, self.target_dir))
            proximity = 1.0 / (1.0 + d / PROXIMITY_SCALE)
            shaping[w] = W_SHAPING * alignment * proximity

        return shaping

    def _compute_terminal_all(
        self, collided: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Optional[str]]]:
        terminal = torch.zeros(self.num_worlds, dtype=torch.float32)
        labels: List[Optional[str]] = [None] * self.num_worlds
        vs = _vscale()
        state = self._global_state()

        for w in range(self.num_worlds):
            if not collided[w].item():
                continue
            npc = state[w, 0].numpy()
            npc_pos = npc[:2]

            for i in range(1, state.shape[1]):
                other = state[w, i].numpy()
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
        return (
            self._env.sim.absolute_self_observation_tensor()
            .to_torch().clone().cpu()
        )

    def _local_speeds(self) -> torch.Tensor:
        return (
            self._env.sim.self_observation_tensor()
            .to_torch().clone().cpu()[:, :, 0]
        )

    def _update_all_ego_targets(self):
        state = self._global_state()
        for w in range(self.num_worlds):
            self._ego_idxs[w] = self._find_target_ego(state[w])

    def _update_ego_targets_for(self, worlds: List[int]):
        state = self._global_state()
        for w in worlds:
            self._ego_idxs[w] = self._find_target_ego(state[w])

    def _find_target_ego(self, world_state: torch.Tensor) -> int:
        npc_xy = world_state[0, :2].numpy()
        best_idx, best_dist = 1, float("inf")
        for i in range(1, world_state.shape[0]):
            xy = world_state[i, :2].numpy()
            d = float(np.linalg.norm(xy - npc_xy))
            if d < _PADDING_DIST and d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def crash_rate(self, window: int = 200) -> float:
        hits = self.crash_hits[-window:]
        return sum(hits) / max(len(hits), 1)
