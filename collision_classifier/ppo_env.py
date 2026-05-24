"""
CrashVecEnv: multi-world GPUDrive VecEnv for crash NPC training.

Runs num_worlds parallel Waymo scenes, each with one controlled NPC.
Compatible with IPPO from gpudrive.integrations.sb3.ppo and FeedForwardPolicy.

Key choices vs single-env DQN:
- num_worlds parallelism → more diverse scene geometry per rollout
- PPO (on-policy) → stable with sparse terminal rewards + dense shaping
- Auto-reset done worlds immediately → standard SB3 VecEnv contract
- 11-float custom obs: pair geometry plus lane-alignment signal
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import json
import os

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

try:
    from data_utils.spawn_families import (
        MIN_SUCCESS_STEP_BY_FAMILY,
        infer_spawn_family_from_path,
    )
except ImportError:
    MIN_SUCCESS_STEP_BY_FAMILY = {"unknown": 15}

    def infer_spawn_family_from_path(path: str) -> str:
        return "unknown"

_DEFAULT_MIN_SUCCESS_STEP = {"ssl": 15, "ssr": 15, "re": 10}
_SPAWN_NORM = 30.0

_VEHICLE_SCALE = None
_REALISTIC_ACCEL_VALUES = (0.5, 1.5, 2.5, 3.5)
_REALISTIC_STEER_VALUES = (-0.2, 0.0, 0.2)
W_LANE = 0.005
MIN_SUCCESS_STEP = 15
W_APPROACH = 0.03        # dense side-swipe guidance; keep << R_MATCH over 91 steps
W_REAR_BLOCK_PENALTY = 0.15
W_EARLY_CLOSE = 0.12     # per-step penalty when closing within 10 m before min_success_step
R_TOO_EARLY = 12.0       # terminal penalty for correct-type contact before min_success_step
R_REAR_SHORTCUT = 10.0
R_OPPOSITE_SIDE = 8.0
R_OFFROAD = 4.0
MIN_BEHIND_SPAWN_GAP = 10.0  # meters; used to gate contact shaping while still behind ego
IDEAL_SIDESWIPE_OFFSET = 2.0
# Discrete action index: accel=0.5, steer=0.0 (mild forward, straight)
PROBE_ACTION_IDX = 1


def _filter_spawn_family_paths(
    paths: List[str],
    spawn_family: str,
) -> List[str]:
    """Drop synthetic scenes whose initial spawn is too close for the family."""
    limits = {
        "behind_near": -11.0,
        "behind_far": -15.0,
    }
    min_rel_x = limits.get(spawn_family)
    if min_rel_x is None:
        return paths
    kept: List[str] = []
    for path in paths:
        try:
            with open(path) as f:
                meta = json.load(f).get("metadata", {}).get("crash_pair", {})
            rel_x = float(meta.get("rel_x", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            kept.append(path)
            continue
        if rel_x <= min_rel_x:
            kept.append(path)
    return kept if kept else paths


def _vscale() -> float:
    global _VEHICLE_SCALE
    if _VEHICLE_SCALE is None:
        from madrona_gpudrive import vehicleScale
        _VEHICLE_SCALE = vehicleScale
    return _VEHICLE_SCALE


class CrashVecEnv(VecEnv):
    """
    num_worlds parallel GPUDrive scenes.
    Each world has one controlled NPC and log-replay background agents.
    """

    OBS_DIM = 11
    OBS_SPAWN_DIM = 4
    DATA_DIR = "data/processed/examples"

    def __init__(
        self,
        crash_type: str,
        num_worlds: int = 64,
        data_dir: str = DATA_DIR,
        dataset_size: int = 50,
        seed: int = 42,
        device: str = _DEFAULT_DEVICE,
        spawn_cond: bool = False,
        spawn_family: str | None = None,
        late_fusion: bool = False,
    ):
        assert crash_type in CRASH_TYPES, f"crash_type must be one of {CRASH_TYPES}"
        self.crash_type = crash_type
        self.spawn_cond = spawn_cond
        self.late_fusion = late_fusion
        self.target_labels = _CRASH_TYPE_MAP[crash_type]
        self.target_dir = np.array(_TARGET_DIRS[crash_type], dtype=np.float32)
        self.target_dir_t = torch.tensor(_TARGET_DIRS[crash_type], dtype=torch.float32)
        self.num_worlds = num_worlds
        self.device = device

        config = EnvConfig(
            dynamics_model="classic",
            collision_behavior="ignore",  # training: NPC keeps moving after early taps
            max_controlled_agents=1,
            num_worlds=num_worlds,
            ego_state=True,
            road_map_obs=late_fusion,
            partner_obs=late_fusion,
            norm_obs=late_fusion,
            obs_radius=50.0 if late_fusion else 50.0,
        )
        self.env_config = config

        if spawn_family is not None:
            needle = f"_{spawn_family}_"
            family_paths = [
                os.path.join(data_dir, scene)
                for scene in sorted(os.listdir(data_dir))
                if scene.startswith("tfrecord") and needle in scene
            ]
            if not family_paths:
                raise FileNotFoundError(
                    f"No scenes matching spawn_family={spawn_family!r} under {data_dir}"
                )
            n_before = len(family_paths)
            family_paths = _filter_spawn_family_paths(family_paths, spawn_family)
            if len(family_paths) < n_before:
                print(
                    f"  spawn filter ({spawn_family}): "
                    f"{len(family_paths)}/{n_before} scenes kept"
                )
            effective_size = min(dataset_size, len(family_paths))
            data_loader = SceneDataLoader(
                root=data_dir,
                batch_size=num_worlds,
                dataset_size=effective_size,
                sample_with_replacement=True,
                shuffle=True,
                seed=seed,
                file_prefix="tfrecord",
            )
            data_loader.dataset = family_paths[:effective_size]
            if data_loader.shuffle:
                data_loader.random_gen.shuffle(data_loader.dataset)
            print(
                f"  spawn_family={spawn_family}  scenes={len(data_loader.dataset)}"
            )
        else:
            data_loader = SceneDataLoader(
                root=data_dir,
                batch_size=num_worlds,
                dataset_size=dataset_size,
                sample_with_replacement=True,
                shuffle=True,
                seed=seed,
                file_prefix="tfrecord",
            )

        self.spawn_family = spawn_family

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
        if late_fusion:
            self._native_obs_dim = int(self._env.observation_space.shape[0])
            self.obs_dim = self._native_obs_dim + (
                self.OBS_SPAWN_DIM if spawn_cond else 0
            )
            print(f"  late_fusion obs: native={self._native_obs_dim}  total={self.obs_dim}")
        else:
            self.obs_dim = self.OBS_DIM + (self.OBS_SPAWN_DIM if spawn_cond else 0)
        self._expert_actions = self._env.get_expert_actions()[0].to(device)
        self._action_values = torch.tensor(
            [
                [accel, steer, 0.0]
                for accel in _REALISTIC_ACCEL_VALUES
                for steer in _REALISTIC_STEER_VALUES
            ],
            dtype=torch.float32,
            device=device,
        )

        observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        action_space = gym.spaces.Discrete(len(self._action_values))
        super().__init__(num_worlds, observation_space, action_space)

        self.controlled_agent_mask = self._env.cont_agent_mask.clone().to(device)
        self._npc_idxs = self._controlled_indices()
        self.dead_agent_mask = torch.zeros(
            num_worlds, self.max_agent_count, dtype=torch.bool, device=device
        )

        self.exp_config = SimpleNamespace(resample_scenes=False)

        # Per-world bookkeeping — kept on device for vectorized indexing
        self._step_counts = torch.zeros(num_worlds, dtype=torch.int32, device=device)
        self._ego_idxs = torch.ones(num_worlds, dtype=torch.long, device=device)
        self._ego_start_h = torch.zeros(num_worlds, dtype=torch.float32, device=device)
        self._arange = torch.arange(num_worlds, dtype=torch.long, device=device)

        # Move target direction tensor to device after super().__init__
        self.target_dir_t = self.target_dir_t.to(device)

        # Stats for external logging
        self.crash_hits: List[int] = []
        self.target_contact_hits: List[int] = []
        self.crash_labels_buf: List[Optional[str]] = []
        self.lane_align_buf: List[float] = []
        self.episode_len_buf: List[int] = []
        self.spawn_family_buf: List[str] = []

        self._spawn_rel_x = torch.zeros(num_worlds, dtype=torch.float32, device=device)
        self._spawn_rel_y = torch.zeros(num_worlds, dtype=torch.float32, device=device)
        self._spawn_angle = torch.zeros(num_worlds, dtype=torch.float32, device=device)
        self._min_success_steps = torch.full(
            (num_worlds,),
            fill_value=_DEFAULT_MIN_SUCCESS_STEP.get(crash_type, MIN_SUCCESS_STEP),
            dtype=torch.int32,
            device=device,
        )
        self._spawn_families: List[str] = ["unknown"] * num_worlds
        self._immature_penalized = torch.zeros(
            num_worlds, dtype=torch.bool, device=device
        )

    # ── VecEnv interface ──────────────────────────────────────────────────────

    def reset(self, world_idx=None, seed=None, **kwargs):
        if world_idx is None:
            self._env.reset()
            self._step_counts.zero_()
            self._immature_penalized.zero_()
            self._refresh_controlled_agents()
            self._update_all_ego_targets()
            self._load_spawn_metadata(list(range(self.num_worlds)))
        else:
            worlds = world_idx.tolist() if isinstance(world_idx, torch.Tensor) else list(world_idx)
            self._env.sim.reset(worlds)
            self._step_counts[world_idx] = 0
            self._immature_penalized[world_idx] = False
            self._update_ego_targets_for(worlds)
            self._load_spawn_metadata(worlds)
        return self._get_obs()

    def swap_scenes(self):
        """Load a new batch of Waymo scenes from the data loader and full-reset."""
        self._env.swap_data_batch()
        return self._finish_scene_load()

    def load_scene_paths(self, paths: List[str]) -> torch.Tensor:
        """Load an explicit list of scene JSON paths (one per world) and reset."""
        if len(paths) != self.num_worlds:
            raise ValueError(
                f"Expected {self.num_worlds} scene paths, got {len(paths)}"
            )
        self._env.swap_data_batch(list(paths))
        return self._finish_scene_load()

    def _finish_scene_load(self) -> torch.Tensor:
        self._expert_actions = self._env.get_expert_actions()[0].to(self.device)
        self._refresh_controlled_agents()
        return self.reset()

    def step(self, actions) -> VecEnvStepReturn:
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1)
        npc_action_values = self._action_values[action_t]
        t_idx = torch.clamp(
            self._step_counts.to(torch.long),
            max=self._expert_actions.shape[2] - 1,
        )
        full_actions = self._expert_actions.permute(0, 2, 1, 3)[self._arange, t_idx].clone()
        full_actions[self._arange, self._npc_idxs] = npc_action_values

        self._env.step_dynamics(full_actions)
        self._step_counts += 1

        # Single state copy for the whole step
        state  = self._global_state()   # (W, A, 14) on device
        speeds = self._local_speeds()   # (W, A)     on device

        # info_tensor layout: [off_road, collidedWithVehicle, collidedWithNonVehicle, goal_achieved, agent_type]
        # get_infos().collided sums columns 1+2 (vehicle + non-vehicle) — we want vehicle-only.
        info_t       = self._env.sim.info_tensor().to_torch().clone()  # (W, A, 5)
        offroad      = info_t[self._arange, self._npc_idxs, 0].bool()
        collided     = info_t[self._arange, self._npc_idxs, 1].bool()   # collidedWithVehicle for NPC
        goal_reached = info_t[self._arange, self._npc_idxs, 3].bool()
        timeout      = self._step_counts >= self.episode_len

        shaping  = self._compute_shaping(state)
        terminal, crash_labels, ego_features = self._compute_terminal(collided, state)
        terminal = terminal - R_OFFROAD * (offroad & ~collided).float()

        immature_contact = torch.zeros(
            self.num_worlds, dtype=torch.bool, device=self.device
        )
        for w in range(self.num_worlds):
            label = crash_labels[w]
            if collided[w] and label is not None and label.startswith("too_early:"):
                immature_contact[w] = True

        first_immature = immature_contact & ~self._immature_penalized
        repeat_immature = immature_contact & self._immature_penalized
        terminal = torch.where(repeat_immature, torch.zeros_like(terminal), terminal)
        terminal = torch.where(
            first_immature,
            torch.full_like(terminal, -R_TOO_EARLY),
            terminal,
        )
        self._immature_penalized |= immature_contact

        delta_xy = state[self._arange, self._npc_idxs, :2] - state[self._arange, self._ego_idxs, :2]
        dist = torch.linalg.norm(delta_xy, dim=1)
        early_close = (
            self._step_counts < self._min_success_steps
        ) & (dist < 10.0)
        rewards = shaping + terminal - W_EARLY_CLOSE * early_close.float()

        collision_done = collided & ~immature_contact
        dones = collision_done | goal_reached | offroad | timeout
        lane_align = self._lane_alignment(state)
        contact_lx_all = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        contact_ly_all = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        cos_h_all = torch.cos(state[self._arange, self._ego_idxs, 7])
        sin_h_all = torch.sin(state[self._arange, self._ego_idxs, 7])
        delta_x = state[self._arange, self._npc_idxs, 0] - state[self._arange, self._ego_idxs, 0]
        delta_y = state[self._arange, self._npc_idxs, 1] - state[self._arange, self._ego_idxs, 1]
        contact_lx_all = cos_h_all * delta_x + sin_h_all * delta_y
        contact_ly_all = sin_h_all * delta_x - cos_h_all * delta_y

        # Collect stats before reset
        for w in range(self.num_worlds):
            if dones[w]:
                label = crash_labels[w]
                if collision_done[w].item() and label is not None:
                    self.crash_labels_buf.append(label)
                    self.crash_hits.append(1 if label in self.target_labels else 0)
                    self.target_contact_hits.append(1 if self._is_target_contact_label(label) else 0)
                else:
                    self.crash_hits.append(0)
                    self.target_contact_hits.append(0)
                self.lane_align_buf.append(float(lane_align[w].item()))
                self.episode_len_buf.append(int(self._step_counts[w].item()))
                self.spawn_family_buf.append(self._spawn_families[w])

        obs = self._get_obs(state, speeds)
        done_worlds = torch.where(dones)[0]
        npc_idxs_for_info = self._npc_idxs.clone()
        ego_idxs_for_info = self._ego_idxs.clone()
        if len(done_worlds) > 0:
            self.reset(done_worlds)
            fresh_state  = self._global_state()
            fresh_speeds = self._local_speeds()
            obs[done_worlds] = self._get_obs(fresh_state, fresh_speeds)[done_worlds]

        info_list = [
            {
                "crash_label": crash_labels[w],
                "collided": bool(collided[w].item()),
                "offroad": bool(offroad[w].item()),
                "goal_reached": bool(goal_reached[w].item()),
                "npc_idx": int(npc_idxs_for_info[w].item()),
                "ego_idx": int(ego_idxs_for_info[w].item()),
                "lane_align": float(lane_align[w].item()),
                "contact_lx": float(contact_lx_all[w].item()),
                "contact_ly": float(contact_ly_all[w].item()),
                "ego_contact_feature": ego_features[w],
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

    def _get_obs(
        self,
        state: torch.Tensor | None = None,
        speeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state is None:
            state = self._global_state()
        if speeds is None:
            speeds = self._local_speeds()
        if self.late_fusion:
            full = self._env.get_obs()
            obs = full[self._arange, self._npc_idxs]
            if self.spawn_cond:
                obs = torch.cat([obs, self._spawn_features()], dim=1)
            return obs
        return self._build_obs(state, speeds)

    def _spawn_features(self) -> torch.Tensor:
        return torch.stack([
            self._spawn_rel_x / _SPAWN_NORM,
            self._spawn_rel_y / _SPAWN_NORM,
            torch.cos(self._spawn_angle),
            torch.sin(self._spawn_angle),
        ], dim=1)

    def _build_obs(self, state: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
        """Vectorized 11-float obs, stays on device — no Python loop."""
        npc_xy  = state[self._arange, self._npc_idxs, :2]             # (W, 2)
        npc_h   = state[self._arange, self._npc_idxs, 7]              # (W,)
        npc_spd = speeds[self._arange, self._npc_idxs]                # (W,)

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
        lane_h = self._ego_start_h
        lane_rel_h = ((npc_h - lane_h + math.pi) % (2 * math.pi)) - math.pi

        base = torch.stack([
            npc_spd,
            torch.cos(npc_h), torch.sin(npc_h),
            rel_x, rel_y,
            torch.cos(rel_h), torch.sin(rel_h),
            ego_spd, dist,
            torch.cos(lane_rel_h), torch.sin(lane_rel_h),
        ], dim=1)
        if self.spawn_cond:
            return torch.cat([base, self._spawn_features()], dim=1)
        return base

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_shaping(self, state: torch.Tensor) -> torch.Tensor:
        """Vectorized approach-to-target shaping, stays on device."""
        npc_xy = state[self._arange, self._npc_idxs, :2]              # (W, 2)
        npc_h  = state[self._arange, self._npc_idxs, 7]               # (W,)
        ego_xy = state[self._arange, self._ego_idxs, :2]             # (W, 2)
        ego_h  = state[self._arange, self._ego_idxs, 7]              # (W,)

        delta = npc_xy - ego_xy                                       # (W, 2)
        d = torch.linalg.norm(delta, dim=1)                          # (W,)

        cos_h = torch.cos(ego_h)
        sin_h = torch.sin(ego_h)
        lx =  cos_h * delta[:, 0] + sin_h * delta[:, 1]
        ly = sin_h * delta[:, 0] - cos_h * delta[:, 1]  # + = ego's right (GPUDrive)

        lane_align = self._lane_alignment(state)
        lane_penalty = W_LANE * torch.clamp(lane_align - 1.0, max=0.0)

        if self.crash_type in ("ssl", "ssr"):
            contact_lx = torch.zeros_like(lx)
            if self.crash_type == "ssl":
                contact_ly = torch.full_like(ly, -IDEAL_SIDESWIPE_OFFSET)
            else:
                contact_ly = torch.full_like(ly, IDEAL_SIDESWIPE_OFFSET)

            heading_align = torch.clamp(torch.cos(npc_h - ego_h), min=0.0, max=1.0)

            # Lateral: hold target flank (±2 m).
            side_score = torch.exp(-((ly - contact_ly) / 2.0).pow(2))

            # Longitudinal: reward alongside ego (lx≈0), not lingering behind.
            alongside_score = torch.exp(-((lx / 5.0).pow(2)))

            # Ramp reward as NPC pulls forward from behind-spawn toward alongside.
            forward_pull = torch.sigmoid((lx + 6.0) / 3.0)

            contact_dist = torch.sqrt((lx - contact_lx).pow(2) + (ly - contact_ly).pow(2))
            contact_score = torch.exp(-contact_dist / 8.0)

            approach = (
                (0.5 * alongside_score + 0.25 * forward_pull) * side_score
                + 0.25 * contact_score
            ) * heading_align

            # Strong penalty for closing from behind while on the correct flank.
            rear_close = (
                torch.sigmoid((-lx - 2.5) / 1.5)
                * torch.sigmoid((18.0 - d) / 4.0)
                * side_score
                * heading_align
            )
            behind_spawn = self._spawn_rel_x < -8.0
            ahead_spawn = self._spawn_rel_x > 8.0

            catch_up = (
                torch.sigmoid((lx + 8.0) / 4.0)
                * side_score
                * heading_align
            )
            # Do not reward "contact geometry" while still directly behind ego.
            alongside = torch.sigmoid((lx + 2.0) / 2.0)
            behind_shaping = W_APPROACH * (
                0.6 * catch_up + 0.4 * contact_score * alongside
            ) + lane_penalty
            side_shaping = W_APPROACH * approach - W_REAR_BLOCK_PENALTY * rear_close + lane_penalty
            ahead_shaping = W_APPROACH * (
                torch.sigmoid((-lx - 4.0) / 3.0) * side_score * heading_align
                + 0.3 * contact_score
            ) + lane_penalty

            shaping = torch.where(behind_spawn, behind_shaping, side_shaping)
            shaping = torch.where(ahead_spawn, ahead_shaping, shaping)
        else:
            # RE: reward closing from behind with lateral alignment.
            behind = torch.sigmoid((-lx - 3.0) / 4.0)
            lateral = torch.exp(-(ly / 1.5).pow(2))
            proximity = 1.0 / (1.0 + d / PROXIMITY_SCALE)
            heading_align = torch.clamp(torch.cos(npc_h - ego_h), min=0.0, max=1.0)
            shaping = W_SHAPING * behind * lateral * proximity * heading_align + lane_penalty

        shaping = shaping * (d >= 1e-6).float()
        return shaping

    def _compute_terminal(
        self, collided: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Optional[str]], List[Optional[str]]]:
        terminal = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        labels: List[Optional[str]] = [None] * self.num_worlds
        ego_features: List[Optional[str]] = [None] * self.num_worlds

        collided_ws = torch.where(collided)[0]
        if len(collided_ws) == 0:
            return terminal, labels, ego_features

        vs = _vscale()
        # Only pay the CPU transfer when there are actual collisions
        state_np = state.cpu().numpy()

        for w in collided_ws.tolist():
            npc_idx = int(self._npc_idxs[w].item())
            ego_idx = int(self._ego_idxs[w].item())
            npc = state_np[w, npc_idx]
            ego = state_np[w, ego_idx]
            npc_pos = npc[:2]
            target_result = check_and_classify(
                ego_pos_x=float(ego[0]), ego_pos_y=float(ego[1]),
                ego_heading=float(ego[7]),
                ego_length=float(ego[10]) * vs, ego_width=float(ego[11]) * vs,
                npc_pos_x=float(npc[0]), npc_pos_y=float(npc[1]),
                npc_heading=float(npc[7]),
                npc_length=float(npc[10]) * vs, npc_width=float(npc[11]) * vs,
            )
            if target_result is not None:
                label = target_result.collision_type
                ego_features[w] = target_result.ego_feature
                step_count = int(self._step_counts[w].item())
                min_step = int(self._min_success_steps[w].item())
                if label in self.target_labels and step_count < min_step:
                    labels[w] = f"too_early:{label}"
                    terminal[w] = -R_TOO_EARLY
                else:
                    labels[w] = label
                    if label in self.target_labels:
                        terminal[w] = R_MATCH
                    elif self.crash_type in ("ssl", "ssr") and label in (
                        "rear-end",
                        "rear-ended",
                    ):
                        terminal[w] = -R_REAR_SHORTCUT
                    elif self.crash_type == "ssl" and label == "side-swipe-right":
                        terminal[w] = -R_OPPOSITE_SIDE
                    elif self.crash_type == "ssr" and label == "side-swipe-left":
                        terminal[w] = -R_OPPOSITE_SIDE
                    else:
                        terminal[w] = -R_WRONG
                continue

            # Log non-target collisions for diagnosis, but never reward them.
            for i in range(state_np.shape[1]):
                if i == npc_idx or i == ego_idx:
                    continue
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
                    labels[w] = f"non_target:{result.collision_type}"
                    break

        return terminal, labels, ego_features

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

    def _refresh_controlled_agents(self) -> None:
        self.controlled_agent_mask = self._env.cont_agent_mask.clone().to(self.device)
        self._npc_idxs = self._controlled_indices()

    def _controlled_indices(self) -> torch.Tensor:
        mask = self.controlled_agent_mask
        if mask.shape[0] != self.num_worlds:
            raise RuntimeError(f"controlled mask world count mismatch: {mask.shape}")
        counts = mask.sum(dim=1)
        if torch.any(counts != 1):
            bad = torch.where(counts != 1)[0].detach().cpu().tolist()
            raise RuntimeError(f"expected exactly one controlled NPC per world, bad worlds={bad}")
        return torch.argmax(mask.to(torch.long), dim=1)

    def _update_all_ego_targets(self):
        state_np = self._global_state().cpu().numpy()
        for w in range(self.num_worlds):
            ego_idx = self._find_target_ego(state_np[w], int(self._npc_idxs[w].item()))
            self._ego_idxs[w] = ego_idx
            self._ego_start_h[w] = float(state_np[w, ego_idx, 7])

    def _update_ego_targets_for(self, worlds: List[int]):
        state_np = self._global_state().cpu().numpy()
        for w in worlds:
            ego_idx = self._find_target_ego(state_np[w], int(self._npc_idxs[w].item()))
            self._ego_idxs[w] = ego_idx
            self._ego_start_h[w] = float(state_np[w, ego_idx, 7])

    def _find_target_ego(self, world_state_np: np.ndarray, npc_idx: int) -> int:
        npc = world_state_np[npc_idx]
        npc_xy = npc[:2]
        best_idx, best_score = 1 if npc_idx != 1 else 0, float("inf")
        fallback_idx, fallback_dist = best_idx, float("inf")
        for i in range(world_state_np.shape[0]):
            if i == npc_idx:
                continue
            xy = world_state_np[i, :2]
            d = float(np.linalg.norm(xy - npc_xy))
            if d >= _PADDING_DIST:
                continue
            if d < fallback_dist:
                fallback_dist = d
                fallback_idx = i
            if not self._pair_viable_np(npc, world_state_np[i]):
                continue
            score = self._pair_score_np(npc, world_state_np[i])
            if score < best_score:
                best_score = score
                best_idx = i
        if best_score == float("inf"):
            return fallback_idx
        return best_idx

    def crash_rate(self, window: int = 200) -> float:
        hits = self.crash_hits[-window:]
        return sum(hits) / max(len(hits), 1)

    def target_contact_rate(self, window: int = 200) -> float:
        hits = self.target_contact_hits[-window:]
        return sum(hits) / max(len(hits), 1)

    def ssl_collision_rate(self, window: int = 200) -> float:
        """Mature target crashes / all labeled collision episodes in window."""
        return self.target_collision_rate(window)

    def target_collision_rate(self, window: int = 200) -> float:
        """Mature target crashes / all labeled collision episodes in window."""
        labels = self.crash_labels_buf[-window:]
        if not labels:
            return 0.0
        hits = sum(1 for lab in labels if lab in self.target_labels)
        return hits / len(labels)

    def ssl_contact_collision_rate(self, window: int = 200) -> float:
        """Any target contact (incl. too_early) / all labeled collision episodes."""
        return self.target_contact_collision_rate(window)

    def target_contact_collision_rate(self, window: int = 200) -> float:
        labels = self.crash_labels_buf[-window:]
        if not labels:
            return 0.0
        hits = sum(1 for lab in labels if self._is_target_contact_label(lab))
        return hits / len(labels)

    def mean_lane_align(self, window: int = 200) -> float:
        values = self.lane_align_buf[-window:]
        return sum(values) / max(len(values), 1)

    def mean_episode_len(self, window: int = 200) -> float:
        values = self.episode_len_buf[-window:]
        return sum(values) / max(len(values), 1)

    def spawn_family_stats(self, window: int = 200) -> Dict[str, Dict[str, float]]:
        """Per spawn_family mature-target rate and episode count in window."""
        n = min(window, len(self.crash_hits))
        if n == 0:
            return {}
        hits = self.crash_hits[-n:]
        families = self.spawn_family_buf[-n:]
        out: Dict[str, Dict[str, float]] = {}
        for fam, hit in zip(families, hits):
            bucket = out.setdefault(fam, {"episodes": 0, "mature_hits": 0})
            bucket["episodes"] += 1
            bucket["mature_hits"] += hit
        for fam, bucket in out.items():
            ep = max(bucket["episodes"], 1)
            bucket["crash_rate"] = bucket["mature_hits"] / ep
        return out

    def _load_spawn_metadata(self, worlds: List[int]) -> None:
        """Read spawn_family / rel offsets from scene JSON for each world."""
        paths = getattr(self._env, "data_batch", None) or []
        default_min = _DEFAULT_MIN_SUCCESS_STEP.get(self.crash_type, MIN_SUCCESS_STEP)
        for w in worlds:
            family = "unknown"
            rel_x = 0.0
            rel_y = 0.0
            min_step = default_min
            path = paths[w] if w < len(paths) else ""
            if path:
                try:
                    with open(path) as f:
                        meta = json.load(f).get("metadata", {}).get("crash_pair", {})
                    family = meta.get("spawn_family") or infer_spawn_family_from_path(path)
                    rel_x = float(meta.get("rel_x", 0.0))
                    rel_y = float(meta.get("rel_y", 0.0))
                    # Always use family table so reward timing stays in sync with code.
                    min_step = int(
                        MIN_SUCCESS_STEP_BY_FAMILY.get(family, default_min)
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    family = infer_spawn_family_from_path(path)
            self._spawn_families[w] = family
            self._spawn_rel_x[w] = rel_x
            self._spawn_rel_y[w] = rel_y
            self._spawn_angle[w] = math.atan2(rel_y, rel_x) if (rel_x or rel_y) else 0.0
            self._min_success_steps[w] = min_step

        if self.spawn_cond:
            self._fill_spawn_from_state(worlds)

    def _fill_spawn_from_state(self, worlds: List[int]) -> None:
        """If metadata lacks rel offsets, derive ego-frame spawn offsets from t=0."""
        state = self._global_state()
        for w in worlds:
            if abs(float(self._spawn_rel_x[w].item())) > 1e-3 or abs(
                float(self._spawn_rel_y[w].item())
            ) > 1e-3:
                continue
            npc_xy = state[w, self._npc_idxs[w], :2]
            ego_xy = state[w, self._ego_idxs[w], :2]
            ego_h = state[w, self._ego_idxs[w], 7]
            c, s = torch.cos(ego_h), torch.sin(ego_h)
            delta = npc_xy - ego_xy
            rel_x = float((c * delta[0] + s * delta[1]).item())
            rel_y = float((s * delta[0] - c * delta[1]).item())
            self._spawn_rel_x[w] = rel_x
            self._spawn_rel_y[w] = rel_y
            self._spawn_angle[w] = math.atan2(rel_y, rel_x) if (rel_x or rel_y) else 0.0

    def _is_target_contact_label(self, label: str) -> bool:
        if label in self.target_labels:
            return True
        return any(label == f"too_early:{target}" for target in self.target_labels)

    def _lane_alignment(self, state: torch.Tensor) -> torch.Tensor:
        npc_h = state[self._arange, self._npc_idxs, 7]
        return torch.cos(npc_h - self._ego_start_h)

    def _pair_viable_np(self, npc: np.ndarray, ego: np.ndarray) -> bool:
        hdiff = abs(float(npc[7] - ego[7])) % (2 * math.pi)
        if hdiff > math.pi:
            hdiff = 2 * math.pi - hdiff
        if hdiff > math.radians(30.0):
            return False
        lx, ly = self._ego_in_npc_frame_np(npc, ego)
        if self.crash_type == "ssl":
            return 1.5 <= ly <= 5.5 and 14.0 <= abs(lx) < 30.0
        if self.crash_type == "ssr":
            return -5.5 <= ly <= -1.5 and 14.0 <= abs(lx) < 30.0
        if self.crash_type == "re":
            return 3.0 <= lx <= 30.0 and abs(ly) < 2.5
        return False

    def _pair_score_np(self, npc: np.ndarray, ego: np.ndarray) -> float:
        lx, ly = self._ego_in_npc_frame_np(npc, ego)
        if self.crash_type in ("ssl", "ssr"):
            return abs(abs(ly) - 3.5) + 0.1 * abs(abs(lx) - 20.0)
        if self.crash_type == "re":
            return abs(lx - 10.0) + 0.5 * abs(ly)
        return float("inf")

    @staticmethod
    def _ego_in_npc_frame_np(npc: np.ndarray, ego: np.ndarray) -> Tuple[float, float]:
        dx = float(ego[0] - npc[0])
        dy = float(ego[1] - npc[1])
        h = float(npc[7])
        lx = math.cos(h) * dx + math.sin(h) * dy
        ly = -math.sin(h) * dx + math.cos(h) * dy
        return lx, ly
