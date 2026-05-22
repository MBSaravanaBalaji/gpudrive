"""
GPUDrive environment wrapper for crash NPC training.

Setup:
  - Agent 0 (NPC): controlled by the DQN — this is what we're training.
  - All other agents: log-replay (follow recorded Waymo trajectories).
  - The "target ego" is the nearest valid log-replay agent to the NPC at episode start.

Observation (9 floats, in NPC's local frame):
  [npc_speed, cos(npc_heading), sin(npc_heading),
   rel_x, rel_y, cos(rel_heading), sin(rel_heading),
   ego_speed, distance]

Reward (sparse):
  +1.0  correct crash type (ssl / ssr / re)
  -0.1  wrong crash type
   0.0  everything else
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv

from collision_classifier.classifier import check_and_classify

# Padding position threshold — agents teleported here after being done
_PADDING_DIST = 5_000.0

# Valid crash type strings
CRASH_TYPES = ("ssl", "ssr", "re")
_CRASH_TYPE_MAP = {
    "ssl": ("side-swipe-left",),
    "ssr": ("side-swipe-right",),
    "re":  ("rear-ended",),   # ego was hit from behind by NPC
}

# ── Reward parameters (mirrors dqn_crasher/sat_reward_wrapper.py) ─────────────
R_MATCH         = 10.0   # terminal bonus for correct crash type
R_WRONG         = 0.0    # no penalty for wrong type — penalizing wrong crashes causes avoidance collapse
W_SHAPING       = 0.1    # per-step alignment shaping weight (small, just enough gradient)
PROXIMITY_SCALE = 10.0   # distance (m) at which proximity ≈ 0.5

# Target direction for each crash type in EGO's LOCAL frame.
# NPC should approach from this direction relative to ego
_TARGET_DIRS = {
    "ssl": np.array([ 0.0, -1.0]),  # classifier side-swipe-left
    "ssr": np.array([ 0.0, +1.0]),  # classifier side-swipe-right
    "re":  np.array([-1.0,  0.0]),  # NPC behind ego     (x<0 in ego frame)
}


class CrashEnv:
    """Single-world GPUDrive wrapper for adversarial crash NPC training."""

    OBS_DIM = 9
    DATA_DIR = "data/processed/examples"

    def __init__(
        self,
        crash_type: str,
        data_dir: str = DATA_DIR,
        dataset_size: int = 50,
        seed: int = 42,
        device: str = "cpu",
    ):
        assert crash_type in CRASH_TYPES, f"crash_type must be one of {CRASH_TYPES}"
        self.crash_type = crash_type
        self.target_labels = _CRASH_TYPE_MAP[crash_type]
        self.target_dir = _TARGET_DIRS[crash_type]
        self.device = device

        config = EnvConfig(
            dynamics_model="classic",
            collision_behavior="stop",    # NPC freezes on contact; position stays readable for classifier
            max_controlled_agents=1,      # only NPC (agent 0) is controlled
            num_worlds=1,
            ego_state=True,
            road_map_obs=False,
            partner_obs=False,            # we compute obs manually for compactness
            norm_obs=False,
        )

        data_loader = SceneDataLoader(
            root=data_dir,
            batch_size=1,
            dataset_size=dataset_size,
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

        self.n_actions = self._env.action_space.n
        self.episode_len = self._env.episode_len
        self._ego_idx: int = 1       # target agent index, refreshed each reset
        self._step_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._env.reset()
        self._step_count = 0
        self._ego_idx = self._find_target_ego()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        actions = torch.zeros(1, self._env.max_agent_count, dtype=torch.long)
        actions[0, 0] = int(action)

        self._env.step_dynamics(actions)
        self._step_count += 1

        obs = self._get_obs()

        # Dense shaping: alignment of NPC→ego approach direction with target, weighted by proximity
        shaping, dist = self._shaping_reward()

        infos = self._env.get_infos()
        collided = bool(infos.collided[0, 0].item())
        goal_reached = bool(infos.goal_achieved[0, 0].item())

        # Terminal reward: only classify when GPUDrive confirms an actual collision
        terminal, crash_label = 0.0, None
        if collided:
            terminal, crash_label = self._compute_reward()

        reward = shaping + terminal

        done = collided or goal_reached or self._step_count >= self.episode_len
        info = {
            "crash_label": crash_label,
            "step": self._step_count,
            "collided": collided,
            "goal_reached": goal_reached,
            "shaping": shaping,
            "dist": dist,
        }

        return obs, reward, done, info

    def render(self, zoom_radius: int = 60) -> np.ndarray:
        """Return BGR numpy image (for OpenCV display)."""
        import cv2
        import matplotlib.pyplot as plt

        figs = self._env.vis.plot_simulator_state(
            env_indices=[0],
            center_agent_indices=[0],
            zoom_radius=zoom_radius,
        )
        fig = figs[0]
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _global_state(self):
        """Return raw absolute_self_observation_tensor as numpy, shape (max_agents, 14)."""
        return (
            self._env.sim.absolute_self_observation_tensor()
            .to_torch()
            .clone()
            .squeeze(0)   # remove world dim → (max_agents, 14)
            .cpu()
            .numpy()
        )

    def _local_speeds(self) -> np.ndarray:
        """Return speed for all agents, shape (max_agents,)."""
        return (
            self._env.sim.self_observation_tensor()
            .to_torch()
            .clone()
            .squeeze(0)   # (max_agents, 8)
            .cpu()
            .numpy()[:, 0]  # column 0 = speed
        )

    def _find_target_ego(self) -> int:
        """Pick the nearest non-NPC agent at episode start as the ego target."""
        state = self._global_state()
        npc_xy = state[0, :2]

        best_idx = 1
        best_dist = float("inf")
        for i in range(1, state.shape[0]):
            xy = state[i, :2]
            dist = float(np.linalg.norm(xy - npc_xy))
            # Skip padding agents (teleported far away)
            if dist < _PADDING_DIST and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _get_obs(self) -> np.ndarray:
        """Build 9-float observation for the NPC."""
        state = self._global_state()
        speeds = self._local_speeds()

        npc = state[0]
        ego = state[self._ego_idx]

        npc_pos = npc[:2]
        npc_heading = float(npc[7])   # rotation_angle column
        npc_speed = float(speeds[0])

        ego_pos = ego[:2]
        ego_heading = float(ego[7])
        ego_speed = float(speeds[self._ego_idx])

        # Relative position in NPC's local frame
        dx, dy = ego_pos - npc_pos
        cos_h, sin_h = np.cos(-npc_heading), np.sin(-npc_heading)
        rel_x = cos_h * dx - sin_h * dy
        rel_y = sin_h * dx + cos_h * dy

        rel_heading = ego_heading - npc_heading
        # Wrap to [-pi, pi]
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

        dist = float(np.linalg.norm([dx, dy]))

        return np.array([
            npc_speed,
            np.cos(npc_heading),
            np.sin(npc_heading),
            rel_x,
            rel_y,
            np.cos(rel_heading),
            np.sin(rel_heading),
            ego_speed,
            dist,
        ], dtype=np.float32)

    def _shaping_reward(self) -> Tuple[float, float]:
        """
        Per-step dense reward: how well is the NPC positioned to make the target crash?
        Mirrors dqn_crasher's SATRewardWrapper._compute_mtv_local logic.

        Returns (shaping_reward, distance_to_target_ego).
        """
        state = self._global_state()
        npc = state[0]
        ego = state[self._ego_idx]

        dx = float(npc[0] - ego[0])
        dy = float(npc[1] - ego[1])
        d = float(np.sqrt(dx * dx + dy * dy))
        if d < 1e-6:
            return 0.0, 0.0

        # Rotate ego→npc vector into ego's local frame (x=forward, y=right in GPUDrive)
        ego_h = float(ego[7])
        cos_h, sin_h = np.cos(ego_h), np.sin(ego_h)
        local_x =  cos_h * dx + sin_h * dy
        local_y = -sin_h * dx + cos_h * dy

        unit = np.array([local_x / d, local_y / d])
        alignment = float(np.dot(unit, self.target_dir))
        proximity = 1.0 / (1.0 + d / PROXIMITY_SCALE)

        return W_SHAPING * alignment * proximity, d

    def _compute_reward(self) -> Tuple[float, Optional[str]]:
        """
        Run the crash classifier against all agents to find which one the NPC hit.
        Returns (reward, crash_label_or_None).
        """
        state = self._global_state()
        from madrona_gpudrive import vehicleScale

        npc = state[0]
        npc_pos = npc[:2]

        # Check NPC against every other agent; pick the one whose OBB overlaps
        for i in range(1, state.shape[0]):
            other = state[i]
            # Skip padding agents (far away)
            if np.linalg.norm(other[:2] - npc_pos) > _PADDING_DIST:
                continue

            result = check_and_classify(
                ego_pos_x=float(other[0]),
                ego_pos_y=float(other[1]),
                ego_heading=float(other[7]),
                ego_length=float(other[10]) * vehicleScale,
                ego_width=float(other[11]) * vehicleScale,
                npc_pos_x=float(npc[0]),
                npc_pos_y=float(npc[1]),
                npc_heading=float(npc[7]),
                npc_length=float(npc[10]) * vehicleScale,
                npc_width=float(npc[11]) * vehicleScale,
            )

            if result is not None:
                label = result.collision_type
                if label in self.target_labels:
                    return R_MATCH, label
                else:
                    return -R_WRONG, label

        return 0.0, None
