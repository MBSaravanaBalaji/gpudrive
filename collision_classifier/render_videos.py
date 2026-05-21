"""
Render MP4 videos of trained crash NPC models (SSL and SSR).

Usage:
  cd /path/to/gpudrive
  .venv/bin/python -m collision_classifier.render_videos [--n_videos 5] [--zoom 50]

Saves:
  collision_classifier/videos/thread 1/ssl_ep01.mp4 ...
  collision_classifier/videos/thread 1/ssr_ep01.mp4 ...
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy
from gpudrive.visualize.utils import img_from_fig

from collision_classifier.ppo_env import CrashVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos", "thread 1")
TRAIN_NUM_WORLDS = 4
TRAIN_SEED = 42
FPS = 25  # matches Waymo's 25 Hz simulation rate


def load_model(crash_type: str, env: CrashVecEnv) -> IPPO:
    path = os.path.join(CHECKPOINT_DIR, f"{crash_type}_ppo_best.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return IPPO.load(
        path,
        env=env,
        custom_objects={"mlp_class": FFN, "policy_class": FeedForwardPolicy},
        device="cpu",
    )


def frames_to_mp4(frames: list[np.ndarray], out_path: str, fps: int = FPS):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def render_episodes(
    crash_type: str,
    n_videos: int = 5,
    zoom: int = 50,
    max_rollouts: int = 200,
):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    print(f"\n=== Rendering {crash_type.upper()} ===", flush=True)

    env = CrashVecEnv(
        crash_type=crash_type,
        num_worlds=TRAIN_NUM_WORLDS,
        dataset_size=50,
        seed=TRAIN_SEED,
        device="cpu",
    )

    model = load_model(crash_type, env)
    target_labels = env.target_labels
    num_worlds = env.num_worlds
    ep_len = env.episode_len  # 91

    saved = 0
    rollout = 0

    while saved < n_videos and rollout < max_rollouts:
        # Load a fresh batch of Waymo scenes each rollout for diversity.
        # Without this, reset() just replays the same 4 scenes every time.
        env._env.swap_data_batch()
        obs = env.reset()

        # Per-world frame buffers and done flags for this rollout
        world_frames: list[list[np.ndarray]] = [[] for _ in range(num_worlds)]
        world_done = [False] * num_worlds

        for step in range(ep_len + 1):
            # Render active (not yet done) worlds
            for w in range(num_worlds):
                if not world_done[w]:
                    figs = env._env.vis.plot_simulator_state(
                        env_indices=[w],
                        center_agent_indices=[0],
                        zoom_radius=zoom,
                    )
                    world_frames[w].append(img_from_fig(figs[0]))

            if all(world_done):
                break

            action, _ = model.predict(obs.numpy(), deterministic=True)
            obs, _, dones, infos = env.step(torch.from_numpy(action))

            for w in range(num_worlds):
                if not world_done[w] and dones[w].item():
                    world_done[w] = True
                    label = infos[w].get("crash_label")
                    collided = infos[w].get("collided", False)
                    status = f"crash:{label}" if collided else "timeout/goal"
                    print(f"  rollout {rollout:3d} w{w} | {status} | frames={len(world_frames[w])} | saved={saved}/{n_videos}", flush=True)

                    if collided and label in target_labels and saved < n_videos:
                        saved += 1
                        out_path = os.path.join(VIDEO_DIR, f"{crash_type}_ep{saved:02d}.mp4")
                        frames_to_mp4(world_frames[w], out_path)
                        print(f"  [{saved}/{n_videos}] saved {os.path.basename(out_path)}  label={label}", flush=True)

        rollout += 1

    if saved < n_videos:
        print(f"  Warning: only {saved}/{n_videos} videos in {rollout} rollouts.", flush=True)
    print(f"Done. Videos → {VIDEO_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash_types", nargs="+", default=["ssl", "ssr"],
                        choices=["ssl", "ssr", "re"])
    parser.add_argument("--n_videos", type=int, default=5)
    parser.add_argument("--zoom",     type=int, default=50)
    args = parser.parse_args()

    for ct in args.crash_types:
        render_episodes(ct, n_videos=args.n_videos, zoom=args.zoom)


if __name__ == "__main__":
    main()
