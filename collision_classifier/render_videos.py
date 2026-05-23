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
import glob
import math
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy
from gpudrive.visualize.utils import img_from_fig

from collision_classifier.classifier import ego_frame_lateral
from collision_classifier.ppo_env import CrashVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos", "thread 1")
TRAIN_NUM_WORLDS = 4
FPS = 25  # matches Waymo's 25 Hz simulation rate


def _scene_key(path: str) -> str:
    """Waymo stem without synthetic variant suffix (one video per base scene)."""
    name = os.path.basename(path).replace(".json", "")
    for marker in ("_behind_near_syn", "_behind_far_syn", "_side_same_syn",
                   "_side_opposite_syn", "_ahead_syn", "_syn"):
        if marker in name:
            return name.split(marker)[0]
    return name


def _scene_pool(data_dir: str, dataset_size: int, seed: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No scene JSONs under {data_dir}")
    rng = random.Random(seed)
    if len(paths) > dataset_size:
        return rng.sample(paths, dataset_size)
    return paths


def _sample_batch(pool: list[str], num_worlds: int, rng: random.Random) -> list[str]:
    if len(pool) >= num_worlds:
        return rng.sample(pool, num_worlds)
    return [rng.choice(pool) for _ in range(num_worlds)]


def _contact_ly_ego_frame(env: CrashVecEnv, state: torch.Tensor, w: int) -> float:
    """Lateral NPC offset in ego frame (+ = ego's right side)."""
    npc = state[w, env._npc_idxs[w]]
    ego = state[w, env._ego_idxs[w]]
    dx = float(npc[0].item() - ego[0].item())
    dy = float(npc[1].item() - ego[1].item())
    h = float(ego[7].item())
    return float(ego_frame_lateral(h, dx, dy))


def _side_ok(
    crash_type: str,
    ly: float,
    label: str | None = None,
    ego_feature: str | None = None,
) -> bool:
    """Trust classifier target labels; ly is a secondary check (NPC center, not contact point)."""
    if label == "side-swipe-left" and crash_type == "ssl":
        return True
    if label == "side-swipe-right" and crash_type == "ssr":
        return True
    if ego_feature:
        ef = ego_feature.lower()
        if crash_type == "ssl" and "left" in ef:
            return True
        if crash_type == "ssr" and "right" in ef:
            return True
    if crash_type == "ssl":
        return ly < -0.5
    if crash_type == "ssr":
        return ly > 0.5
    return True


def load_model(crash_type: str, env: CrashVecEnv, checkpoint: str | None = None) -> tuple[IPPO, str]:
    if checkpoint:
        path = checkpoint
    else:
        finetune = os.path.join(CHECKPOINT_DIR, f"{crash_type}_ppo_finetune_best.zip")
        base = os.path.join(CHECKPOINT_DIR, f"{crash_type}_ppo_best.zip")
        if os.path.exists(finetune):
            path = finetune
        elif os.path.exists(base):
            path = base
        else:
            raise FileNotFoundError(f"Checkpoint not found for {crash_type}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    print(f"  checkpoint: {path}", flush=True)
    model = IPPO.load(
        path,
        env=env,
        custom_objects={"mlp_class": FFN, "policy_class": FeedForwardPolicy},
        device="cpu",
    )
    return model, path


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
    data_dir: str = CrashVecEnv.DATA_DIR,
    dataset_size: int = 50,
    spawn_cond: bool = False,
    center: str = "ego",
    verify_side: bool = True,
    deterministic: bool = False,
    checkpoint: str | None = None,
    seed: int | None = None,
    unique_scenes: bool = True,
):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    if seed is None:
        seed = int(time.time()) % 1_000_000
    scene_pool = _scene_pool(data_dir, dataset_size, seed)
    rng = random.Random(seed)
    print(
        f"\n=== Rendering {crash_type.upper()} | data={data_dir} | "
        f"pool={len(scene_pool)} scenes | seed={seed} | unique_scenes={unique_scenes} | "
        f"spawn_cond={spawn_cond} | center={center} | verify_side={verify_side} | "
        f"deterministic={deterministic} ===",
        flush=True,
    )

    env = CrashVecEnv(
        crash_type=crash_type,
        num_worlds=TRAIN_NUM_WORLDS,
        data_dir=data_dir,
        dataset_size=dataset_size,
        seed=seed,
        device="cpu",
        spawn_cond=spawn_cond,
    )

    model, _ = load_model(crash_type, env, checkpoint=checkpoint)
    target_labels = env.target_labels
    num_worlds = env.num_worlds
    ep_len = env.episode_len  # 91

    saved = 0
    rollout = 0
    saved_scene_keys: set[str] = set()

    while saved < n_videos and rollout < max_rollouts:
        batch = _sample_batch(scene_pool, num_worlds, rng)
        obs = env.load_scene_paths(batch)

        # Per-world frame buffers and done flags for this rollout
        world_frames: list[list[np.ndarray]] = [[] for _ in range(num_worlds)]
        world_done = [False] * num_worlds

        for step in range(ep_len + 1):
            state = env._global_state()
            # Render active (not yet done) worlds
            for w in range(num_worlds):
                if not world_done[w]:
                    npc_idx = int(env._npc_idxs[w].item())
                    ego_idx = int(env._ego_idxs[w].item())
                    center_idx = ego_idx if center == "ego" else npc_idx
                    spawn_y = float(env._spawn_rel_y[w].item())
                    ly = _contact_ly_ego_frame(env, state, w)
                    scene_path = env._env.data_batch[w]
                    scene_id = _scene_key(scene_path)
                    overlay = (
                        f"{crash_type.upper()} | step {step} | "
                        f"scene={scene_id[-24:]} | "
                        f"spawn_y={spawn_y:+.1f}m | npc_ly={ly:+.1f}m "
                        f"(−=left, +=right)"
                    )
                    figs = env._env.vis.plot_simulator_state(
                        env_indices=[w],
                        center_agent_indices=[center_idx],
                        zoom_radius=zoom,
                    )
                    ax = figs[0].axes[0]
                    ax.text(
                        0.02,
                        0.98,
                        overlay,
                        transform=ax.transAxes,
                        fontsize=9,
                        va="top",
                        color="white",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.65),
                    )
                    world_frames[w].append(img_from_fig(figs[0]))
                    plt.close(figs[0])

            if all(world_done):
                break

            action, _ = model.predict(obs.numpy(), deterministic=deterministic)
            obs, _, dones, infos = env.step(torch.from_numpy(action))

            for w in range(num_worlds):
                if not world_done[w] and dones[w].item():
                    world_done[w] = True
                    label = infos[w].get("crash_label")
                    collided = infos[w].get("collided", False)
                    contact_ly = float(infos[w].get("contact_ly", 0.0))
                    if not collided:
                        contact_ly = _contact_ly_ego_frame(env, env._global_state(), w)
                    status = f"crash:{label}" if collided else "timeout/goal"
                    print(
                        f"  rollout {rollout:3d} w{w} | {status} | ly={contact_ly:+.2f} | "
                        f"frames={len(world_frames[w])} | saved={saved}/{n_videos}",
                        flush=True,
                    )

                    side_valid = (not verify_side) or _side_ok(
                        crash_type,
                        contact_ly,
                        label=label,
                        ego_feature=infos[w].get("ego_contact_feature"),
                    )
                    scene_id = _scene_key(env._env.data_batch[w])
                    if (
                        collided
                        and label in target_labels
                        and side_valid
                        and saved < n_videos
                    ):
                        if unique_scenes and scene_id in saved_scene_keys:
                            print(
                                f"  skip w{w}: duplicate scene {scene_id} "
                                f"(already saved)",
                                flush=True,
                            )
                            continue
                        saved += 1
                        if unique_scenes:
                            saved_scene_keys.add(scene_id)
                        out_path = os.path.join(VIDEO_DIR, f"{crash_type}_ep{saved:02d}.mp4")
                        frames_to_mp4(world_frames[w], out_path)
                        print(
                            f"  [{saved}/{n_videos}] saved {os.path.basename(out_path)}  "
                            f"scene={scene_id}  label={label}  contact_ly={contact_ly:+.2f}",
                            flush=True,
                        )
                    elif collided and label in target_labels and not side_valid:
                        print(
                            f"  skip w{w}: label={label} but contact_ly={contact_ly:+.2f} "
                            f"wrong side for {crash_type.upper()}",
                            flush=True,
                        )

        rollout += 1

    if saved < n_videos:
        print(f"  Warning: only {saved}/{n_videos} videos in {rollout} rollouts.", flush=True)
    print(f"Done. Videos → {VIDEO_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash_types",  nargs="+", default=["ssl", "ssr"],
                        choices=["ssl", "ssr", "re"])
    parser.add_argument("--n_videos",     type=int, default=5)
    parser.add_argument("--max_rollouts", type=int, default=200,
                        help="Max scene rollouts before giving up (default 200)")
    parser.add_argument("--zoom",         type=int, default=50)
    parser.add_argument("--data_dir",     type=str, default=CrashVecEnv.DATA_DIR,
                        help="Path to processed scene JSONs (e.g. data/processed/validation)")
    parser.add_argument("--dataset_size", type=int, default=1500,
                        help="Random pool size sampled from data_dir (use 1500+ for variety)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for scene sampling (default: random each run)")
    parser.add_argument(
        "--no_unique_scenes",
        action="store_true",
        help="Allow multiple videos from the same base Waymo scene",
    )
    parser.add_argument(
        "--spawn_cond",
        action="store_true",
        help="15-dim obs for multi-spawn checkpoints (must match training)",
    )
    parser.add_argument(
        "--center",
        choices=["ego", "npc"],
        default="ego",
        help="Camera center: ego makes left/right contact geometry obvious",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit .zip path (default: finetune_best, else best)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use mean policy action (default: stochastic, matches training exploration)",
    )
    parser.add_argument(
        "--no_verify_side",
        action="store_true",
        help="Save target-label crashes even if contact is on wrong lateral side",
    )
    args = parser.parse_args()

    for ct in args.crash_types:
        render_episodes(
            ct,
            n_videos=args.n_videos,
            max_rollouts=args.max_rollouts,
            zoom=args.zoom,
            data_dir=args.data_dir,
            dataset_size=args.dataset_size,
            spawn_cond=args.spawn_cond,
            center=args.center,
            verify_side=not args.no_verify_side,
            deterministic=args.deterministic,
            checkpoint=args.checkpoint,
            seed=args.seed,
            unique_scenes=not args.no_unique_scenes,
        )


if __name__ == "__main__":
    main()
