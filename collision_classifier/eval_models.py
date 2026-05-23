"""
Quick batch evaluation for SSL / SSR / RE crash NPC checkpoints.

Runs each crash type in a separate process so all three can evaluate in parallel
on CPU (or cuda if --device cuda and VRAM allows one env per process).

Usage:
  .venv/bin/python -m collision_classifier.eval_models
  .venv/bin/python -m collision_classifier.eval_models --crash_types ssl ssr re --episodes 300
  .venv/bin/python -m collision_classifier.eval_models --parallel --device cpu
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from typing import Dict, List

import torch

from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy

from collision_classifier.ppo_env import CrashVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_DATA = {
    "ssl": "data/processed/synthetic/training_ssl",
    "ssr": "data/processed/synthetic/training_ssr",
    "re": "data/processed/synthetic/training_re",
}


def _resolve_checkpoint(crash_type: str) -> str:
    for name in (f"{crash_type}_ppo_finetune_best", f"{crash_type}_ppo_best"):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.zip")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint for {crash_type} in {CHECKPOINT_DIR}")


def evaluate_one(
    crash_type: str,
    episodes: int,
    num_worlds: int,
    data_dir: str,
    device: str,
    spawn_cond: bool,
    result_queue: mp.Queue | None = None,
) -> Dict:
    ckpt = _resolve_checkpoint(crash_type)
    env = CrashVecEnv(
        crash_type=crash_type,
        num_worlds=num_worlds,
        data_dir=data_dir,
        dataset_size=max(50, num_worlds * 4),
        device=device,
        spawn_cond=spawn_cond,
    )
    model = IPPO.load(
        ckpt,
        env=env,
        custom_objects={"mlp_class": FFN, "policy_class": FeedForwardPolicy},
        device=device,
    )

    target_labels = set(env.target_labels)
    labels: Counter = Counter()
    mature_hits = 0
    contact_hits = 0
    episode_lens: List[int] = []
    lane_aligns: List[float] = []
    done_episodes = 0
    rollouts = 0
    max_rollouts = max(50, (episodes // num_worlds) * 4)

    t0 = time.time()
    obs = env.swap_scenes()
    world_steps = [0] * num_worlds

    while done_episodes < episodes and rollouts < max_rollouts:
        action, _ = model.predict(obs.cpu().numpy(), deterministic=True)
        obs, _, dones, infos = env.step(torch.as_tensor(action, device=env.device))

        for w in range(num_worlds):
            world_steps[w] += 1
            if not dones[w].item():
                continue
            done_episodes += 1
            info = infos[w]
            label = info.get("crash_label")
            if label:
                labels[label] += 1
            if info.get("collided") and label in target_labels:
                mature_hits += 1
            if info.get("collided") and label and (
                label in target_labels
                or str(label).startswith("too_early:")
            ):
                contact_hits += 1
            lane_aligns.append(float(info.get("lane_align", 0.0)))
            episode_lens.append(world_steps[w])
            world_steps[w] = 0

        if done_episodes >= episodes:
            break
        if all(d.item() for d in dones):
            obs = env.swap_scenes()
            world_steps = [0] * num_worlds
            rollouts += 1

    elapsed = time.time() - t0
    collision_eps = sum(
        c for lab, c in labels.items() if lab and not str(lab).startswith("non_target")
    )
    result = {
        "crash_type": crash_type,
        "checkpoint": ckpt,
        "episodes": done_episodes,
        "crash_rate": mature_hits / max(done_episodes, 1),
        "target_contact_rate": contact_hits / max(done_episodes, 1),
        "target_collision_purity": mature_hits / max(collision_eps, 1),
        "mean_episode_len": sum(episode_lens) / max(len(episode_lens), 1),
        "mean_lane_align": sum(lane_aligns) / max(len(lane_aligns), 1),
        "label_counts": dict(labels.most_common(12)),
        "elapsed_s": round(elapsed, 1),
    }
    if result_queue is not None:
        result_queue.put(result)
    return result


def _print_result(r: Dict) -> None:
    ct = r["crash_type"].upper()
    print(f"\n=== {ct} eval ({r['episodes']} episodes, {r['elapsed_s']}s) ===")
    print(f"  checkpoint: {os.path.basename(r['checkpoint'])}")
    print(f"  crash_rate={r['crash_rate']:.3f}  target_contact={r['target_contact_rate']:.3f}")
    print(f"  collision_purity={r['target_collision_purity']:.3f}  mean_len={r['mean_episode_len']:.1f}")
    print(f"  lane_align={r['mean_lane_align']:.3f}")
    print(f"  labels: {r['label_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-eval crash NPC checkpoints")
    parser.add_argument("--crash_types", nargs="+", default=["ssl", "ssr", "re"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--num_worlds", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--parallel", action="store_true",
                        help="Evaluate each crash type in a separate process")
    parser.add_argument("--spawn_cond", action="store_true")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    data_dirs = {ct: DEFAULT_DATA[ct] for ct in args.crash_types}
    results: List[Dict] = []

    if args.parallel and len(args.crash_types) > 1:
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        procs: List[mp.Process] = []
        for ct in args.crash_types:
            p = ctx.Process(
                target=evaluate_one,
                args=(
                    ct,
                    args.episodes,
                    args.num_worlds,
                    data_dirs[ct],
                    args.device,
                    args.spawn_cond,
                    queue,
                ),
            )
            p.start()
            procs.append(p)
        for _ in procs:
            results.append(queue.get())
        for p in procs:
            p.join()
        results.sort(key=lambda r: r["crash_type"])
    else:
        for ct in args.crash_types:
            results.append(
                evaluate_one(
                    ct,
                    args.episodes,
                    args.num_worlds,
                    data_dirs[ct],
                    args.device,
                    args.spawn_cond,
                )
            )

    for r in results:
        _print_result(r)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
