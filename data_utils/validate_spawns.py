"""
Batch drivability check for synthetic spawn scenes.

Rolls each scene briefly with mild forward / straight NPC steering and
flags worlds where the controlled NPC goes off-road immediately.

Usage:
  python -m data_utils.validate_spawns \\
      --data_dir data/processed/multi_spawn/training_ssl \\
      --crash_type ssl \\
      --num_worlds 32 \\
      --probe_steps 15

  # Remove failed scenes (writes list to --reject_log)
  python -m data_utils.validate_spawns \\
      --data_dir data/processed/multi_spawn/training_ssl \\
      --crash_type ssl \\
      --delete_invalid
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import torch

from collision_classifier.ppo_env import PROBE_ACTION_IDX, CrashVecEnv
from data_utils.spawn_families import infer_spawn_family_from_path


def _load_scene_paths(data_dir: str) -> List[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".json") and f.startswith("tfrecord")
    )


def validate_batch(
    env: CrashVecEnv,
    paths: List[str],
    probe_steps: int,
) -> List[Tuple[str, bool, str]]:
    """Return (path, ok, reason) for each scene in paths."""
    env.load_scene_paths(paths)
    offroad_any = torch.zeros(env.num_worlds, dtype=torch.bool, device=env.device)
    for _ in range(probe_steps):
        actions = torch.full(
            (env.num_worlds,),
            PROBE_ACTION_IDX,
            dtype=torch.long,
            device=env.device,
        )
        _, _, _, infos = env.step(actions)
        for w, info in enumerate(infos):
            if info.get("offroad"):
                offroad_any[w] = True

    results: List[Tuple[str, bool, str]] = []
    for w, path in enumerate(paths):
        if offroad_any[w].item():
            results.append((path, False, "offroad"))
        else:
            results.append((path, True, "ok"))
    return results


def run(
    data_dir: str,
    crash_type: str,
    num_worlds: int = 32,
    probe_steps: int = 15,
    dataset_size: int = 10_000,
    delete_invalid: bool = False,
    reject_log: str | None = None,
    device: str = "cuda",
) -> None:
    paths = _load_scene_paths(data_dir)
    if not paths:
        print(f"No scenes under {data_dir}")
        return

    ok_count = 0
    bad: List[str] = []
    family_stats: dict[str, dict[str, int]] = {}

    for start in range(0, len(paths), num_worlds):
        chunk = paths[start : start + num_worlds]
        env = CrashVecEnv(
            crash_type=crash_type,
            num_worlds=len(chunk),
            data_dir=data_dir,
            dataset_size=min(dataset_size, len(chunk)),
            device=device,
        )
        results = validate_batch(env, chunk, probe_steps)
        for path, ok, _reason in results:
            family = "unknown"
            try:
                with open(path) as f:
                    family = json.load(f).get("metadata", {}).get("crash_pair", {}).get(
                        "spawn_family"
                    ) or infer_spawn_family_from_path(path)
            except OSError:
                family = infer_spawn_family_from_path(path)
            bucket = family_stats.setdefault(family, {"ok": 0, "bad": 0})
            if ok:
                ok_count += 1
                bucket["ok"] += 1
            else:
                bad.append(path)
                bucket["bad"] += 1
                if delete_invalid:
                    os.remove(path)

    total = len(paths)
    print(f"\n=== Drivability validation ({crash_type.upper()}) ===")
    print(f"  dir={data_dir}")
    print(f"  scenes={total}  ok={ok_count}  bad={len(bad)}  "
          f"pass_rate={100.0 * ok_count / max(total, 1):.1f}%")
    print("  by spawn_family:")
    for family in sorted(family_stats):
        s = family_stats[family]
        n = s["ok"] + s["bad"]
        pct = 100.0 * s["ok"] / max(n, 1)
        print(f"    {family:16s}  ok={s['ok']:4d}  bad={s['bad']:4d}  pass={pct:.1f}%")

    if reject_log and bad:
        os.makedirs(os.path.dirname(reject_log) or ".", exist_ok=True)
        with open(reject_log, "w") as f:
            f.write("\n".join(bad) + "\n")
        print(f"  reject_log → {reject_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic spawn drivability")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--crash_type", required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--num_worlds", type=int, default=32)
    parser.add_argument("--probe_steps", type=int, default=15)
    parser.add_argument("--dataset_size", type=int, default=10_000)
    parser.add_argument("--delete_invalid", action="store_true")
    parser.add_argument("--reject_log", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        crash_type=args.crash_type,
        num_worlds=args.num_worlds,
        probe_steps=args.probe_steps,
        dataset_size=args.dataset_size,
        delete_invalid=args.delete_invalid,
        reject_log=args.reject_log,
        device=args.device,
    )


if __name__ == "__main__":
    main()
