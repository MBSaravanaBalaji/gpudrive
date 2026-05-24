"""
Tier-3 heuristic rollout validation in GPUDrive sim.

Rolls a spawn-family-specific open-loop controller for up to 91 steps and
keeps scenes where the NPC either:
  - achieves a target-type collision, or
  - reaches within 3m of the ideal contact manifold by step 80.

Usage:
  python -m data_utils.validate_heuristic \\
      --data_dir data/processed/multi_spawn_v2/training_ssr \\
      --crash_type ssr \\
      --num_worlds 16 \\
      --delete_invalid
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Tuple

import torch

from collision_classifier.ppo_env import CrashVecEnv, _REALISTIC_ACCEL_VALUES, _REALISTIC_STEER_VALUES
from data_utils.coords import ideal_contact_ly
from data_utils.spawn_families import infer_spawn_family_from_path

MAX_STEPS = 91
CONTACT_DIST = 3.0


def _action_idx(accel: float, steer: float) -> int:
    ai = _REALISTIC_ACCEL_VALUES.index(accel)
    sj = _REALISTIC_STEER_VALUES.index(steer)
    return ai * len(_REALISTIC_STEER_VALUES) + sj


def _family_action(spawn_family: str, crash_type: str, step: int) -> int:
    max_a = _action_idx(3.5, 0.0)
    mid_a = _action_idx(2.5, 0.0)
    low_a = _action_idx(0.5, 0.0)
    left = _action_idx(2.5, -0.2)
    right = _action_idx(2.5, 0.2)

    if crash_type == "re":
        if spawn_family == "ahead":
            return low_a if step < 25 else max_a
        return max_a

    if spawn_family in ("behind_near", "behind_far"):
        return right if crash_type == "ssr" else left if step > 8 else max_a
    if spawn_family == "side_same":
        return right if crash_type == "ssr" else left
    if spawn_family == "side_opposite":
        steer = right if crash_type == "ssr" else left
        return steer if step < 45 else mid_a
    if spawn_family == "ahead":
        return low_a if step < 20 else (right if crash_type == "ssr" else left)
    return mid_a


def _load_paths(data_dir: str) -> List[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".json") and f.startswith("tfrecord")
    )


def validate_batch(
    env: CrashVecEnv,
    paths: List[str],
    crash_type: str,
) -> List[Tuple[str, bool, str]]:
    env.load_scene_paths(paths)
    outcomes: List[Tuple[str, bool, str]] = [(p, False, "no_contact") for p in paths]
    target_label = {
        "ssl": "side-swipe-left",
        "ssr": "side-swipe-right",
        "re": "rear-ended",
    }[crash_type]

    for step in range(MAX_STEPS):
        actions = []
        for w, path in enumerate(paths):
            fam = env._spawn_families[w]
            if not fam or fam == "unknown":
                try:
                    with open(path) as f:
                        fam = json.load(f).get("metadata", {}).get("crash_pair", {}).get(
                            "spawn_family", "unknown"
                        )
                except OSError:
                    fam = infer_spawn_family_from_path(path)
            actions.append(_family_action(fam, crash_type, step))
        _, _, _, infos = env.step(torch.tensor(actions, dtype=torch.long, device=env.device))
        for w, info in enumerate(infos):
            if outcomes[w][1]:
                continue
            if info.get("offroad"):
                outcomes[w] = (paths[w], False, "offroad")
                continue
            label = info.get("crash_label")
            if info.get("collided") and label == target_label:
                outcomes[w] = (paths[w], True, "target_collision")
                continue
            lx = float(info.get("contact_lx", 0.0))
            ly = float(info.get("contact_ly", 0.0))
            if crash_type == "re":
                dist = math.hypot(lx + 4.0, ly)
            else:
                dist = math.hypot(lx, ly - ideal_contact_ly(crash_type))
            if dist <= CONTACT_DIST and step >= 10:
                outcomes[w] = (paths[w], True, "near_manifold")
        if all(ok for _, ok, _ in outcomes):
            break

    return outcomes


def run(
    data_dir: str,
    crash_type: str,
    num_worlds: int = 16,
    delete_invalid: bool = False,
    reject_log: str | None = None,
    device: str = "cuda",
    limit: int | None = None,
) -> None:
    paths = _load_paths(data_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        print(f"No scenes under {data_dir}")
        return

    ok_count = 0
    bad: List[str] = []
    family_stats: dict[str, dict[str, int]] = {}

    def _make_env(n_worlds: int) -> CrashVecEnv:
        return CrashVecEnv(
            crash_type=crash_type,
            num_worlds=n_worlds,
            data_dir=data_dir,
            dataset_size=n_worlds,
            device=device,
        )

    env = _make_env(num_worlds)
    for start in range(0, len(paths), num_worlds):
        chunk = paths[start : start + num_worlds]
        batch_env = env if len(chunk) == num_worlds else _make_env(len(chunk))
        batch = validate_batch(batch_env, chunk, crash_type)
        for path, ok, reason in batch:
            family = infer_spawn_family_from_path(path)
            bucket = family_stats.setdefault(family, {"ok": 0, "bad": 0})
            if ok:
                ok_count += 1
                bucket["ok"] += 1
            else:
                bad.append(f"{path}\t{reason}")
                bucket["bad"] += 1
                if delete_invalid:
                    os.remove(path)

    total = len(paths)
    print(f"\n=== Heuristic rollout validation ({crash_type.upper()}) ===")
    print(f"  dir={data_dir}")
    print(f"  scenes={total}  ok={ok_count}  bad={len(bad)}  pass_rate={100.0 * ok_count / max(total, 1):.1f}%")
    print("  by spawn_family:")
    for family in sorted(family_stats):
        s = family_stats[family]
        n = s["ok"] + s["bad"]
        print(f"    {family:16s}  ok={s['ok']:4d}  bad={s['bad']:4d}  pass={100.0 * s['ok'] / max(n, 1):.1f}%")

    if reject_log and bad:
        os.makedirs(os.path.dirname(reject_log) or ".", exist_ok=True)
        with open(reject_log, "w") as f:
            f.write("\n".join(bad) + "\n")
        print(f"  reject_log → {reject_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-3 heuristic sim validation")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--crash_type", required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--num_worlds", type=int, default=16)
    parser.add_argument("--delete_invalid", action="store_true")
    parser.add_argument("--reject_log", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Debug: validate first N scenes only")
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        crash_type=args.crash_type,
        num_worlds=args.num_worlds,
        delete_invalid=args.delete_invalid,
        reject_log=args.reject_log,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
