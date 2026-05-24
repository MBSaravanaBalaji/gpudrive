"""
Tier-2 static reachability checks for synthetic multi-spawn scenes.

Reads scene JSON metadata (no GPU) and rejects spawns that cannot plausibly
reach a typed contact within the ego log horizon (~91 steps).

Usage:
  python -m data_utils.validate_reachability \\
      --data_dir data/processed/multi_spawn_v2/training_ssr \\
      --crash_type ssr \\
      --delete_invalid
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Tuple

from data_utils.coords import ideal_contact_ly
from data_utils.spawn_families import MIN_SUCCESS_STEP_BY_FAMILY, infer_spawn_family_from_path

SIM_DT = 0.04
EPISODE_HORIZON = 91
MIN_EGO_VALID_STEPS = 80


def _load_paths(data_dir: str) -> List[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".json") and f.startswith("tfrecord")
    )


def _ego_valid_steps(scene: dict) -> int:
    objects = scene.get("objects", [])
    if len(objects) < 2:
        return 0
    ego = objects[1]
    valid = ego.get("valid", [])
    if valid:
        return sum(1 for v in valid if v)
    pos = ego.get("position", [])
    return len(pos) if isinstance(pos, list) else 0


def _estimate_contact_step(
    crash_type: str,
    spawn_family: str,
    rel_x: float,
    rel_y: float,
    speed_delta: float,
) -> Tuple[float, str]:
    """Return (estimated_step, reason_tag). Lower is better."""
    ly_target = ideal_contact_ly(crash_type)

    if crash_type == "re":
        if spawn_family == "ahead":
            # NPC must fall behind ego then close — conservative budget
            return 35.0 + abs(rel_x) / max(0.8, 1.0 - speed_delta), "re_ahead"
        long_dist = abs(rel_x) if rel_x < 0 else abs(rel_x) + 6.0
        closing = max(0.8, speed_delta)
        return long_dist / (closing * SIM_DT), "re_behind"

    # SSL / SSR
    long_dist = abs(rel_x) if rel_x <= 0 else rel_x + 4.0
    lat_dist = abs(rel_y - ly_target)
    closing = max(0.8, speed_delta)

    if spawn_family in ("behind_near", "behind_far"):
        long_steps = long_dist / (closing * SIM_DT)
        lat_steps = lat_dist / (0.8 * SIM_DT)
        return max(long_steps, lat_steps), "behind_overtake"
    if spawn_family == "side_same":
        lat_steps = lat_dist / (0.7 * SIM_DT)
        long_steps = abs(rel_x) / (0.6 * SIM_DT)
        return max(lat_steps, long_steps), "side_same"
    if spawn_family == "side_opposite":
        lat_dist = abs(rel_y - ly_target)
        lat_steps = lat_dist / (1.2 * SIM_DT)
        long_steps = abs(rel_x) / (0.8 * SIM_DT)
        return max(lat_steps, long_steps) + 8.0, "side_opposite"
    if spawn_family == "ahead":
        closing = max(0.8, 1.5 - speed_delta)
        long_steps = abs(rel_x) / (closing * SIM_DT)
        lat_steps = abs(rel_y - ly_target) / (1.0 * SIM_DT)
        return max(long_steps, lat_steps) + 6.0, "ahead_cut"
    return long_dist / (closing * SIM_DT), "generic"


def validate_scene(path: str, crash_type: str) -> Tuple[bool, str]:
    try:
        with open(path) as f:
            scene = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"parse_error:{exc}"

    meta = scene.get("metadata", {}).get("crash_pair", {})
    if meta.get("crash_type") and meta["crash_type"] != crash_type:
        return False, "wrong_crash_type"

    family = meta.get("spawn_family") or infer_spawn_family_from_path(path)
    rel_x = float(meta.get("rel_x", 0.0))
    rel_y = float(meta.get("rel_y", 0.0))
    speed_delta = float(meta.get("speed_delta", 1.5))
    min_step = int(MIN_SUCCESS_STEP_BY_FAMILY.get(family, 15))

    ego_steps = _ego_valid_steps(scene)
    if ego_steps < MIN_EGO_VALID_STEPS:
        return False, f"ego_valid_short:{ego_steps}"

    est_step, tag = _estimate_contact_step(crash_type, family, rel_x, rel_y, speed_delta)
    if est_step > EPISODE_HORIZON - 6:
        return False, f"too_slow:{tag}:{est_step:.1f}"
    if est_step < max(5, min_step - 8):
        return False, f"too_fast:{tag}:{est_step:.1f}"

    return True, "ok"


def run(
    data_dir: str,
    crash_type: str,
    delete_invalid: bool = False,
    reject_log: str | None = None,
) -> None:
    paths = _load_paths(data_dir)
    if not paths:
        print(f"No scenes under {data_dir}")
        return

    ok = 0
    bad: List[str] = []
    family_stats: dict[str, dict[str, int]] = {}

    for path in paths:
        valid, reason = validate_scene(path, crash_type)
        family = infer_spawn_family_from_path(path)
        try:
            with open(path) as f:
                family = (
                    json.load(f).get("metadata", {}).get("crash_pair", {}).get("spawn_family")
                    or family
                )
        except OSError:
            pass
        bucket = family_stats.setdefault(family, {"ok": 0, "bad": 0})
        if valid:
            ok += 1
            bucket["ok"] += 1
        else:
            bad.append(f"{path}\t{reason}")
            bucket["bad"] += 1
            if delete_invalid:
                os.remove(path)

    total = len(paths)
    print(f"\n=== Reachability validation ({crash_type.upper()}) ===")
    print(f"  dir={data_dir}")
    print(f"  scenes={total}  ok={ok}  bad={len(bad)}  pass_rate={100.0 * ok / max(total, 1):.1f}%")
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
    parser = argparse.ArgumentParser(description="Tier-2 static reachability validation")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--crash_type", required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--delete_invalid", action="store_true")
    parser.add_argument("--reject_log", type=str, default=None)
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        crash_type=args.crash_type,
        delete_invalid=args.delete_invalid,
        reject_log=args.reject_log,
    )


if __name__ == "__main__":
    main()
