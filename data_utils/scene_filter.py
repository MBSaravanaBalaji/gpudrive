"""
Filter processed Waymo JSON scenes by geometric viability for crash NPC training.

For each crash type, checks whether any vehicle pair in the scene has the geometry
needed for an NPC (agent 0 or any vehicle in loose mode) to cause that crash type.

Geometric criteria (NPC frame: x=forward, y=right):
  SSL  — same heading (±30°), ego to NPC's RIGHT:  ly ∈ [1.5, 5.5]m, |lx| < 20m
  SSR  — same heading (±30°), ego to NPC's LEFT:   ly ∈ [-5.5, -1.5]m, |lx| < 20m
  RE   — same heading (±30°), ego AHEAD of NPC:    lx ∈ [3, 30]m,  |ly| < 2.5m

Usage:
  python -m data_utils.scene_filter \\
      --input_dir  data/processed/training \\
      --output_dir data/processed/filtered \\
      --crash_type all \\
      --num_workers 8

  # dry run — stats only, no copying
  python -m data_utils.scene_filter --input_dir data/processed/training --dry_run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it

# ── Geometric thresholds ──────────────────────────────────────────────────────

HEADING_TOL_DEG = 30.0          # max heading difference between NPC and ego
HEADING_TOL_RAD = math.radians(HEADING_TOL_DEG)

SSL_LY_MIN  =  1.5              # ego to NPC's right (NPC frame y > 0 → ego is rightward)
SSL_LY_MAX  =  5.5
SSL_LX_ABS  = 20.0              # |longitudinal offset| doesn't matter much

SSR_LY_MIN  = -5.5              # ego to NPC's left
SSR_LY_MAX  = -1.5
SSR_LX_ABS  = 20.0

RE_LX_MIN   =  3.0              # ego ahead of NPC
RE_LX_MAX   = 30.0
RE_LY_ABS   =  2.5              # laterally aligned

MIN_VEHICLES = 2                # scene must have at least this many moving vehicles

CRASH_TYPES = ["ssl", "ssr", "re"]


# ── Per-scene result ──────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    path: str
    viable: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""          # why skipped entirely (< MIN_VEHICLES, parse error, etc.)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _heading_diff(h_npc: float, h_ego: float) -> float:
    """Absolute angular difference in [0, pi]."""
    diff = abs(h_npc - h_ego)
    diff = diff % (2 * math.pi)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff


def _to_npc_frame(
    npc_x: float, npc_y: float, npc_h: float,
    ego_x: float, ego_y: float,
) -> Tuple[float, float]:
    """Transform ego position into NPC body frame (x=forward, y=right)."""
    dx = ego_x - npc_x
    dy = ego_y - npc_y
    # Rotate by -npc_h
    lx =  math.cos(npc_h) * dx + math.sin(npc_h) * dy
    ly = -math.sin(npc_h) * dx + math.cos(npc_h) * dy
    return lx, ly


def _pair_viable(
    npc_x: float, npc_y: float, npc_h: float,
    ego_x: float, ego_y: float, ego_h: float,
    crash_type: str,
) -> bool:
    if _heading_diff(npc_h, ego_h) > HEADING_TOL_RAD:
        return False
    lx, ly = _to_npc_frame(npc_x, npc_y, npc_h, ego_x, ego_y)
    if crash_type == "ssl":
        return SSL_LY_MIN <= ly <= SSL_LY_MAX and abs(lx) < SSL_LX_ABS
    if crash_type == "ssr":
        return SSR_LY_MIN <= ly <= SSR_LY_MAX and abs(lx) < SSR_LX_ABS
    if crash_type == "re":
        return RE_LX_MIN <= lx <= RE_LX_MAX and abs(ly) < RE_LY_ABS
    return False


# ── Scene-level filter ────────────────────────────────────────────────────────

def _extract_vehicles(scene: dict) -> List[dict]:
    """Return list of vehicle objects that have at least one valid timestep."""
    vehicles = []
    for obj in scene.get("objects", []):
        if obj.get("type", "").lower() not in ("vehicle", "car"):
            continue
        # position may be a list of per-timestep [x,y] or just [x,y]
        pos = obj.get("position", [])
        heading = obj.get("heading", [])
        valid = obj.get("valid", [])
        # accept if any timestep is valid (or no validity mask = always valid)
        if valid and not any(valid):
            continue
        vehicles.append(obj)
    return vehicles


def _first_valid_pose(obj: dict) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, heading) from the first valid timestep."""
    pos = obj.get("position", [])
    heading = obj.get("heading", [])
    valid = obj.get("valid", [])

    # Scalar case: pos = [x, y]
    if pos and not isinstance(pos[0], (list, tuple)):
        x, y = float(pos[0]), float(pos[1])
        h = float(heading) if not isinstance(heading, (list, tuple)) else float(heading[0])
        return x, y, h

    # List-of-timesteps case
    for t in range(len(pos)):
        v = valid[t] if t < len(valid) else True
        if v:
            p = pos[t]
            x, y = float(p[0]), float(p[1])
            h = float(heading[t]) if t < len(heading) else 0.0
            return x, y, h

    return None


def filter_scene(
    path: str,
    crash_types: List[str],
    strict: bool = False,
) -> FilterResult:
    """
    Args:
        strict: if True, only consider agent-0 as NPC (matches CrashVecEnv).
                if False, any vehicle pair (more scenes pass).
    """
    result = FilterResult(path=path)
    try:
        with open(path) as f:
            scene = json.load(f)
    except Exception as e:
        result.reason = f"parse_error: {e}"
        result.viable = {ct: False for ct in crash_types}
        return result

    vehicles = _extract_vehicles(scene)
    if len(vehicles) < MIN_VEHICLES:
        result.reason = f"too_few_vehicles:{len(vehicles)}"
        result.viable = {ct: False for ct in crash_types}
        return result

    poses = [_first_valid_pose(v) for v in vehicles]

    for ct in crash_types:
        found = False
        npc_range = range(0, 1) if strict else range(len(poses))
        for ni in npc_range:
            if poses[ni] is None:
                continue
            nx, ny, nh = poses[ni]
            for ei in range(len(poses)):
                if ei == ni or poses[ei] is None:
                    continue
                ex, ey, eh = poses[ei]
                if _pair_viable(nx, ny, nh, ex, ey, eh, ct):
                    found = True
                    break
            if found:
                break
        result.viable[ct] = found

    return result


# ── Worker (top-level for pickling) ──────────────────────────────────────────

_WORKER_CRASH_TYPES: List[str] = []
_WORKER_STRICT: bool = False


def _worker_init(crash_types: List[str], strict: bool) -> None:
    global _WORKER_CRASH_TYPES, _WORKER_STRICT
    _WORKER_CRASH_TYPES = crash_types
    _WORKER_STRICT = strict


def _worker(path: str) -> FilterResult:
    return filter_scene(path, _WORKER_CRASH_TYPES, strict=_WORKER_STRICT)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    input_dir: str,
    output_dir: str,
    crash_types: List[str],
    num_workers: int = 4,
    dry_run: bool = False,
    strict: bool = False,
) -> None:
    json_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.endswith(".json")
    )
    if not json_files:
        print(f"No JSON files found under {input_dir}")
        return

    print(f"Found {len(json_files)} scenes in {input_dir}")
    print(f"Crash types: {crash_types}  |  strict={strict}  |  dry_run={dry_run}  |  workers={num_workers}\n")

    if not dry_run:
        for ct in crash_types:
            os.makedirs(os.path.join(output_dir, f"training_{ct}"), exist_ok=True)

    counts: Dict[str, int] = {ct: 0 for ct in crash_types}
    total = len(json_files)

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(crash_types, strict),
    ) as pool:
        for result in tqdm(pool.imap_unordered(_worker, json_files), total=total, desc="filtering"):
            for ct in crash_types:
                if result.viable.get(ct, False):
                    counts[ct] += 1
                    if not dry_run:
                        dst = os.path.join(output_dir, f"training_{ct}", os.path.basename(result.path))
                        if not os.path.exists(dst):
                            shutil.copy2(result.path, dst)

    print("\n=== Filter results ===")
    for ct in crash_types:
        pct = 100.0 * counts[ct] / max(total, 1)
        print(f"  {ct.upper():3s}: {counts[ct]:>6d} / {total} scenes viable  ({pct:.1f}%)")
    if not dry_run:
        print(f"\nCopied to: {output_dir}/training_{{ssl,ssr,re}}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Waymo scenes by crash-type geometry")
    parser.add_argument("--input_dir",   required=True,  help="Root dir of processed JSON scenes")
    parser.add_argument("--output_dir",  default="data/processed/filtered",
                        help="Destination root for filtered subdirs")
    parser.add_argument("--crash_type",  default="all", choices=["all", "ssl", "ssr", "re"],
                        help="Which crash type(s) to filter for")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run",     action="store_true",
                        help="Report stats only — no files are copied")
    parser.add_argument("--strict",      action="store_true",
                        help="Only consider agent-0 as NPC (matches CrashVecEnv)")
    args = parser.parse_args()

    crash_types = CRASH_TYPES if args.crash_type == "all" else [args.crash_type]

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crash_types=crash_types,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
