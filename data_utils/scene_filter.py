"""
Filter processed Waymo JSON scenes by geometric viability for crash NPC training.

For each crash type, checks whether any vehicle pair in the scene has the geometry
needed for an NPC to cause that crash type. Filtered outputs keep only the selected
NPC and replay ego, preserving the ego's logged trajectory while removing third-party
vehicles that would pollute the crash objective.

Geometric criteria (NPC frame: x=forward, y=right):
  SSL  — same heading (±30°), ego to NPC's RIGHT:  ly ∈ [1.5, 5.5]m, |lx| ∈ [14, 30]m
  SSR  — same heading (±30°), ego to NPC's LEFT:   ly ∈ [-5.5, -1.5]m, |lx| ∈ [14, 30]m
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
import copy
import json
import math
import os
import random
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

SSL_LY_MIN  =  1.5              # ego to NPC's right (NPC is left of ego)
SSL_LY_MAX  =  5.5
SSL_LX_MIN_ABS = 14.0           # require a visible approach; near side-by-side starts crash instantly
SSL_LX_ABS  = 30.0

SSR_LY_MIN  = -5.5              # ego to NPC's left (NPC is right of ego)
SSR_LY_MAX  = -1.5
SSR_LX_MIN_ABS = 14.0
SSR_LX_ABS  = 30.0

RE_LX_MIN   =  3.0              # ego ahead of NPC
RE_LX_MAX   = 30.0
RE_LY_ABS   =  2.5              # laterally aligned

MIN_VEHICLES = 2                # scene must have at least this many moving vehicles
MIN_VALID_STEPS = 80            # keep ego visible for most of the 91-step episode
MIN_INITIAL_SPEED = 2.0         # m/s; avoid parked/creeping pairs that look static
SIM_DT = 0.04                   # Waymo/GPUDrive runs at 25 Hz

CRASH_TYPES = ["ssl", "ssr", "re"]


# ── Per-scene result ──────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    path: str
    viable: Dict[str, bool] = field(default_factory=dict)
    pairs: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
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
        return SSL_LY_MIN <= ly <= SSL_LY_MAX and SSL_LX_MIN_ABS <= abs(lx) < SSL_LX_ABS
    if crash_type == "ssr":
        return SSR_LY_MIN <= ly <= SSR_LY_MAX and SSR_LX_MIN_ABS <= abs(lx) < SSR_LX_ABS
    if crash_type == "re":
        return RE_LX_MIN <= lx <= RE_LX_MAX and abs(ly) < RE_LY_ABS
    return False


def _pair_score(
    npc_x: float, npc_y: float, npc_h: float,
    ego_x: float, ego_y: float,
    crash_type: str,
) -> float:
    """Lower is better; prefer pairs near the middle of the viable envelope."""
    lx, ly = _to_npc_frame(npc_x, npc_y, npc_h, ego_x, ego_y)
    if crash_type == "ssl":
        return abs(abs(ly) - 3.5) + 0.1 * abs(abs(lx) - 20.0)
    if crash_type == "ssr":
        return abs(abs(ly) - 3.5) + 0.1 * abs(abs(lx) - 20.0)
    if crash_type == "re":
        return abs(lx - 10.0) + 0.5 * abs(ly)
    return float("inf")


def _xy_at(pos, t: int) -> Optional[Tuple[float, float]]:
    if not pos:
        return None
    p = pos[t] if isinstance(pos, list) and pos and isinstance(pos[0], (dict, list, tuple)) else pos
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    return None


def _heading_at(heading, t: int) -> float:
    if isinstance(heading, list):
        return float(heading[t]) if t < len(heading) else 0.0
    return float(heading) if heading is not None else 0.0


def _speed_at(obj: dict, t: int) -> float:
    vel = obj.get("velocity", [])
    if not vel:
        return 0.0
    v = vel[t] if isinstance(vel, list) and vel and isinstance(vel[0], (dict, list, tuple)) else vel
    if isinstance(v, dict):
        return math.hypot(float(v.get("x", 0.0)), float(v.get("y", 0.0)))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return math.hypot(float(v[0]), float(v[1]))
    return 0.0


def _velocity_at(obj: dict, t: int) -> Tuple[float, float]:
    vel = obj.get("velocity", [])
    if not vel:
        return 0.0, 0.0
    v = vel[t] if isinstance(vel, list) and vel and isinstance(vel[0], (dict, list, tuple)) else vel
    if isinstance(v, dict):
        return float(v.get("x", 0.0)), float(v.get("y", 0.0))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return 0.0, 0.0


def _z_at(obj: dict, t: int) -> float:
    pos = obj.get("position", [])
    if not pos:
        return 0.0
    p = pos[t] if isinstance(pos, list) and pos and isinstance(pos[0], (dict, list, tuple)) else pos
    if isinstance(p, dict):
        return float(p.get("z", 0.0))
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return float(p[2])
    return 0.0


def _valid_run_length(obj: dict, start_t: int = 0) -> int:
    valid = obj.get("valid", [])
    if not valid:
        pos = obj.get("position", [])
        return len(pos) if isinstance(pos, list) else MIN_VALID_STEPS
    count = 0
    for t in range(start_t, len(valid)):
        if not valid[t]:
            break
        count += 1
    return count


def _set_pose_at(obj: dict, t: int, x: float, y: float, z: float, h: float, vx: float, vy: float) -> None:
    obj["position"][t] = {"x": x, "y": y, "z": z}
    obj["heading"][t] = h
    obj["velocity"][t] = {"x": vx, "y": vy}
    if "valid" in obj:
        obj["valid"][t] = True


# ── Scene-level filter ────────────────────────────────────────────────────────

def _extract_vehicles(scene: dict) -> List[Tuple[int, dict]]:
    """Return (object index, object) vehicles with at least one valid timestep."""
    vehicles = []
    for idx, obj in enumerate(scene.get("objects", [])):
        if obj.get("type", "").lower() not in ("vehicle", "car"):
            continue
        # position may be a list of per-timestep [x,y] or just [x,y]
        pos = obj.get("position", [])
        heading = obj.get("heading", [])
        valid = obj.get("valid", [])
        # accept if any timestep is valid (or no validity mask = always valid)
        if valid and not any(valid):
            continue
        vehicles.append((idx, obj))
    return vehicles


def _first_valid_pose(obj: dict) -> Optional[Tuple[float, float, float, float]]:
    """Return (x, y, heading, speed) from the first valid timestep."""
    pos = obj.get("position", [])
    heading = obj.get("heading", [])
    valid = obj.get("valid", [])

    # Scalar case: pos = [x, y]
    if pos and not isinstance(pos[0], (dict, list, tuple)):
        xy = _xy_at(pos, 0)
        if xy is None:
            return None
        x, y = xy
        h = _heading_at(heading, 0)
        return x, y, h, _speed_at(obj, 0)

    # List-of-timesteps case
    for t in range(len(pos)):
        v = valid[t] if t < len(valid) else True
        if v:
            xy = _xy_at(pos, t)
            if xy is None:
                continue
            x, y = xy
            h = _heading_at(heading, t)
            return x, y, h, _speed_at(obj, t)

    return None


def _find_best_pair(
    vehicles: List[Tuple[int, dict]],
    poses: List[Optional[Tuple[float, float, float, float]]],
    crash_type: str,
    strict: bool = False,
    min_valid_steps: int = MIN_VALID_STEPS,
    min_initial_speed: float = MIN_INITIAL_SPEED,
) -> Optional[Tuple[int, int]]:
    best_pair: Optional[Tuple[int, int]] = None
    best_score = float("inf")
    npc_range = range(0, 1) if strict else range(len(poses))

    for ni in npc_range:
        if poses[ni] is None:
            continue
        npc_obj_idx, npc_obj = vehicles[ni]
        if npc_obj.get("mark_as_expert", False):
            continue
        nx, ny, nh, ns = poses[ni]
        if ns < min_initial_speed or _valid_run_length(npc_obj) < min_valid_steps:
            continue
        for ei in range(len(poses)):
            if ei == ni or poses[ei] is None:
                continue
            ego_obj_idx, ego_obj = vehicles[ei]
            if ego_obj.get("mark_as_expert", False):
                continue
            ex, ey, eh, es = poses[ei]
            if es < min_initial_speed or _valid_run_length(ego_obj) < min_valid_steps:
                continue
            if not _pair_viable(nx, ny, nh, ex, ey, eh, crash_type):
                continue
            score = _pair_score(nx, ny, nh, ex, ey, crash_type)
            if score < best_score:
                best_score = score
                best_pair = (npc_obj_idx, ego_obj_idx)

    return best_pair


def _find_synthetic_pair(
    vehicles: List[Tuple[int, dict]],
    poses: List[Optional[Tuple[float, float, float, float]]],
    min_valid_steps: int = MIN_VALID_STEPS,
    min_initial_speed: float = MIN_INITIAL_SPEED,
) -> Optional[Tuple[int, int]]:
    """Pick a template NPC and replay ego for synthetic controlled-NPC spawning."""
    valid = []
    for local_idx, (obj_idx, obj) in enumerate(vehicles):
        if poses[local_idx] is None or obj.get("mark_as_expert", False):
            continue
        _, _, _, speed = poses[local_idx]
        if speed < min_initial_speed or _valid_run_length(obj) < min_valid_steps:
            continue
        valid.append((local_idx, obj_idx, obj, speed))

    if len(valid) < 2:
        return None

    # Use the fastest stable replay vehicle as ego; clone another vehicle for NPC dimensions.
    ego_local, ego_idx, _, _ = max(valid, key=lambda item: item[3])
    npc_candidates = [item for item in valid if item[0] != ego_local]
    if not npc_candidates:
        return None
    _, npc_idx, _, _ = min(npc_candidates, key=lambda item: item[3])
    return npc_idx, ego_idx


def _synthetic_spawn_params(crash_type: str, rng: random.Random) -> Tuple[float, float, float]:
    """Return (rel_x, rel_y, speed_delta) in ego frame where y>0 is ego's right."""
    rel_x = rng.uniform(-20.0, -10.0)
    if crash_type == "ssl":
        rel_y = rng.uniform(-2.2, -1.6)  # near-side corridor that stays drivable
    elif crash_type == "ssr":
        rel_y = rng.uniform(1.6, 2.2)    # near-side corridor that stays drivable
    else:
        rel_y = rng.uniform(-0.8, 0.8)
    speed_delta = rng.uniform(1.2, 2.6)
    return rel_x, rel_y, speed_delta


def _scene_with_synthetic_spawn(
    scene: dict,
    pair: Tuple[int, int],
    crash_type: str,
    variant_idx: int,
    rng: random.Random,
) -> dict:
    """Return a pair-only scene with a controlled NPC spawned relative to replay ego."""
    objects = list(scene.get("objects", []))
    npc_idx, ego_idx = pair
    ego = copy.deepcopy(objects[ego_idx])
    npc = copy.deepcopy(objects[npc_idx])

    ego_pose = _first_valid_pose(ego)
    if ego_pose is None:
        raise ValueError("synthetic spawn requires a valid ego pose")
    ego_x, ego_y, ego_h, ego_speed = ego_pose
    rel_x, rel_y, speed_delta = _synthetic_spawn_params(crash_type, rng)

    c, s = math.cos(ego_h), math.sin(ego_h)
    spawn_x = ego_x + c * rel_x - s * rel_y
    spawn_y = ego_y + s * rel_x + c * rel_y
    spawn_h = ego_h + rng.uniform(-0.05, 0.05)
    spawn_speed = max(ego_speed + speed_delta, MIN_INITIAL_SPEED)
    vx = math.cos(spawn_h) * spawn_speed
    vy = math.sin(spawn_h) * spawn_speed

    horizon = min(len(ego.get("position", [])), len(npc.get("position", [])))
    if horizon <= 0:
        raise ValueError("synthetic spawn requires non-empty trajectories")

    npc["mark_as_expert"] = False
    npc["valid"] = [True] * horizon
    npc["position"] = list(npc.get("position", []))[:horizon]
    npc["heading"] = list(npc.get("heading", []))[:horizon]
    npc["velocity"] = list(npc.get("velocity", []))[:horizon]

    for t in range(horizon):
        x = spawn_x + vx * SIM_DT * t
        y = spawn_y + vy * SIM_DT * t
        z = _z_at(ego, min(t, len(ego.get("position", [])) - 1))
        _set_pose_at(npc, t, x, y, z, spawn_h, vx, vy)

    npc["goalPosition"] = {
        "x": spawn_x + vx * SIM_DT * (horizon - 1),
        "y": spawn_y + vy * SIM_DT * (horizon - 1),
        "z": _z_at(ego, min(horizon - 1, len(ego.get("position", [])) - 1)),
    }

    out = dict(scene)
    out["objects"] = [npc, ego]
    metadata = dict(out.get("metadata", {}))
    metadata["sdc_track_index"] = 0
    metadata["tracks_to_predict"] = []
    metadata["objects_of_interest"] = []
    metadata["crash_pair"] = {
        "crash_type": crash_type,
        "npc_original_index": npc_idx,
        "ego_original_index": ego_idx,
        "npc_id": npc.get("id"),
        "ego_id": ego.get("id"),
        "pair_only": True,
        "synthetic_spawn": True,
        "variant_idx": variant_idx,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "spawn_speed": spawn_speed,
    }
    out["metadata"] = metadata
    return out


def _scene_with_pair_first(scene: dict, pair: Tuple[int, int], crash_type: str) -> dict:
    """Return a pair-only copy with NPC first and replay ego second."""
    objects = list(scene.get("objects", []))
    npc_idx, ego_idx = pair

    out = dict(scene)
    out["objects"] = [objects[npc_idx], objects[ego_idx]]
    metadata = dict(out.get("metadata", {}))
    metadata["sdc_track_index"] = 0
    metadata["tracks_to_predict"] = []
    metadata["objects_of_interest"] = []
    metadata["crash_pair"] = {
        "crash_type": crash_type,
        "npc_original_index": npc_idx,
        "ego_original_index": ego_idx,
        "npc_id": objects[npc_idx].get("id"),
        "ego_id": objects[ego_idx].get("id"),
        "pair_only": True,
    }
    out["metadata"] = metadata
    return out


def filter_scene(
    path: str,
    crash_types: List[str],
    strict: bool = False,
    synthetic_spawn: bool = False,
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

    poses = [_first_valid_pose(obj) for _, obj in vehicles]

    for ct in crash_types:
        pair = (
            _find_synthetic_pair(vehicles, poses)
            if synthetic_spawn
            else _find_best_pair(vehicles, poses, ct, strict=strict)
        )
        result.pairs[ct] = pair
        result.viable[ct] = pair is not None

    return result


# ── Worker (top-level for pickling) ──────────────────────────────────────────

_WORKER_CRASH_TYPES: List[str] = []
_WORKER_STRICT: bool = False
_WORKER_SYNTHETIC_SPAWN: bool = False


def _worker_init(crash_types: List[str], strict: bool, synthetic_spawn: bool) -> None:
    global _WORKER_CRASH_TYPES, _WORKER_STRICT, _WORKER_SYNTHETIC_SPAWN
    _WORKER_CRASH_TYPES = crash_types
    _WORKER_STRICT = strict
    _WORKER_SYNTHETIC_SPAWN = synthetic_spawn


def _worker(path: str) -> FilterResult:
    return filter_scene(
        path,
        _WORKER_CRASH_TYPES,
        strict=_WORKER_STRICT,
        synthetic_spawn=_WORKER_SYNTHETIC_SPAWN,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    input_dir: str,
    output_dir: str,
    crash_types: List[str],
    num_workers: int = 4,
    dry_run: bool = False,
    strict: bool = False,
    synthetic_spawn: bool = False,
    variants_per_scene: int = 1,
    spawn_seed: int = 42,
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
    print(
        f"Crash types: {crash_types}  |  strict={strict}  |  dry_run={dry_run}  |  "
        f"synthetic_spawn={synthetic_spawn}  |  variants={variants_per_scene}  |  workers={num_workers}\n"
    )

    if not dry_run:
        for ct in crash_types:
            os.makedirs(os.path.join(output_dir, f"training_{ct}"), exist_ok=True)

    counts: Dict[str, int] = {ct: 0 for ct in crash_types}
    total = len(json_files)

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(crash_types, strict, synthetic_spawn),
    ) as pool:
        for result in tqdm(pool.imap_unordered(_worker, json_files), total=total, desc="filtering"):
            for ct in crash_types:
                if result.viable.get(ct, False):
                    counts[ct] += variants_per_scene if synthetic_spawn else 1
                    if not dry_run:
                        with open(result.path) as f:
                            scene = json.load(f)
                        pair = result.pairs.get(ct)
                        if pair is None:
                            dst = os.path.join(output_dir, f"training_{ct}", os.path.basename(result.path))
                            shutil.copy2(result.path, dst)
                        elif synthetic_spawn:
                            stem, ext = os.path.splitext(os.path.basename(result.path))
                            for variant_idx in range(variants_per_scene):
                                rng = random.Random(f"{spawn_seed}:{result.path}:{ct}:{variant_idx}")
                                dst = os.path.join(
                                    output_dir,
                                    f"training_{ct}",
                                    f"{stem}_syn{variant_idx:02d}{ext}",
                                )
                                spawned = _scene_with_synthetic_spawn(
                                    scene, pair, ct, variant_idx, rng
                                )
                                with open(dst, "w") as f:
                                    json.dump(spawned, f)
                        else:
                            dst = os.path.join(output_dir, f"training_{ct}", os.path.basename(result.path))
                            with open(dst, "w") as f:
                                json.dump(_scene_with_pair_first(scene, pair, ct), f)

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
    parser.add_argument("--synthetic_spawn", action="store_true",
                        help="Clone a vehicle as controlled NPC and spawn it relative to replay ego")
    parser.add_argument("--variants_per_scene", type=int, default=1,
                        help="Synthetic spawn variants to write per source scene")
    parser.add_argument("--spawn_seed", type=int, default=42,
                        help="Seed for deterministic synthetic spawn variants")
    args = parser.parse_args()

    crash_types = CRASH_TYPES if args.crash_type == "all" else [args.crash_type]

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crash_types=crash_types,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
        strict=args.strict,
        synthetic_spawn=args.synthetic_spawn,
        variants_per_scene=args.variants_per_scene,
        spawn_seed=args.spawn_seed,
    )


if __name__ == "__main__":
    main()
