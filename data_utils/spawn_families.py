"""
Spawn family definitions for synthetic crash-NPC datasets.

All positions are in the replay ego's local frame at t=0:
  rel_x > 0  → spawn ahead of ego
  rel_x < 0  → spawn behind ego
  rel_y > 0  → spawn to ego's right
  rel_y < 0  → spawn to ego's left

Curriculum order (easiest → hardest): behind_near, behind_far, side_same,
side_opposite, ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

SPAWN_FAMILY_NAMES: Tuple[str, ...] = (
    "behind_near",
    "behind_far",
    "side_same",
    "side_opposite",
    "ahead",
)

# Suggested MIN_SUCCESS_STEP gates per family (used by ppo_env when spawn_cond on).
MIN_SUCCESS_STEP_BY_FAMILY: Dict[str, int] = {
    "behind_near": 25,
    "behind_far": 28,
    "side_same": 22,
    "side_opposite": 30,
    "ahead": 32,
    "unknown": 20,
}


@dataclass(frozen=True)
class SpawnEnvelope:
    rel_x: Tuple[float, float]
    rel_y: Tuple[float, float]
    speed_delta: Tuple[float, float] = (1.2, 2.6)


# Shared longitudinal bands (tightened for 91-step ego log reachability)
_BEHIND_NEAR = SpawnEnvelope((-14.0, -10.0), (0.0, 0.0), (1.2, 2.8))
_BEHIND_FAR = SpawnEnvelope((-22.0, -14.0), (0.0, 0.0), (2.0, 3.5))
_SIDE_X = SpawnEnvelope((-6.0, 6.0), (0.0, 0.0), (1.2, 3.0))
_AHEAD = SpawnEnvelope((5.0, 12.0), (0.0, 0.0), (0.5, 2.0))

# Lateral bands per crash type (ego frame)
_SSL_SAME_Y = (-2.2, -1.6)       # NPC left of ego
_SSL_OPPOSITE_Y = (1.6, 2.2)     # NPC right of ego — must lane-change for SSL
_SSR_SAME_Y = (1.6, 2.2)
_SSR_OPPOSITE_Y = (-1.8, -1.2)
_RE_CENTER_Y = (-0.8, 0.8)
_RE_AHEAD_Y = (-0.6, 0.6)
_RE_AHEAD_SPEED = (-1.0, 0.5)    # slower or slightly faster when ahead


def spawn_envelopes(crash_type: str) -> Dict[str, SpawnEnvelope]:
    """Return spawn envelopes keyed by family name for a crash type."""
    if crash_type == "ssl":
        same_y, opp_y = _SSL_SAME_Y, _SSL_OPPOSITE_Y
    elif crash_type == "ssr":
        same_y, opp_y = _SSR_SAME_Y, _SSR_OPPOSITE_Y
    elif crash_type == "re":
        return {
            "behind_near": SpawnEnvelope(_BEHIND_NEAR.rel_x, _RE_CENTER_Y, _BEHIND_NEAR.speed_delta),
            "behind_far": SpawnEnvelope(_BEHIND_FAR.rel_x, _RE_CENTER_Y, _BEHIND_FAR.speed_delta),
            "side_same": SpawnEnvelope(_SIDE_X.rel_x, _RE_CENTER_Y, _SIDE_X.speed_delta),
            "side_opposite": SpawnEnvelope(_SIDE_X.rel_x, _RE_CENTER_Y, _SIDE_X.speed_delta),
            "ahead": SpawnEnvelope(_AHEAD.rel_x, _RE_AHEAD_Y, _RE_AHEAD_SPEED),
        }
    else:
        raise ValueError(f"unknown crash_type: {crash_type}")

    return {
        "behind_near": SpawnEnvelope(_BEHIND_NEAR.rel_x, same_y, _BEHIND_NEAR.speed_delta),
        "behind_far": SpawnEnvelope(_BEHIND_FAR.rel_x, same_y, _BEHIND_FAR.speed_delta),
        "side_same": SpawnEnvelope(_SIDE_X.rel_x, same_y, _SIDE_X.speed_delta),
        "side_opposite": SpawnEnvelope(_SIDE_X.rel_x, opp_y, (1.5, 3.0)),
        "ahead": SpawnEnvelope(_AHEAD.rel_x, same_y, (0.0, 1.5)),
    }


def resolve_spawn_families(
    crash_type: str,
    spawn_family_arg: str,
) -> List[str]:
    """Parse --spawn_family value into an ordered list of family names."""
    available = set(spawn_envelopes(crash_type).keys())
    if spawn_family_arg in ("default", "behind_near"):
        return ["behind_near"]
    if spawn_family_arg == "all":
        return [f for f in SPAWN_FAMILY_NAMES if f in available]
    names = [s.strip() for s in spawn_family_arg.split(",") if s.strip()]
    bad = [n for n in names if n not in available]
    if bad:
        raise ValueError(
            f"Unknown spawn families for {crash_type}: {bad}. "
            f"Available: {sorted(available)}"
        )
    return names


def infer_spawn_family_from_path(path: str) -> str:
    """Parse spawn family from synthetic scene filename, else unknown."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    for fam in SPAWN_FAMILY_NAMES:
        if f"_{fam}_" in name:
            return fam
    if "_syn" in name:
        return "behind_near"
    return "unknown"
