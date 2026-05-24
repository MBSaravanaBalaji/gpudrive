"""Shared ego-frame coordinate helpers for spawn / env / validation."""
from __future__ import annotations

import math


def ego_frame(
    npc_x: float,
    npc_y: float,
    ego_x: float,
    ego_y: float,
    ego_h: float,
) -> tuple[float, float]:
    """NPC position in ego frame: rel_x +ahead, rel_y +right."""
    dx = npc_x - ego_x
    dy = npc_y - ego_y
    c, s = math.cos(ego_h), math.sin(ego_h)
    rel_x = c * dx + s * dy
    rel_y = s * dx - c * dy
    return rel_x, rel_y


def spawn_from_ego(
    rel_x: float,
    rel_y: float,
    ego_x: float,
    ego_y: float,
    ego_h: float,
) -> tuple[float, float]:
    """Place NPC from ego-frame offsets (matches scene_filter synthetic spawn)."""
    c, s = math.cos(ego_h), math.sin(ego_h)
    spawn_x = ego_x + c * rel_x + s * rel_y
    spawn_y = ego_y + s * rel_x - c * rel_y
    return spawn_x, spawn_y


def ideal_contact_ly(crash_type: str) -> float:
    if crash_type == "ssl":
        return -2.0
    if crash_type == "ssr":
        return 2.0
    return 0.0
