"""
Verify ego-frame coordinate conventions across spawn, env, and classifier.

Convention (canonical):
  delta = npc_world - ego_world
  rel_x =  cos(h)*dx + sin(h)*dy   (longitudinal in ego frame, + = NPC ahead of ego)
  rel_y =  sin(h)*dx - cos(h)*dy   (lateral in ego frame, + = NPC to ego's right)

Spawn placement (scene_filter._scene_with_synthetic_spawn):
  spawn = ego + rel_x * forward + rel_y * right
  forward = (cos h, sin h), right = (sin h, -cos h)
"""
from __future__ import annotations

import math
import sys


def ego_frame(npc_x: float, npc_y: float, ego_x: float, ego_y: float, ego_h: float) -> tuple[float, float]:
    dx = npc_x - ego_x
    dy = npc_y - ego_y
    c, s = math.cos(ego_h), math.sin(ego_h)
    rel_x = c * dx + s * dy
    rel_y = s * dx - c * dy
    return rel_x, rel_y


def spawn_from_ego(rel_x: float, rel_y: float, ego_x: float, ego_y: float, ego_h: float) -> tuple[float, float]:
    c, s = math.cos(ego_h), math.sin(ego_h)
    spawn_x = ego_x + c * rel_x + s * rel_y
    spawn_y = ego_y + s * rel_x - c * rel_y
    return spawn_x, spawn_y


def npc_frame_ego_pos(
    npc_x: float, npc_y: float, npc_h: float, ego_x: float, ego_y: float
) -> tuple[float, float]:
    """scene_filter._to_npc_frame: ego position in NPC body frame."""
    dx = ego_x - npc_x
    dy = ego_y - npc_y
    lx = math.cos(npc_h) * dx + math.sin(npc_h) * dy
    ly = -math.sin(npc_h) * dx + math.cos(npc_h) * dy
    return lx, ly


def classifier_lateral(ego_h: float, dx: float, dy: float) -> float:
    return math.sin(ego_h) * dx - math.cos(ego_h) * dy


def run_checks() -> None:
    failures: list[str] = []

    # 1) Round-trip spawn ↔ ego frame at several headings
    cases = [
        (10.0, 0.0, 0.0, 0.0, 0.0),
        (-15.0, -1.8, 5.0, 3.0, 0.0),
        (-15.0, 1.8, 5.0, 3.0, 0.0),
        (12.0, 0.5, -2.0, 7.0, 0.7),
        (-18.0, -2.0, 1.0, -4.0, -1.2),
    ]
    for rel_x, rel_y, ego_x, ego_y, ego_h in cases:
        sx, sy = spawn_from_ego(rel_x, rel_y, ego_x, ego_y, ego_h)
        rx, ry = ego_frame(sx, sy, ego_x, ego_y, ego_h)
        if not (math.isclose(rx, rel_x, abs_tol=1e-6) and math.isclose(ry, rel_y, abs_tol=1e-6)):
            failures.append(f"round-trip failed h={ego_h:.2f} in=({rel_x},{rel_y}) out=({rx},{ry})")

    # 2) SSL geometry: NPC left of ego → rel_y < 0
    ego_x, ego_y, ego_h = 0.0, 0.0, 0.0
    sx, sy = spawn_from_ego(-12.0, -1.8, ego_x, ego_y, ego_h)  # SSL same-side left
    rx, ry = ego_frame(sx, sy, ego_x, ego_y, ego_h)
    ly_cls = classifier_lateral(ego_h, sx - ego_x, sy - ego_y)
    if not (ry < 0 and ly_cls < 0):
        failures.append(f"SSL left spawn expected rel_y<0, got ry={ry}, ly_cls={ly_cls}")

    # 3) SSR geometry: NPC right of ego → rel_y > 0
    sx, sy = spawn_from_ego(-12.0, 1.8, ego_x, ego_y, ego_h)
    rx, ry = ego_frame(sx, sy, ego_x, ego_y, ego_h)
    ly_cls = classifier_lateral(ego_h, sx - ego_x, sy - ego_y)
    if not (ry > 0 and ly_cls > 0):
        failures.append(f"SSR right spawn expected rel_y>0, got ry={ry}, ly_cls={ly_cls}")

    # 4) RE geometry: NPC behind → rel_x < 0
    sx, sy = spawn_from_ego(-12.0, 0.0, ego_x, ego_y, ego_h)
    rx, ry = ego_frame(sx, sy, ego_x, ego_y, ego_h)
    if not rx < 0:
        failures.append(f"RE behind spawn expected rel_x<0, got rx={rx}")

    # 5) NPC-frame ly equals ego-frame rel_y at h=0 (scene_filter natural filter uses NPC frame)
    npc_x, npc_y, npc_h = 0.0, 0.0, 0.0
    ego_x, ego_y, ego_h = 20.0, 3.5, 0.0
    lx, ly = npc_frame_ego_pos(npc_x, npc_y, npc_h, ego_x, ego_y)
    rx, ry = ego_frame(npc_x, npc_y, ego_x, ego_y, ego_h)
    if not (1.5 <= ly <= 5.5 and 14.0 <= abs(lx) < 30.0):
        failures.append(f"SSL natural NPC-frame expected |lx|≈20 & ly in [1.5,5.5], got lx={lx}, ly={ly}")
    if not math.isclose(ry, ly, abs_tol=1e-6):
        failures.append(f"At h=0 expected rel_y==ly_npc_frame, got ry={ry}, ly={ly}")

    # 6) Ahead spawn: rel_x > 0
    sx, sy = spawn_from_ego(15.0, 0.0, ego_x, ego_y, ego_h)
    rx, ry = ego_frame(sx, sy, ego_x, ego_y, ego_h)
    if not rx > 0:
        failures.append(f"Ahead spawn expected rel_x>0, got rx={rx}")

    if failures:
        print("COORD CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("All coordinate checks passed.")
    print("  Canonical ego frame: rel_x=forward (+ahead), rel_y=lateral (+right)")
    print("  spawn_from_ego ↔ ego_frame round-trip: OK")
    print("  SSL/SSR/RE/ahead sign conventions: OK")
    print("  scene_filter NPC-frame SSL viability: OK")


if __name__ == "__main__":
    run_checks()
