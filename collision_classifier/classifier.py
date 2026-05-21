"""
Crash collision classifier for GPUDrive.

Ported from dqn_crasher's highway-env classifier. Uses the same SAT+MTV approach
with vertex/edge IDs to label collisions as rear-end, side-swipe-left, or side-swipe-right.

The only addition is compute_polygon() and compute_mtv(), which reconstruct the
geometry from GPUDrive's state tensors since GPUDrive doesn't expose MTV natively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ── Vertex / edge naming ──────────────────────────────────────────────────────
# Matches the dqn_crasher convention exactly.

VERTEX_NAMES = {
    0: "rear-left corner",
    1: "rear-right corner",
    2: "front-right corner",
    3: "front-left corner",
}

EDGE_NAMES = {
    0: "rear edge",
    1: "right edge",
    2: "front edge",
    3: "left edge",
}

EDGE_VERTICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),  # rear edge
    (1, 2),  # right edge
    (2, 3),  # front edge
    (3, 0),  # left edge
)


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CollisionClassification:
    contact_type: str   # "edge-edge", "vertex-edge", "vertex-vertex"
    collision_type: str  # "rear-end", "side-swipe-left", "side-swipe-right", "head-on", ...
    ego_feature: str
    npc_feature: str
    ego_vertices: Tuple[int, ...]
    npc_vertices: Tuple[int, ...]
    ego_edges: Tuple[int, ...]
    npc_edges: Tuple[int, ...]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def compute_polygon(
    pos_x: float,
    pos_y: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """
    Build the 4 OBB corners (world frame) from a vehicle's state.
    Matches RoadObject.polygon() from dqn_crasher (without closing the loop).

    Vertex order: rear-left(0), rear-right(1), front-right(2), front-left(3)
    """
    corners = np.array([
        [-length / 2, -width / 2],  # rear-left
        [-length / 2, +width / 2],  # rear-right
        [+length / 2, +width / 2],  # front-right
        [+length / 2, -width / 2],  # front-left
    ])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return (R @ corners.T).T + np.array([pos_x, pos_y])


def compute_mtv(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Separating Axis Test with Minimum Translation Vector.

    Returns the MTV that would push poly_a out of poly_b, or None if not intersecting.
    poly_a, poly_b: (4, 2) arrays of OBB corners.
    """
    min_overlap = float("inf")
    best_axis = None

    for poly in [poly_a, poly_b]:
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            # Normal to edge
            axis = np.array([-edge[1], edge[0]])
            length = np.linalg.norm(axis)
            if length < 1e-10:
                continue
            axis = axis / length

            proj_a = poly_a @ axis
            proj_b = poly_b @ axis

            overlap = min(proj_a.max(), proj_b.max()) - max(proj_a.min(), proj_b.min())
            if overlap <= 0:
                return None  # Separating axis found — no collision

            if overlap < min_overlap:
                min_overlap = overlap
                # Ensure axis points from poly_b centroid toward poly_a centroid
                d = poly_a.mean(axis=0) - poly_b.mean(axis=0)
                if np.dot(d, axis) < 0:
                    axis = -axis
                best_axis = axis

    if best_axis is None:
        return None

    return best_axis * min_overlap


def check_and_classify(
    ego_pos_x: float,
    ego_pos_y: float,
    ego_heading: float,
    ego_length: float,
    ego_width: float,
    npc_pos_x: float,
    npc_pos_y: float,
    npc_heading: float,
    npc_length: float,
    npc_width: float,
) -> Optional[CollisionClassification]:
    """
    Full pipeline: build polygons, run SAT+MTV, classify.
    Returns None if the two vehicles are not overlapping.
    """
    ego_poly = compute_polygon(ego_pos_x, ego_pos_y, ego_heading, ego_length, ego_width)
    npc_poly = compute_polygon(npc_pos_x, npc_pos_y, npc_heading, npc_length, npc_width)

    mtv = compute_mtv(ego_poly, npc_poly)
    if mtv is None:
        return None

    return classify_collision(ego_poly, npc_poly, mtv)


# ── Classifier (identical logic to dqn_crasher) ───────────────────────────────

def classify_collision(
    ego_polygon: np.ndarray,
    npc_polygon: np.ndarray,
    mtv: np.ndarray,
) -> CollisionClassification:
    ego_verts = _get_vertices(ego_polygon)
    npc_verts = _get_vertices(npc_polygon)

    overlap = np.linalg.norm(mtv)
    axis = mtv / overlap if overlap > 1e-10 else np.array([1.0, 0.0])

    ego_proj = ego_verts @ axis
    npc_proj = npc_verts @ axis

    ego_vertices, npc_vertices = _find_extremal_vertices(ego_proj, npc_proj)
    ego_edges = _find_edges(ego_vertices)
    npc_edges = _find_edges(npc_vertices)

    contact_type, collision_type, ego_feature, npc_feature = _classify(
        ego_vertices, npc_vertices, ego_edges, npc_edges
    )

    return CollisionClassification(
        contact_type=contact_type,
        collision_type=collision_type,
        ego_feature=ego_feature,
        npc_feature=npc_feature,
        ego_vertices=tuple(ego_vertices),
        npc_vertices=tuple(npc_vertices),
        ego_edges=tuple(ego_edges),
        npc_edges=tuple(npc_edges),
    )


def _get_vertices(polygon: np.ndarray) -> np.ndarray:
    verts = np.asarray(polygon, dtype=float)
    if verts.shape[0] > 4 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    if verts.shape[0] != 4:
        raise ValueError(f"Expected 4 vertices, got {verts.shape[0]}")
    return verts


def _find_extremal_vertices(
    ego_proj: np.ndarray,
    npc_proj: np.ndarray,
) -> Tuple[List[int], List[int]]:
    ego_min, ego_max = float(ego_proj.min()), float(ego_proj.max())
    npc_min, npc_max = float(npc_proj.min()), float(npc_proj.max())

    if ego_min < npc_min:
        ego_target, npc_target = ego_max, npc_min
    else:
        ego_target, npc_target = ego_min, npc_max

    ego_vertices = _vertices_at_value(ego_proj, ego_target, tolerance=0.001)
    npc_vertices = _vertices_at_value(npc_proj, npc_target, tolerance=0.001)
    return ego_vertices, npc_vertices


def _vertices_at_value(
    projections: np.ndarray, target: float, tolerance: float
) -> List[int]:
    distances = np.abs(projections - target)
    min_dist = float(distances.min())
    return [int(i) for i, d in enumerate(distances) if d <= min_dist + tolerance]


def _find_edges(vertex_ids: List[int]) -> List[int]:
    if len(vertex_ids) < 2:
        return []
    v_set = set(vertex_ids)
    return [
        edge_id
        for edge_id, (v1, v2) in enumerate(EDGE_VERTICES)
        if v1 in v_set and v2 in v_set
    ]


def _classify(
    ego_v: List[int],
    npc_v: List[int],
    ego_e: List[int],
    npc_e: List[int],
) -> Tuple[str, str, str, str]:
    # Edge-edge
    if ego_e and npc_e:
        e_edge, n_edge = ego_e[0], npc_e[0]
        if e_edge == 2 and n_edge == 0:
            coll_type = "rear-end"
        elif e_edge == 0 and n_edge == 2:
            coll_type = "rear-ended"
        elif e_edge == 2 and n_edge == 2:
            coll_type = "head-on"
        elif e_edge in (1, 3) and n_edge in (1, 3):
            # Determine SSL vs SSR from which side of ego
            coll_type = "side-swipe-left" if e_edge == 3 else "side-swipe-right"
        else:
            coll_type = "side-swipe"
        return "edge-edge", coll_type, EDGE_NAMES[e_edge], EDGE_NAMES[n_edge]

    # Vertex-edge: NPC vertex hits EGO edge
    if ego_e and npc_v:
        e_edge = ego_e[0]
        n_vertex = npc_v[0]
        if e_edge == 2:
            coll_type = "rear-end"
        elif e_edge == 0:
            coll_type = "rear-ended"
        elif e_edge == 3:
            coll_type = "side-swipe-left"
        else:
            coll_type = "side-swipe-right"
        return "vertex-edge", coll_type, EDGE_NAMES[e_edge], VERTEX_NAMES[n_vertex]

    # Vertex-edge: EGO vertex hits NPC edge
    if npc_e and ego_v:
        n_edge = npc_e[0]
        e_vertex = ego_v[0]
        if n_edge == 0:
            coll_type = "rear-end"
        elif n_edge == 2:
            coll_type = "rear-ended"
        elif n_edge == 3:
            coll_type = "side-swipe-left"
        else:
            coll_type = "side-swipe-right"
        return "vertex-edge", coll_type, VERTEX_NAMES[e_vertex], EDGE_NAMES[n_edge]

    # Vertex-vertex
    if ego_v and npc_v:
        e_v, n_v = ego_v[0], npc_v[0]
        e_front = e_v in (2, 3)
        n_front = n_v in (2, 3)
        e_left = e_v in (0, 3)
        n_left = n_v in (0, 3)
        if e_left != n_left:
            coll_type = "side-swipe-left" if e_left else "side-swipe-right"
        elif e_front and not n_front:
            coll_type = "rear-end"
        elif not e_front and n_front:
            coll_type = "rear-ended"
        elif e_front and n_front:
            coll_type = "head-on"
        else:
            coll_type = "angled"
        return "vertex-vertex", coll_type, VERTEX_NAMES[e_v], VERTEX_NAMES[n_v]

    return "complex", "angled", f"{len(ego_v)} verts", f"{len(npc_v)} verts"
