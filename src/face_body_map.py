"""
Spatial face-to-body mapping.

A face belongs to a body if it sits in the head zone of that body's box:
horizontally inside it, vertically in the upper part, and with a plausible
size relative to the body width. Mapping is one-to-one (a body has one face,
a face belongs to one body); the best-scoring pairs win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


@dataclass
class FaceBodyLink:
    face_index: int
    track_id: int
    score: float


def face_body_score(face: Sequence[float], body: Sequence[float]) -> float:
    """0..1 plausibility that `face` is the face of `body`. 0 = impossible."""
    fx1, fy1, fx2, fy2 = face
    bx1, by1, bx2, by2 = body
    fw, fh = fx2 - fx1, fy2 - fy1
    bw, bh = bx2 - bx1, by2 - by1
    if fw <= 0 or fh <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    # containment: how much of the face lies inside the body box
    ix = max(0.0, min(fx2, bx2) - max(fx1, bx1))
    iy = max(0.0, min(fy2, by2) - max(fy1, by1))
    containment = (ix * iy) / (fw * fh)
    if containment < 0.8:
        return 0.0

    # vertical: face centre must be in the upper half of the body
    fcy = (fy1 + fy2) / 2.0
    if (fcy - by1) / bh > 0.5:
        return 0.0

    # horizontal centrality: 1 at the body's centre line, 0 at its edge
    fcx = (fx1 + fx2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    centrality = max(0.0, 1.0 - abs(fcx - bcx) / (bw / 2.0))

    # size plausibility: face width is ~0.25-0.5 of a body's width
    ratio = fw / bw
    if 0.2 <= ratio <= 0.55:
        size_ok = 1.0
    elif ratio < 0.2:
        size_ok = max(0.0, ratio / 0.2)
    else:
        size_ok = max(0.0, 1.0 - (ratio - 0.55) / 0.45)

    return float(containment * (0.6 * centrality + 0.4 * size_ok))


def map_faces_to_bodies(
    faces: Sequence[Sequence[float]],
    bodies: Dict[int, Sequence[float]],
    min_score: float = 0.25,
) -> Tuple[List[FaceBodyLink], List[int]]:
    """One-to-one greedy assignment.

    Returns (links, unmatched_face_indices).
    """
    pairs = []
    for fi, f in enumerate(faces):
        for tid, b in bodies.items():
            s = face_body_score(f, b)
            if s >= min_score:
                pairs.append((s, fi, tid))
    pairs.sort(reverse=True)

    links: List[FaceBodyLink] = []
    used_f, used_t = set(), set()
    for s, fi, tid in pairs:
        if fi in used_f or tid in used_t:
            continue
        used_f.add(fi)
        used_t.add(tid)
        links.append(FaceBodyLink(fi, tid, s))

    unmatched = [i for i in range(len(faces)) if i not in used_f]
    return links, unmatched
