"""Render NTU skeleton frames from demo_data.json as inline SVG.

Uses the same hip-centering + 3/4-view rotation as scripts/skeleton_view.py so the
poster figures match the paper figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO_JSON = ROOT / "demo_data.json"

# Body-part colouring, matching scripts/visualize_retargeting.py
PART_COLORS = {
    "torso": "#4a5568",
    "head": "#9467bd",
    "l_arm": "#1f77b4",
    "r_arm": "#ff7f0e",
    "l_leg": "#2ca02c",
    "r_leg": "#d62728",
}

L_ARM = {4, 5, 6, 7, 21, 22}
R_ARM = {8, 9, 10, 11, 23, 24}
L_LEG = {12, 13, 14, 15}
R_LEG = {16, 17, 18, 19}
HEAD = {2, 3}


def bone_part(i: int, j: int) -> str:
    joints = {i, j}
    if joints & L_ARM:
        return "l_arm"
    if joints & R_ARM:
        return "r_arm"
    if joints & L_LEG:
        return "l_leg"
    if joints & R_LEG:
        return "r_leg"
    if joints & HEAD:
        return "head"
    return "torso"


def center_at_hip(joints: np.ndarray) -> np.ndarray:
    return joints - joints[0]


def rotate_to_view(joints: np.ndarray, elev_deg: float = 8.0,
                   azim_deg: float = 0.0) -> np.ndarray:
    # NTU cross-view cameras already sit at +/-45 deg, so we project near-frontal
    # and let the capture angle supply the 3/4 look. Adding azimuth on top of that
    # compounds the two rotations and makes the figures unreadable.
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    ry = np.array([[np.cos(az), 0, np.sin(az)],
                   [0, 1, 0],
                   [-np.sin(az), 0, np.cos(az)]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    return (rx @ ry @ joints.T).T


def transform(joints: np.ndarray) -> np.ndarray:
    return rotate_to_view(center_at_hip(joints))


def load_examples() -> dict:
    with open(DEMO_JSON) as f:
        return json.load(f)


def sequence_bounds(seqs: list[np.ndarray], pad: float = 1.10) -> tuple[float, float, float]:
    """Shared 2D bounds across several (T, V, 3) sequences after transform."""
    xs, ys = [], []
    for seq in seqs:
        for t in range(seq.shape[0]):
            p = transform(seq[t])
            xs.append(p[:, 0])
            ys.append(p[:, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    cx = (xs.max() + xs.min()) / 2
    cy = (ys.max() + ys.min()) / 2
    half = max(np.abs(xs - cx).max(), np.abs(ys - cy).max()) * pad
    return float(cx), float(cy), float(half)


def frame_svg(joints: np.ndarray, bones: list, cx: float, cy: float, half: float,
              size: float, color: str | None = None, stroke: float = 2.6,
              dot: float = 2.2, opacity: float = 1.0,
              part_colors: bool = False) -> str:
    """Render one frame as an SVG <g>, mapped into a size x size box."""
    p = transform(joints)

    def sx(v):  # world x -> svg x
        return (v - (cx - half)) / (2 * half) * size

    def sy(v):  # world y -> svg y (flip: NTU +y is up)
        return size - (v - (cy - half)) / (2 * half) * size

    out = []
    for (i, j) in bones:
        c = PART_COLORS[bone_part(i, j)] if part_colors else color
        out.append(
            f'<line x1="{sx(p[i,0]):.2f}" y1="{sy(p[i,1]):.2f}" '
            f'x2="{sx(p[j,0]):.2f}" y2="{sy(p[j,1]):.2f}" '
            f'stroke="{c}" stroke-width="{stroke}" stroke-linecap="round" '
            f'opacity="{opacity}"/>'
        )
    joint_c = "#101820" if part_colors else color
    for v in range(p.shape[0]):
        out.append(
            f'<circle cx="{sx(p[v,0]):.2f}" cy="{sy(p[v,1]):.2f}" r="{dot}" '
            f'fill="{joint_c}" opacity="{opacity}"/>'
        )
    return "".join(out)


def filmstrip(seq: np.ndarray, bones: list, frames: list[int], cx: float, cy: float,
              half: float, cell: float, color: str, part_colors: bool = False,
              ghost: np.ndarray | None = None) -> str:
    """A horizontal row of frames. Optional ghost sequence drawn underneath."""
    parts = []
    for k, fi in enumerate(frames):
        t = min(fi, seq.shape[0] - 1)
        parts.append(f'<g transform="translate({k * cell:.2f},0)">')
        if ghost is not None:
            gt = min(fi, ghost.shape[0] - 1)
            parts.append(frame_svg(ghost[gt], bones, cx, cy, half, cell,
                                   color="#C3CBC6", stroke=1.8, dot=1.3,
                                   opacity=0.34))
        parts.append(frame_svg(seq[t], bones, cx, cy, half, cell, color=color,
                               part_colors=part_colors))
        parts.append("</g>")
    return "".join(parts)
