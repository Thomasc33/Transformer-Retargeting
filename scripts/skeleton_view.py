"""Shared 3/4-view skeleton transform utilities.

Provides hip-centering and 3/4-view rotation for all TMR visualization scripts.
Import and call these before projecting to XY in draw_skeleton / compute_limits.
"""

import numpy as np


def center_at_hip(joints_3d):
    """Center skeleton at hip joint (joint 0) and return a copy.

    Args:
        joints_3d: (V, 3) single frame of joint positions.

    Returns:
        Centered copy of joints_3d with joint 0 at the origin.
    """
    centered = joints_3d.copy()
    hip = centered[0].copy()
    centered -= hip
    return centered


def rotate_to_view(joints_3d, elev_deg=15, azim_deg=45):
    """Rotate 3D joints to a 3/4 viewing angle.

    Applies Y-axis rotation (azimuth) then X-axis rotation (elevation).
    Returns the full rotated 3D array (caller uses [:, 0] and [:, 1] for XY).

    Args:
        joints_3d: (V, 3) single frame of joint positions.
        elev_deg: Elevation angle in degrees (tilt up/down).
        azim_deg: Azimuth angle in degrees (rotate left/right).

    Returns:
        Rotated copy of joints_3d, same shape (V, 3).
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    # Y-axis rotation (azimuth)
    Ry = np.array([
        [np.cos(az), 0, np.sin(az)],
        [0, 1, 0],
        [-np.sin(az), 0, np.cos(az)],
    ])
    # X-axis rotation (elevation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)],
    ])
    rotated = (Rx @ Ry @ joints_3d.T).T
    return rotated


def transform_frame(joints_3d, center=True, rotate=True):
    """Apply hip-centering and 3/4-view rotation to a single frame.

    Args:
        joints_3d: (V, 3) single frame of joint positions.
        center: Whether to center at hip (joint 0).
        rotate: Whether to apply 3/4-view rotation.

    Returns:
        Transformed copy of joints_3d, same shape (V, 3).
    """
    joints = joints_3d.copy()
    if center:
        joints = center_at_hip(joints)
    if rotate:
        joints = rotate_to_view(joints)
    return joints
