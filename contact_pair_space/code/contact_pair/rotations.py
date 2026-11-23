"""Rotations used by the map: unit vectors, skew matrices, rotations about an axis, and the smallest rotation taking one unit vector
onto another (Rodrigues). All functions are vectorised over a leading axis of length n."""
import numpy as np

EZ = np.array([0.0, 0.0, 1.0])


def unit(v, eps=1e-300):
    """v / |v| along the last axis"""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def skew(v):
    """[v]x with [v]x w = v x w, for v (..., 3)"""
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1), np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rot_axis(axis, ang):
    """rotation by ang (n,) about the unit axis (n, 3)"""
    K = skew(axis); s = np.sin(ang)[:, None, None]; c = (1.0 - np.cos(ang))[:, None, None]
    return np.eye(3) + s * K + c * (K @ K)


def minrot(a, b):
    """smallest rotation taking unit a onto unit b, both (n, 3): R0(a, b) = I + [v]x + [v]x^2 / (1 + c), v = a x b, c = a . b;
    for a = -b a half turn about a fixed axis orthogonal to a"""
    v = np.cross(a, b); c = (a * b).sum(1); n = len(a); R = np.empty((n, 3, 3)); reg = c > -1 + 1e-12
    K = skew(v[reg]); R[reg] = np.eye(3) + K + K @ K / (1.0 + c[reg])[:, None, None]
    if (~reg).any():
        a2 = a[~reg]; e = np.where(np.abs(a2[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0.0, 1.0, 0]])); w = unit(np.cross(a2, e)); R[~reg] = 2.0 * w[:, :, None] * w[:, None, :] - np.eye(3)
    return R


def section(u):
    """S(u): the smallest rotation with S(u) u = -e_z (gravity u turned to point straight down)"""
    return minrot(u, np.broadcast_to(-EZ, u.shape).copy())
