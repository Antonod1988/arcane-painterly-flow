"""Tiny numba ray tracer for oriented boxes (fork C, picture 20's base render).

Boxes: centre c (N,3), half extents h (N,3), rotation R (N,3,3) whose ROWS are the box's local axes in world
space, material id m (N,). trace() gives depth, world normal, box index and local hit point per pixel;
light() gives diffuse irradiance with ray-traced shadows, specular and ambient occlusion."""
import math
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
from numba import njit, prange


class Scene:
    def __init__(self):
        self.c, self.h, self.R, self.m, self.tint = [], [], [], [], []

    def box(self, lo, hi, mat, tint=(1, 1, 1), rot=None, center=None, half=None):
        """Axis-aligned box from corners, or an oriented one from center/half/rot (3x3 rows = local axes)."""
        if center is None:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            center, half = (lo + hi) / 2, np.abs(hi - lo) / 2
        self.c.append(center)
        self.h.append(half)
        self.R.append(np.eye(3) if rot is None else np.asarray(rot, float))
        self.m.append(mat)
        self.tint.append(tint)
        return len(self.m) - 1

    def arrays(self):
        return (np.array(self.c, np.float64), np.array(self.h, np.float64), np.array(self.R, np.float64),
                np.array(self.m, np.int32), np.array(self.tint, np.float64))


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])


@njit(cache=True, fastmath=True)
def _hit(ox, oy, oz, dx, dy, dz, c, h, R, i, tmax):
    """Ray vs oriented box i: returns (t, axis, sign) or t=-1."""
    # bounding-sphere early out
    rx, ry, rz = c[i, 0] - ox, c[i, 1] - oy, c[i, 2] - oz
    tca = rx * dx + ry * dy + rz * dz
    rad2 = h[i, 0] * h[i, 0] + h[i, 1] * h[i, 1] + h[i, 2] * h[i, 2]
    d2 = rx * rx + ry * ry + rz * rz - tca * tca
    if d2 > rad2:
        return -1.0, 0, 0.0
    if tca - math.sqrt(rad2 - d2) > tmax:
        return -1.0, 0, 0.0
    t0, t1 = -1e30, 1e30
    ax, sg = 0, 0.0
    for k in range(3):
        lo = -(R[i, k, 0] * rx + R[i, k, 1] * ry + R[i, k, 2] * rz)     # local origin (o - c) on axis k
        ld = R[i, k, 0] * dx + R[i, k, 1] * dy + R[i, k, 2] * dz
        if abs(ld) < 1e-12:
            if lo < -h[i, k] or lo > h[i, k]:
                return -1.0, 0, 0.0
            continue
        ta = (-h[i, k] - lo) / ld
        tb = (h[i, k] - lo) / ld
        s = -1.0
        if ta > tb:
            ta, tb = tb, ta
            s = 1.0
        if ta > t0:
            t0 = ta
            ax = k
            sg = s
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return -1.0, 0, 0.0
    if t1 < 1e-4:
        return -1.0, 0, 0.0
    if t0 < 1e-4:
        return -1.0, 0, 0.0          # starting inside: ignore (we never trace from inside geometry on purpose)
    if t0 > tmax:
        return -1.0, 0, 0.0
    return t0, ax, sg


@njit(cache=True, fastmath=True)
def _any(ox, oy, oz, dx, dy, dz, c, h, R, tmax, skip):
    for i in range(c.shape[0]):
        if i == skip:
            continue
        t, a, s = _hit(ox, oy, oz, dx, dy, dz, c, h, R, i, tmax)
        if t > 0:
            return True
    return False


@njit(parallel=True, cache=True, fastmath=True)
def trace(ro, dirs, c, h, R):
    Hh, Ww = dirs.shape[0], dirs.shape[1]
    T = np.full((Hh, Ww), -1.0)
    Nn = np.zeros((Hh, Ww, 3))
    ID = np.full((Hh, Ww), -1, np.int32)
    LP = np.zeros((Hh, Ww, 3))
    for y in prange(Hh):
        for x in range(Ww):
            dx, dy, dz = dirs[y, x, 0], dirs[y, x, 1], dirs[y, x, 2]
            best, bi, ba, bs = 1e30, -1, 0, 0.0
            for i in range(c.shape[0]):
                t, a, s = _hit(ro[0], ro[1], ro[2], dx, dy, dz, c, h, R, i, best)
                if t > 0 and t < best:
                    best, bi, ba, bs = t, i, a, s
            if bi >= 0:
                T[y, x] = best
                ID[y, x] = bi
                for k in range(3):
                    Nn[y, x, k] = R[bi, ba, k] * bs
                px = ro[0] + dx * best - c[bi, 0]
                py = ro[1] + dy * best - c[bi, 1]
                pz = ro[2] + dz * best - c[bi, 2]
                for k in range(3):
                    LP[y, x, k] = R[bi, k, 0] * px + R[bi, k, 1] * py + R[bi, k, 2] * pz
    return T, Nn, ID, LP


@njit(parallel=True, cache=True, fastmath=True)
def light(P, Nn, ID, VD, c, h, R, lpos, lcol, lrad, sun_dir, sun_col, ao_n, ao_len, seed, shininess, specw):
    """Diffuse irradiance (point lights + sun, shadowed), Blinn specular and AO for every hit pixel."""
    Hh, Ww = ID.shape
    E = np.zeros((Hh, Ww, 3))
    S = np.zeros((Hh, Ww, 3))
    AO = np.ones((Hh, Ww))
    for y in prange(Hh):
        st = np.uint64(seed + y * 7919)
        for x in range(Ww):
            i = ID[y, x]
            if i < 0:
                continue
            nx, ny, nz = Nn[y, x, 0], Nn[y, x, 1], Nn[y, x, 2]
            px = P[y, x, 0] + nx * 2e-3
            py = P[y, x, 1] + ny * 2e-3
            pz = P[y, x, 2] + nz * 2e-3
            vx, vy, vz = -VD[y, x, 0], -VD[y, x, 1], -VD[y, x, 2]
            sw = specw[i]
            for L in range(lpos.shape[0]):
                lx, ly, lz = lpos[L, 0] - px, lpos[L, 1] - py, lpos[L, 2] - pz
                d2 = lx * lx + ly * ly + lz * lz
                d = math.sqrt(d2)
                lx, ly, lz = lx / d, ly / d, lz / d
                ndl = nx * lx + ny * ly + nz * lz
                if ndl <= 0:
                    continue
                if _any(px, py, pz, lx, ly, lz, c, h, R, d - lrad[L], -1):
                    continue
                f = 1.0 / (d2 + 0.25)
                for k in range(3):
                    E[y, x, k] += lcol[L, k] * ndl * f
                if sw > 0:
                    hx, hy, hz = lx + vx, ly + vy, lz + vz
                    hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
                    ndh = (nx * hx + ny * hy + nz * hz) / hl
                    if ndh > 0:
                        sp = sw * ndh ** shininess * f * 4.0
                        for k in range(3):
                            S[y, x, k] += lcol[L, k] * sp
            ndl = nx * sun_dir[0] + ny * sun_dir[1] + nz * sun_dir[2]
            if ndl > 0:
                if not _any(px, py, pz, sun_dir[0], sun_dir[1], sun_dir[2], c, h, R, 200.0, -1):
                    for k in range(3):
                        E[y, x, k] += sun_col[k] * ndl
            # ambient occlusion: cosine-ish hemisphere samples
            occ = 0
            for s in range(ao_n):
                st = st * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                u1 = float((st >> np.uint64(33)) & np.uint64(0xFFFFFF)) / 16777216.0
                st = st * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                u2 = float((st >> np.uint64(33)) & np.uint64(0xFFFFFF)) / 16777216.0
                r = math.sqrt(u1)
                phi = 6.283185307 * u2
                a1, a2, a3 = r * math.cos(phi), r * math.sin(phi), math.sqrt(max(0.0, 1 - u1))
                # basis around n
                if abs(nx) > 0.5:
                    tx, ty, tz = 0.0, 1.0, 0.0
                else:
                    tx, ty, tz = 1.0, 0.0, 0.0
                bx, by, bz = ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx
                bl = math.sqrt(bx * bx + by * by + bz * bz)
                bx, by, bz = bx / bl, by / bl, bz / bl
                cx, cy, cz = ny * bz - nz * by, nz * bx - nx * bz, nx * by - ny * bx
                dx = a1 * bx + a2 * cx + a3 * nx
                dy = a1 * by + a2 * cy + a3 * ny
                dz = a1 * bz + a2 * cz + a3 * nz
                if _any(px, py, pz, dx, dy, dz, c, h, R, ao_len, -1):
                    occ += 1
            AO[y, x] = 1.0 - occ / max(1, ao_n)
    return E, S, AO
