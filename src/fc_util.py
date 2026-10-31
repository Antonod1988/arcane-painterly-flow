"""Fork C helpers (pictures 17-20): fast blurs, anti-aliased line/shape masks, neon glow, tone curves."""
import math

import cv2
import numpy as np

import hp

W, H = hp.W, hp.H


def fblur(a, sigma):
    """Gaussian blur that downsamples first for big sigmas (glows), so it stays fast at 1440x2560."""
    if sigma <= 0:
        return a
    if sigma < 10:
        return hp.blur(a, sigma)
    f = int(sigma // 5)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = hp.blur(small, sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def soft_clip(x, knee=0.8):
    """Identity below knee, smooth shoulder to 1 above it (per channel, so hot colours whiten)."""
    over = np.maximum(x - knee, 0.0)
    span = 1.0 - knee
    return np.where(x < knee, x, knee + span * (1.0 - np.exp(-over / span)))


class Mask:
    """An 8-bit drawing surface for anti-aliased lines, polygons and circles at full resolution."""

    def __init__(self, w=W, h=H):
        self.m = np.zeros((h, w), np.uint8)

    def line(self, pts, thick=3, closed=False):
        p = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.m, [p], closed, 255, max(1, int(round(thick))), cv2.LINE_AA, shift=4)
        return self

    def poly(self, pts):
        p = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(self.m, [p], 255, cv2.LINE_AA, shift=4)
        return self

    def circle(self, c, r, thick=-1):
        cv2.circle(self.m, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(r * 16)), 255,
                   thick if thick < 0 else max(1, int(thick)), cv2.LINE_AA, shift=4)
        return self

    def ellipse(self, c, axes, angle=0, thick=-1):
        cv2.ellipse(self.m, (int(round(c[0] * 16)), int(round(c[1] * 16))),
                    (int(round(axes[0] * 16)), int(round(axes[1] * 16))), angle, 0, 360, 255,
                    thick if thick < 0 else max(1, int(thick)), cv2.LINE_AA, shift=4)
        return self

    @property
    def f(self):
        return self.m.astype(np.float32) / 255.0


def neon(core, color, gain=1.0, whiten=0.65, sig=(1.5, 6, 18, 48, 110), wts=(1.1, 0.9, 0.7, 0.45, 0.3), hot_gain=1.6):
    """Neon tube light from a core mask: white-hot centre plus layered coloured glow (additive, HDR)."""
    color = np.asarray(color, np.float32)
    hot = color * (1 - whiten) + whiten
    out = core[..., None] * hot * hot_gain
    for s, w_ in zip(sig, wts):
        out = out + fblur(core, s)[..., None] * color * w_
    return out * gain


def bat_poly(cx, cy, span, angle=0.0, flap=0.0):
    """Bat silhouette polygon (wingspan `span` px). flap in [-1, 1] raises/lowers the wing tips."""
    half = [(0.0, -0.13), (0.035, -0.235), (0.06, -0.12), (0.1, -0.07), (0.32, -0.2 - 0.15 * flap),
            (0.62, -0.3 - 0.25 * flap), (1.0, -0.36 - 0.3 * flap), (0.86, -0.1 - 0.15 * flap),
            (0.74, 0.02 - 0.08 * flap), (0.6, -0.02 - 0.06 * flap), (0.47, 0.1 - 0.03 * flap),
            (0.33, 0.04), (0.2, 0.15), (0.09, 0.09), (0.05, 0.2), (0.0, 0.25)]
    right = half
    left = [(-x, y) for x, y in reversed(half)]
    pts = left + right
    ca, sa = math.cos(angle), math.sin(angle)
    s = span / 2.0
    return [(cx + s * (x * ca - y * sa), cy + s * (x * sa + y * ca)) for x, y in pts]


def tree_segments(x, y, ang, length, depth, rng, spread=0.55, decay=0.72, curl=0.25):
    """Gnarled dead-tree branches as [(polyline, thickness_level)], growing from (x, y) at angle ang (rad)."""
    segs = []

    def grow(x, y, ang, length, d):
        pts = [(x, y)]
        n = 5
        a = ang
        for i in range(n):
            a += rng.normal(0, curl * 0.35)
            x += math.cos(a) * length / n
            y += math.sin(a) * length / n
            pts.append((x, y))
        segs.append((pts, d))
        if d <= 0:
            return
        k = 2 if rng.random() < 0.7 else 3
        for j in range(k):
            off = (j - (k - 1) / 2) * spread * 2 / max(1, k - 1) + rng.normal(0, 0.18)
            grow(x, y, a + off, length * (decay + rng.normal(0, 0.06)), d - 1)

    grow(x, y, ang, length, depth)
    return segs


def tapered(pts, w0, w1):
    """Polygon around a polyline whose width goes from w0 to w1."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    left, right = [], []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        d /= np.linalg.norm(d) + 1e-9
        nrm = np.array([-d[1], d[0]])
        w = (w0 + (w1 - w0) * i / max(1, n - 1)) / 2
        left.append(pts[i] + nrm * w)
        right.append(pts[i] - nrm * w)
    return [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]


def catmull(pts, n=12):
    """Catmull-Rom spline through the points."""
    p = np.asarray(pts, np.float64)
    p = np.vstack([p[0] * 2 - p[1], p, p[-1] * 2 - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p[i]) + (-p[i - 1] + p[i + 1]) * t
                              + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t2
                              + (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t3))
    out.append(p[-2])
    return np.array(out)


def wobble(pts, amp=2.0, step=8.0, seed=0, closed=False):
    """Resample a polyline every `step` px and add a smooth hand-drawn sideways wobble."""
    p = np.asarray(pts, np.float64)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step))
    t = np.linspace(0, s[-1], n)
    x = np.interp(t, s, p[:, 0])
    y = np.interp(t, s, p[:, 1])
    r = np.random.default_rng(seed)
    k = max(4, n // 6)
    ctrl = r.normal(0, 1, k)
    off = np.interp(np.linspace(0, k - 1, n), np.arange(k), ctrl) * amp
    dx, dy = np.gradient(x), np.gradient(y)
    ln = np.hypot(dx, dy) + 1e-9
    return np.stack([x - dy / ln * off, y + dx / ln * off], 1)
