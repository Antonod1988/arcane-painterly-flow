"""Arcane-like painterly finish for pictures 29-33, generalised from 20_painterly_alley.py (which the user liked).

A Cycles render is tone mapped, repainted with oriented brush strokes from coarse to fine, inked along
geometry edges and colour designed (teal shadows, warm light), so every picture of the set shares one hand:

    import arcane as A
    P = A.load(tag)                          # work/<tag>/*.npy -> dict: image (linear HDR), depth, normal, AOV masks
    img = A.tonemap(P["image"], exposure=0.85)
    with A.paint_slot():                     # at most two paintings at once (RAM is tight)
        img = A.paint(img, depth=P["depth"], detail=P.get("actor"), seed=29)
    img = A.ink(img, P["depth"], P["normal"], near=2, far=20, keep_out=P.get("actor"), outline=P.get("actor"))
    img = A.finish(img)
    hp.save(A.full(img), 29, "slug")         # or A.preview(img, tag) while testing

Blender side (scene script):
    B.passes(extra=("z", "normal")); B.aov_mask(mat, "actor")      # actor = anti-aliased mask of characters etc.
    B.render(tag, outputs=("Image", "Depth", "Normal", "actor"))

All pixel sizes (stroke radii, blurs) are given for 1440 px width and scale with the input, so a 50% test
render paints like a small version of the final.
"""
import glob
import math
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np
from numba import njit

import hp
from fc_util import fblur, soft_clip

try:
    import msvcrt

    def _try_lock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
except ImportError:                                   # Linux / macOS
    import fcntl

    def _try_lock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
LAYERS = ((26, 0.0), (14, 0.035), (7, 0.045), (3.5, 0.06))     # (stroke radius px, error threshold) as in 20


# ------------------------------------------------------------------ io
def load(tag):
    """Every pass bl_common.render dumped for this tag, as float32 arrays keyed by lower-case pass name."""
    out = {}
    for p in glob.glob(f"{ROOT}/work/{tag}/*.npy"):
        out[os.path.basename(p)[:-4]] = np.load(p).astype(np.float32)
    if "depth" in out:
        out["depth"] = np.minimum(out["depth"], 1e4)               # the world reads as 1e10
    return out


def scale_of(img):
    return img.shape[1] / 1440.0


def full(img):
    """Up to the delivery size (1440x2560) if the input was a test render."""
    if img.shape[1] == hp.W:
        return img
    return cv2.resize(img.astype(np.float32), (hp.W, hp.H), interpolation=cv2.INTER_CUBIC)


def preview(img, tag, name="arcane"):
    p = f"{ROOT}/work/{tag}/{name}.jpg"
    im = hp.to_pil(np.clip(img, 0, 1))
    im.resize((720, 1280)).save(p, quality=90)
    print("preview", p)
    return p


@contextmanager
def paint_slot(slots=2):
    """Cross-process limit on simultaneous paintings (each needs ~1.5 GB at full size)."""
    f = None
    waited = 0
    os.makedirs(f"{ROOT}/work", exist_ok=True)
    while f is None:
        for i in range(slots):
            g = open(f"{ROOT}/work/paint{i}.lock", "a+")
            try:
                _try_lock(g)
                f = g
                break
            except OSError:
                g.close()
        if f is None:
            if waited == 0:
                print("waiting for a paint slot ...", flush=True)
            time.sleep(2)
            waited += 2
    try:
        yield
    finally:
        _unlock(f)
        f.close()


# ------------------------------------------------------------------ tone
def filmic(x):
    x = np.maximum(x, 0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def tonemap(hdr, exposure=0.85, bloom=(0.5, 0.45, 0.35), sig=(6, 22, 70), knee=1.0):
    """Linear HDR -> display 0..1: bloom from everything above `knee`, then the filmic curve used in 20."""
    s = scale_of(hdr)
    hdr = hdr.astype(np.float32)
    hot = np.clip(hdr - knee, 0, None)
    out = hdr.copy()
    for sg, w in zip(sig, bloom):
        out += fblur(hot, sg * s) * w
    return (filmic(out * exposure) ** (1 / 2.2)).astype(np.float32)


# ------------------------------------------------------------------ strokes
@njit(cache=True, fastmath=True)
def _strokes(canvas, cx, cy, ang, ln, wd, colr, alpha, seed):
    Hh, Ww = canvas.shape[0], canvas.shape[1]
    st = np.uint64(seed)
    for k in range(cx.shape[0]):
        ca, sa = math.cos(ang[k]), math.sin(ang[k])
        hl, hw = ln[k] * 0.5, wd[k] * 0.5
        ext = hl + hw + 2
        x0, x1 = max(0, int(cx[k] - ext)), min(Ww - 1, int(cx[k] + ext))
        y0, y1 = max(0, int(cy[k] - ext)), min(Hh - 1, int(cy[k] + ext))
        st = st * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        ph = float((st >> np.uint64(33)) & np.uint64(0xFFFF)) / 65536.0 * 6.28
        freq = 2.4 / max(1.0, hw * 0.35)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx, dy = x + 0.5 - cx[k], y + 0.5 - cy[k]
                u = dx * ca + dy * sa
                v = -dx * sa + dy * ca
                uu = abs(u) - hl
                if uu < 0:
                    uu = 0.0
                d = math.sqrt(uu * uu + v * v)
                cov = (hw - d) / 1.1 + 0.5
                if cov <= 0:
                    continue
                if cov > 1:
                    cov = 1.0
                br = 0.72 + 0.28 * math.sin(v * freq + ph + 0.15 * u / max(1.0, hw))   # bristle streaks
                tail = 1.0
                if u > hl * 0.6:
                    tail = 1.0 - 0.6 * (u - hl * 0.6) / (hl * 0.4 + hw + 1e-6)          # dry-brush tail
                a = alpha * cov * br * tail
                if a <= 0:
                    continue
                for ch in range(3):
                    canvas[y, x, ch] = canvas[y, x, ch] * (1 - a) + colr[k, ch] * a


def _layer(canvas, ref_full, lum, rad, thr, rng, seed, alpha, farw=None, far_k=0.0, detail=None, avoid=False,
           only=None, short=False):
    H, W = lum.shape
    ref = hp.blur(ref_full, rad * 0.35) if rad > 4 else ref_full
    bl = hp.blur(lum, rad * 0.5)
    gx = cv2.Sobel(bl, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bl, cv2.CV_32F, 0, 1, ksize=3)
    Jxx, Jxy, Jyy = hp.blur(gx * gx, rad), hp.blur(gx * gy, rad), hp.blur(gy * gy, rad)
    ori = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + math.pi / 2
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)
    k = max(1, int(rad))
    err = cv2.blur(np.abs(canvas - ref).sum(axis=2) / 3, (k, k))
    sp = rad * 0.8
    gyy, gxx = np.mgrid[0:H:sp, 0:W:sp]
    cxs = (gxx + rng.uniform(0, sp, gxx.shape)).ravel()
    cys = (gyy + rng.uniform(0, sp, gyy.shape)).ravel()
    keep = (cxs < W - 1) & (cys < H - 1)
    cxs, cys = cxs[keep], cys[keep]
    ix, iy = cxs.astype(int), cys.astype(int)
    sel = np.ones(len(ix), bool)
    if thr > 0:
        t = thr * (1 + far_k * farw[iy, ix]) if farw is not None else thr
        sel &= err[iy, ix] > t
    if avoid and detail is not None:                       # coarse strokes leave the detailed areas alone
        sel &= detail[iy, ix] < rng.uniform(0.25, 0.75, len(ix))
    if only is not None:
        sel &= only[iy, ix] > 0.3
    cxs, cys, ix, iy = cxs[sel], cys[sel], ix[sel], iy[sel]
    order = rng.permutation(len(cxs))
    cxs, cys, ix, iy = cxs[order], cys[order], ix[order], iy[order]
    n = len(ix)
    c3 = np.clip(coh[iy, ix] * 3, 0, 1)
    a = ori[iy, ix] + rng.normal(0, 0.12, n) + (1 - c3) * rng.normal(0, 0.5, n)
    lns = rad * (rng.uniform(1.4, 2.8, n) if short else rng.uniform(2.2, 4.2, n)) * (0.6 + 0.8 * c3)
    wds = rad * rng.uniform(1.4, 2.0, n)
    cols = ref[iy, ix] * (1 + rng.normal(0, 0.035, (n, 1))) + rng.normal(0, 0.012, (n, 3))
    _strokes(canvas, cxs.astype(np.float64), cys.astype(np.float64), a.astype(np.float64), lns.astype(np.float64),
             wds.astype(np.float64), cols.astype(np.float64), alpha, seed)
    return n


def paint(img, depth=None, detail=None, seed=0, layers=LAYERS, alpha=0.88, far=(14.0, 80.0), far_k=1.6, fine=2.2,
          base_blur=10.0):
    """Repaint a display-referred image (H, W, 3, 0..1) with oriented brush strokes, coarse to fine.

    depth : optional (H, W) camera distance; far areas (far[0] -> far[1]) keep their bigger strokes, because the
            finer layers need `far_k` times more error there before they paint over them.
    detail: optional (H, W) 0..1 mask (characters, faces, carved pumpkins): the coarse layers avoid it and an extra
            fine layer (radius `fine`) finishes it, so the focal point stays crisp while the rest stays loose."""
    s = scale_of(img)
    rng = np.random.default_rng(seed)
    ref_full = np.ascontiguousarray(img, dtype=np.float32)
    lum = (ref_full @ np.array([0.3, 0.55, 0.15], np.float32)).astype(np.float32)
    canvas = hp.blur(ref_full, base_blur * s).astype(np.float32).copy()
    farw = hp.smoothstep(far[0], far[1], depth).astype(np.float32) if depth is not None else None
    det = None if detail is None else np.clip(detail, 0, 1).astype(np.float32)
    for li, (rad, thr) in enumerate(layers):
        n = _layer(canvas, ref_full, lum, rad * s, thr, rng, seed * 100 + li, alpha, farw, far_k, det,
                   avoid=rad >= 7)
        print(f"  layer r={rad}: {n} strokes", flush=True)
    if det is not None:
        grown = cv2.dilate(det, np.ones((5, 5), np.uint8))
        n = _layer(canvas, ref_full, lum, fine * s, 0.03, rng, seed * 100 + 9, alpha, only=grown, short=True)
        print(f"  detail layer r={fine}: {n} strokes", flush=True)
    return canvas


# ------------------------------------------------------------------ ink
def ink(img, depth, normal=None, near=1.0, far=16.0, keep_out=None, outline=None, color="#14161c", amount=0.75,
        dthr=0.12, nthr=1.2, seed=50, broken=(0.3, 0.55)):
    """Selective dark linework from depth and normal discontinuities: near and mid ground only, broken up by
    noise so it reads hand-made. keep_out: masks the ink inside (characters get only their `outline`)."""
    s = scale_of(img)
    H, W = depth.shape
    logd = np.log(depth + 0.5).astype(np.float32)
    # the Depth pass is sample-averaged, so depth of field and glossy speckle make it noisy: median it first
    logd = hp.blur(cv2.medianBlur(logd, 5), 0.7 * s)
    gxd = cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)
    gyd = cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)
    edge_d = np.hypot(gxd, gyd)
    lines = np.clip((edge_d - dthr) * 4, 0, 1)
    if normal is not None:
        # the Normal pass carries bump mapping (water ripples, stone grain): blur it first so only real creases count
        nf = np.dstack([cv2.medianBlur(np.ascontiguousarray(normal[..., k], np.float32), 5) for k in range(3)])
        nf = hp.blur(nf, max(1.0, 1.5 * s))
        edge_n = sum(np.abs(cv2.Sobel(nf[..., k], cv2.CV_32F, 1, 0, ksize=3))
                     + np.abs(cv2.Sobel(nf[..., k], cv2.CV_32F, 0, 1, ksize=3)) for k in range(3))
        lines = lines + np.clip((edge_n - nthr) * 0.8, 0, 1) * 0.7
    lines *= hp.smoothstep(far, near, depth)
    lines *= hp.smoothstep(broken[0], broken[1], hp.noise((H, W), 60 * s, seed=seed, octaves=2))
    lines = np.clip(hp.blur(lines.astype(np.float32), 0.8 * s) * 1.3, 0, 1)
    if keep_out is not None:
        lines *= 1 - np.clip(fblur(keep_out, 1.5 * s) * 1.5, 0, 1)
    if outline is not None:
        o = np.clip(outline, 0, 1).astype(np.float32)
        e = max(3, int(round(3 * s)) | 1)
        lines = np.clip(lines + hp.blur(np.clip(o - cv2.erode(o, np.ones((e, e), np.uint8)), 0, 1), 0.7 * s) * 0.9, 0, 1)
    return hp.mix(img, hp.rgb(color), (lines * amount)[..., None]).astype(np.float32)


# ------------------------------------------------------------------ effects painted on top
def motes(img, pts, color="#ffb060", size=2.0, glow=1.0, streak=(0.0, 0.0), seed=0):
    """Glowing specks (embers, dust in a light shaft, fireflies, spray): pts = [(x, y, brightness)] in pixels.
    streak = (dx, dy) motion smear in px."""
    s = scale_of(img)
    H, W = img.shape[:2]
    core = np.zeros((H, W), np.float32)
    for (x, y, b) in pts:
        x2, y2 = x + streak[0] * s, y + streak[1] * s
        cv2.line(core, (int(x * 16), int(y * 16)), (int(x2 * 16), int(y2 * 16)), float(b),
                 max(1, int(round(size * s))), cv2.LINE_AA, shift=4)
    col = hp.rgb(color)
    add = (core * 1.2)[..., None] * (col * 0.5 + 0.5) + (fblur(core, 3 * s) * 2.0 + fblur(core, 10 * s) * 1.4)[..., None] * col
    return (img + add * glow).astype(np.float32)


def finish(img, teal=0.5, warm=0.08, sat=1.18, vig=0.38, knee=0.86, canvas=0.035, seed=60,
           shadow_tint=(0.5, 1.0, 1.18), shadow_lift=(0.0, 0.03, 0.045)):
    """Colour design of 20: teal-shifted shadows, warm highlights, extra saturation, canvas weave, vignette and a
    soft shoulder so hot lights whiten instead of clipping."""
    s = scale_of(img)
    H, W = img.shape[:2]
    lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
    sh_w = hp.smoothstep(0.62, 0.08, lum)[..., None]
    hi_w = hp.smoothstep(0.35, 0.85, lum)[..., None]
    img = img * (1 - teal * sh_w) + sh_w * teal * img * np.array(shadow_tint, np.float32) \
        + sh_w * np.array(shadow_lift, np.float32) * (teal / 0.5)
    img = img * (1 + hi_w * np.array([warm, -0.01, -warm], np.float32))
    mean = img.mean(axis=2, keepdims=True)
    img = mean + (img - mean) * sat
    X, Y = hp.grid(W, H)
    canv = hp.noise((H, W), 2.2 * s, seed=seed) - 0.5
    weave = (np.sin(X * 1.9 / s) * np.sin(Y * 1.9 / s)) * 0.5
    img = img * (1 + (canvas * canv + 0.012 * weave)[..., None])
    img = hp.vignette(img, vig, 2.2)
    img = soft_clip(img, knee)
    return np.clip(img, 0, 1).astype(np.float32)
