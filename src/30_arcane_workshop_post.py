"""30 · post: the Arcane painterly finish for the pumpkin carver's workshop (blender/30_arcane_workshop.py).

    python src/30_arcane_workshop_post.py <tag> [final] [ev=1.0] [haze=1.0] [dust=1.0] [embers=1.0] [ink=0.7]

Haze over the town outside the window (from depth), tone map, oriented brush strokes (fine on the pumpkins, the knife
and the raven, loose on the far town), ink along the near geometry, dust specks drifting in the moonbeams and warm
motes over the lanterns, then the shared teal/orange colour design."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import arcane as A  # noqa: E402
import hp  # noqa: E402
from fc_util import fblur  # noqa: E402

tag = sys.argv[1]
FINAL = "final" in sys.argv[2:]
kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)


def f(k, d):
    return float(kv.get(k, d))


P = A.load(tag)
hdr = P["image"]
depth = P["depth"]
nrm = P["normal"]
act = np.clip(P["actor"], 0, 1) if "actor" in P else None
H, W = depth.shape
s = A.scale_of(hdr)
rng = np.random.default_rng(30)
lum_l = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
print("linear luminance p50/p90/p99/max:", *np.round(np.percentile(lum_l, [50, 90, 99, 100]), 3))

# ---------------------------------------------------------------- moonbeams: keep the view through the window clear
win = hp.smoothstep(4.52, 4.75, depth).astype(np.float32)                 # seen through the window (planar depth)
if act is not None:
    win = np.maximum(win, act * hp.smoothstep(4.3, 4.42, depth))           # the raven stays a dark silhouette
win = fblur(win, 3 * s)
if "volumedir" in P:
    vol = P["volumedir"]                     # noisy (not denoised): median, then a wide blur; the beams are soft
    vol = np.dstack([cv2.medianBlur(np.ascontiguousarray(vol[..., c]), 5) for c in range(3)])
    vol = fblur(vol, 9 * s)
    k = f("beams", 1.5) * (1 - win) + f("winveil", 0.15) * win
    hdr = hdr + vol * (k - 1)[..., None]

# ---------------------------------------------------------------- aerial perspective over the town outside
out = hp.smoothstep(6.0, 9.0, depth) * (depth < 5000)                   # the town beyond the window, not the sky
haze_t = (1 - np.exp(-np.maximum(depth - 8.0, 0) / 80.0)) * out * f("haze", 0.35)
ys = (np.arange(H, dtype=np.float32) / H)[:, None]
haze_col = hp.mix(np.array([0.10, 0.20, 0.26], np.float32), np.array([0.17, 0.17, 0.27], np.float32),
                  hp.smoothstep(0.25, 0.5, ys)[..., None] * np.ones((1, W, 1), np.float32))
hdr = hdr * (1 - haze_t[..., None] * 0.85) + haze_col * (haze_t[..., None] * 0.85)



def tm(hdr, ev, bloom=(0.34, 0.28, 0.2), sig=(6, 22, 70), knee=1.0, white=(2.0, 30.0, 0.8)):
    """Bloom, then the filmic curve of 20 applied to the brightest channel only (keeps the oranges orange
    instead of sliding to yellow), with hot lights whitening toward their own brightness."""
    x = hdr * ev
    hot = np.clip(x - knee, 0, None)
    for sg, w in zip(sig, bloom):
        x = x + fblur(hot, sg * s) * w
    m = x.max(axis=2)
    mc = A.filmic(m)
    out = x * (mc / np.maximum(m, 1e-6))[..., None]
    wt = (hp.smoothstep(white[0], white[1], m) * white[2])[..., None]
    out = out * (1 - wt) + mc[..., None] * wt
    return (np.clip(out, 0, 1) ** (1 / 2.2)).astype(np.float32)


sky = depth > 1000
hdr = np.where(sky[..., None], np.minimum(hdr, f("moonclamp", 1.7)), hdr)    # a big moon, not a blown-out flare
img = tm(hdr, f("ev", 2.9))

# ---------------------------------------------------------------- painted god rays fanning from the window into the dust
X, Y = hp.grid(W, H)
SX, SY = 0.625 * W, 0.36 * H                                           # just below the moon, inside the window
dx, dy = X - SX, Y - SY
r = np.hypot(dx, dy) / H
ang = np.arctan2(dy, dx)
NB = 260
bins = rng.uniform(0, 1, NB).astype(np.float32)
bins = np.convolve(np.concatenate([bins[-4:], bins, bins[:4]]), np.ones(3) / 3, mode="same")[4:-4]
pos = (ang + np.pi) / (2 * np.pi) * NB
i0 = np.floor(pos).astype(int) % NB
fr = hp.smoothstep(0.0, 1.0, (pos - np.floor(pos)).astype(np.float32))
streak = bins[i0] * (1 - fr) + bins[(i0 + 1) % NB] * fr
rays = hp.smoothstep(0.42, 0.75, streak) * hp.smoothstep(0.07, 0.2, r) * np.exp(-r / 0.32)
rays *= hp.smoothstep(-0.35, 0.25, np.sin(ang))                          # fan downward and sideways, not up
rays *= 1 - win                                                           # in the room's air, not over the sky
warmth = np.clip(img[..., 0] - img[..., 2], 0, 1)
rays *= 1 - np.clip(warmth * 1.6, 0, 0.85)                                # the lantern-lit side stays warm
rays = fblur(rays.astype(np.float32), 4 * s)
img = np.clip(img + rays[..., None] * hp.rgb("#8fd2e0") * f("rays", 0.5), 0, 1)
pre = img.copy()
with A.paint_slot():
    img = A.paint(img, depth=depth, detail=act, seed=30, far=(6.0, 40.0), far_k=1.6)
img = A.ink(img, depth, nrm, near=0.4, far=5.0, keep_out=act, outline=None, amount=f("ink", 0.7))
if act is not None:                       # the raven: a dark silhouette against the sky with a cold moon rim
    rv = np.clip(act * hp.smoothstep(4.3, 4.42, depth), 0, 1).astype(np.float32)
    img = hp.mix(img, hp.rgb("#0e1119"), (hp.blur(rv, 0.6 * s) * f("raven", 0.8))[..., None])
    sh = max(1, int(round(2 * s)))
    rim = np.clip(rv - np.roll(rv, -sh, axis=1), 0, 1) * hp.smoothstep(0.5, 0.0, np.roll(rv, -sh, axis=1))
    img = img + (hp.blur(rim, 0.7 * s) * 0.55)[..., None] * hp.rgb("#cfe6f0")
if act is not None:                       # a broken, soft outline around the pumpkins, the knife and the raven
    e = max(3, int(round(3 * s)) | 1)
    edge = np.clip(act - cv2.erode(act, np.ones((e, e), np.uint8)), 0, 1)
    brk = hp.smoothstep(0.35, 0.6, hp.noise((H, W), 40 * s, seed=31, octaves=2))
    edge = hp.blur(edge, 0.7 * s) * (0.3 + 0.7 * brk)
    img = hp.mix(img, hp.rgb("#1a1416"), (edge * f("outline", 0.55))[..., None])

# ---------------------------------------------------------------- dust in the moonbeams, warm motes over the lanterns
inside = hp.smoothstep(4.6, 4.2, depth)
cold = np.clip(pre[..., 2] - pre[..., 0] * 0.75, 0, 1) * inside
if act is not None:
    cold *= 1 - np.clip(fblur(act, 4 * s) * 2, 0, 1)
cold = fblur(cold, 6 * s)
cand = rng.uniform(0, 1, (int(9000 * f("dust", 1.0)), 2)) * [W, H]
ix, iy = cand[:, 0].astype(int), cand[:, 1].astype(int)
pc = cold[iy, ix] / (cold.max() + 1e-6)
keep = rng.uniform(0, 1, len(ix)) < pc ** 1.3
pts = [(x, y, float(b)) for (x, y), b in zip(cand[keep], 0.35 + 0.65 * rng.uniform(0, 1, keep.sum()) * pc[keep])]
print("dust specks:", len(pts))
img = A.motes(img, pts, color="#cdeeff", size=1.6, glow=0.45)
warm = fblur(np.clip(pre[..., 0] - pre[..., 2] * 1.1, 0, 1) * inside, 40 * s)       # glow of the lanterns
air = hp.smoothstep(0.42, 0.22, pre @ np.array([0.3, 0.55, 0.15], np.float32))         # specks show on dark air
if act is not None:
    air *= 1 - np.clip(fblur(act, 8 * s) * 3, 0, 1)
warm = warm * air
cand = rng.uniform(0, 1, (int(2500 * f("embers", 1.0)), 2)) * [W, H]
ix, iy = cand[:, 0].astype(int), cand[:, 1].astype(int)
pw = warm[iy, ix] / (warm.max() + 1e-6)
keep = rng.uniform(0, 1, len(ix)) < pw ** 1.5 * 0.25
pts = [(x, y, float(b)) for (x, y), b in zip(cand[keep], 0.4 + 0.6 * rng.uniform(0, 1, keep.sum()))]
print("warm motes:", len(pts))
img = A.motes(img, pts, color="#ffb45a", size=1.4, glow=0.4, streak=(0.0, -2.0))

img = A.finish(img, teal=f("teal", 0.6), warm=0.08, sat=f("sat", 1.16), vig=f("vig", 0.42),
               shadow_tint=(0.46, 0.9, 1.3), shadow_lift=(0.0, 0.02, 0.06))
if FINAL:
    hp.save(A.full(img), 30, "arcane-workshop")
else:
    A.preview(img, tag)
    A.preview(np.clip(pre, 0, 1), tag, "pre")
