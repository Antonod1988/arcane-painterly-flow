"""33 · Arcane-like painterly finish for the overgrown glasshouse (render: blender/33_arcane_greenhouse.py).

    python src/33_arcane_greenhouse_post.py <tag> [final] [ev=0.9] [teal=0.5] [sat=1.15] [dust=500]

tone map -> brush strokes (the gardener, the jacks and the wisps stay crisp) -> ink -> painted FX (wisp halos and
sparks, dust specks in the moon shafts) -> colour design. Writes work/<tag>/arcane.jpg, or with `final`
out/33-arcane-greenhouse.png and captioned/33-arcane-greenhouse.png."""
import json
import sys

import cv2
import numpy as np

import arcane as A
import hp
from fc_util import fblur

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
tag = sys.argv[1]
FINAL = "final" in sys.argv[2:]
KV = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)


def kv(k, d):
    return float(KV.get(k, d))


P = A.load(tag)
hdr = P["image"]
H, W = hdr.shape[:2]
s = A.scale_of(hdr)
depth = P["depth"]
actor = P.get("actor", np.zeros((H, W), np.float32))
glow = P.get("glow", np.zeros((H, W), np.float32))
fx = json.load(open(f"{ROOT}/work/{tag}/fx.json"))
rng = np.random.default_rng(33)

img = A.tonemap(hdr, exposure=kv("ev", 1.05), bloom=(0.5, 0.45, 0.4))
base = img.copy()
detail = np.clip(actor + glow, 0, 1)
with A.paint_slot():
    img = A.paint(img, depth=depth, detail=detail, seed=33, far=(12.0, 40.0))
img = A.ink(img, depth, P["normal"], near=1.5, far=kv("inkfar", 8.0), keep_out=detail, outline=actor,
            amount=kv("ink", 0.4), dthr=kv("dthr", 0.28))

# ---------------------------------------------------------------- a crisper silhouette for the gardener
# (the haze in front of him greys him out; darken his unlit parts, keep the lantern-lit coat)
fu, fv = fx["gardener_feet"][:2], fx["gardener_head"][:2]
gh = (fv[1] - fu[1]) * H
X, Y = hp.grid(W, H)
box = ((np.abs(X - fu[0] * W) < -gh * 0.45) & (Y > fv[1] * H + gh * 0.08) & (Y < fu[1] * H + 4 * s)).astype(np.float32)
gl = img @ np.array([0.3, 0.55, 0.15], np.float32)
dark = actor * hp.blur(box, 3 * s) * hp.smoothstep(0.45, 0.2, gl)
img = img * (1 - kv("silhouette", 0.35) * dark)[..., None]

# ---------------------------------------------------------------- will-o'-wisps: halos and a few sparks
core = np.zeros((H, W), np.float32)
for i, (u, v, z, rv) in enumerate(fx["wisps"]):
    if not (0 <= u <= 1 and 0 <= v <= 1):
        continue
    x, y, r = u * W, v * H, max(1.5 * s, rv * H)
    cv2.circle(core, (int(x * 16), int(y * 16)), int(r * 16), 1.0, -1, cv2.LINE_AA, shift=4)
    for k in range(rng.integers(2, 5)):                         # a few tiny sparks drifting around each wisp
        d, a2 = rng.uniform(2.5, 7) * r, rng.uniform(0, 2 * np.pi)
        cv2.circle(core, (int((x + d * np.cos(a2)) * 16), int((y + d * np.sin(a2)) * 16)), int(max(0.6 * s, r * 0.25) * 16),
                   float(rng.uniform(0.3, 0.7)), -1, cv2.LINE_AA, shift=4)
wcol = hp.rgb("#62ffd2")
img = img + (core[..., None] * (wcol * 0.4 + 0.6) * 0.9
             + (fblur(core, 4 * s) * 1.6 + fblur(core, 14 * s) * 1.1 + fblur(core, 45 * s) * 0.6)[..., None] * wcol) * kv("wisp", 1.0)

# ---------------------------------------------------------------- dust in the moon shafts: pale specks where the air glows
air = (depth > 2.0) & (glow < 0.1) & (actor < 0.1)
lum = hp.blur(base @ np.array([0.3, 0.55, 0.15], np.float32), 6 * s)
n = int(kv("dust", 500))
xs = rng.uniform(0, W, n * 6)
ys = rng.uniform(0, H * 0.8, n * 6)
ix, iy = xs.astype(int), ys.astype(int)
p = np.clip((lum[iy, ix] - 0.22) * 3.0, 0, 1) * air[iy, ix]
keep = rng.random(len(p)) < p
pts = [(x, y, b) for x, y, b in zip(xs[keep][:n], ys[keep][:n], rng.uniform(0.15, 0.55, n))]
img = A.motes(img, pts, color="#bdf5ea", size=1.2, glow=0.35, streak=(0.0, 2.0))

img = A.finish(img, teal=kv("teal", 0.5), warm=kv("warm", 0.08), sat=kv("sat", 1.15), vig=kv("vig", 0.42))
if FINAL:
    hp.save(A.full(img), 33, "arcane-greenhouse")
else:
    A.preview(img, tag)
