"""29 arcane-canal post: Cycles passes -> Arcane-like painting (src/arcane.py), plus specks of light in the haze.

    python src/29_arcane_canal_post.py <tag> [final] [ev=0.85] [ink=0.7] [teal=0.5] [sat=1.15] [motes=1]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
import numpy as np  # noqa: E402

import arcane as A  # noqa: E402
import hp  # noqa: E402

tag = sys.argv[1]
final = "final" in sys.argv[2:]
kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)


def F(k, d):
    return float(kv.get(k, d))


P = A.load(tag)
depth, actor = P["depth"], P.get("actor")
s = A.scale_of(P["image"])
img = A.tonemap(P["image"], exposure=F("ev", 1.0), knee=F("knee", 1.0))
with A.paint_slot():
    img = A.paint(img, depth=depth, detail=actor, seed=29, far=(F("far0", 16.0), F("far1", 90.0)))
outline = np.clip((actor - 0.65) * 4, 0, 1)            # ferryman and lanterns get an outline, the moon and pole not
img = A.ink(img, depth, P.get("normal"), near=2.0, far=F("inkfar", 24.0), keep_out=actor, outline=outline,
            amount=F("ink", 0.7))

# specks of light drifting in the lit haze: warm ones near the lanterns, pink ones under the neon
if F("motes", 1.0) > 0:
    H, W = img.shape[:2]
    rng = np.random.default_rng(29)
    glow = hp.blur(img, 30 * s)
    lum = glow.mean(axis=2)
    warm, pink = [], []
    for _ in range(int(900 * F("motes", 1.0))):
        x, y = rng.uniform(0, W), rng.uniform(H * 0.3, H * 0.8)
        b = float(lum[int(y), int(x)])
        if depth[int(y), int(x)] > 500 or b < 0.3 or rng.random() > b * 1.2:     # not in the sky
            continue
        r_, g_, b_ = glow[int(y), int(x)]
        pt = (x, y, min(1.0, (b - 0.18) * 1.6) * rng.uniform(0.25, 1.0))
        (pink if b_ > g_ * 1.1 and r_ > g_ else warm).append(pt)
    img = A.motes(img, warm, color="#ffbf70", size=1.5, glow=0.45, streak=(0.0, -3.0))
    img = A.motes(img, pink, color="#ff7ac0", size=1.5, glow=0.4, streak=(0.0, -3.0))

img = A.finish(img, teal=F("teal", 0.5), sat=F("sat", 1.15), vig=F("vig", 0.4))
if final:
    hp.save(A.full(img), 29, "arcane-canal")
else:
    A.preview(img, tag)
