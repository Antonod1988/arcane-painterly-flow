"""31 · Arcane-style painting from the rooftop render (blender/31_arcane_rooftop.py).

    python src/31_arcane_rooftop_post.py <tag> [final] [key=value ...]

1. a painted night sky goes behind the transparent render: teal-blue gradient, a big low moon with soft maria,
   thin cloud bands that turn magenta and silver near the moon, a few bats;
2. atmospheric perspective from the depth pass (far city sinks into blue haze, a warm glow hangs over the
   horizon where the moon and the city lights meet);
3. arcane.py: filmic tone map + bloom, oriented brush strokes (kids, jack-o'-lantern and lanterns kept crisp),
   ink on the near roof, chimney and kids, crisp dabs for the far lanterns, colour design, canvas."""
import json
import math
import sys

import cv2
import numpy as np

import arcane as A
import hp
from fc_util import Mask, bat_poly, fblur

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
tag = sys.argv[1]
FINAL = "final" in sys.argv[2:]
KW = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:] if "=" in a)}


def kw(k, d):
    return KW.get(k, d)


P = A.load(tag)
S = json.load(open(f"{ROOT}/work/{tag}/scene.json"))
img, alpha, depth = P["image"], np.clip(P["alpha"], 0, 1), P["depth"]
H, W = alpha.shape
s = W / 1440.0
X, Y = hp.grid(W, H)
mx, my = S["moon"][0] * W / S["W"], S["moon"][1] * H / S["H"]
mr = S["moon_r"] * W / S["W"] * kw("moonk", 1.0)
hy = S["horizon"] * H / S["H"]
rng = np.random.default_rng(31)

# ------------------------------------------------------------------ sky (linear light)
up_t = np.clip((hy - Y) / hy, 0, 1)[..., None]
sky = hp.mix(np.array([0.08, 0.15, 0.2], np.float32), np.array([0.022, 0.06, 0.1], np.float32), hp.smoothstep(0.0, 0.35, up_t))
sky = hp.mix(sky, np.array([0.006, 0.02, 0.045], np.float32), hp.smoothstep(0.3, 1.0, up_t))
dm = np.hypot(X - mx, Y - my)
near_moon = np.exp(-np.maximum(dm - mr, 0) / (mr * 1.3))
wide_moon = np.exp(-np.maximum(dm - mr, 0) / (mr * 4.5))
sky = sky + (near_moon * kw("halo", 0.32))[..., None] * np.array([0.85, 0.62, 0.7], np.float32) \
    + (wide_moon * 0.2)[..., None] * np.array([0.45, 0.3, 0.5], np.float32)
# thin cloud bands, stretched sideways; lit silver/magenta near the moon, dark further out
cn = hp.noise((H, max(8, W // 7)), 26 * s, seed=301, octaves=4)
cn = cv2.resize(cn, (W, H), interpolation=cv2.INTER_CUBIC)
fine = hp.noise((H, max(8, W // 3)), 6 * s, seed=302, octaves=3)
fine = cv2.resize(fine, (W, H), interpolation=cv2.INTER_CUBIC)
band = np.exp(-((Y - (my + 0.25 * mr)) / (mr * 2.2)) ** 2) + 0.35 * np.exp(-((Y - (my - 3.2 * mr)) / (mr * 1.6)) ** 2)
band *= 0.45 + 0.55 * np.exp(-((X - mx) / (W * 0.45)) ** 2)
cloud = np.clip(hp.smoothstep(0.5, 0.72, cn * 0.8 + fine * 0.3) * band * kw("clouds", 1.0), 0, 1)
lit = np.clip(near_moon * 1.6 + wide_moon * 0.5, 0, 1.6)
cloud_col = hp.mix(np.array([0.012, 0.02, 0.04], np.float32), np.array([0.95, 0.5, 0.78], np.float32),
                   np.clip(lit * 0.75, 0, 1)[..., None])
cloud_col = cloud_col + (np.clip(near_moon * 2.0 - 0.9, 0, 1) * 1.2)[..., None] * np.array([0.9, 0.85, 0.9], np.float32)
# the moon itself: limb darkening and soft maria, veiled where a cloud band crosses it
md = dm / mr
disc = np.clip((1 - md) * mr * 0.9 + 0.5, 0, 1)
maria = hp.noise((H, W), 0.33 * mr, seed=303, octaves=3)
mcol = np.array([1.0, 0.95, 0.84], np.float32) * (kw("moon", 2.1) * (1 - 0.25 * np.clip(md, 0, 1) ** 3)
                                                  * (1 - 0.22 * hp.smoothstep(0.5, 0.72, maria)))[..., None]
sky = hp.mix(sky, mcol, disc[..., None])
sky = hp.mix(sky, cloud_col, (cloud * 0.85)[..., None])
# a handful of faint stars up high
st = np.zeros((H, W), np.float32)
for _ in range(int(160 * s * s) + 40):
    x, y = rng.uniform(0, W), rng.uniform(0, hy - 2.5 * mr)
    st[int(y), int(x)] = rng.uniform(0.3, 1.0) * (1 - 0.8 * np.exp(-math.hypot(x - mx, y - my) / (4 * mr)))
sky = sky + (fblur(st, 0.7 * s) * 2.2)[..., None] * np.array([0.7, 0.8, 1.0], np.float32)
# bats crossing the moon
bat = Mask(W, H)
for (bx, by, sp, an, fl) in [(mx - 0.55 * mr, my - 0.25 * mr, 0.42 * mr, -0.2, 0.6), (mx + 0.3 * mr, my - 0.75 * mr, 0.26 * mr, 0.25, -0.6),
                             (mx - 1.6 * mr, my - 1.3 * mr, 0.2 * mr, 0.1, 0.3), (mx + 1.25 * mr, my + 0.05 * mr, 0.16 * mr, -0.3, 0.9)]:
    bat.poly(bat_poly(bx, by, sp, an, fl))
sky = hp.mix(sky, np.array([0.006, 0.006, 0.012], np.float32), bat.f[..., None])

hdr = img + (1 - alpha)[..., None] * sky

# a thin ribbon of chimney smoke curling up and drifting left, moonlit on top, warm from the city below
cx, cy = S["chimney_top"][0] * W / S["W"], S["chimney_top"][1] * H / S["H"]
sm = Mask(W, H)
L = kw("smokel", 620.0) * s
for k, t in enumerate(np.linspace(0, 1, 140)):
    px = cx - 0.38 * L * t ** 1.2 + 26 * s * math.sin(t * 7.0) * (0.3 + t)
    py = cy - L * t
    sm.circle((px, py), (8 + 85 * t ** 1.1) * s)
smoke = fblur(sm.f, 12 * s) * hp.smoothstep(0.25, 0.75, hp.noise((H, W), 30 * s, seed=312, octaves=3))
smoke *= np.clip(1.0 - np.clip((cy - Y) / L, 0, 1) ** 0.6, 0, 1) * kw("smoke", 1.1)
smoke_col = hp.mix(np.array([0.22, 0.13, 0.09], np.float32), np.array([0.2, 0.24, 0.32], np.float32),
                   np.clip((cy - Y) / (0.4 * L), 0, 1)[..., None]) + (wide_moon * 0.2)[..., None]
hdr = hp.mix(hdr, smoke_col, np.clip(smoke, 0, 1)[..., None] * (1 - P.get("actor", 0 * smoke))[..., None])
print("smoke", round(cx), round(cy), float(smoke.max()), float((smoke > 0.1).mean()))

# ------------------------------------------------------------------ atmosphere from depth
d = np.where(alpha > 0.5, depth, 1e4)
fog_t = (1 - np.exp(-np.maximum(d - kw("fog0", 90.0), 0) / kw("fogl", 1700.0))) * alpha
horizon_glow = np.exp(-np.abs(Y - hy) / (60 * s))
fog_col = np.array([0.035, 0.07, 0.11], np.float32) + (wide_moon * 0.35)[..., None] * np.array([0.5, 0.3, 0.4], np.float32) \
    + horizon_glow[..., None] * np.array([0.12, 0.09, 0.1], np.float32)
hdr = hdr * (1 - fog_t[..., None] * kw("fogk", 0.85)) + fog_col * (fog_t[..., None] * kw("fogk", 0.85))
# warm light pollution over the city, a soft band hugging the horizon
hdr = hdr + (np.exp(-np.maximum(Y - hy, 0) / (90 * s) - np.maximum(hy - Y, 0) / (45 * s)) * kw("haze", 0.05))[..., None] \
    * np.array([1.0, 0.55, 0.35], np.float32)
# the moon's glitter path on the river: broken horizontal dashes in a column under the moon, widening toward us
water = P.get("water", np.zeros((H, W), np.float32))
below = np.clip((Y - hy) / (H - hy), 0, 1)
col_w = mr * (0.18 + 1.2 * below)
column = np.exp(-((X - mx) / col_w) ** 2) * np.exp(-below / 0.45) * hp.smoothstep(0.0, 0.04, below)
dash = hp.noise((H, max(8, W // 9)), 2.2 * s, seed=311, octaves=2)
dash = cv2.resize(dash, (W, H), interpolation=cv2.INTER_CUBIC)
glit = hp.smoothstep(0.6, 0.78, dash) * column * water * kw("glitter", 0.7)
hdr = hdr + glit[..., None] * np.array([1.0, 0.9, 0.82], np.float32) + (fblur(glit, 6 * s) * 0.5)[..., None] * np.array([0.9, 0.55, 0.7], np.float32)
hdr = hdr.astype(np.float32)

# ------------------------------------------------------------------ paint
disp = A.tonemap(hdr, exposure=kw("ev", 1.0), bloom=(kw("b1", 0.35), kw("b2", 0.25), kw("b3", 0.12)))
actor = P.get("actor", np.zeros((H, W), np.float32))
lamp = P.get("lamp", np.zeros((H, W), np.float32))
detail = np.clip(actor + 0.8 * lamp + 0.6 * disc, 0, 1)
with A.paint_slot():
    out = A.paint(disp, depth=d if kw("far0", 0) > 0 else None, detail=detail, seed=31,
                  far=(kw("far0", 400.0), kw("far1", 4000.0)), far_k=kw("fark", 0.8))
out = A.ink(out, d, P["normal"], near=1.0, far=kw("inkfar", 9.0), keep_out=actor, outline=actor, amount=0.7)

# far lanterns: crisp painted dabs over the loose strokes, so the rising stream stays readable
half_v = math.atan(18.0 / S["lens"])
pts = []
for (px, py, z, r) in S["lanterns"]:
    if z <= 0:
        continue
    rp = r / z / math.tan(half_v) * (H / 2)
    if rp > 3.5 * s:
        continue
    x, y = px * W / S["W"], py * H / S["H"]
    if 0 <= x < W and 0 <= y < H and alpha[int(y), int(x)] > 0.5 and depth[int(y), int(x)] > z * 0.97:
        pts.append((x, y, float(np.clip(0.35 + rp / (2.5 * s), 0.35, 1.0))))
out = A.motes(out, pts, color="#ff9a3c", size=1.6, glow=kw("dab", 0.22), streak=(0.0, 1.2))
print("lantern dabs", len(pts))

out = A.finish(out, teal=kw("teal", 0.5), warm=0.08, sat=kw("sat", 1.15), vig=kw("vig", 0.4))
if FINAL:
    hp.save(A.full(out), 31, "arcane-rooftop")
else:
    A.preview(out, tag)
    if W == 1440:
        hp.to_pil(out).save(f"{ROOT}/work/{tag}/arcane_full.png")
    A.preview(np.clip(disp, 0, 1), tag, "tonemapped")
