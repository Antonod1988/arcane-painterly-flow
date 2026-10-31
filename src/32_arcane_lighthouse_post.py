"""32 · Arcane lighthouse, post: beams from the jack-o'-lantern's eyes (analytic light cones from scene.json, occluded
by the depth pass), storm fog, bats in the beams, then the shared Arcane painter, wind-blown rain lit by the beams,
softened spray and drifting spray specks.

    python src/32_arcane_lighthouse_post.py <tag> [final] [ev=1.3] [beam=1.0] [rain=1.0]
"""
import json
import math
import sys

import cv2
import numpy as np

import arcane as A
import hp
from fc_util import Mask, bat_poly, fblur

TAG = sys.argv[1]
FINAL = "final" in sys.argv
KW = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
EV = float(KW.get("ev", 1.3))
BEAM = float(KW.get("beam", 1.0))
RAIN = float(KW.get("rain", 1.0))
rng = np.random.default_rng(32)

P = A.load(TAG)
S = json.load(open(f"{A.ROOT}/work/{TAG}/scene.json"))
hdr = P["image"]
depth = P["depth"]
H, W = depth.shape
s = W / 1440.0
actor = P.get("actor", np.zeros((H, W), np.float32))
spray = P.get("spray", np.zeros((H, W), np.float32))
sky = depth >= 9e3

# ------------------------------------------------------------------ camera rays (world space)
M = np.array(S["cam"], np.float32)
R, C = M[:3, :3], M[:3, 3]
fwd = -R[:, 2]
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
sw, sh = 36.0 * W / H, 36.0
cx = ((xs + 0.5) / W - 0.5) * sw
cy = (0.5 - (ys + 0.5) / H) * sh + S["shift_y"] * 36.0
V = cx[..., None] * R[:, 0] + cy[..., None] * R[:, 1] - S["lens"] * R[:, 2]
V /= np.linalg.norm(V, axis=2, keepdims=True)
kz = V @ fwd                                               # planar depth per unit of ray length
del xs, ys, cx, cy


def project(p):
    d = R.T @ (np.asarray(p, np.float32) - C)
    return np.array([(0.5 + d[0] / -d[2] * S["lens"] / sw) * W, (0.5 - d[1] / -d[2] * S["lens"] / sh - S["shift_y"] * 36 / sh) * H])


# ------------------------------------------------------------------ beams
def beam(O, D, r0=0.32, ang=float(KW.get("ang", 2.8)), L=80.0):
    """In-scattered light of a soft cone from O along unit D, seen along every pixel ray (relative units)."""
    O, D = np.asarray(O, np.float32), np.asarray(D, np.float32)
    D = D / np.linalg.norm(D)
    w0 = C - O
    b = V @ D
    d = V @ w0
    e = float(D @ w0)
    den = np.maximum(1 - b * b, 1e-4)
    sr = (b * e - d) / den
    tb = (e - b * d) / den
    q = w0[None, None] + sr[..., None] * V - tb[..., None] * D[None, None]
    dist = np.linalg.norm(q, axis=2)
    rt = r0 + np.maximum(tb, 0) * math.tan(math.radians(ang))
    inten = np.exp(-(dist / rt) ** 2) * (r0 / rt) ** 0.85 * np.exp(-np.maximum(tb, 0) / L) / np.maximum(np.sqrt(den), 0.3)
    inten *= hp.smoothstep(0.0, 1.5, tb) * (sr > 0)
    vis = np.where(sky, 1.0, hp.smoothstep(-1.5, 1.5, depth - sr * kz))   # hidden behind nearer geometry
    # texture: streaky density along the beam (rain and spray in the light)
    tex = np.clip(0.55 + 0.9 * cv2.remap(NZ, (np.clip(tb, 0, 199) * 4.0).astype(np.float32),
                                         ((dist / rt) * 40 + 200).astype(np.float32), cv2.INTER_LINEAR), 0.2, 1.6)
    core = np.exp(-(dist / (rt * 0.75)) ** 6) * np.exp(-np.maximum(tb, 0) / (L * 0.7)) * hp.smoothstep(-0.3, 0.6, tb) * (sr > 0)
    return (inten * vis * tex).astype(np.float32), tb, (core * vis * np.clip(tex, 0.5, 1.3)).astype(np.float32)


NZ = (hp.noise((600, 800), 9, seed=5, octaves=4) - 0.5).astype(np.float32)
NZ = cv2.resize(NZ, (800, 600))
beams = np.zeros((H, W), np.float32)
cores = np.zeros((H, W), np.float32)
tbs = []
for O, D in zip(S["eyes"], S["beams"]):
    bi, tb, co = beam(O, D)
    beams += bi
    cores = np.maximum(cores, co)
    tbs.append(tb)
beam_col = np.array([1.0, 0.58, 0.22], np.float32)
hot = hp.smoothstep(6.0, 0.0, np.minimum(*tbs))[..., None] * np.array([0.25, 0.2, 0.12], np.float32)   # whiter near the eyes
pumpkin = np.clip(fblur(actor, 1.5 * s) * 1.3, 0, 1)
beams *= 1 - pumpkin                                         # the beams leave the eyes; the carved face stays readable
cores *= 1 - pumpkin
hdr = hdr + (beams[..., None] * (beam_col + hot)) * 0.7 * BEAM
pk = np.minimum(1.0, float(KW.get("pclamp", 1.6)) / np.maximum(hdr.max(axis=2, keepdims=True), 1e-6))   # keep the hue
hdr = np.where(pumpkin[..., None] > 0.5, hdr * pk, hdr)                     # carved holes stay yellow, no giant bloom

# ------------------------------------------------------------------ land vs sky: a pale break low behind the cliff line, a near-black cliff
cliff = P.get("cliff", np.zeros((H, W), np.float32))
cols = np.nonzero((cliff > 0.5).any(axis=0))[0]
first = np.argmax(cliff > 0.5, axis=0).astype(np.float32)
ycl = np.interp(np.arange(W), cols, first[cols]) if len(cols) else np.full(W, H * 0.55, np.float32)
ycl = cv2.GaussianBlur(ycl.reshape(1, -1).astype(np.float32), (0, 0), 40 * s).ravel()
yy = np.arange(H, dtype=np.float32)[:, None]
band = np.exp(-np.maximum(ycl[None, :] - yy, 0) / (0.085 * H)) * sky
band *= 0.75 + 0.5 * hp.noise((H, W), 160 * s, seed=8, octaves=2)
hdr = hdr + band[..., None] * np.array([0.085, 0.19, 0.165], np.float32) * float(KW.get("glow", 1.0))
lum_h = hdr @ np.array([0.3, 0.55, 0.15], np.float32)
keep = 0.4 + 0.6 * hp.smoothstep(0.03, 0.25, lum_h)                # darks to near-black, wet highlights survive
cl = np.clip(cliff, 0, 1) * (1 - np.clip(spray * 2, 0, 1))
hdr = hdr * (1 - cl * (1 - keep))[..., None]
# big facets: sea-facing (west) planes catch a cool wet sheen, up-facing planes near the lights a warm spill
Nn = P["normal"]
X0, Y0 = hp.grid(W, H)
spill_pts = [project((5.5 + 7.0, 30.0 + 3.5 - 2.42, 12.5 + 1.6)), project((5.5, 30.0 - 2.6, 13.5))]
spill = np.clip(sum(np.exp(-np.hypot(X0 - q[0], Y0 - q[1]) / (0.22 * W)) for q in spill_pts), 0, 1)
cool = np.clip(-Nn[..., 0] - 0.15, 0, 1) * (0.6 + 0.8 * hp.noise((H, W), 30 * s, seed=12))
warmf = np.clip(Nn[..., 2] - 0.15, 0, 1) * spill
hdr = hdr + (cl * cool)[..., None] * np.array([0.02, 0.05, 0.055], np.float32) * float(KW.get("sheen", 1.0))     + (cl * warmf)[..., None] * np.array([0.16, 0.07, 0.02], np.float32)

# ------------------------------------------------------------------ storm fog and the lantern's halo
jack_s = project(S["jack"])
fog_t = (np.where(sky, 0.0, 1 - np.exp(-np.maximum(depth - 6.0, 0) * 0.008)) * (1 - pumpkin))[..., None]
X, Y = hp.grid(W, H)
rj = np.hypot(X - jack_s[0], Y - jack_s[1]) / W
halo = (np.exp(-rj / 0.09) * 0.55 + np.exp(-rj / 0.3) * 0.25)[..., None] * np.array([1.0, 0.45, 0.14], np.float32)
fog_col = np.array([0.03, 0.062, 0.068], np.float32) + halo * 0.6
hdr = hdr * (1 - fog_t * 0.85) + fog_col * fog_t * 0.85
hdr = hdr + halo * 0.35 * (0.4 + 0.6 * sky[..., None]) * (1 - pumpkin[..., None])
img = A.tonemap(hdr, exposure=EV)

# ------------------------------------------------------------------ spray: melt the foam balls into mist before painting
spm = np.clip(fblur(spray, 3.0 * s) * 1.3, 0, 1)[..., None]
soft = fblur(img, 7.0 * s)
img = hp.mix(img, np.clip(soft * 1.15 + np.array([0.03, 0.05, 0.055], np.float32), 0, 1), spm * 0.85)
img = img + fblur(spray, 30 * s)[..., None] * np.array([0.1, 0.16, 0.17], np.float32) * 0.7

# ------------------------------------------------------------------ bats in the beams (silhouettes, kept crisp by the painter)
bt = Mask(W, H)
for k, (bi, t_, off, sp_, an, fl) in enumerate(((0, 7, -0.5, 0.06, -0.3, 0.6), (0, 13, 0.7, 0.05, 0.25, -0.4),
                                                (1, 7.0, 0.4, 0.045, -0.1, 0.2), (1, 11.5, -0.9, 0.07, 0.35, 0.9),
                                                (0, 16.5, 1.2, 0.08, 0.1, -0.7), (1, 4.5, -0.3, 0.032, -0.25, 0.3))):
    O, D = np.array(S["eyes"][bi]), np.array(S["beams"][bi])
    side = np.cross(D, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    q = project(O + D * t_ + side * off)
    zc = (R.T @ (O + D * t_ + side * off - C))[2]
    if zc < -1.0 and -0.05 * W < q[0] < 1.05 * W and -0.05 * H < q[1] < 0.5 * H:
        bt.poly(bat_poly(q[0], q[1], sp_ * W, an, fl))
batm = bt.f
img = hp.mix(img, hp.rgb("#120a12"), batm[..., None] * 0.95)

# ------------------------------------------------------------------ paint
detail = np.clip(actor + batm, 0, 1)
with A.paint_slot():
    img = A.paint(img, depth=depth, detail=detail, seed=32, far=(40.0, 400.0), far_k=1.2)
img = A.ink(img, depth, P.get("normal"), near=3.0, far=60.0, keep_out=np.clip(actor + fblur(spray, 4 * s) * 3, 0, 1),
            outline=P.get("keeper"), amount=0.6)
# crisp light cones over the painted glow (painted look: textured by the strokes underneath)
lum_p = img @ np.array([0.3, 0.55, 0.15], np.float32)
img = img + (fblur(cores, 1.2 * s) * (0.55 + 0.6 * lum_p))[..., None] * np.array([1.0, 0.7, 0.36], np.float32) * 0.32 * BEAM

# ------------------------------------------------------------------ flying spray specks
sy, sx = np.nonzero(spray > 0.5)
if len(sx):
    k = rng.choice(len(sx), min(len(sx), int(900 * s * s)), replace=False)
    pts = [(sx[i] + rng.normal(0, 18 * s), sy[i] + rng.normal(0, 18 * s) - rng.uniform(0, 40 * s), rng.uniform(0.15, 0.55)) for i in k]
    img = A.motes(img, pts, color="#cfe6ea", size=1.5, glow=0.3, streak=(-3.0, -1.5))

# ------------------------------------------------------------------ rain, lit where it crosses the beams
lightm = np.clip(fblur(beams, 3 * s) * 0.9 + np.exp(-rj / 0.12) * 0.4, 0, 1.5)
for (n, ln, th, a, blur_) in ((6500, 30, 1.0, 0.3, 0.0), (1100, 80, 1.6, 0.38, 1.2)):
    rm = np.zeros((H, W), np.float32)
    n = int(n * s * s * RAIN)
    x0 = rng.uniform(-0.1 * W, 1.05 * W, n)
    y0 = rng.uniform(-0.05 * H, H, n)
    L_ = ln * s * rng.uniform(0.6, 1.4, n)
    ang = math.radians(14) + rng.normal(0, 0.03, n)
    for i in range(n):
        cv2.line(rm, (int(x0[i] * 16), int(y0[i] * 16)),
                 (int((x0[i] + math.sin(ang[i]) * L_[i]) * 16), int((y0[i] + math.cos(ang[i]) * L_[i]) * 16)),
                 float(rng.uniform(0.4, 1.0)), max(1, int(round(th * s))), cv2.LINE_AA, shift=4)
    if blur_:
        rm = hp.blur(rm, blur_ * s)
    lit = (0.25 + lightm)[..., None]
    rm *= 1 - 0.8 * pumpkin
    img = img + rm[..., None] * a * lit * np.array([0.75, 0.82, 0.86], np.float32) * (1 + lightm[..., None] * np.array([0.6, 0.15, -0.3], np.float32))

# ------------------------------------------------------------------ painted rim along the cliff top: warm by the cottage and tower, cool elsewhere
landm = (~sky).astype(np.float32)
k = max(2, int(round(5 * s)))
top = np.clip(landm - np.roll(landm, k, axis=0), 0, 1) * (cv2.dilate(cliff, np.ones((9, 9), np.uint8)) > 0.3)
top *= 1 - np.clip(spray * 3, 0, 1)
warm_pts = [project((5.5 + 7.0, 30.0 + 3.5 - 2.42, 12.5 + 1.6)), project((5.5, 30.0 - 2.6, 13.5))]
wgt = sum(np.exp(-np.hypot(X - q[0], Y - q[1]) / (0.25 * W)) for q in warm_pts)
wgt = np.clip(wgt, 0, 1)[..., None]
rim = hp.blur(top, 1.0 * s) * hp.smoothstep(0.35, 0.6, hp.noise((H, W), 14 * s, seed=21, octaves=2))
rim_col = hp.mix(np.array([0.55, 0.85, 0.82], np.float32), np.array([1.0, 0.62, 0.28], np.float32), wgt)
img = img + rim[..., None] * rim_col * (0.3 + 0.6 * wgt)
img = img + fblur(top, 6 * s)[..., None] * rim_col * 0.12 * wgt

img = A.finish(img, teal=0.5, warm=0.08, sat=1.15, vig=0.42)
if FINAL:
    hp.save(A.full(img), 32, "arcane-lighthouse")
else:
    A.preview(img, TAG)
    for i, c in enumerate(KW.get("crop", "").split(";")):
        if c:
            x0, y0, x1, y1 = [int(v) for v in c.split(",")]
            hp.to_pil(img[y0:y1, x0:x1]).save(f"{A.ROOT}/work/{TAG}/crop{i}.png")
