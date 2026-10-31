"""20 · Arcane-like painterly: a steep old-town stair alley at dusk, lanterns, a hooded figure with a jack-o'-lantern.

1. a small box scene is ray traced (numba): stairs, leaning houses, windows, lanterns, a tower at the top;
2. shaded with shadowed lantern light, a low sunset sun, sky ambient + AO, wet-stone speculars and fog;
3. figure, pumpkins, garland and bats are painted on in 2D at projected positions;
4. then the whole frame is repainted with oriented brush strokes (coarse to fine, numba),
   followed by selective dark linework, crisp highlights and a teal/orange colour design."""
import math
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import cv2
import numpy as np
from numba import njit

import hp
from fc_ray import Scene, light, rot_y, trace
from fc_util import Mask, W, H, bat_poly, catmull, fblur, soft_clip, tapered

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root

PREVIEW = "--preview" in sys.argv
RS = 4 if PREVIEW else 2                                   # base render runs at 1/RS resolution
BW, BH = W // RS, H // RS
rng = np.random.default_rng(2020)

# ------------------------------------------------------------------ scene
AW = 1.45                                                  # half width of the alley
STEP_D, STEP_H, Z0, NSTEP = 0.34, 0.17, 0.6, 56
TOP = NSTEP * STEP_H


def ground(z):
    if z < Z0:
        return 0.0
    i = int((z - Z0) / STEP_D)
    return TOP if i >= NSTEP else (i + 1) * STEP_H


PLASTER, TIMBER, STONE, ROOF, WIN_LIT, WIN_DARK, IRON, LAMP, TOWER, CLOCK, BRICK, DOOR = range(1, 13)
sc = Scene()
sc.box([-AW - 0.3, -3, -4], [AW + 0.3, 0, Z0], STONE, (0.9, 0.9, 0.95))
for i in range(NSTEP):
    j = rng.uniform(-0.012, 0.012)
    sc.box([-AW - 0.3, -3, Z0 + i * STEP_D], [AW + 0.3, (i + 1) * STEP_H + j, Z0 + (i + 1) * STEP_D + 0.002],
           STONE, tuple(rng.uniform(0.88, 1.08, 3) * [1, 1, 1.03]))
sc.box([-AW - 0.3, -3, Z0 + NSTEP * STEP_D], [AW + 0.3, TOP, 45], STONE, (0.95, 0.95, 1.0))

tints = [(0.58, 0.44, 0.32), (0.34, 0.44, 0.44), (0.58, 0.38, 0.35), (0.38, 0.40, 0.53), (0.53, 0.48, 0.37),
         (0.50, 0.35, 0.44)]
lights = []                                                 # (pos, colour, radius) point lights
lamp_world = []


def building(side, z0, z1, sb, floors, tint, oh, seed, lantern_z=None, balcony=False, chimney=True):
    r = np.random.default_rng(seed)
    s = side
    g0 = ground(z0) if z0 > Z0 else 0.0
    face0 = s * (AW + sb)                                     # ground floor face x
    face1 = s * (AW + sb - oh)                                # upper floors face x (overhang)
    deep = s * 9.0
    gf_top = g0 + 3.3
    top = gf_top + (floors - 1) * 2.9
    lo_x, hi_x = sorted([face0, deep])
    sc.box([lo_x, g0 - 4, z0], [hi_x, gf_top, z1], BRICK, tint)
    lo_x, hi_x = sorted([face1, deep])
    sc.box([lo_x, gf_top, z0 + 0.04], [hi_x, top + 0.5, z1 - 0.04], PLASTER, tint)
    if oh > 0:                                                # jetty beam under the overhang
        lo_x, hi_x = sorted([face0 + s * 0.02, face1 - s * 0.04])
        sc.box([lo_x, gf_top - 0.18, z0 + 0.04], [hi_x, gf_top + 0.02, z1 - 0.04], TIMBER)
    # roof slab, eave hanging over the alley
    th = 0.62
    u = np.array([-s * math.cos(th), -math.sin(th), 0.0])
    v = np.array([-u[1], u[0], 0.0])
    if v[1] < 0:
        v = -v
    w = np.cross(u, v)
    eave = np.array([face1 - s * 0.18, top + 0.45, (z0 + z1) / 2])
    sc.box(None, None, ROOF, (1, 1, 1), rot=np.array([u, v, w]), center=eave - u * 2.6 + v * 0.17,
           half=np.array([2.6, 0.17, (z1 - z0) / 2 + 0.12]))
    if chimney and r.random() < 0.7:
        cz = r.uniform(z0 + 0.6, z1 - 0.6)
        cx = face1 - s * r.uniform(1.6, 2.6)
        lo_x, hi_x = sorted([cx, cx + s * 0.6])
        sc.box([lo_x, top, cz - 0.3], [hi_x, top + 2.8, cz + 0.3], BRICK, (0.85, 0.75, 0.7))
    # windows (upper floors) and shutters
    nwin = max(1, int((z1 - z0) / 1.9))
    for f in range(1, floors):
        fy = gf_top + (f - 1) * 2.9
        for k in range(nwin):
            zc = z0 + (k + 0.5) * (z1 - z0) / nwin + r.uniform(-0.2, 0.2)
            lit = r.random() < 0.42
            x_in, x_out = face1 + s * 0.05, face1 - s * 0.07
            lo_x, hi_x = sorted([x_in, x_out])
            sc.box([lo_x, fy + 0.85, zc - 0.36], [hi_x, fy + 2.15, zc + 0.36], WIN_LIT if lit else WIN_DARK)
            if r.random() < 0.55:
                for sgn in (-1, 1):
                    zz = zc + sgn * 0.6
                    x2 = face1 - s * 0.05
                    lo_x, hi_x = sorted([face1 + s * 0.02, x2])
                    sc.box([lo_x, fy + 0.85, zz - 0.2], [hi_x, fy + 2.15, zz + 0.2], TIMBER, tuple(r.uniform(0.7, 1.3, 3)))
            if balcony and f == 1 and k == 0:
                x2 = face1 - s * 0.75
                lo_x, hi_x = sorted([face1, x2])
                sc.box([lo_x, fy + 0.62, zc - 0.9], [hi_x, fy + 0.74, zc + 0.9], TIMBER)
                lo_x, hi_x = sorted([x2, x2 + s * 0.05])
                sc.box([lo_x, fy + 0.74, zc - 0.9], [hi_x, fy + 1.55, zc + 0.9], IRON)
    # ground floor: a door, sometimes a lit shop window
    dz = r.uniform(z0 + 0.8, z1 - 0.8)
    gd = ground(dz)
    lo_x, hi_x = sorted([face0 + s * 0.05, face0 - s * 0.05])
    sc.box([lo_x, gd - 0.1, dz - 0.5], [hi_x, gd + 2.15, dz + 0.5], DOOR)
    if r.random() < 0.6 and z1 - z0 > 3.2:
        sz = dz + (1.4 if dz < (z0 + z1) / 2 else -1.4)
        gs = ground(sz)
        sc.box([lo_x, gs + 0.8, sz - 0.5], [hi_x, gs + 2.0, sz + 0.5], WIN_LIT)
    if lantern_z is not None:
        gl = ground(lantern_z)
        y = gl + 2.95
        xb0, xb1 = face0, face0 - s * 0.7
        lo_x, hi_x = sorted([xb0, xb1])
        sc.box([lo_x, y, lantern_z - 0.025], [hi_x, y + 0.05, lantern_z + 0.025], IRON)
        sc.box([xb1 - 0.12, y - 0.42, lantern_z - 0.12], [xb1 + 0.12, y - 0.08, lantern_z + 0.12], LAMP)
        sc.box([xb1 - 0.15, y - 0.1, lantern_z - 0.15], [xb1 + 0.15, y - 0.04, lantern_z + 0.15], IRON)
        lights.append(((xb1, y - 0.27, lantern_z), (9.0, 4.6, 1.7), 0.3))
        lamp_world.append((xb1, y - 0.27, lantern_z))
    return top


L, R_ = -1, 1
building(L, -3, 2.6, 0.0, 6, tints[0], 0.30, 1)
building(L, 2.6, 6.8, 0.22, 6, tints[1], 0.45, 2, lantern_z=4.3)
building(L, 6.8, 10.6, 0.05, 5, tints[2], 0.35, 3, balcony=True)
building(L, 10.6, 15.0, 0.25, 4, tints[3], 0.40, 4, lantern_z=12.6)
building(L, 15.0, 19.8, 0.08, 4, tints[4], 0.30, 5)
building(L, 19.8, 26.0, 0.18, 3, tints[5], 0.35, 6, lantern_z=23.5)
building(L, 26.0, 31.0, 0.0, 4, tints[0], 0.3, 7)
building(R_, -3, 1.8, 0.1, 6, tints[3], 0.35, 11)
building(R_, 1.8, 5.9, 0.0, 6, tints[4], 0.30, 12)
building(R_, 5.9, 9.4, 0.2, 5, tints[0], 0.50, 13, lantern_z=8.1, balcony=True)
building(R_, 9.4, 13.8, 0.05, 4, tints[5], 0.30, 14)
building(R_, 13.8, 18.4, 0.22, 4, tints[1], 0.45, 15, lantern_z=17.0)
building(R_, 18.4, 25.0, 0.05, 3, tints[2], 0.30, 16)
building(R_, 25.0, 31.0, 0.1, 4, tints[4], 0.3, 17)
# tower closing the view at the top of the stairs
sc.box([-2.4, TOP - 1, 31.5], [0.9, TOP + 15.5, 34.5], TOWER, (0.9, 0.85, 0.95))
for sgn in (-1, 1):
    a = 1.2
    u = np.array([sgn * math.cos(a), -math.sin(a), 0.0])
    v = np.array([-u[1], u[0], 0.0])
    w = np.cross(u, v)
    apex = np.array([-0.75, TOP + 21.5, 33.0])
    sc.box(None, None, ROOF, (0.8, 0.8, 0.9), rot=np.array([u, v, w]), center=apex + u * 3.3,
           half=np.array([3.3, 0.15, 1.8]))
sc.box([-1.35, TOP + 11.4, 31.3], [-0.15, TOP + 12.6, 31.5], CLOCK)
sc.box([-1.05, TOP + 6.0, 31.3], [-0.45, TOP + 7.8, 31.5], WIN_LIT)
# the figure's jack-o'-lantern and the pumpkins on the steps also light the scene
FIG_Z, FIG_X = 6.4, 0.18
fig_g = ground(FIG_Z)
fig_lamp = (FIG_X + 0.38, fig_g + 0.72, FIG_Z - 0.05)
lights.append((fig_lamp, (5.0, 1.9, 0.45), 0.25))
step_pumpkins = [(-1.12, 2.9, 0.24), (-0.86, 3.25, 0.16), (1.12, 5.5, 0.27), (0.88, 5.95, 0.17), (-1.1, 13.4, 0.21),
                 (1.1, 15.4, 0.23), (0.9, 15.75, 0.15)]
for (px, pz, pr) in step_pumpkins:
    lights.append(((px, ground(pz) + pr * 0.9, pz), (1.2, 0.5, 0.12), 0.2))

c, hh, RR, mats, tint = sc.arrays()
print("boxes:", len(mats), "lights:", len(lights))

# ------------------------------------------------------------------ camera + trace
ro = np.array([0.1, 1.5, -1.6])
target = np.array([0.0, 10.8, 18.0])
fwd = (target - ro) / np.linalg.norm(target - ro)
right = np.cross(fwd, [0, 1, 0])
right /= np.linalg.norm(right)
right = -right if right[0] < 0 else right
upv = np.cross(right, fwd)
upv = upv if upv[1] > 0 else -upv
FOV = math.radians(72)
TV = math.tan(FOV / 2)
ASP = W / H
ys, xs = np.mgrid[0:BH, 0:BW].astype(np.float64)
ndx = ((xs + 0.5) / BW * 2 - 1) * TV * ASP
ndy = (1 - (ys + 0.5) / BH * 2) * TV
dirs = fwd[None, None] + ndx[..., None] * right[None, None] + ndy[..., None] * upv[None, None]
dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)


def project(p):
    d = np.asarray(p, float) - ro
    zc = d @ fwd
    return np.array([((d @ right) / (zc * TV * ASP) + 1) / 2 * W, (1 - (d @ upv) / (zc * TV)) / 2 * H, zc])


T, N, ID, LP = trace(ro, dirs, c, hh, RR)
hit = ID >= 0
P = ro[None, None] + dirs * T[..., None]
lpos = np.array([l[0] for l in lights], float)
lcol = np.array([l[1] for l in lights], float)
lrad = np.array([l[2] for l in lights], float)
sun_dir = np.array([0.62, 0.30, 0.50])
sun_dir /= np.linalg.norm(sun_dir)
specw = np.where(mats == STONE, 1.0, np.where(mats == ROOF, 0.25, np.where(mats == WIN_DARK, 0.6, 0.0)))
E, S, AO = light(P, N, ID, dirs, c, hh, RR, lpos, lcol, lrad, sun_dir, np.array([1.25, 0.5, 0.38]),
                 4 if PREVIEW else 8, 1.1, 7, 60.0, specw.astype(np.float64))
print("traced")

# ------------------------------------------------------------------ materials
idc = np.clip(ID, 0, None)
mat = np.where(hit, mats[idc], 0)
tin = tint[idc]
nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
# face coordinates: (along, up) on the plane facing the normal's dominant axis
side = np.abs(nx) > 0.6
fu = np.where(side, P[..., 2], P[..., 0])
fv = np.where(np.abs(ny) > 0.6, P[..., 2], P[..., 1])
nz_small = hp.noise((BH, BW), 6, seed=31)
nz_big = hp.noise((BH, BW), 40, seed=32, octaves=3)


def brick_pattern(u, v, bw=0.26, bh=0.085):
    row = np.floor(v / bh)
    uu = u / bw + 0.5 * (row % 2)
    mu = np.abs(uu - np.round(uu)) * bw
    mv = np.abs(v / bh - np.round(v / bh)) * bh
    mortar = np.clip(1 - np.minimum(mu, mv) / 0.012, 0, 1)
    rnd = np.sin(np.floor(uu) * 12.9898 + row * 78.233) * 43758.5453
    rnd = rnd - np.floor(rnd)
    return mortar, rnd


alb = np.zeros((BH, BW, 3))
mort, rnd = brick_pattern(fu, fv)
brick_col = (np.array([0.46, 0.28, 0.2])[None, None] * (0.8 + 0.4 * rnd[..., None])) * (1 - 0.5 * mort[..., None]) \
    + mort[..., None] * np.array([0.42, 0.38, 0.34]) * 0.5
stain = hp.smoothstep(0.35, 0.8, nz_big)
plaster = tin * (0.86 + 0.22 * nz_small[..., None]) * (1 - 0.22 * stain[..., None])
streak = hp.noise((BH, BW), 3, seed=33)
streak = cv2.resize(cv2.resize(streak, (BW, BH // 12)), (BW, BH))
plaster = plaster * (0.88 + 0.18 * streak[..., None])
alb = np.where((mat == PLASTER)[..., None], plaster, alb)
alb = np.where((mat == BRICK)[..., None], brick_col * (0.75 + 0.35 * tin), alb)
wood = np.array([0.30, 0.18, 0.11]) * (0.75 + 0.5 * hp.noise((BH, BW), 3, seed=34)[..., None])
alb = np.where(np.isin(mat, [TIMBER, DOOR])[..., None], wood * np.where(mat == DOOR, 0.75, 1.0)[..., None] * tin, alb)
st_edge = np.clip(1 - np.abs(((P[..., 2] - Z0) / STEP_D) % 1 - 0.0) / 0.12, 0, 1)
stone = np.array([0.40, 0.40, 0.43]) * tin * (0.82 + 0.3 * nz_small[..., None]) * (1 + 0.15 * st_edge[..., None])
alb = np.where((mat == STONE)[..., None], stone, alb)
tiles = 0.75 + 0.25 * ((np.floor(fu / 0.22) + np.floor(fv / 0.18)) % 2)[..., None]
alb = np.where((mat == ROOF)[..., None], np.array([0.26, 0.15, 0.13]) * tin * tiles, alb)
alb = np.where((mat == IRON)[..., None], np.array([0.05, 0.05, 0.06]), alb)
alb = np.where((mat == TOWER)[..., None], np.array([0.46, 0.44, 0.48]) * (0.8 + 0.35 * nz_small[..., None]), alb)
# windows: frame and mullions from the local hit point
lu = np.where(side, LP[..., 2], LP[..., 0])
lv = LP[..., 1]
hu = np.where(side, hh[idc, 2], hh[idc, 0])
hv = hh[idc, 1]
frame = (np.abs(lu) > hu - 0.065) | (np.abs(lv) > hv - 0.065) | (np.abs(lu) < 0.028) | (np.abs(lv - 0.12) < 0.028)
win = np.isin(mat, [WIN_LIT, WIN_DARK])
alb = np.where((win & frame)[..., None], np.array([0.22, 0.14, 0.09]), alb)
alb = np.where((win & ~frame)[..., None], np.array([0.03, 0.05, 0.07]), alb)
emit = np.zeros((BH, BW, 3))
glass_lit = (mat == WIN_LIT) & ~frame
curtain = (np.abs(lu) > hu * 0.5) & glass_lit
emit = np.where(glass_lit[..., None], np.array([1.45, 0.62, 0.22]) * (0.8 + 0.35 * hp.smoothstep(0.6, -0.6, lv))[..., None],
                emit)
emit = np.where(curtain[..., None], np.array([0.55, 0.12, 0.08]) * (0.7 + 0.3 * np.cos(lu * 60))[..., None], emit)
emit = np.where((mat == LAMP)[..., None], np.array([6.0, 3.6, 1.5]), emit)
emit = np.where((mat == CLOCK)[..., None], np.array([2.6, 1.9, 1.0]), emit)

# ------------------------------------------------------------------ lighting
sky_up = np.array([0.20, 0.19, 0.34])
sky_side = np.array([0.06, 0.15, 0.20])
gnd = np.array([0.06, 0.05, 0.05])
canyon = np.clip(0.3 + 0.7 * (P[..., 1] - np.vectorize(ground)(np.clip(P[..., 2], -5, 40))) / 9.0, 0.3, 1.0)
amb = (sky_up * np.clip(ny, 0, 1)[..., None] + sky_side * (1 - np.abs(ny))[..., None]
       + gnd * np.clip(-ny, 0, 1)[..., None]) * (AO * canyon)[..., None]
col = alb * (E + amb * 1.6) + S * 0.9 + emit
# sky for misses
el = np.degrees(np.arcsin(np.clip(dirs[..., 1], -1, 1)))
sky_t = np.clip((el - 26) / 40, 0, 1)
sky = np.stack([np.interp(sky_t, [0, 0.25, 0.5, 0.75, 1.0], ch) for ch in
                np.array([[1.9, 1.15, 0.6], [1.6, 0.62, 0.55], [0.85, 0.32, 0.6], [0.32, 0.2, 0.5], [0.12, 0.1, 0.3]]).T], -1)
clouds = hp.noise((BH, BW), 26, seed=40, octaves=4)
sky *= (0.85 + 0.35 * hp.smoothstep(0.45, 0.8, clouds))[..., None]
col = np.where(hit[..., None], col, sky)
# fog (distance), warmer toward the bright end of the alley
dist = np.where(hit, T, 80.0)
fog_t = 1 - np.exp(-np.maximum(dist - 4.0, 0) * 0.042)
fog_col = hp.mix(np.broadcast_to(np.array([0.06, 0.15, 0.21]), col.shape), np.broadcast_to(np.array([0.62, 0.26, 0.40]), col.shape),
                 hp.smoothstep(0.35, 0.7, dirs[..., 1]))
col = np.where(hit[..., None], col * (1 - fog_t[..., None]) + fog_col * fog_t[..., None], col)
base_hdr = col.astype(np.float32)
np.save(ROOT + "/work/fc20_base.npy", base_hdr)
np.save(ROOT + "/work/fc20_geo.npy", np.dstack([T, N, ID.astype(np.float64), mat]).astype(np.float32))
print("shaded")


# ------------------------------------------------------------------ to display, then up to full res
def filmic(x):
    x = np.maximum(x, 0)
    a, b, c_, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c_ * x + d) + e), 0, 1)


def up(a):
    return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


hdr = up(base_hdr)
# bloom from lanterns and windows, done in HDR before tone mapping
hot = np.clip(hdr - 1.0, 0, None)
hdr = hdr + fblur(hot, 6) * 0.5 + fblur(hot, 22) * 0.45 + fblur(hot, 70) * 0.35
img = filmic(hdr * 0.85) ** (1 / 2.2)
hp.to_pil(img).resize((540, 960)).save(ROOT + "/work/fc20_base.jpg")

# ------------------------------------------------------------------ 2D painted actors at projected positions
X, Y = hp.grid()
fig_feet = project((FIG_X, fig_g, FIG_Z))
fig_head = project((FIG_X, fig_g + 1.78, FIG_Z))
fh = fig_feet[1] - fig_head[1]                     # figure height in px
fx0, fy0 = fig_feet[0], fig_feet[1]
lamp_s = project(fig_lamp)
print("figure px", fx0, fy0, fh, "lamp", lamp_s[:2])


def F(u, v):                                        # figure-local coords: u across (fraction of height), v up
    return (fx0 + u * fh, fy0 - v * fh)


# the figure stands on the steps: contact shadow plus a long shadow thrown back toward us by its own lantern
cast = Mask().poly([F(-0.2, 0.02), F(0.18, 0.02), F(0.02, -0.75), F(-0.42, -0.68)]).f
contact = Mask().ellipse(F(0.0, 0.0), (fh * 0.24, fh * 0.045)).f
shadow2d = np.clip(fblur(cast, fh * 0.05) * 0.55 + fblur(contact, fh * 0.02) * 0.75, 0, 0.85)
img = img * (1 - shadow2d)[..., None] + shadow2d[..., None] * np.array([0.02, 0.05, 0.07]) * 0.5
cloak = Mask()
cloak.poly([F(-0.16, 0.0), F(-0.21, 0.08), F(-0.2, 0.3), F(-0.17, 0.55), F(-0.15, 0.72), F(-0.12, 0.84),
            F(-0.07, 0.95), F(0.0, 1.0), F(0.07, 0.96), F(0.11, 0.86), F(0.13, 0.74), F(0.16, 0.56),
            F(0.19, 0.3), F(0.22, 0.08), F(0.18, 0.0), F(0.1, 0.03), F(0.03, -0.01), F(-0.05, 0.03), F(-0.11, -0.01)])
hood_shadow = Mask().poly([F(-0.07, 0.93), F(0.0, 0.985), F(0.07, 0.93), F(0.06, 0.85), F(-0.06, 0.85)])
arm = Mask().poly(tapered([F(0.13, 0.72), F(0.21, 0.56), F(0.26, 0.45)], fh * 0.09, fh * 0.06))
cm = np.clip(cloak.f + arm.f, 0, 1)
# cloak colour: deep plum/teal with the lantern's orange rim on the right edge, cool rim on the left
dxl = X - lamp_s[0]
rim_r = np.clip(cm - np.roll(cm, -int(max(3, fh * 0.02)), 1), 0, 1)
rim_l = np.clip(cm - np.roll(cm, int(max(3, fh * 0.015)), 1), 0, 1)
cloak_col = hp.vgrad([(0, "#2a2033"), (1, "#2a2033")])
cloak_col = cloak_col * (0.8 + 0.25 * hp.noise((H, W), 18, seed=41)[..., None])
lamp_glow = np.exp(-np.hypot(X - lamp_s[0], Y - lamp_s[1]) / (fh * 0.35))
cloak_col = cloak_col + lamp_glow[..., None] * hp.rgb("#ff7a2a") * 0.5
img = hp.mix(img, cloak_col, cm)
img = hp.mix(img, hp.rgb("#120c16"), hood_shadow.f * 0.9)
img = img + (fblur(rim_r, 1.2) * 1.3)[..., None] * hp.rgb("#ffb050") * (0.5 + lamp_glow[..., None])
img = img + (fblur(rim_l, 1.2) * 0.5)[..., None] * hp.rgb("#7fd0e0")
# the jack-o'-lantern in the figure's hand
pr = fh * 0.085
pk_all = Mask()
pk_face = Mask()
pumps = [(lamp_s[0], lamp_s[1], pr, True)]
for (px, pz, prr) in step_pumpkins:
    q = project((px, ground(pz) + prr * 0.82, pz))
    q2 = project((px + prr, ground(pz) + prr * 0.82, pz))
    pumps.append((q[0], q[1], abs(q2[0] - q[0]), True))
for (qx, qy, qr, lit) in pumps:
    for k, sx_ in ((-0.42, 0.62), (0.42, 0.62), (0.0, 0.7)):
        pk_all.ellipse((qx + k * qr, qy), (qr * sx_, qr * 0.8))
    e = qr * 0.18
    for s_ in (-1, 1):
        pk_face.poly([(qx + s_ * qr * 0.36, qy - qr * 0.3), (qx + s_ * qr * 0.36 - e, qy), (qx + s_ * qr * 0.36 + e, qy)])
    pk_face.poly([(qx - qr * 0.48, qy + qr * 0.16), (qx - qr * 0.24, qy + qr * 0.3), (qx, qy + qr * 0.2),
                  (qx + qr * 0.24, qy + qr * 0.3), (qx + qr * 0.48, qy + qr * 0.16), (qx + qr * 0.3, qy + qr * 0.5),
                  (qx, qy + qr * 0.56), (qx - qr * 0.3, qy + qr * 0.5)])
pkb = pk_all.f
pkf = pk_face.f
shade_pk = np.ones((H, W), np.float32)
for (qx, qy, qr, lit) in pumps:
    shade_pk *= 1 - 0.45 * np.clip(((X - qx) / qr) ** 2 + 0, 0, 1) * (np.hypot(X - qx, Y - qy) < qr * 1.2)
pk_col = hp.rgb("#e2661e") * (0.55 + 0.6 * shade_pk)[..., None]
img = hp.mix(img, pk_col, pkb)
img = hp.mix(img, hp.rgb("#ffd36a") * 1.15, pkf * pkb.clip(0, 1))
pglow = fblur(pkf, 4) * 0.8 + fblur(pkf, 18) * 0.6 + fblur(pkb * 0.3 + pkf, 60) * 0.5
img = img + pglow[..., None] * hp.rgb("#ff8a30") * 0.7
stem = Mask()
for (qx, qy, qr, lit) in pumps:
    stem.poly(tapered([(qx, qy - qr * 0.72), (qx + qr * 0.06, qy - qr * 0.95), (qx + qr * 0.18, qy - qr * 1.05)], qr * 0.2, qr * 0.1))
img = hp.mix(img, hp.rgb("#3a4a22"), stem.f)
# handle from the hand to the lantern
hand = F(0.26, 0.45)
hm = Mask().line([hand, (lamp_s[0], lamp_s[1] - pr * 0.8)], max(2, fh * 0.012))
img = hp.mix(img, hp.rgb("#1a1214"), hm.f)

# garland of small paper lanterns across the alley, and bats in the strip of sky
ga = project((-AW + 0.3, ground(9.3) + 4.9, 9.3))
gb = project((AW - 0.35, ground(10.1) + 4.7, 10.1))
gl_pts = [(ga[0] + (gb[0] - ga[0]) * t, ga[1] + (gb[1] - ga[1]) * t + 4 * t * (1 - t) * fh * 0.35) for t in np.linspace(0, 1, 40)]
gm = Mask().line(gl_pts, 2)
img = hp.mix(img, hp.rgb("#140e12"), gm.f * 0.9)
gls = Mask()
for k, t in enumerate(np.linspace(0.08, 0.92, 8)):
    x = ga[0] + (gb[0] - ga[0]) * t
    y = ga[1] + (gb[1] - ga[1]) * t + 4 * t * (1 - t) * fh * 0.35 + 16
    gls.ellipse((x, y), (11, 15))
glf = gls.f
img = hp.mix(img, hp.rgb("#ffb24a"), glf)
img = img + (fblur(glf, 8) * 0.6 + fblur(glf, 30) * 0.4)[..., None] * hp.rgb("#ff7a30")
skym = up((~hit).astype(np.float32))
mx_, my_, mr_ = 742.0, 560.0, 64.0
moon_d = np.hypot(X - mx_, Y - my_)
moon_m = np.clip(mr_ - moon_d + 0.5, 0, 1) * np.clip(skym * 1.5 - 0.25, 0, 1)
img = img + (np.exp(-np.maximum(moon_d - mr_, 0) / 60.0) * 0.35 * skym)[..., None] * hp.rgb("#ffb6a0")
img = hp.mix(img, hp.rgb("#fff0d6") * (0.92 + 0.1 * hp.noise((H, W), 9, seed=77))[..., None], moon_m)
bt = Mask()
for (bx, by, sp_, an, fl) in [(600, 520, 70, -0.2, 0.6), (690, 470, 44, 0.15, -0.5), (850, 560, 54, 0.1, 0.3),
                              (770, 400, 30, -0.1, 0.9), (930, 470, 36, 0.25, -0.7)]:
    bt.poly(bat_poly(bx, by, sp_, an, fl))
img = hp.mix(img, hp.rgb("#1a1020"), bt.f)
np.save(ROOT + "/work/fc20_pre.npy", img.astype(np.float32))
print("actors done")


# ------------------------------------------------------------------ stroke painting
@njit(cache=True, fastmath=True)
def paint(canvas, cx, cy, ang, ln, wd, colr, alpha, seed):
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
                br = 0.72 + 0.28 * math.sin(v * freq + ph + 0.15 * u / max(1.0, hw))
                tail = 1.0
                if u > hl * 0.6:
                    tail = 1.0 - 0.6 * (u - hl * 0.6) / (hl * 0.4 + hw + 1e-6)
                a = alpha * cov * br * tail
                if a <= 0:
                    continue
                for ch in range(3):
                    canvas[y, x, ch] = canvas[y, x, ch] * (1 - a) + colr[k, ch] * a


ref_full = img.astype(np.float32)
lum = ref_full @ np.array([0.3, 0.55, 0.15], np.float32)
canvas = hp.blur(ref_full, 10).copy()
for li, (rad, thr) in enumerate([(26, 0.0), (14, 0.035), (7, 0.045), (3.5, 0.06)]):
    ref = hp.blur(ref_full, rad * 0.35) if rad > 4 else ref_full
    gx_ = cv2.Sobel(hp.blur(lum, rad * 0.5), cv2.CV_32F, 1, 0, ksize=3)
    gy_ = cv2.Sobel(hp.blur(lum, rad * 0.5), cv2.CV_32F, 0, 1, ksize=3)
    Jxx, Jxy, Jyy = hp.blur(gx_ * gx_, rad), hp.blur(gx_ * gy_, rad), hp.blur(gy_ * gy_, rad)
    ori = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + math.pi / 2
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)
    err = cv2.blur(np.abs(canvas - ref).sum(axis=2) / 3, (int(rad), int(rad)))
    sp = rad * 0.8
    gyy, gxx = np.mgrid[0:H:sp, 0:W:sp]
    cxs = (gxx + rng.uniform(0, sp, gxx.shape)).ravel()
    cys = (gyy + rng.uniform(0, sp, gyy.shape)).ravel()
    keep = (cxs < W - 1) & (cys < H - 1)
    cxs, cys = cxs[keep], cys[keep]
    ix, iy = cxs.astype(int), cys.astype(int)
    if thr > 0:
        sel = err[iy, ix] > thr
        cxs, cys, ix, iy = cxs[sel], cys[sel], ix[sel], iy[sel]
    order = rng.permutation(len(cxs))
    cxs, cys, ix, iy = cxs[order], cys[order], ix[order], iy[order]
    a = ori[iy, ix] + rng.normal(0, 0.12, len(ix)) + (1 - np.clip(coh[iy, ix] * 3, 0, 1)) * rng.normal(0, 0.5, len(ix))
    lns = rad * rng.uniform(2.2, 4.2, len(ix)) * (0.6 + 0.8 * np.clip(coh[iy, ix] * 3, 0, 1))
    wds = rad * rng.uniform(1.4, 2.0, len(ix))
    cols = ref[iy, ix] * (1 + rng.normal(0, 0.035, (len(ix), 1))) + rng.normal(0, 0.012, (len(ix), 3))
    paint(canvas, cxs.astype(np.float64), cys.astype(np.float64), a.astype(np.float64), lns.astype(np.float64),
          wds.astype(np.float64), cols.astype(np.float64), 0.88, 1000 + li)
    print("layer", rad, len(ix), "strokes")
img = canvas

# ------------------------------------------------------------------ selective dark linework from the geometry
ysf, xsf = np.mgrid[0:H, 0:W].astype(np.float64)
dirs_f = (fwd[None, None] + (((xsf + 0.5) / W * 2 - 1) * TV * ASP)[..., None] * right[None, None]
          + ((1 - (ysf + 0.5) / H * 2) * TV)[..., None] * upv[None, None])
dirs_f /= np.linalg.norm(dirs_f, axis=2, keepdims=True)
Tfull, Nfull, IDfull, _ = trace(ro, dirs_f, c, hh, RR)
del dirs_f, ysf, xsf
Tf = np.where(IDfull >= 0, Tfull, 80.0).astype(np.float32)
logd = np.log(Tf + 0.5)
gxd = cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)
gyd = cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)
edge_d = np.hypot(gxd, gyd)
Nf = Nfull.astype(np.float32)
edge_n = sum(np.abs(cv2.Sobel(Nf[..., k], cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(Nf[..., k], cv2.CV_32F, 0, 1, ksize=3))
             for k in range(3))
lines = np.clip((edge_d - 0.12) * 4, 0, 1) + np.clip((edge_n - 1.2) * 0.8, 0, 1) * 0.7
lines *= hp.smoothstep(16, 6, Tf)                                       # only near/mid ground gets ink
lines *= hp.smoothstep(0.3, 0.55, hp.noise((H, W), 60, seed=50, octaves=2))     # broken, hand-made
lines = np.clip(hp.blur(lines, 0.8) * 1.3, 0, 1)
actors = np.clip(fblur(np.clip(cm + pkb + glf + gm.f, 0, 1), 1.5) * 1.5, 0, 1)
lines *= 1 - actors
fig_edge = np.clip(cm - cv2.erode(cm, np.ones((3, 3), np.uint8)), 0, 1) + np.clip(pkb - cv2.erode(pkb, np.ones((3, 3), np.uint8)), 0, 1)
lines = np.clip(lines + hp.blur(fig_edge, 0.7) * 0.9, 0, 1)
img = hp.mix(img, hp.rgb("#14161c"), lines * 0.75)

# ------------------------------------------------------------------ crisp painted highlights on step edges and rims
Sf = up(S.astype(np.float32))
spec_l = Sf.max(axis=2)
glint = np.clip(spec_l * 1.2, 0, 1)
glint = cv2.blur(glint, (9, 1))
img = img + (glint * (1 - actors))[..., None] * hp.rgb("#ffc070") * 0.55
Ef = up(E.astype(np.float32)).max(axis=2)
nosing = np.clip(-gyd * 3, 0, 1) * (mats[np.clip(IDfull, 0, None)] == STONE) * (IDfull >= 0) * np.clip(Ef * 0.6, 0, 1)
img = img + (hp.blur(nosing, 0.8) * (1 - actors))[..., None] * hp.rgb("#ffb873") * 0.6

# ------------------------------------------------------------------ colour design: teal shadows, orange/magenta light
lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
sh_w = hp.smoothstep(0.62, 0.08, lum)[..., None]
hi_w = hp.smoothstep(0.35, 0.85, lum)[..., None]
img = img * (1 - 0.5 * sh_w) + sh_w * 0.5 * img * np.array([0.5, 1.0, 1.18]) + sh_w * np.array([0.0, 0.03, 0.045])
img = img * (1 + hi_w * np.array([0.08, -0.01, -0.08]))
mean = img.mean(axis=2, keepdims=True)
img = mean + (img - mean) * 1.18
# canvas texture and vignette
canv = hp.noise((H, W), 2.2, seed=60) - 0.5
weave = (np.sin(X * 1.9) * np.sin(Y * 1.9)) * 0.5
img = img * (1 + (0.035 * canv + 0.012 * weave)[..., None])
img = hp.vignette(img, 0.38, 2.2)
img = soft_clip(img, 0.86)
hp.save(np.clip(img, 0, 1), 20, "painterly-alley")
