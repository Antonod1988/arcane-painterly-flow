"""32 · Arcane lighthouse: a lighthouse on a sea cliff in a storm, seen from wet rocks at the waterline. A giant
jack-o'-lantern fills the lantern room; its eyes throw two beams (painted in post from work/<tag>/scene.json).
Waves break on the foreground rocks, a keeper with a lantern stands on the gallery.

cd src && bash rt.sh 32_arcane_lighthouse.py 32-test pct=35 samples=16   (check=1: build only)
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/blender")
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import bl_build as K  # noqa: E402
import bl_common as B  # noqa: E402

A = B.args()
PCT = int(A.get("pct", 35))
SPP = int(A.get("samples", 16))
TAG = A.get("tag", "32-test")
WORK = ROOT + "/work"
rnd = random.Random(32)

sc = B.empty_scene()
B.cycles(samples=SPP, pct=PCT, bounces=4, clamp_indirect=3.0, transparent_bounces=16)

CAM = Vector((0.0, 0.0, float(A.get("camz", 2.0))))
LH = Vector((5.5, 30.0, 12.5))              # lighthouse base centre (cliff top)
TOP = 31.0                                   # gallery floor
CLIFF_C, CLIFF_R = (15.0, 44.0), 24.0       # crag centre and top radius


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def dir_noise(P, seed, octs=((1.6, 1.0), (3.4, 0.5), (7.0, 0.25), (14.0, 0.12)), n=8):
    """Sum-of-sinusoids noise on 3D points (cheap, smooth, isotropic enough for rocks)."""
    r = np.random.default_rng(seed)
    out = np.zeros(len(P))
    for f, a in octs:
        for _ in range(n):
            d = r.normal(size=3)
            d /= np.linalg.norm(d)
            out += a * np.sin(f * (P @ d) + r.uniform(0, 6.28)) / math.sqrt(n)
    return out


def grid_faces(nr, nc, wrap=False):
    f = []
    for j in range(nr - 1):
        for i in range(nc if wrap else nc - 1):
            a, b = j * nc + i, j * nc + (i + 1) % nc
            f.append((a, b, b + nc, a + nc))
    return f


# ---------------------------------------------------------------- materials
def rock_mat(name, base=(0.03, 0.03, 0.032), wet_top=3.5, grass=False):
    m, t = K.mat(name)
    geo = t.n("ShaderNodeNewGeometry")
    pos = geo.outputs["Position"]
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(pos, sep.inputs[0])
    n1 = t.n("ShaderNodeTexNoise", Scale=0.9, Detail=10.0, Roughness=0.65)
    n2 = t.n("ShaderNodeTexNoise", Scale=7.0, Detail=6.0, Roughness=0.6)
    for nn in (n1, n2):
        t.link(pos, nn.inputs["Vector"])
    col = t.mixc(t.mapr(n1.outputs["Fac"], 0.35, 0.7), base, tuple(c * 2.2 for c in base))
    col = t.mixc(t.mapr(n2.outputs["Fac"], 0.55, 0.75, 0, 0.6), col, (0.06, 0.07, 0.065))
    wet = t.mapr(sep.outputs["Z"], wet_top, wet_top - 2.5)                    # 1 near the water
    col = t.mixc(t.math("MULTIPLY", wet, 0.6), col, tuple(c * 0.45 for c in base))
    if grass:
        up = t.n("ShaderNodeSeparateXYZ")
        t.link(geo.outputs["Normal"], up.inputs[0])
        g = t.math("MULTIPLY", t.mapr(up.outputs["Z"], 0.55, 0.85), t.mapr(sep.outputs["Z"], 9.0, 11.5))
        col = t.mixc(g, col, (0.03, 0.04, 0.018))
    rough = t.mixf(wet, 0.72, 0.18)
    h = t.math("ADD", n1.outputs["Fac"], t.math("MULTIPLY", n2.outputs["Fac"], 0.5))
    t.out(t.bsdf(col, rough, normal=t.bump(h, 0.6, 0.12), spec=0.5))
    return m


ROCK = rock_mat("rock")
CLIFF = rock_mat("cliff", (0.05, 0.05, 0.052), wet_top=4.0, grass=True)
B.aov_mask(CLIFF, "cliff")


def plaster_mat():
    m, t = K.mat("plaster")
    geo = t.n("ShaderNodeNewGeometry")
    pos = geo.outputs["Position"]
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(pos, sep.inputs[0])
    z = sep.outputs["Z"]
    mp = t.n("ShaderNodeMapping")
    t.link(pos, mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (6.0, 6.0, 0.35)                       # vertical rain stains
    stain = t.n("ShaderNodeTexNoise", Scale=1.0, Detail=6.0, Roughness=0.6)
    t.link(mp.outputs[0], stain.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=2.5, Detail=8.0)
    t.link(pos, nz.inputs["Vector"])
    cream = t.mixc(t.mapr(stain.outputs["Fac"], 0.4, 0.75, 0, 0.75), (0.56, 0.52, 0.44), (0.24, 0.24, 0.22))
    band = t.math("ADD",
                  t.math("MULTIPLY", t.mapr(z, 18.4, 18.5), t.mapr(z, 20.6, 20.5)),
                  t.math("MULTIPLY", t.mapr(z, 23.6, 23.7), t.mapr(z, 25.6, 25.5)))
    col = t.mixc(band, cream, (0.13, 0.025, 0.02))
    t.out(t.bsdf(col, 0.7, normal=t.bump(nz.outputs["Fac"], 0.25, 0.05)))
    return m


PLASTER = plaster_mat()
IRON = B.principled("iron", (0.018, 0.02, 0.022), rough=0.4, metal=0.6)
PATINA = B.principled("patina", (0.05, 0.1, 0.085), rough=0.45, metal=0.3)
GLASS, t = K.mat("glass")                       # clear: mostly transparent, a faint mirror sheen, no diffuse
gm = t.n("ShaderNodeMixShader")
gm.inputs[0].default_value = float(A.get("sheen", 0.07))
t.link(t.n("ShaderNodeBsdfTransparent").outputs[0], gm.inputs[1])
t.link(t.bsdf((1.0, 1.0, 1.0), 0.04, metal=1.0), gm.inputs[2])
t.out(gm.outputs[0])
SLATE = B.principled("slate", (0.02, 0.024, 0.03), rough=0.5)
WALL = B.principled("cottage", (0.3, 0.28, 0.25), rough=0.85)
WARM = B.emission("window", (1.0, 0.55, 0.2), 7.0)
SPRAY, t = K.mat("spray")
t.out(t.bsdf((0.86, 0.92, 0.95), 0.55, subsurface=0.35, ss_radius=(0.4, 0.45, 0.5), spec=0.4))
B.aov_mask(SPRAY, "spray")
COAT = B.principled("coat", (0.55, 0.34, 0.03), rough=0.3)
DARK = B.principled("boots", (0.02, 0.018, 0.016), rough=0.6)
SKIN = B.principled("skin", (0.45, 0.28, 0.2), rough=0.6)
LAMPGL = B.emission("lamp", (1.0, 0.6, 0.25), 25.0)
for m in (COAT, DARK, SKIN, LAMPGL):
    B.aov_mask(m, "actor")
    B.aov_mask(m, "keeper")


# ---------------------------------------------------------------- geometry helpers
def sphere(mb, c, r, mat, nu=12, nv=8, sz=1.0):
    vs = [(c[0], c[1], c[2] - r * sz)]
    for j in range(1, nv):
        ph = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            vs.append((c[0] + r * math.sin(ph) * math.cos(th), c[1] + r * math.sin(ph) * math.sin(th),
                       c[2] - r * sz * math.cos(ph)))
    vs.append((c[0], c[1], c[2] + r * sz))
    fs = [(0, 1 + (i + 1) % nu, 1 + i) for i in range(nu)]
    for j in range(nv - 2):
        for i in range(nu):
            a, b = 1 + j * nu + i, 1 + j * nu + (i + 1) % nu
            fs.append((a, b, b + nu, a + nu))
    top = len(vs) - 1
    o = 1 + (nv - 2) * nu
    fs += [(o + i, o + (i + 1) % nu, top) for i in range(nu)]
    mb.add(vs, fs, mat)


def rock(c, size, seed, nu=56, nv=30, amp=0.34, flat=0.3):
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    ph = np.linspace(0.02, np.pi - 0.02, nv)
    TH, PH = np.meshgrid(th, ph)
    d = np.stack([np.sin(PH) * np.cos(TH), np.sin(PH) * np.sin(TH), np.cos(PH)], -1).reshape(-1, 3)
    rad = 1 + amp * dir_noise(d, seed) / 1.4
    p = d * rad[:, None] * np.array(size) / 2
    top = size[2] / 2 * (1 - flat)
    p[:, 2] = np.where(p[:, 2] > top, top + (p[:, 2] - top) * 0.35, p[:, 2])
    p += np.array(c)
    f = grid_faces(nv, nu, wrap=True)
    f.append(tuple(range(nu - 1, -1, -1)))
    f.append(tuple(range((nv - 1) * nu, nv * nu)))
    return p.tolist(), f


def crag(cx, cy, z0, z1, r_top, r_bot, seed, nth=128, nz=90, amp=0.22, ledges=0.05, cap=True, zj=0.08):
    """Rock column with layered radial noise (from 02_fog_manor.py)."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    zs = np.linspace(z0, z1, nz)
    R = np.zeros((nz, nth))
    for cells, a in ((7, 1.0), (15, 0.6), (31, 0.35), (63, 0.18)):
        gz = max(3, int(cells * (z1 - z0) / 60))
        g = rng.standard_normal((gz + 1, cells))
        zi = np.linspace(0, gz, nz)
        ti = np.linspace(0, cells, nth, endpoint=False)
        z0i = np.floor(zi).astype(int).clip(0, gz - 1)
        fz = (zi - z0i)[:, None]
        t0i = np.floor(ti).astype(int)
        ft = (ti - t0i)[None, :]
        t1i = (t0i + 1) % cells
        fz, ft = fz * fz * (3 - 2 * fz), ft * ft * (3 - 2 * ft)
        v = (g[z0i][:, t0i] * (1 - ft) + g[z0i][:, t1i] * ft) * (1 - fz) + \
            (g[z0i + 1][:, t0i] * (1 - ft) + g[z0i + 1][:, t1i] * ft) * fz
        R += a * v
    R = R / 1.6
    frac = (zs - z0) / (z1 - z0)
    base = r_bot + (r_top - r_bot) * frac ** 0.7
    strata = 1 + ledges * np.sign(np.sin(zs * 1.3 + rng.uniform(0, 6)))[:, None] * (np.abs(np.sin(zs * 1.3)) > 0.3)[:, None]
    rad = base[:, None] * (1 + amp * R) * strata
    rad[-1] *= 0.97
    X = cx + rad * np.cos(th)[None, :]
    Y = cy + rad * np.sin(th)[None, :]
    Z = np.repeat(zs[:, None], nth, axis=1) + rng.normal(0, zj, (nz, nth))
    Z[-1] = z1 + 0.35 * (np.sin(th * 9 + rng.uniform(0, 6)) + 0.6 * np.sin(th * 23 + rng.uniform(0, 6))) + 0.25
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).tolist()
    faces = grid_faces(nz, nth, wrap=True)
    if cap:
        verts.append((cx, cy, z1))
        c = len(verts) - 1
        top = (nz - 1) * nth
        faces += [(top + i, top + (i + 1) % nth, c) for i in range(nth)]
    return verts, faces


# ---------------------------------------------------------------- cliff, stacks, far headland
v, f = crag(CLIFF_C[0], CLIFF_C[1], -6.0, LH.z, CLIFF_R, CLIFF_R + 3.0, seed=7, amp=0.16, ledges=0.035,
            nth=int(A.get("cnth", 46)), nz=int(A.get("cnz", 15)), zj=0.45)
B.mesh("cliff", v, f, CLIFF, smooth=False)
for i, (x, y, top, rt, rb, nth) in enumerate(((-21, 74, 15, 2.6, 6.5, 48), (-46, 118, 22, 3.5, 8.0, 48),
                                              (-12, 150, 10, 2.2, 5.0, 40), (-95, 260, 30, 9.0, 20.0, 64))):
    v, f = crag(x, y, -6.0, top, rt, rb, seed=20 + i, amp=0.25, ledges=0.04, nth=nth, nz=50)
    B.mesh(f"stack{i}", v, f, CLIFF, smooth=True)
v, f = crag(-230, 620, -10.0, 55, 60.0, 120.0, seed=31, amp=0.18, nth=96, nz=40)
B.mesh("headland", v, f, CLIFF, smooth=True)

# ---------------------------------------------------------------- foreground rocks (camera stands on one)
ROCKS = [((-4.6, 14.0, 0.0), (4.2, 3.2, 2.8)), ((2.8, 6.6, -0.1), (3.8, 2.9, 1.5)), ((6.2, 11.5, -0.3), (4.4, 3.4, 2.6)),
         ((-7.6, 12.5, -0.4), (3.2, 2.8, 2.8)), ((-0.6, 10.8, -0.5), (1.6, 1.4, 0.9)), ((10.5, 16.0, -0.3), (3.0, 2.4, 2.0)),
         ((-11.0, 8.0, -0.2), (2.6, 2.2, 1.8)), ((1.2, 15.5, -0.6), (1.8, 1.5, 1.0))]
for i, (c, s) in enumerate(ROCKS):
    v, f = rock(c, s, 100 + i)
    B.mesh(f"rock{i}", v, f, ROCK, smooth=True)

# ---------------------------------------------------------------- the storm sea (polar grid around the camera)
nr, na = 380, 300
rr = np.geomspace(1.2, 3200, nr)
aa = np.radians(np.linspace(-50, 55, na))
RR, AA = np.meshgrid(rr, aa, indexing="ij")
X = CAM.x + RR * np.sin(AA)
Y = CAM.y + RR * np.cos(AA)
dR = RR * (rr[1] / rr[0] - 1)
near = 0.3 + 0.7 * smooth(4.0, 18.0, RR)
Hs = np.zeros_like(RR)
asum = np.zeros_like(RR)
wr = np.random.default_rng(5)
WAVE = float(A.get("wave", 1.0))
for lam, a, ang in ((46, 0.62, 0), (29, 0.45, -18), (17, 0.3, 22), (10, 0.18, -8), (6, 0.1, 35), (3.2, 0.05, 12)):
    a *= WAVE
    th = math.radians(-60 + ang)
    k = 2 * np.pi / lam
    ph = k * (X * math.cos(th) + Y * math.sin(th)) + wr.uniform(0, 6.28)
    ph += 0.6 * np.sin(k * 0.37 * (X * math.sin(th) - Y * math.cos(th)) + wr.uniform(0, 6.28))   # bend the crests
    keep = smooth(lam / 2.5, lam / 6.0, dR)
    Hs += a * (np.cos(ph) + 0.32 * np.cos(2 * ph)) * keep * near
    asum += a * keep * near
foam = smooth(0.42, 1.0, Hs / (asum * 1.32 + 1e-3)) * 0.85
for (c, s) in ROCKS:
    d = np.hypot(X - c[0], Y - c[1]) - max(s[0], s[1]) * 0.5
    foam += 0.75 * smooth(2.2, 0.0, d)
dcl = np.hypot(X - CLIFF_C[0], Y - CLIFF_C[1]) - (CLIFF_R + 2.5)
foam += 1.3 * smooth(4.5, 0.0, dcl + 0.9)
for (x, y, rb) in ((-21, 74, 6.5), (-46, 118, 8.0), (-12, 150, 5.0)):
    foam += smooth(6.0, 0.0, np.hypot(X - x, Y - y) - rb)
Z = Hs - 0.15
verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
me = bpy.data.meshes.new("sea")
me.from_pydata(verts.tolist(), [], grid_faces(nr, na))
me.validate()
at = me.attributes.new("foam", "FLOAT", "POINT")
at.data.foreach_set("value", np.clip(foam, 0, 1.5).astype(np.float32).ravel())
me.shade_smooth()
sea = bpy.data.objects.new("sea", me)
B.link(sea)
WATER, t = K.mat("water")
pos = t.n("ShaderNodeNewGeometry").outputs["Position"]
fa = t.n("ShaderNodeAttribute", _attribute_type="GEOMETRY", _attribute_name="foam")
mp = t.n("ShaderNodeMapping")
t.link(pos, mp.inputs["Vector"])
mp.inputs["Scale"].default_value = (0.5, 1.6, 1.0)                           # wind streaks
nf = t.n("ShaderNodeTexNoise", Scale=1.3, Detail=10.0, Roughness=0.62)
t.link(mp.outputs[0], nf.inputs["Vector"])
chop = t.n("ShaderNodeTexNoise", Scale=1.8, Detail=8.0, Roughness=0.6)
t.link(pos, chop.inputs["Vector"])
fm = t.mapr(t.math("MULTIPLY", fa.outputs["Fac"], t.mapr(nf.outputs["Fac"], 0.3, 0.7, 0.35, 1.35)), 0.55, 0.85)
bump = t.bump(chop.outputs["Fac"], 0.25, 0.25)
water = t.bsdf((0.003, 0.014, 0.016), 0.07, normal=bump, spec=0.5)
foamsh = t.bsdf((0.7, 0.78, 0.8), 0.8, normal=bump)
mix = t.n("ShaderNodeMixShader")
t.link(fm, mix.inputs[0])
t.link(water, mix.inputs[1])
t.link(foamsh, mix.inputs[2])
t.out(mix.outputs[0])
sea.data.materials.append(WATER)

# ---------------------------------------------------------------- spray plumes (clusters of foam balls)
sp = K.MB()


def plume(base, height, width, n, seed, lean=(-1.0, 0.0)):
    r = random.Random(seed)
    for _ in range(n):
        u = r.random() ** 0.9
        z = base[2] + u * height
        spread = width * (0.2 + 1.0 * u)
        a = r.uniform(0, 2 * math.pi)
        rr_ = spread * math.sqrt(r.random())
        x = base[0] + rr_ * math.cos(a) + lean[0] * u * u * height * 0.3
        y = base[1] + rr_ * math.sin(a) * 0.6 + lean[1] * u * u * height * 0.3
        rad = width * (0.022 + 0.12 * r.random() ** 3) * (1.25 - u)
        sphere(sp, (x, y, z), max(0.03, rad), 0, nu=8, nv=6, sz=r.uniform(0.75, 1.2))


plume((-4.8, 15.8, 0.3), float(A.get("splash", 7.5)), 2.6, 1500, 1, lean=(-1.2, 0.2))
plume((-2.6, 15.3, 0.2), 3.5, 1.4, 220, 2, lean=(-0.6, 0.0))
plume((-7.5, 29.6, 0.0), 5.5, 2.2, 600, 3, lean=(-1.0, 0.0))
for k in range(13):                                            # surf along the cliff foot
    a = math.radians(196 + k * 8.5 + rnd.uniform(-2, 2))
    rb = CLIFF_R + 3.2
    plume((CLIFF_C[0] + rb * math.cos(a), CLIFF_C[1] + rb * math.sin(a), 0.0), rnd.uniform(0.8, 2.2), 1.5, 110, 40 + k, lean=(-0.6, 0.0))
for k in range(7):                                             # a wave breaking along the reef in front of the cliff
    u = k / 6
    plume((-10.0 + 11.0 * u, 21.5 - 2.5 * u, 0.0), 1.3 + 1.4 * math.sin(u * 3.0 + 0.4), 1.6, 160, 10 + k, lean=(-0.8, 0.0))
sp.build("spray", [SPRAY], smooth=True)

# ---------------------------------------------------------------- lighthouse
lh = K.MB()   # 0 plaster, 1 iron, 2 patina, 3 glass, 4 window glow


def r_at(z):
    return 3.3 - (z - LH.z) / (TOP - LH.z) * 0.8


# tower (tapered, 48 sides) up to the pumpkin; a gallery for the keeper a little below it
GZ = TOP - float(A.get("gdrop", 2.8))                                        # gallery floor
GR = r_at(GZ) + 0.75                                                         # railing radius
lh.cyl((LH.x, LH.y, LH.z - 1.5), (LH.x, LH.y, TOP), r_at(LH.z - 1.5), r_at(TOP), n=48, mat=0)
for i in range(28):
    a = 2 * math.pi * i / 28
    rr_ = r_at(GZ - 0.6)
    lh.box((LH.x + (rr_ + 0.2) * math.cos(a), LH.y + (rr_ + 0.2) * math.sin(a), GZ - 0.62), (0.45, 0.22, 0.6), 1, rz=a)
lh.cyl((LH.x, LH.y, GZ - 0.3), (LH.x, LH.y, GZ), GR + 0.12, GR + 0.12, n=72, mat=1)            # gallery floor
for i in range(48):                                                                              # railing posts
    a = 2 * math.pi * i / 48
    p = (LH.x + GR * math.cos(a), LH.y + GR * math.sin(a))
    lh.cyl((p[0], p[1], GZ), (p[0], p[1], GZ + 1.1), 0.035, n=5, mat=1, caps=False)
for zr in (GZ + 0.55, GZ + 1.1):                                                                 # rails
    pts = [(LH.x + GR * math.cos(2 * math.pi * i / 72), LH.y + GR * math.sin(2 * math.pi * i / 72), zr) for i in range(73)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        lh.cyl(p0, p1, 0.04, n=5, mat=1, caps=False)
lh.cyl((LH.x, LH.y, TOP - 0.25), (LH.x, LH.y, TOP + 0.05), r_at(TOP) + 0.25, r_at(TOP) + 0.25, n=48, mat=1)   # iron crown
# the lantern is a giant jack-o'-lantern overhanging the tower top, held by an iron cage
PR = float(A.get("pr", 3.5))
PZ = TOP + PR * 0.72 - 0.35
FACE_DIR = math.radians(float(A.get("face", 226)))                           # world azimuth the face looks at


def adiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


for k in range(12):                                                                              # cage meridians
    a = FACE_DIR + 2 * math.pi * (k + 0.5) / 12
    if adiff(a, FACE_DIR) < math.radians(40):
        continue
    pts = []
    for ph in np.linspace(0.32, math.pi - 0.3, 16):
        rr_ = PR * math.sin(ph) * 0.98 + 0.06
        pts.append((LH.x + rr_ * math.cos(a), LH.y + rr_ * math.sin(a), PZ + PR * 0.72 * math.cos(ph) * 1.02))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        lh.cyl(p0, p1, 0.055, n=5, mat=1, caps=False)
for zr_, rr_ in ((PZ - PR * 0.45, PR * 0.9 + 0.07), (PZ + PR * 0.42, PR * 0.92 + 0.07)):         # cage hoops, open at the face
    pts = []
    for i in range(73):
        a = FACE_DIR + math.radians(42) + (2 * math.pi - math.radians(84)) * i / 72
        pts.append((LH.x + rr_ * math.cos(a), LH.y + rr_ * math.sin(a), zr_))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        lh.cyl(p0, p1, 0.05, n=5, mat=1, caps=False)
rod0 = (LH.x + 0.55, LH.y + 0.3, PZ + PR * 0.5)
lh.cyl(rod0, (rod0[0], rod0[1], PZ + PR * 1.45), 0.045, n=5, mat=1)
sphere(lh, (rod0[0], rod0[1], PZ + PR * 1.05), 0.3, 2, nu=12, nv=8)
half = [(0.0, -0.13), (0.035, -0.235), (0.06, -0.12), (0.1, -0.07), (0.32, -0.2), (0.62, -0.3), (1.0, -0.36),
        (0.86, -0.1), (0.74, 0.02), (0.6, -0.02), (0.47, 0.1), (0.33, 0.04), (0.2, 0.15), (0.09, 0.09), (0.05, 0.2), (0.0, 0.25)]
bat = [(-x, y) for x, y in reversed(half)] + half
va = math.radians(30)
bv = []
for x, y in bat:
    for off in (-0.02, 0.02):
        bv.append((rod0[0] + x * 0.6 * math.cos(va) - off * math.sin(va), rod0[1] + x * 0.6 * math.sin(va) + off * math.cos(va),
                   PZ + PR * 1.4 - y * 0.6))
nb = len(bat)
lh.add(bv, [tuple(2 * i for i in range(nb))[::-1], tuple(2 * i + 1 for i in range(nb))] +
       [(2 * i, 2 * ((i + 1) % nb), 2 * ((i + 1) % nb) + 1, 2 * i + 1) for i in range(nb)], 1)
# small warm windows up the tower, on the side facing the camera
face_a = math.atan2(CAM.y - LH.y, CAM.x - LH.x)
for k, (z, da) in enumerate(((16.2, 0.12), (22.1, -0.1), (27.0, 0.16), (GZ + 1.0, -0.42))):
    a = face_a + da
    rr_ = r_at(z)
    lh.box((LH.x + rr_ * math.cos(a), LH.y + rr_ * math.sin(a), z), (0.16, 0.95, 1.95) if k == 3 else (0.16, 0.62, 1.05), 4, rz=a)
    lh.box((LH.x + (rr_ + 0.02) * math.cos(a), LH.y + (rr_ + 0.02) * math.sin(a), z + 0.6), (0.2, 0.8, 0.12), 1, rz=a)
lh.build("lighthouse", [PLASTER, IRON, PATINA, GLASS, WARM], smooth=False)

# keeper's cottage beside the tower
ck = K.MB()
cx_, cy_ = LH.x + 7.0, LH.y + 3.5
ck.boxr(cx_ - 3.2, cx_ + 3.2, cy_ - 2.4, cy_ + 2.4, LH.z - 1, LH.z + 3.2, 0)
ck.add([(cx_ - 3.5, cy_ - 2.8, LH.z + 3.2), (cx_ + 3.5, cy_ - 2.8, LH.z + 3.2), (cx_ + 3.5, cy_, LH.z + 5.4), (cx_ - 3.5, cy_, LH.z + 5.4),
        (cx_ - 3.5, cy_ + 2.8, LH.z + 3.2), (cx_ + 3.5, cy_ + 2.8, LH.z + 3.2)],
       [(0, 1, 2, 3), (3, 2, 5, 4), (0, 3, 4), (1, 5, 2)], 1)
ck.boxr(cx_ + 1.6, cx_ + 2.3, cy_ - 0.6, cy_ + 0.1, LH.z + 3.5, LH.z + 6.6, 0)
for wx in (cx_ - 1.6, cx_ + 0.2, cx_ + 2.0):
    ck.boxr(wx - 0.4, wx + 0.4, cy_ - 2.46, cy_ - 2.38, LH.z + 1.0, LH.z + 2.2, 2)
ck.build("cottage", [WALL, SLATE, WARM])

# ---------------------------------------------------------------- the giant jack-o'-lantern
rz = FACE_DIR + math.pi / 2                                                  # local -Y -> FACE_DIR
jack = K.jack("giant_jack", (LH.x, LH.y, PZ), PR, f"{WORK}/face_classic.png", rz=rz, seed=3,
              light_w=float(A.get("lamp", 1500)), strength=float(A.get("glow", 30)))
jm = jack.data.materials[0]
for nd in jm.node_tree.nodes:
    if nd.type == "BSDF_PRINCIPLED":
        nd.inputs["Emission Color"].default_value = (1.0, 0.33, 0.05, 1)
        nd.inputs["Emission Strength"].default_value = float(A.get("skin", 0.18))
B.aov_mask(jm, "actor")
jl = bpy.data.objects["giant_jack_candle"]
jl.data.shadow_soft_size = 0.6
jl.data.color = (1.0, 0.5, 0.16)


def local_to_world(p):
    x, y, z = p
    c, s = math.cos(rz), math.sin(rz)
    return (LH.x + x * c - y * s, LH.y + x * s + y * c, PZ + z)


EYES = [local_to_world((sx * 0.34 * PR, -0.9 * PR, 0.22 * PR)) for sx in (-1, 1)]

# ---------------------------------------------------------------- the keeper on the gallery
kp = K.MB()     # 0 coat, 1 boots, 2 skin, 3 lamp
ka = math.radians(float(A.get("keeper", 238)))
kx, ky, kz = LH.x + (GR - 0.42) * math.cos(ka), LH.y + (GR - 0.42) * math.sin(ka), GZ
hd = math.radians(float(A.get("heading", 235)))
hx, hy = math.cos(hd), math.sin(hd)
sx, sy = -hy, hx
for s in (-1, 1):
    kp.cyl((kx + s * 0.09 * sx, ky + s * 0.09 * sy, kz), (kx + s * 0.09 * sx, ky + s * 0.09 * sy, kz + 0.8), 0.075, n=8, mat=1)
kp.cyl((kx, ky, kz + 0.38), (kx, ky, kz + 1.42), 0.3, 0.19, n=14, mat=0)
sphere(kp, (kx, ky, kz + 1.42), 0.2, 0, nu=12, nv=8, sz=0.6)
sphere(kp, (kx + 0.03 * hx, ky + 0.03 * hy, kz + 1.6), 0.115, 2, nu=12, nv=8)
kp.cyl((kx, ky, kz + 1.68), (kx, ky, kz + 1.71), 0.25, n=16, mat=0)
kp.cyl((kx, ky, kz + 1.68), (kx, ky, kz + 1.85), 0.135, 0.11, n=14, mat=0)
sh = (kx + 0.17 * sx, ky + 0.17 * sy, kz + 1.36)
hand = (sh[0] + 0.55 * hx + 0.1 * sx, sh[1] + 0.55 * hy + 0.1 * sy, kz + 1.25)
kp.cyl(sh, hand, 0.06, 0.05, n=8, mat=0)
kp.cyl((kx - 0.17 * sx, ky - 0.17 * sy, kz + 1.36), (kx - 0.22 * sx + 0.1 * hx, ky - 0.22 * sy + 0.1 * hy, kz + 0.9), 0.06, 0.05, n=8, mat=0)
lamp_c = (hand[0], hand[1], hand[2] - 0.24)
kp.cyl(hand, (lamp_c[0], lamp_c[1], lamp_c[2] + 0.12), 0.01, n=4, mat=1)
kp.box(lamp_c, (0.16, 0.16, 0.24), 3)
kp.build("keeper", [COAT, DARK, SKIN, LAMPGL])
L = B.light("POINT", lamp_c, float(A.get("keeplamp", 12)), color=(1.0, 0.55, 0.2), size=0.05, name="keeper_lamp")

# ---------------------------------------------------------------- sky (storm clouds, a magenta break) and light
BAZ, BEL = math.radians(float(A.get("baz", -20))), math.radians(float(A.get("bel", 50)))
BREAK = Vector((math.sin(BAZ) * math.cos(BEL), math.cos(BAZ) * math.cos(BEL), math.sin(BEL))).normalized()
w = sc.world
wn = K.NT(w)
gen = wn.n("ShaderNodeTexCoord").outputs["Generated"]
sep = wn.n("ShaderNodeSeparateXYZ")
wn.link(gen, sep.inputs[0])
el = sep.outputs["Z"]
mp = wn.n("ShaderNodeMapping")
wn.link(gen, mp.inputs["Vector"])
mp.inputs["Scale"].default_value = (1.0, 1.0, 2.4)
n1 = wn.n("ShaderNodeTexNoise", Scale=2.2, Detail=12.0, Roughness=0.62, Distortion=0.3)
n2 = wn.n("ShaderNodeTexNoise", Scale=8.0, Detail=6.0, Roughness=0.55)
for nn in (n1, n2):
    wn.link(mp.outputs[0], nn.inputs["Vector"])
cloud = wn.mapr(wn.math("ADD", n1.outputs["Fac"], wn.math("MULTIPLY", wn.math("SUBTRACT", n2.outputs["Fac"], 0.5), 0.4)), 0.42, 0.6)
sky = wn.ramp(el, [(0.0, (0.09, 0.15, 0.16)), (0.1, (0.04, 0.08, 0.095)), (0.4, (0.012, 0.028, 0.038)), (0.9, (0.004, 0.009, 0.016))])
dark = wn.n("ShaderNodeVectorMath", _operation="SCALE")
wn.link(sky, dark.inputs[0])
dark.inputs["Scale"].default_value = 0.25
gap = wn.n("ShaderNodeVectorMath", _operation="SCALE")
wn.link(sky, gap.inputs[0])
gap.inputs["Scale"].default_value = 1.6
col = wn.mixc(cloud, gap.outputs["Vector"], dark.outputs["Vector"])
dot = wn.n("ShaderNodeVectorMath", _operation="DOT_PRODUCT")
wn.link(gen, dot.inputs[0])
dot.inputs[1].default_value = tuple(BREAK)
nb = wn.n("ShaderNodeTexNoise", Scale=3.2, Detail=8.0, Roughness=0.62)
wn.link(mp.outputs[0], nb.inputs["Vector"])
dotn = wn.math("ADD", dot.outputs["Value"], wn.math("MULTIPLY", wn.math("SUBTRACT", nb.outputs["Fac"], 0.5), float(A.get("bnoise", 0.2))))
bm = wn.mapr(dotn, math.cos(math.radians(float(A.get("bsize", 17)))), math.cos(math.radians(2)))
bm = wn.math("POWER", bm, 1.4)
glow_k = wn.math("MULTIPLY", bm, wn.mapr(cloud, 0.0, 1.0, 1.0, 0.12))
magenta = wn.n("ShaderNodeVectorMath", _operation="SCALE")
magenta.inputs[0].default_value = (0.55, 0.08, 0.32)
wn.link(wn.math("MULTIPLY", glow_k, float(A.get("brk", 0.6))), magenta.inputs["Scale"])
add = wn.n("ShaderNodeVectorMath", _operation="ADD")
wn.link(col, add.inputs[0])
wn.link(magenta.outputs["Vector"], add.inputs[1])
bg = wn.n("ShaderNodeBackground", Strength=float(A.get("skyk", 1.0)))
wn.link(add.outputs["Vector"], bg.inputs["Color"])
wo = wn.n("ShaderNodeOutputWorld")
wn.link(bg.outputs[0], wo.inputs["Surface"])
B.light("SUN", tuple(LH + BREAK * 100), float(A.get("key", 0.3)), color=(1.0, 0.45, 0.7), size=math.radians(12),
        target=tuple(LH), name="break_key")
B.light("SUN", (-100, 40, 45), float(A.get("fill", 1.2)), color=(0.45, 0.78, 0.85), size=math.radians(20), target=(0, 40, 0), name="sea_fill")

glint_coll = bpy.data.collections.new("glint_receivers")
sc.collection.children.link(glint_coll)
for ob in [o for o in bpy.data.objects if o.name.startswith(("sea", "rock", "cliff", "spray"))]:
    glint_coll.objects.link(ob)
gl_ = B.light("AREA", (LH.x, LH.y, PZ), float(A.get("glint", 60000)), color=(1.0, 0.5, 0.18), size=6.0, target=tuple(CAM), name="glint")
gl_.data.shape = "DISK"
gl_.light_linking.receiver_collection = glint_coll

# ---------------------------------------------------------------- camera, metadata, render
TGT = Vector((float(A.get("tx", 2.4)), 30.0, float(A.get("tz", 15.0))))
LENS = float(A.get("lens", 24))
SHIFT = float(A.get("shift", 0.0))
cam = B.camera(tuple(CAM), tuple(TGT), lens=LENS, shift_y=SHIFT)
bpy.context.view_layer.update()
fd = Vector((math.cos(FACE_DIR), math.sin(FACE_DIR), 0.0))
beams = []
for k, (sgn, elv) in enumerate(((-1, float(A.get("el0", 7))), (1, float(A.get("el1", 2))))):
    a = FACE_DIR + math.radians(sgn * float(A.get("spread", 7.0)))
    beams.append([math.cos(a) * math.cos(math.radians(elv)), math.sin(a) * math.cos(math.radians(elv)), math.sin(math.radians(elv))])
info = {"cam": [list(r) for r in cam.matrix_world], "lens": LENS, "sensor": 36.0, "shift_x": 0.0, "shift_y": SHIFT,
        "eyes": [list(e) for e in EYES], "beams": beams, "jack": [LH.x, LH.y, PZ], "jack_r": PR,
        "keeper_lamp": list(lamp_c), "splash": [-3.9, 9.8, 4.0], "break": list(BREAK)}
os.makedirs(f"{WORK}/{TAG}", exist_ok=True)
with open(f"{WORK}/{TAG}/scene.json", "w") as fh:
    json.dump(info, fh, indent=1)
B.passes(mist=(1.0, 300.0, "LINEAR"), extra=("z", "normal"))
if "border" in A:                                   # x0,y0,x1,y1 fractions (y from the bottom), for 1:1 checks
    bx = [float(q) for q in A["border"].split(",")]
    sc.render.use_border, sc.render.use_crop_to_border = True, True
    sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = bx
if "check" in A:
    nv = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == "MESH")
    print("CHECK objects", len(bpy.data.objects), "verts", nv, "lights", len(bpy.data.lights), "eyes", EYES)
    sys.exit(0)
B.render(TAG, outputs=("Image", "Depth", "Normal", "actor", "spray", "keeper", "cliff"))
