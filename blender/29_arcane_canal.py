"""29 arcane-canal: an undercity canal at night, built for the Arcane-like painted finish (src/arcane.py).

A hooded ferryman poles a punt full of pumpkins away from us between tall jettied canal houses, toward a magenta
neon crescent over a tunnel mouth. Jack-o'-lanterns hang from arch bridges and wall brackets and float on the
water; teal haze lies on the canal, a big moon sits in the strip of sky.

    python src/gpu_run.py blender/29_arcane_canal.py -- tag=29-test pct=35 samples=16
    blender -b --factory-startup --python blender/29_arcane_canal.py -- check=1   (build only)
"""
import math
import random
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/blender")
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import bl_build as K  # noqa: E402
import bl_common as B  # noqa: E402

A = B.args()
PCT = int(A.get("pct", 35))
SPP = int(A.get("samples", 16))
TAG = A.get("tag", "29-test")
WORK = ROOT + "/work"


def F(k, d):
    return float(A.get(k, d))



rnd = random.Random(29)

sc = B.empty_scene()
B.cycles(samples=SPP, pct=PCT, bounces=4, clamp_indirect=3.0, vol_bounces=0, vol_step=F("volstep", 1.5))
sc.cycles.volume_max_steps = int(A.get("volmax", 160))

CW, FX, QZ = 3.5, 5.0, 1.0          # canal half width, base facade line, quay / walkway height
Y_END = 98.0                         # the tunnel building closes the view here


# ---------------------------------------------------------------- materials
def wall_mat(name, base, brick=False, grime=0.45):
    m, t = K.mat(name)
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    n1 = t.n("ShaderNodeTexNoise", Scale=0.4, Detail=8.0, Roughness=0.6)
    t.link(tc, n1.inputs["Vector"])
    mp = t.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (3.0, 3.0, 0.12)            # long vertical rain streaks
    t.link(tc, mp.inputs["Vector"])
    n2 = t.n("ShaderNodeTexNoise", Scale=1.5, Detail=3.0)
    t.link(mp.outputs[0], n2.inputs["Vector"])
    col = t.mixc(t.mapr(n1.outputs["Fac"], 0.3, 0.7), tuple(c * 0.5 for c in base), base)
    if brick:
        vec = t.n("ShaderNodeCombineXYZ")
        t.link(t.math("ADD", sep.outputs["X"], sep.outputs["Y"]), vec.inputs["X"])
        t.link(sep.outputs["Z"], vec.inputs["Y"])
        br = t.n("ShaderNodeTexBrick", **{"Scale": 1.0, "Mortar Size": 0.012, "Brick Width": 0.26, "Row Height": 0.085})
        t.link(vec.outputs[0], br.inputs["Vector"])
        t.set(br.inputs["Color1"], base)
        t.set(br.inputs["Color2"], tuple(c * 0.68 for c in base))
        t.set(br.inputs["Mortar"], (0.05, 0.045, 0.042))
        col = t.mixc(0.65, col, br.outputs["Color"])
    g = t.math("ADD", t.mapr(n2.outputs["Fac"], 0.45, 0.72, 0.0, grime), t.mapr(sep.outputs["Z"], QZ + 0.2, QZ + 4.0, 0.45, 0.0))
    col = t.mixc(g, col, (0.02, 0.025, 0.025))
    t.out(t.bsdf(col, 0.86, normal=t.bump(n1.outputs["Fac"], 0.25, 0.05)))
    return m


def stone_mat(name, base):
    m, t = K.mat(name)
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    vor = t.n("ShaderNodeTexVoronoi", Scale=1.6)
    vec = t.n("ShaderNodeCombineXYZ")
    t.link(t.math("ADD", sep.outputs["X"], sep.outputs["Y"]), vec.inputs["X"])
    t.link(t.math("MULTIPLY", sep.outputs["Z"], 1.8), vec.inputs["Y"])
    t.link(vec.outputs[0], vor.inputs["Vector"])
    n1 = t.n("ShaderNodeTexNoise", Scale=2.5, Detail=8.0)
    t.link(tc, n1.inputs["Vector"])
    col = t.mixc(t.mapr(n1.outputs["Fac"], 0.3, 0.7), tuple(c * 0.55 for c in base), base)
    col = t.mixc(t.mapr(vor.outputs["Distance"], 0.0, 0.06, 0.7, 0.0), col, (0.03, 0.03, 0.03))   # joints
    wet = t.mapr(sep.outputs["Z"], 0.9, 0.15, 0.0, 1.0)                                         # waterline
    col = t.mixc(t.math("MULTIPLY", wet, 0.75), col, (0.012, 0.03, 0.026))
    rough = t.mapr(sep.outputs["Z"], 0.9, 0.15, 0.85, 0.25)
    t.out(t.bsdf(col, rough, normal=t.bump(n1.outputs["Fac"], 0.35, 0.04)))
    return m


def water_mat():
    m, t = K.mat("water")
    tc = t.coords("Object")
    mp = t.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 7.0, 1.0)      # ripples run across the canal: lights streak toward us
    t.link(tc, mp.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=1.6, Detail=6.0, Roughness=0.55)
    t.link(mp.outputs[0], nz.inputs["Vector"])
    t.out(t.bsdf((0.006, 0.013, 0.013), 0.045, spec=0.6, normal=t.bump(nz.outputs["Fac"], F("ripple", 0.22), 0.05)))
    return m


def haze_mat():
    m, t = K.mat("haze")
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    nz = t.n("ShaderNodeTexNoise", Scale=0.07, Detail=3.0)
    t.link(tc, nz.inputs["Vector"])
    low = t.math("EXPONENT", t.math("MULTIPLY", sep.outputs["Z"], -1.0 / F("hazeh", 1.8)))
    dens = t.math("ADD", t.math("MULTIPLY", low, F("haze0", 0.025)), F("haze1", 0.002))
    dens = t.math("MULTIPLY", dens, t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.45, 1.5))
    pv = t.n("ShaderNodeVolumePrincipled", Color=(0.5, 0.76, 0.95), Anisotropy=F("aniso", 0.45))
    t.link(dens, pv.inputs["Density"])
    o = t.n("ShaderNodeOutputMaterial")
    t.link(pv.outputs[0], o.inputs["Volume"])
    m.cycles.volume_step_rate = F("matstep", 2.0)
    return m


def neon(name, color, strength):
    return B.emission(name, color, strength)


STONE = stone_mat("stone", (0.20, 0.19, 0.18))
BRIDGE = stone_mat("bridge_stone", (0.11, 0.1, 0.095))
IVY = B.principled("ivy", (0.035, 0.08, 0.035), rough=0.7)
TUNNEL = B.emission("tunnel_glow", (0.3, 0.9, 0.8), F("tunnel", 1.2))
WALLS = [wall_mat("brick_red", (0.30, 0.12, 0.08), brick=True), wall_mat("brick_dark", (0.15, 0.10, 0.09), brick=True),
         wall_mat("plaster_teal", (0.15, 0.27, 0.27)), wall_mat("plaster_ochre", (0.42, 0.30, 0.16)),
         wall_mat("plaster_violet", (0.25, 0.20, 0.30)), wall_mat("brick_brown", (0.22, 0.14, 0.10), brick=True)]
WOOD = B.principled("wood", (0.11, 0.065, 0.04), rough=0.72)
METAL = B.principled("metal", (0.035, 0.038, 0.04), rough=0.45, metal=0.85)
COPPER = B.principled("verdigris", (0.07, 0.2, 0.17), rough=0.6, metal=0.3)
SLATE = B.principled("slate", (0.045, 0.045, 0.055), rough=0.55)
WIN = [B.emission("win_warm", (1.0, 0.52, 0.2), F("win", 8.0)), B.emission("win_dim", (0.95, 0.4, 0.14), F("win", 8.0) * 0.45),
       B.emission("win_teal", (0.3, 0.85, 0.78), F("win", 8.0) * 0.4), B.emission("win_pink", (0.95, 0.3, 0.55), F("win", 8.0) * 0.45)]
WIN_DARK = B.principled("win_dark", (0.008, 0.01, 0.012), rough=0.12)
NEON_MAG = neon("neon_mag", (1.0, 0.16, 0.6), F("neon", 28.0))
NEON_TEAL = neon("neon_teal", (0.2, 1.0, 0.82), F("neon", 28.0) * 0.6)
NEON_ORA = neon("neon_ora", (1.0, 0.42, 0.08), F("neon", 28.0) * 0.6)
CLOTH = [B.principled("cloth_r", (0.42, 0.08, 0.07), rough=0.9), B.principled("cloth_t", (0.1, 0.26, 0.28), rough=0.9),
         B.principled("cloth_c", (0.55, 0.47, 0.34), rough=0.9)]
FACES = [f"{WORK}/face_classic.png", f"{WORK}/face_grin.png", f"{WORK}/face_scary.png"]
CLOAK = B.principled("cloak", (0.05, 0.035, 0.06), rough=0.8)
B.aov_mask(CLOAK, "actor")
POLE_M = B.principled("pole", (0.09, 0.06, 0.035), rough=0.6)
B.aov_mask(POLE_M, "actor", 0.6)

# material slots of the two big building meshes
MATS = [STONE, BRIDGE, IVY, TUNNEL] + WALLS + [WOOD, METAL, COPPER, SLATE] + WIN + [WIN_DARK, NEON_MAG, NEON_TEAL, NEON_ORA] + CLOTH
MI = {m.name: i for i, m in enumerate(MATS)}


def mi(m):
    return MI[m.name]


# ---------------------------------------------------------------- geometry helpers
def prism_xz(mb, outline, y0, y1, mat):
    """Extrude a counter-clockwise (x, z) outline (may be concave) along y from y0 to y1."""
    n = len(outline)
    vs = [(x, y0, z) for x, z in outline] + [(x, y1, z) for x, z in outline]
    fs = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    fs += [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    mb.add(vs, fs, mat)


def prism_yz(mb, outline, x0, x1, mat):
    """Extrude a counter-clockwise (y, z) outline along x from x0 to x1 > x0 (gables facing the canal)."""
    n = len(outline)
    vs = [(x0, y, z) for y, z in outline] + [(x1, y, z) for y, z in outline]
    fs = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    fs += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    mb.add(vs, fs, mat)


def catenary(p0, p1, sag, n=24):
    p0, p1 = Vector(p0), Vector(p1)
    return [p0.lerp(p1, i / n) - Vector((0, 0, sag * 4 * (i / n) * (1 - i / n))) for i in range(n + 1)]


def wire(name, pts, r=0.012, mat=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    cu.bevel_resolution = 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1)
    ob = bpy.data.objects.new(name, cu)
    if mat:
        ob.data.materials.append(mat)
    return B.link(ob)


# ---------------------------------------------------------------- canal, quays, steps
mb = K.MB()
for s in (-1, 1):
    x0, x1 = sorted([s * CW, s * (FX + 0.8)])
    mb.boxr(x0, x1, -10, Y_END + 2, -3.0, QZ, mi(STONE))
    xk0, xk1 = sorted([s * CW, s * (CW + 0.35)])          # kerb stones along the edge
    mb.boxr(xk0, xk1, -10, Y_END + 2, QZ, QZ + 0.12, mi(STONE))
# steps down to the water on the left, near us (they carry pumpkins and candles)
STEP_Y0, STEP_D, STEP_H = 5.0, 0.42, 0.19


def step_top(y):
    """Top of the stair step under y (the stair descends along +y from the walkway into the water)."""
    k = 6 - int((y - STEP_Y0) / STEP_D)
    return QZ - (k + 1) * STEP_H


for k in range(7):                                                   # descends toward us
    mb.boxr(-CW, -CW + 1.15, STEP_Y0 + (6 - k) * STEP_D, STEP_Y0 + (7 - k) * STEP_D, -3.0, QZ - (k + 1) * STEP_H, mi(STONE))
# bollards and a railing run on the quays
for s in (-1, 1):
    for y in range(-6, int(Y_END), 7):
        if s < 0 and 3 < y < 10:
            continue
        mb.cyl((s * (CW + 0.25), y + 0.5 * s, QZ), (s * (CW + 0.25), y + 0.5 * s, QZ + 0.55), 0.14, 0.12, n=10, mat=mi(METAL))

# ---------------------------------------------------------------- canal houses
lights = []          # (pos, energy, colour) for wall lanterns


def window(mb, s, fx, y, z, w=1.0, h=1.6, lit_p=0.38):
    """A window on a facade facing the canal (s = -1 left side, +1 right; fx = facade x)."""
    d = -s                                                   # toward the canal
    lit = rnd.random() < lit_p
    m = mi(rnd.choices(WIN, weights=(5, 4, 2, 1))[0]) if lit else mi(WIN_DARK)
    gx0, gx1 = sorted([fx + d * 0.02, fx - d * 0.05])
    mb.boxr(gx0, gx1, y - w / 2, y + w / 2, z, z + h, m)
    fx0, fx1 = sorted([fx, fx + d * 0.07])
    for (ya, yb, za, zb) in ((y - w / 2 - 0.08, y - w / 2, z - 0.05, z + h + 0.05), (y + w / 2, y + w / 2 + 0.08, z - 0.05, z + h + 0.05),
                             (y - w / 2, y + w / 2, z + h, z + h + 0.08), (y - 0.025, y + 0.025, z, z + h),
                             (y - w / 2, y + w / 2, z + h * 0.62, z + h * 0.62 + 0.05)):
        mb.boxr(fx0, fx1, ya, yb, za, zb, mi(WOOD))
    sx0, sx1 = sorted([fx, fx + d * 0.18])
    mb.boxr(sx0, sx1, y - w / 2 - 0.12, y + w / 2 + 0.12, z - 0.12, z - 0.02, mi(STONE))
    if rnd.random() < 0.35:                                     # shutters
        for sg in (-1, 1):
            hx0, hx1 = sorted([fx, fx + d * 0.06])
            mb.boxr(hx0, hx1, y + sg * (w / 2 + 0.1) - 0.25, y + sg * (w / 2 + 0.1) + 0.25, z, z + h,
                    mi(rnd.choice([WOOD, COPPER])))


def house(mb, s, y0, y1, seed, floors=None, setback=None, oh=None, gable=None, wall=None, shop=None, lantern=False):
    r = random.Random(seed)
    floors = floors or r.randint(4, 7)
    setback = r.uniform(0.0, 0.5) if setback is None else setback
    oh = (r.choice([0.0, 0.0, 0.5, 0.8, 1.1]) if oh is None else oh)
    wall = WALLS[r.randrange(len(WALLS))] if wall is None else wall
    d = -s
    fx = s * (FX + setback)
    deep = s * (FX + 14.0)
    zj = QZ + 3.4
    top = zj + (floors - 1) * 2.9
    x0, x1 = sorted([fx, deep])
    mb.boxr(x0, x1, y0, y1, -1.0, zj, mi(wall))
    fu = fx + d * oh
    x0, x1 = sorted([fu, deep])
    mb.boxr(x0, x1, y0 + 0.05, y1 - 0.05, zj, top, mi(wall))
    if oh > 0:
        bx0, bx1 = sorted([fx, fu + d * 0.05])
        mb.boxr(bx0, bx1, y0 + 0.05, y1 - 0.05, zj - 0.25, zj + 0.05, mi(WOOD))
        for k in range(int((y1 - y0) / 1.2) + 1):                  # brackets under the jetty
            yy = y0 + 0.3 + k * (y1 - y0 - 0.6) / max(1, int((y1 - y0) / 1.2))
            mb.cyl((fx, yy, zj - 1.1), (fu, yy, zj - 0.2), 0.07, n=6, mat=mi(WOOD))
    # cornice
    cx0, cx1 = sorted([fu - s * 0.0, fu + d * 0.3])
    mb.boxr(cx0, cx1, y0, y1, top - 0.1, top + 0.2, mi(STONE))
    # gable facing the canal, or a flat roof with a water tank
    gable = r.random() < 0.7 if gable is None else gable
    gx0, gx1 = sorted([fu, deep])
    if gable:
        yc, hw = (y0 + y1) / 2, (y1 - y0) / 2
        kind = r.choice(["point", "step", "bell"])
        ya, yb = y0 + 0.05, y1 - 0.05
        if kind == "point":
            ol = [(ya, top), (yb, top), (yc, top + hw * r.uniform(1.0, 1.5))]
        elif kind == "step":                                        # stepped gable, counter-clockwise in (y, z)
            n, h = 3, 1.0
            sw = hw / (n + 0.5)
            rs = []
            for i in range(n):
                rs += [(yb - i * sw, top + (i + 1) * h), (yb - (i + 1) * sw, top + (i + 1) * h)]
            cap = min(0.35, 0.8 * (yb - n * sw - yc))
            cp = [(yc + cap, top + n * h), (yc + cap, top + n * h + 0.8), (yc - cap, top + n * h + 0.8), (yc - cap, top + n * h)]
            ls = [(2 * yc - y, z) for (y, z) in reversed(rs)]
            ol = [(ya, top), (yb, top)] + rs + cp + ls
        else:
            ol = [(ya, top), (yb, top), (y1 - 0.4, top + 1.2), (yc + 0.7, top + 2.6), (yc + 0.3, top + 3.6),
                  (yc - 0.3, top + 3.6), (yc - 0.7, top + 2.6), (y0 + 0.4, top + 1.2)]
        prism_yz(mb, ol, gx0, gx1, mi(wall))
        if r.random() < 0.6:                                         # hoist beam at the gable top (canal-house style)
            hz = max(z for _, z in ol) - 0.6
            hx0, hx1 = sorted([fu, fu + d * 1.0])
            mb.boxr(hx0, hx1, yc - 0.1, yc + 0.1, hz - 0.1, hz + 0.1, mi(WOOD))
        if r.random() < 0.6:
            window(mb, s, fu, yc, top + 0.4, 0.8, 1.3, 0.5)
    else:
        mb.boxr(gx0, gx1, y0, y1, top + 0.2, top + 0.7, mi(SLATE))
        if r.random() < 0.7:                                         # water tank on legs
            tx = fu - d * r.uniform(2.5, 4.0)
            ty = r.uniform(y0 + 1.2, y1 - 1.2)
            for (ax, ay) in ((-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)):
                mb.cyl((tx + ax, ty + ay, top + 0.7), (tx + ax * 0.8, ty + ay * 0.8, top + 3.2), 0.06, n=6, mat=mi(METAL))
            mb.cyl((tx, ty, top + 3.2), (tx, ty, top + 5.4), 1.2, 1.2, n=16, mat=mi(WOOD))
            mb.cyl((tx, ty, top + 5.4), (tx, ty, top + 6.2), 1.25, 0.1, n=16, mat=mi(SLATE))
    if r.random() < 0.8:                                             # chimneys
        for _ in range(r.randint(1, 2)):
            cx = fu - d * r.uniform(2.0, 5.0)
            cy = r.uniform(y0 + 0.5, y1 - 0.5)
            ch0, ch1 = sorted([cx, cx + 0.7])
            mb.boxr(ch0, ch1, cy - 0.35, cy + 0.35, top, top + r.uniform(2.5, 5.0), mi(WALLS[1]))
    # windows: upper floors on the jetty face, ground floor shop or door
    nwin = max(1, int((y1 - y0) / 2.3))
    for f in range(1, floors):
        zf = zj + (f - 1) * 2.9 + 0.75
        for k in range(nwin):
            yc = y0 + (k + 0.5) * (y1 - y0) / nwin + r.uniform(-0.15, 0.15)
            window(mb, s, fu, yc, zf, 1.0, 1.6)
            if f >= 2 and r.random() < 0.16:                         # iron balcony with a slab and railing
                bx0, bx1 = sorted([fu, fu + d * 0.85])
                mb.boxr(bx0, bx1, yc - 0.9, yc + 0.9, zf - 0.32, zf - 0.2, mi(STONE))
                xo = fu + d * 0.82
                mb.cyl((xo, yc - 0.9, zf + 0.75), (xo, yc + 0.9, zf + 0.75), 0.025, n=4, mat=mi(METAL))
                for q in range(9):
                    yq = yc - 0.9 + q * 0.225
                    mb.cyl((xo, yq, zf - 0.2), (xo, yq, zf + 0.75), 0.012, n=4, mat=mi(METAL))
                if r.random() < 0.5:                                  # a potted plant or a small pumpkin on it
                    K.pumpkin_parts(mb, (fu + d * 0.5, yc + 0.5, zf - 0.2 + 0.12), 0.16, mi(CLOTH[0]), mi(WOOD), nu=16, nv=8,
                                    seed=int(yc * 10))
    shop = r.random() < 0.55 if shop is None else shop
    yd = r.uniform(y0 + 1.0, y1 - 1.0)
    dx0, dx1 = sorted([fx + d * 0.03, fx - d * 0.05])
    mb.boxr(dx0, dx1, yd - 0.55, yd + 0.55, QZ, QZ + 2.3, mi(WOOD))
    if shop and y1 - y0 > 4.0:
        ys = yd + (1.9 if yd < (y0 + y1) / 2 else -1.9)
        mb.boxr(dx0, dx1, ys - 1.0, ys + 1.0, QZ + 0.7, QZ + 2.5, mi(WIN[0]))
        ax0, ax1 = sorted([fx, fx + d * 1.2])                        # awning
        for i in range(6):
            z = QZ + 3.0 - i * 0.07
            xa = fx + d * (i + 1) * 0.2
            xb0, xb1 = sorted([fx + d * i * 0.2, xa])
            mb.boxr(xb0, xb1, ys - 1.2, ys + 1.2, z - 0.05, z, mi(CLOTH[r.randrange(3)]))
    if lantern:
        ly = r.uniform(y0 + 0.8, y1 - 0.8)
        ax = fx + d * 0.9
        a0, a1 = sorted([fx, ax])
        mb.boxr(a0, a1, ly - 0.03, ly + 0.03, QZ + 3.05, QZ + 3.12, mi(METAL))
        mb.cyl((ax, ly, QZ + 3.08), (ax, ly, QZ + 2.65), 0.012, n=4, mat=mi(METAL))
        lights.append((ax, ly, QZ + 2.42))
    # drain pipe at the block edge
    px = fx + d * 0.12
    mb.cyl((px, y0 + 0.12, QZ), (px, y0 + 0.12, top), 0.07, n=8, mat=mi(COPPER if r.random() < 0.4 else METAL))
    return top, fu


tops = {}
for s in (-1, 1):
    y = -10.0
    i = 0
    while y < Y_END - 2:
        L = rnd.uniform(4.6, 8.5)
        y1 = min(y + L, Y_END - 0.5)
        tops[(s, i)] = house(mb, s, y, y1, 1000 + 37 * i + (0 if s < 0 else 500),
                             lantern=(-4 < y < 50) or (i % 2 == 0 and y < 80))
        y, i = y1, i + 1

# ---------------------------------------------------------------- the tunnel building closing the view
TB0 = Y_END
mb.boxr(-24, -CW, TB0, TB0 + 14, -3.0, 30.0, mi(WALLS[1]))
mb.boxr(CW, 24, TB0, TB0 + 14, -3.0, 30.0, mi(WALLS[1]))
ol = [(-CW, 3.0)] + [(CW * math.cos(t), 3.0 + 2.6 * math.sin(t)) for t in [math.pi * (1 - k / 20) for k in range(1, 20)]] \
    + [(CW, 3.0), (CW, 30.0), (-CW, 30.0)]                           # wall over the tunnel mouth (counter-clockwise)
prism_xz(mb, ol, TB0, TB0 + 14, mi(WALLS[1]))
# tunnel glow deep inside, plus arched windows and a cornice
mb.boxr(-CW, CW, TB0 + 12, TB0 + 13, -1, 6, mi(TUNNEL))
for k in range(-3, 4):
    if k == 0:
        continue
    xw = k * 2.6
    mb.boxr(xw - 0.55, xw + 0.55, TB0 - 0.06, TB0 + 0.02, 11.5, 13.9, mi(WIN[0] if k % 2 else WIN[1]))
mb.boxr(-24, 24, TB0 - 0.5, TB0 + 14, 30.0, 30.8, mi(STONE))
# neon crescent moon with three stars over the tunnel (the far focal point)
NEON_Y = TB0 - 0.25


def ring(cx, cz, r, n=40, a0=0.0, a1=2 * math.pi):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), NEON_Y, cz + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


MZ = F("neonz", 7.9)
outer = ring(0.0, MZ, 1.8, 48, math.radians(60), math.radians(300))
inner = ring(0.7, MZ + 0.2, 1.46, 40, math.radians(255), math.radians(105))
crescent = outer + inner + [outer[0]]
wire("neon_moon", crescent, 0.06, NEON_MAG)
for (sx, sz, sr) in ((1.46, MZ + 1.24, 0.31), (1.97, MZ + 0.11, 0.225), (1.24, MZ - 1.01, 0.19)):
    star = []
    for k in range(11):
        rr = sr if k % 2 == 0 else sr * 0.42
        a = math.pi / 2 + k * math.pi / 5
        star.append((sx + rr * math.cos(a), NEON_Y, sz + rr * math.sin(a)))
    wire("neon_star", star, 0.045, NEON_MAG)
B.light("POINT", (0.3, TB0 - 3.0, MZ - 0.5), F("neonw", 2500.0), color=(1.0, 0.25, 0.65), size=2.0, name="neon_light")
B.light("AREA", (0.0, TB0 + 6.0, 2.0), F("tunnelw", 180.0), color=(0.3, 1.0, 0.85), size=5.0, name="tunnel_light",
        target=(0.0, TB0 - 20.0, 1.0))

# ---------------------------------------------------------------- bridges, pipes, a footbridge, wires, laundry
BRIDGES = [float(v) for v in A.get("bridges", "18,46").split(",")]
for bi, yb in enumerate(BRIDGES):
    rise, spring, deck = 3.8, 1.2, 6.2
    ol = [(-FX - 1.0, -0.6), (-CW, -0.6), (-CW, spring)]
    ol += [(CW * math.cos(t), spring + rise * math.sin(t)) for t in [math.pi * (1 - k / 24) for k in range(1, 24)]]
    ol += [(CW, spring), (CW, -0.6), (FX + 1.0, -0.6), (FX + 1.0, deck), (-FX - 1.0, deck)]
    prism_xz(mb, ol, yb, yb + 2.6, mi(BRIDGE))
    for q in range(28):                                              # ivy hanging over the parapet and spandrels
        xq = rnd.uniform(-FX - 0.8, FX + 0.8)
        top_q = deck + 1.0
        ln = rnd.uniform(0.6, 2.8)
        if abs(xq) < CW:
            arch_z = spring + rise * math.sqrt(max(0.0, 1 - (xq / CW) ** 2))
            ln = min(ln, top_q - arch_z - 0.1)
        wq = rnd.uniform(0.15, 0.4)
        mb.boxr(xq - wq / 2, xq + wq / 2, yb - 0.09, yb - 0.04, top_q - ln, top_q, mi(IVY))
    for (ya, yb2) in ((yb - 0.05, yb + 0.35), (yb + 2.25, yb + 2.65)):
        mb.boxr(-FX - 1.0, FX + 1.0, ya, yb2, deck, deck + 1.0, mi(BRIDGE))
    # keystone lantern hanging under the arch, chain from the crown
    crown = spring + rise
    mb.cyl((0.0, yb - 0.05, crown), (0.0, yb - 0.05, crown - 0.9), 0.015, n=4, mat=mi(METAL))
    lights.append((0.0, yb - 0.05, crown - 1.15))
    # lamp posts on the parapet
    for sx in (-1, 1):
        px = sx * (FX - 0.2)
        mb.cyl((px, yb + 0.15, deck + 1.0), (px, yb + 0.15, deck + 3.4), 0.06, n=8, mat=mi(METAL))
        mb.boxr(px - 0.16, px + 0.16, yb - 0.01, yb + 0.31, deck + 3.4, deck + 3.85, mi(WIN[0]))
        mb.boxr(px - 0.2, px + 0.2, yb - 0.05, yb + 0.35, deck + 3.85, deck + 3.95, mi(METAL))
# big pipes across the canal high up (an industrial undercity)
for (yp, zp, rp) in ((27.0, 13.5, 0.38), (31.0, 9.8, 0.22), (55.0, 19.0, 0.5), (62.0, 11.0, 0.25)):
    mb.cyl((-FX - 0.5, yp, zp), (FX + 0.5, yp, zp), rp, n=16, mat=mi(COPPER if yp > 50 else METAL))
    for xf in (-2.5, 0.0, 2.5):
        mb.cyl((xf - 0.08, yp, zp), (xf + 0.08, yp, zp), rp * 1.25, n=16, mat=mi(METAL))
# iron footbridge between upper floors
yf, zf = 35.5, 10.6
mb.boxr(-FX, FX, yf, yf + 1.4, zf, zf + 0.15, mi(METAL))
for yy in (yf, yf + 1.4):
    mb.cyl((-FX, yy, zf + 1.05), (FX, yy, zf + 1.05), 0.04, n=6, mat=mi(METAL))
    for k in range(13):
        xx = -FX + k * 2 * FX / 12
        mb.cyl((xx, yy, zf), (xx, yy, zf + 1.05), 0.025, n=4, mat=mi(METAL))
for k in range(9):                                                   # truss under it
    x0 = -FX + k * 2 * FX / 9
    mb.cyl((x0, yf + 0.7, zf), (x0 + 2 * FX / 9, yf + 0.7, zf - 0.6 if k % 2 == 0 else zf), 0.04, n=4, mat=mi(METAL))
# a blade sign on the right facade with an orange neon jack-o'-lantern
SX0, SX1, SY, SZ = 3.7, 4.95, 24.0, F("signz", 10.3)
mb.boxr(SX0, SX1, SY, SY + 0.08, SZ - 1.0, SZ + 1.0, mi(METAL))
mb.boxr(SX0 + 0.3, SX0 + 0.34, SY + 0.02, SY + 0.06, SZ + 1.0, SZ + 1.6, mi(METAL))
mb.boxr(SX1 - 0.34, SX1 - 0.3, SY + 0.02, SY + 0.06, SZ + 1.0, SZ + 1.6, mi(METAL))
building = mb.build("canal_city", MATS)
scx, scz, sr = (SX0 + SX1) / 2, SZ - 0.05, 0.5
pumpkin = [(scx + sr * 1.15 * math.cos(2 * math.pi * k / 40), SY - 0.03, scz + sr * 0.9 * math.sin(2 * math.pi * k / 40)) for k in range(41)]
wire("neon_pumpkin", pumpkin, 0.03, NEON_ORA)
wire("neon_stem", [(scx, SY - 0.03, scz + sr * 0.88), (scx + 0.06, SY - 0.03, scz + sr * 1.25), (scx + 0.2, SY - 0.03, scz + sr * 1.32)], 0.03, NEON_ORA)
for sg in (-1, 1):
    wire("neon_eye", [(scx + sg * 0.32, SY - 0.03, scz + 0.05), (scx + sg * 0.18, SY - 0.03, scz + 0.05),
                      (scx + sg * 0.25, SY - 0.03, scz + 0.24), (scx + sg * 0.32, SY - 0.03, scz + 0.05)], 0.022, NEON_ORA)
wire("neon_mouth", [(scx - 0.36 + 0.09 * k, SY - 0.03, scz - 0.18 - 0.07 * (k % 2)) for k in range(9)], 0.022, NEON_ORA)
B.light("POINT", (scx, SY - 0.6, scz), F("signw", 60.0), color=(1.0, 0.45, 0.12), size=0.6, name="sign_light")

# wires and laundry lines across the canal
for k, yw in enumerate([8.0, 12.5, 23.0, 29.5, 46.0, 52.0, 66.0, 78.0]):
    z0 = rnd.uniform(8.0, 18.0)
    z1 = z0 + rnd.uniform(-1.5, 1.5)
    pts = catenary((-FX - 0.2, yw, z0), (FX + 0.2, yw + rnd.uniform(-1.0, 1.0), z1), rnd.uniform(0.6, 1.4))
    wire(f"wire{k}", pts, 0.014, METAL)
    if k in (2, 5):                                                  # laundry
        lm = K.MB()
        for j in range(3, len(pts) - 3, 2):
            if rnd.random() < 0.3:
                continue
            p = pts[j]
            w, h = rnd.uniform(0.25, 0.5), rnd.uniform(0.3, 0.6)
            lm.boxr(p.x - w / 2, p.x + w / 2, p.y - 0.01, p.y + 0.01, p.z - h, p.z - 0.02, rnd.randrange(3))
        lm.build(f"laundry{k}", CLOTH)

# ---------------------------------------------------------------- water, haze, sky, moon
wm = K.MB()
wm.boxr(-CW, CW, -12, TB0 + 13, -0.6, 0.0, 0)
water = wm.build("water", [water_mat()])
hz = K.MB()
hz.boxr(-FX - 0.4, FX + 0.4, -12, TB0 + 12.5, 0.0, 34.0, 0)
haze = hz.build("haze", [haze_mat()])

cam_loc = Vector((F("camx", -0.5), F("camy", -2.0), F("camz", 1.45)))
MOON_EL, MOON_AZ = math.radians(F("moon_el", 37.0)), math.radians(F("moon_az", 7.0))
mdir = Vector((math.sin(MOON_AZ) * math.cos(MOON_EL), math.cos(MOON_AZ) * math.cos(MOON_EL), math.sin(MOON_EL)))
mpos = cam_loc + mdir * 3000.0
# sky: teal-to-navy gradient, a layer of torn clouds whose edges catch the moon, a halo around it
wt = K.NT(sc.world)
dirv = wt.n("ShaderNodeTexCoord").outputs["Generated"]
sep = wt.n("ShaderNodeSeparateXYZ")
wt.link(dirv, sep.inputs[0])
sky = wt.ramp(sep.outputs["Z"], [(0.0, (0.03, 0.07, 0.08)), (0.25, (0.02, 0.03, 0.07)), (0.7, (0.004, 0.006, 0.02))])
mp = wt.n("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (2.0, 2.0, 5.0)
wt.link(dirv, mp.inputs["Vector"])
cn = wt.n("ShaderNodeTexNoise", Scale=2.4, Detail=8.0, Roughness=0.58)
wt.link(mp.outputs[0], cn.inputs["Vector"])
cloud = wt.mapr(cn.outputs["Fac"], F("cloud0", 0.54), F("cloud1", 0.72))
dot = wt.n("ShaderNodeVectorMath", _operation="DOT_PRODUCT")
wt.link(dirv, dot.inputs[0])
dot.inputs[1].default_value = tuple(mdir)
near = wt.mapr(dot.outputs["Value"], 0.94, 1.0)
cl_col = wt.mixc(near, (0.008, 0.013, 0.026), (0.2, 0.19, 0.28))
col = wt.mixc(cloud, sky, cl_col)
col = wt.mixc(wt.math("MULTIPLY", wt.math("POWER", near, 3.0), 0.18), col, (0.9, 0.8, 0.88))
bg = wt.n("ShaderNodeBackground", Strength=F("sky", 1.0))
wt.link(col, bg.inputs["Color"])
wo = wt.n("ShaderNodeOutputWorld")
wt.link(bg.outputs[0], wo.inputs["Surface"])

bpy.ops.mesh.primitive_uv_sphere_add(radius=3000.0 * math.tan(math.radians(F("moon_r", 2.0))), location=mpos, segments=48,
                                     ring_count=24)
moon = bpy.context.object
mm, t = K.mat("moon")
nz = t.n("ShaderNodeTexNoise", Scale=0.012, Detail=6.0, Roughness=0.6)
t.link(t.coords("Object"), nz.inputs["Vector"])
em = t.n("ShaderNodeEmission", Color=(1.0, 0.93, 0.85))
t.link(t.mapr(nz.outputs["Fac"], 0.3, 0.75, 7.0, 4.0), em.inputs["Strength"])
t.out(em.outputs[0])
B.aov_mask(mm, "actor", 0.6)
moon.data.materials.append(mm)
moon.visible_shadow = False
B.light("SUN", tuple(mpos), F("moonlight", 0.25), color=(0.6, 0.8, 1.0), size=math.radians(1.0), target=tuple(cam_loc))

# ---------------------------------------------------------------- lanterns: wall brackets, under bridges, floating
for k, p in enumerate(lights):
    j = K.jack(f"lantern{k}", p, 0.24, FACES[k % 3], rz=rnd.uniform(-0.5, 0.5), seed=50 + k,
               light_w=F("lanw", 140.0), strength=F("jackglow", 30.0))
for k in range(int(A.get("floaters", 9))):
    x = rnd.uniform(-2.6, 2.9)
    y = rnd.uniform(1.5, 24.0)
    if abs(x - 0.9) < 1.2 and 6.0 < y < 15.0:
        continue
    K.jack(f"floater{k}", (x, y, 0.06), rnd.uniform(0.09, 0.13), FACES[k % 3], rz=rnd.uniform(-1, 1), seed=80 + k,
           light_w=F("floatw", 6.0), strength=F("jackglow", 30.0))
# pumpkins and candles on the steps at the lower left
for k, (x, y, r_) in enumerate([(-CW + 0.5, 5.25, 0.26), (-CW + 0.82, 6.1, 0.17), (-CW + 0.45, 7.0, 0.2)]):
    K.jack(f"step{k}", (x, y, step_top(y) + r_ * 0.7), r_, FACES[(k + 1) % 3], rz=-0.5 + k * 0.3, seed=90 + k,
           light_w=F("stepw", 25.0), strength=F("jackglow", 30.0))
for k, (x, y) in enumerate([(-CW + 0.95, 5.6), (-CW + 0.25, 6.5), (-CW + 0.9, 7.35)]):
    z = step_top(y)
    cm = K.MB()
    cm.cyl((x, y, z), (x, y, z + 0.12), 0.03, n=8, mat=0)
    cm.build(f"candle{k}", [B.principled("wax", (0.85, 0.8, 0.65), rough=0.5, subsurface=0.3)])
    cm2 = K.MB()
    cm2.cyl((x, y, z + 0.12), (x, y, z + 0.17), 0.012, 0.002, n=6, mat=0)
    cm2.build(f"flame{k}", [B.emission("flame", (1.0, 0.6, 0.2), 40.0)])
    B.light("POINT", (x, y, z + 0.16), 1.5, color=(1.0, 0.55, 0.2), size=0.02, name=f"candle_light{k}")

# ---------------------------------------------------------------- the pumpkin ferry
BX, BY0, BY1 = F("boatx", 1.0), F("boaty", 6.0), F("boaty", 6.0) + 5.6       # stern to bow


def hull_w(t):                    # half width along the hull, t = 0 stern .. 1 bow
    return 0.62 * (1 - max(0.0, (t - 0.72) / 0.28) ** 1.8) + 0.04


hm = K.MB()
N = 20
for i in range(N + 1):
    t = i / N
    y = BY0 + (BY1 - BY0) * t
    wdt = hull_w(t)
    zt = 0.34 + 0.35 * max(0.0, (t - 0.75) / 0.25) ** 2          # bow sweeps up
    zb = -0.12 + 0.2 * max(0.0, (t - 0.8) / 0.2)
    hm.v += [(BX - wdt, y, zt), (BX - wdt * 0.78, y, zb), (BX + wdt * 0.78, y, zb), (BX + wdt, y, zt)]
for i in range(N):
    a, b = i * 4, (i + 1) * 4
    hm.f += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1), (a + 2, a + 3, b + 3, b + 2), (a + 3, a, b, b + 3)]
    hm.m += [0, 0, 0, 1]
hm.f += [(0, 3, 2, 1), (N * 4, N * 4 + 1, N * 4 + 2, N * 4 + 3)]
hm.m += [0, 0]
DECK = B.principled("deck", (0.05, 0.03, 0.02), rough=0.8)
hull = hm.build("hull", [WOOD, DECK])
gm = K.MB()
pts_l = [(BX - hull_w(i / N), BY0 + (BY1 - BY0) * i / N, 0.34 + 0.35 * max(0.0, (i / N - 0.75) / 0.25) ** 2) for i in range(N + 1)]
pts_r = [(2 * BX - p[0], p[1], p[2]) for p in pts_l]
for pts in (pts_l, pts_r):
    for a, b in zip(pts[:-1], pts[1:]):
        gm.cyl(a, b, 0.045, n=6, mat=0)
gm.build("gunwale", [B.principled("gunwale", (0.2, 0.12, 0.06), rough=0.55)])
# pumpkin cargo (one lit), the prow lantern on a hooked pole leaning out to the left
cargo = K.MB()
SKIN = K.pumpkin_material("cargo_skin", None, skin=(0.6, 0.2, 0.03))
STEM = B.principled("stem2", (0.12, 0.1, 0.04), rough=0.8)
for k, (dx, dy, r_) in enumerate([(0.2, 2.0, 0.3), (-0.22, 2.25, 0.24), (0.18, 2.75, 0.26), (0.25, 3.4, 0.22), (-0.05, 3.9, 0.2),
                                  (0.08, 1.45, 0.19)]):
    K.pumpkin_parts(cargo, (BX + dx, BY0 + dy, 0.34 + r_ * 0.62), r_, 0, 1, seed=200 + k)
cargo.build("cargo", [SKIN, STEM], smooth=True)
K.jack("cargo_jack", (BX - 0.27, BY0 + 3.15, 0.34 + 0.2 * 0.7), 0.2, FACES[1], rz=-0.35, seed=211,
       light_w=F("cargow", 60.0), strength=F("jackglow", 30.0))
pm = K.MB()
bow = Vector((BX, BY1 - 0.35, 0.6))
hook_top = bow + Vector((-0.45, 0.25, 1.85))
pm.cyl(bow, hook_top, 0.035, n=6, mat=0)
pm.cyl(hook_top, hook_top + Vector((-0.35, 0.45, 0.05)), 0.03, n=6, mat=0)
pm.cyl(hook_top + Vector((-0.35, 0.45, 0.05)), hook_top + Vector((-0.35, 0.45, -0.25)), 0.012, n=4, mat=0)
pm.build("prow_pole", [METAL])
K.jack("prow_jack", tuple(hook_top + Vector((-0.35, 0.45, -0.5))), 0.22, FACES[0], rz=F("prowrz", -0.6), seed=212,
       light_w=F("prow", 160.0), strength=F("jackglow", 30.0))

# the ferryman at the stern, back to us, leaning into the pole: hem flare, cape over the shoulders, pointed hood
fm = K.MB()
FY, FZ = BY0 + 0.75, 0.34
prof = [(0.0, 0.0), (0.41, 0.0), (0.39, 0.12), (0.34, 0.45), (0.29, 0.85), (0.28, 1.08), (0.36, 1.28), (0.33, 1.38),
        (0.25, 1.46), (0.14, 1.52), (0.0, 1.55)]
NU = 32


def lean(z):
    return 0.14 * (z / 1.55) ** 1.5                                   # leans forward (+y) into the push


for j, (rr, zz) in enumerate(prof):
    for i in range(NU):
        th = 2 * math.pi * i / NU
        wob = 1 + 0.07 * math.sin(7 * th + j) * max(0.0, 1 - zz / 1.1)       # cloth folds near the hem
        fm.v.append((BX + rr * wob * math.cos(th) * 1.12, FY + lean(zz) + rr * wob * math.sin(th) * 0.8, FZ + zz))
for j in range(len(prof) - 1):
    for i in range(NU):
        a_, b_ = j * NU + i, j * NU + (i + 1) % NU
        fm.f.append((a_, b_, b_ + NU, a_ + NU))
        fm.m.append(0)
hood_c = Vector((BX, FY + lean(1.55) + 0.06, FZ + 1.67))                # head bowed forward
for j in range(15):
    ph = math.pi * j / 14
    for i in range(NU):
        th = 2 * math.pi * i / NU
        r0 = 0.195 * math.sin(ph)
        pk = F("hoodpk", 0.17) * max(0.0, -math.sin(th)) ** 2 * max(0.0, math.cos(ph)) ** 1.5    # peak up and back (-y)
        fm.v.append((hood_c.x + r0 * math.cos(th), hood_c.y + r0 * math.sin(th) * 1.08 - pk * 0.8,
                     hood_c.z + 0.23 * math.cos(ph) + pk * 0.55))
off = len(prof) * NU
for j in range(14):
    for i in range(NU):
        a_, b_ = off + j * NU + i, off + j * NU + (i + 1) % NU
        fm.f.append((a_, b_, b_ + NU, a_ + NU))
        fm.m.append(0)
# arms holding the pole: it enters the water just behind the stern and rises past the figure's right side
pole_lo = Vector((BX + 0.66, FY - 1.5, -1.0))
pole_hi = Vector((BX + 0.16, FY + 1.1, 3.7))


def on_pole(z):
    t = (z - pole_lo.z) / (pole_hi.z - pole_lo.z)
    return pole_lo.lerp(pole_hi, t)


grip_lo, grip_hi = on_pole(FZ + 1.0), on_pole(FZ + 1.48)
for sh, g in (((BX + 0.26, FY + lean(1.3), FZ + 1.33), grip_lo), ((BX - 0.24, FY + lean(1.3), FZ + 1.33), grip_hi)):
    sh = Vector(sh)
    el_ = sh.lerp(g, 0.5) + Vector((0.04, -0.05, -0.14))
    fm.cyl(sh, el_, 0.1, 0.09, n=12, mat=0)
    fm.cyl(el_, g, 0.09, 0.075, n=12, mat=0)                       # wide sleeve
fm.build("ferryman", [CLOAK], smooth=True)
pl = K.MB()
pl.cyl(pole_lo, pole_hi, 0.03, n=8, mat=0)
pl.build("pole", [POLE_M])
# lantern faces stay crisp in the painting: the shared jack material carries the actor AOV too
for key, m in list(K._CACHE.items()):
    if key != "stem":
        B.aov_mask(m, "actor", 0.8)

# ---------------------------------------------------------------- camera, passes, render
tgt = cam_loc + Vector((math.sin(math.radians(F("yaw", 0.0))), 1.0, math.tan(math.radians(F("pitch", 6.0))))) * 30
cam = B.camera(tuple(cam_loc), tuple(tgt), lens=F("lens", 24.0), focus=F("focus", 10.0), fstop=F("fstop", 64.0),
               shift_y=F("shift", 0.06))
B.passes(mist=(2.0, 120.0, "LINEAR"), extra=("z", "normal"))
if "check" in A:
    nv = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == "MESH")
    print("CHECK objects", len(bpy.data.objects), "verts", nv, "lights", len([o for o in bpy.data.objects if o.type == "LIGHT"]))
    sys.exit(0)
B.render(TAG, outputs=("Image",))
B.render_data(TAG, hide=[haze])
