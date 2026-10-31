"""30 · Arcane painterly: the pumpkin carver's attic workshop at night.
A long, worn workbench runs diagonally toward a tall lancet window. Jack-o'-lanterns, a half-carved pumpkin with a
knife in it, a tool roll, candles and an oil lamp glow warm on the near side; cold teal moonlight pours through the
window in dusty shafts. A raven stands by the latch, dried herbs hang from the collar beam, and outside lie steep
roofs, a spire and a big moon.
python src/gpu_run.py blender/30_arcane_workshop.py -- tag=30-test pct=35 samples=16 [check=1]"""
import math
import random
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/blender")
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

import bl_build as K  # noqa: E402
import bl_common as B  # noqa: E402

A = B.args()
PCT, SPP, TAG = int(A.get("pct", 35)), int(A.get("samples", 16)), A.get("tag", "30-test")
CHECK = A.get("check") == "1"
WORK = ROOT + "/work"
rnd = random.Random(30)
sc = B.empty_scene()
B.cycles(samples=SPP, pct=PCT, bounces=5, clamp_indirect=4.0, vol_bounces=0, vol_step=float(A.get("volstep", 1.0)))
sc.cycles.volume_max_steps = int(A.get("volmax", 256))

HW, KNEE, RIDGE = 2.2, 1.3, 4.0            # attic half width, knee wall height, ridge height (x = 0)
YF = -6.5                                  # front wall (behind the camera); the gable with the window is at y = 0
WT = 0.42                                  # gable wall thickness = window reveal depth
WX, WW, WS, WA = 0.35, 0.48, 1.0, 2.3      # window centre x, half width, sill z, arch spring z
BT = 0.84                                  # bench top height
SLOPE = (RIDGE - KNEE) / HW


def gable_z(x):
    return RIDGE - abs(x) * SLOPE


def arch_z(x):
    """Top of the lancet opening at x (equilateral pointed arch springing at WA)."""
    r = 2 * WW
    cx = WX + WW if x < WX else WX - WW
    return WA + math.sqrt(max(0.0, r * r - (x - cx) ** 2))


# ---------------------------------------------------------------- materials


def wood_mat(name, c0, c1, scale=1.0, rough=0.7, bands="X", worn=0.0):
    m, t = K.mat(name)
    tc = t.coords("Object")
    mp = t.n("ShaderNodeMapping")
    s = {"X": (9.0, 1.0, 1.0), "Y": (1.0, 9.0, 1.0), "Z": (1.0, 1.0, 9.0)}[bands]
    mp.inputs["Scale"].default_value = tuple(v * scale for v in s)
    t.link(tc, mp.inputs["Vector"])
    wave = t.n("ShaderNodeTexWave", Scale=3.0, Distortion=9.0, Detail=4.0)
    wave.bands_direction = "X"
    t.link(mp.outputs[0], wave.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=5.0 * scale, Detail=8.0)
    t.link(tc, nz.inputs["Vector"])
    col = t.mixc(t.mapr(wave.outputs["Fac"], 0.2, 0.9), c0, c1)
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.5, 0.75, 0, 0.6), col, (c0[0] * 0.55, c0[1] * 0.55, c0[2] * 0.55))
    r = t.mapr(nz.outputs["Fac"], 0.3, 0.7, rough - 0.12 - worn, rough + 0.08)
    h = t.math("ADD", wave.outputs["Fac"], t.math("MULTIPLY", nz.outputs["Fac"], 0.5))
    t.out(t.bsdf(col, r, normal=t.bump(h, 0.3, 0.004), spec=0.45))
    return m


def plaster_mat(name, base):
    m, t = K.mat(name)
    tc = t.coords("Object")
    n1 = t.n("ShaderNodeTexNoise", Scale=1.2, Detail=8.0, Roughness=0.6)
    n2 = t.n("ShaderNodeTexNoise", Scale=9.0, Detail=4.0)
    for n in (n1, n2):
        t.link(tc, n.inputs["Vector"])
    stain = t.mapr(n1.outputs["Fac"], 0.5, 0.72, 0.0, 0.7)
    col = t.mixc(stain, base, (base[0] * 0.45, base[1] * 0.42, base[2] * 0.38))
    col = t.mixc(t.mapr(n2.outputs["Fac"], 0.3, 0.7, 0, 0.25), col, (base[0] * 1.2, base[1] * 1.2, base[2] * 1.15))
    t.out(t.bsdf(col, 0.9, normal=t.bump(t.math("ADD", n1.outputs["Fac"], n2.outputs["Fac"]), 0.25, 0.01)))
    return m


def glow_mat(name, color, strength):
    return B.emission(name, color, strength)


def glass_mat(name, tint=(0.85, 0.95, 0.9)):
    return B.principled(name, tint, rough=0.06, transmission=1.0, ior=1.08)      # thin walls: barely refract


BENCH = wood_mat("bench", (0.20, 0.115, 0.06), (0.32, 0.19, 0.10), scale=1.4, rough=0.52, worn=0.18)
FLOOR = wood_mat("floor", (0.07, 0.045, 0.03), (0.12, 0.075, 0.045), scale=0.8, rough=0.75, bands="Y")
BEAM = wood_mat("beam", (0.06, 0.04, 0.028), (0.11, 0.07, 0.045), scale=0.9, rough=0.8, bands="Y")
BOARDS = wood_mat("boards", (0.05, 0.034, 0.025), (0.09, 0.06, 0.04), scale=0.7, rough=0.85, bands="Y")
FRAMEW = wood_mat("framew", (0.05, 0.035, 0.028), (0.08, 0.055, 0.04), scale=2.0, rough=0.7, bands="Z")
PLASTER = plaster_mat("plaster", (0.24, 0.29, 0.33))
IRON = B.principled("iron", (0.02, 0.02, 0.022), rough=0.45, metal=0.7)
STEEL = B.principled("steel", (0.62, 0.62, 0.64), rough=0.22, metal=1.0)
BRASS = B.principled("brass", (0.78, 0.52, 0.22), rough=0.28, metal=1.0)
LEATHER = B.principled("leather", (0.34, 0.16, 0.07), rough=0.55)
HANDLE = B.principled("handle", (0.30, 0.16, 0.07), rough=0.5)
WAX = B.principled("wax", (0.85, 0.78, 0.62), rough=0.45, subsurface=0.3)
SEED = B.principled("seed", (0.82, 0.74, 0.55), rough=0.45)
RIND = B.principled("rind", (0.62, 0.20, 0.03), rough=0.5)
FLESH = B.principled("flesh", (0.95, 0.62, 0.18), rough=0.55, subsurface=0.2)
HERB = B.principled("herb", (0.10, 0.10, 0.045), rough=0.85)
STRING = B.principled("string", (0.25, 0.2, 0.14), rough=0.9)
RAVEN = B.principled("raven", (0.010, 0.010, 0.014), rough=0.38, coat=0.35)
BOOKS = [B.principled(f"book{i}", c, rough=0.7) for i, c in
         enumerate([(0.22, 0.04, 0.05), (0.05, 0.12, 0.1), (0.25, 0.17, 0.07), (0.08, 0.06, 0.12)])]
GLASS = glass_mat("glass")
FLAME = glow_mat("flame", (1.0, 0.55, 0.16), 45.0)
POTION = glow_mat("potion", (1.0, 0.16, 0.55), float(A.get("potion", 3.0)))
AMBER = B.principled("amber", (0.6, 0.25, 0.03), rough=0.2, transmission=0.6)
GREENL = B.principled("greenl", (0.12, 0.3, 0.12), rough=0.2, transmission=0.6)
TOWN = B.principled("town", (0.004, 0.0045, 0.006), rough=0.8, spec=0.0)   # no sheen: the town reads as silhouettes
TOWNROOF = B.principled("townroof", (0.003, 0.0035, 0.005), rough=0.8, spec=0.0)
TOWNWIN = glow_mat("townwin", (1.0, 0.55, 0.2), float(A.get("townwin", 6.0)))
B.aov_mask(RAVEN, "actor")
B.aov_mask(FLAME, "actor")

# ---------------------------------------------------------------- shell: floor, gable with the window, roof, walls
f6 = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def boxv(x0, x1, y0, y1, z0, z1):
    return [(x, y, z) for z in (z0, z1) for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def box(name, x0, x1, y0, y1, z0, z1, mat):
    return B.mesh(name, boxv(x0, x1, y0, y1, z0, z1), f6, mat)


def bar_xz(mb, p0, p1, w, y0, y1, mat=0):
    """A bar in the window plane between xz points p0 and p1 (width w in the plane, depth y0..y1)."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(dx, dz)
    nx, nz = -dz / ln * w / 2, dx / ln * w / 2
    q = [(p0[0] + nx, p0[1] + nz), (p1[0] + nx, p1[1] + nz), (p1[0] - nx, p1[1] - nz), (p0[0] - nx, p0[1] - nz)]
    vs = [(x, y, z) for y in (y0, y1) for (x, z) in q]
    mb.add(vs, f6, mat)


box("floor", -HW - 0.4, HW + 0.4, YF - 0.3, WT + 0.3, -0.3, 0.0, FLOOR)
box("frontwall", -HW - 0.4, HW + 0.4, YF - 0.3, YF, -0.3, RIDGE + 0.5, PLASTER)
for s in (-1, 1):
    x0, x1 = sorted((s * HW, s * (HW + 0.3)))
    box(f"knee{s}", x0, x1, YF - 0.3, WT + 0.3, -0.3, KNEE + 0.4, PLASTER)
# window outline (counter-clockwise in xz), used for the cutter and the frame
outline = [(WX - WW, WS), (WX + WW, WS), (WX + WW, WA)]
for i in range(1, 15):
    a = math.radians(60.0 * i / 15)
    outline.append((WX - WW + 2 * WW * math.cos(a), WA + 2 * WW * math.sin(a)))
for i in range(0, 15):
    a = math.radians(120.0 + 60.0 * i / 15)
    outline.append((WX + WW + 2 * WW * math.cos(a), WA + 2 * WW * math.sin(a)))
outline.append((WX - WW, WA))
gable_poly = [(-HW - 0.4, -0.3), (HW + 0.4, -0.3), (HW + 0.4, KNEE - 0.4 * SLOPE), (0.0, RIDGE + 0.15),
              (-HW - 0.4, KNEE - 0.4 * SLOPE)]
n = len(gable_poly)
gv = [(x, y, z) for y in (0.0, WT) for (x, z) in gable_poly]
gf = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
gable = B.mesh("gable", gv, gf, PLASTER)
m = len(outline)
cv = [(x, y, z) for y in (-0.2, WT + 0.2) for (x, z) in outline]
cf = [tuple(range(m)), tuple(range(2 * m - 1, m - 1, -1))] + [(i, m + i, m + (i + 1) % m, (i + 1) % m) for i in range(m)]
cutter = B.mesh("cutter", cv, cf)
cutter.hide_render = True
cutter.display_type = "WIRE"
bm = gable.modifiers.new("window", "BOOLEAN")
bm.operation = "DIFFERENCE"
bm.solver = "EXACT"
bm.object = cutter

# roof slabs (inner surfaces are the attic's sloping ceilings), rafters, purlins, collar beams
for s in (-1, 1):
    d = Vector((-s * HW, RIDGE - KNEE)).normalized()
    nrm = Vector((s * (RIDGE - KNEE), HW)).normalized()                  # outward, away from the room
    p0 = Vector((s * HW, KNEE)) - d * 0.6
    p1 = Vector((0.0, RIDGE)) + d * 0.35
    q = [p0, p1, p1 + nrm * 0.3, p0 + nrm * 0.3]
    rv = [(p.x, y, p.y) for y in (YF - 0.3, WT + 0.4) for p in q]
    B.mesh(f"roof{s}", rv, f6, BOARDS)
mb = K.MB()
y = YF + 0.4
while y < -0.2:
    for s in (-1, 1):
        a = Vector((s * HW, KNEE - 0.05))
        b = Vector((0.0, RIDGE - 0.05))
        dd = (b - a).normalized()
        nn = Vector((-s * (RIDGE - KNEE), -HW)).normalized() * 0.09      # hang below the boards
        p0, p1 = a + nn, b + nn
        bar_xz(mb, (p0.x, p0.y), (p1.x, p1.y), 0.16, y - 0.05, y + 0.05, 0)
    y += 0.68
for s in (-1, 1):                                    # purlins along the slopes
    xm = s * HW * 0.5
    zm = gable_z(xm) - 0.2
    mb.boxr(xm - 0.07, xm + 0.07, YF, 0.0, zm - 0.09, zm + 0.09, 0)
mb.boxr(-0.08, 0.08, YF, 0.0, RIDGE - 0.32, RIDGE - 0.12, 0)        # ridge beam
COLLARS = (-1.45, -2.7)
for yc in COLLARS:
    zc = 2.62
    xe = HW * (RIDGE - zc) / (RIDGE - KNEE) + 0.05
    mb.boxr(-xe, xe, yc - 0.07, yc + 0.07, zc - 0.08, zc + 0.08, 0)
mb.build("timbers", [BEAM])

# ---------------------------------------------------------------- the window: frame, mullion, transoms, latch
fm = K.MB()
FY0, FY1 = 0.16, 0.26
for i in range(len(outline)):
    p0, p1 = outline[i], outline[(i + 1) % len(outline)]
    if p0[1] == WS and p1[1] == WS:
        continue
    bar_xz(fm, p0, p1, 0.09, FY0, FY1, 0)
bar_xz(fm, (WX - WW - 0.02, WS + 0.03), (WX + WW + 0.02, WS + 0.03), 0.08, FY0 - 0.02, FY1 + 0.02, 0)   # bottom rail
bar_xz(fm, (WX, WS), (WX, arch_z(WX) - 0.02), 0.07, FY0 - 0.01, FY1 + 0.01, 0)                        # mullion
for zt in (WA,):
    bar_xz(fm, (WX - WW, zt), (WX + WW, zt), 0.06, FY0, FY1, 0)                                      # transom
for xo in (-0.24, 0.24):                                                                               # arch lights
    x = WX + xo
    bar_xz(fm, (x, WA), (x, arch_z(x) - 0.01), 0.022, FY0 + 0.03, FY1 - 0.03, 0)
fm.build("winframe", [FRAMEW])
lm = K.MB()                                           # iron latch on the mullion, hinge straps
STAY_Z = float(A.get("stay_z", 1.72))
lm.boxr(WX - 0.05, WX + 0.05, FY0 - 0.03, FY0, STAY_Z - 0.06, STAY_Z + 0.02, 0)        # latch plate
lm.cyl((WX + 0.04, FY0 - 0.04, STAY_Z), (WX + 0.04, FY0 - 0.04, STAY_Z - 0.14), 0.012, 0.012, n=8)
lm.cyl((WX - 0.02, FY0 - 0.05, STAY_Z), (WX - WW + 0.03, FY0 - 0.05, STAY_Z), 0.011, 0.011, n=8)   # casement stay
lm.cyl((WX - WW + 0.03, FY0 - 0.05, STAY_Z), (WX - WW + 0.03, FY0 - 0.01, STAY_Z), 0.011, 0.011, n=8)
for zh in (1.2, 2.0):
    lm.boxr(WX - WW + 0.02, WX - WW + 0.16, FY0 - 0.02, FY0, zh - 0.015, zh + 0.015, 0)
    lm.boxr(WX + WW - 0.16, WX + WW - 0.02, FY0 - 0.02, FY0, zh - 0.015, zh + 0.015, 0)
lm.build("latch", [IRON])

# ---------------------------------------------------------------- the raven on the sill by the latch (actor)
def ellipsoid(name, c, r, rot=(0, 0, 0), mat=None, nu=20, nv=12):
    vs, fs = [], []
    for j in range(nv + 1):
        ph = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            vs.append((math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)))
    for j in range(nv):
        for i in range(nu):
            a_, b_ = j * nu + i, j * nu + (i + 1) % nu
            fs.append((a_, b_, b_ + nu, a_ + nu))
    ob = B.mesh(name, [(x * r[0], y * r[1], z * r[2]) for x, y, z in vs], fs, mat, smooth=True)
    ob.location = c
    ob.rotation_euler = rot
    return ob


RVX, RVY = WX - float(A.get("raven_dx", 0.2)), FY0 - 0.05
rz = STAY_Z + 0.011                                    # feet on the iron stay
facing = math.radians(float(A.get("raven_rz", 200)))  # body axis: head toward -x (into the room's side), turned to us
rv = K.MB()
ca, sa = math.cos(facing), math.sin(facing)


def RV(u, v, w):
    """Raven-local (u forward to the head, v left, w up) -> world."""
    return (RVX + u * ca - v * sa, RVY + u * sa + v * ca, rz + w)


rv.cyl(RV(0.0, 0.025, 0.0), RV(0.0, 0.025, 0.07), 0.006, 0.005, n=6)
rv.cyl(RV(0.0, -0.025, 0.0), RV(0.0, -0.025, 0.07), 0.006, 0.005, n=6)
for s in (-1, 1):                                      # toes gripping the sill
    for ang in (-0.5, 0.0, 0.5):
        rv.cyl(RV(0.0, s * 0.025, 0.004), RV(0.045 * math.cos(ang), s * 0.025 + 0.045 * math.sin(ang), 0.002),
               0.004, 0.003, n=5)
    rv.cyl(RV(0.0, s * 0.025, 0.004), RV(-0.035, s * 0.025, 0.002), 0.004, 0.003, n=5)
tail = [RV(-0.07, 0.0, 0.12), RV(-0.26, 0.0, 0.03)]
rv.cyl(tail[0], tail[1], 0.045, 0.02, n=10)
rv.cyl(RV(0.12, 0.0, 0.245), RV(0.205, 0.0, 0.228), 0.022, 0.003, n=10)      # beak
rv.build("raven_parts", [RAVEN], smooth=True)
ellipsoid("raven_body", RV(0.0, 0.0, 0.15), (0.13, 0.07, 0.075), rot=(0, math.radians(-28), facing), mat=RAVEN)
ellipsoid("raven_head", RV(0.085, 0.0, 0.245), (0.05, 0.042, 0.046), rot=(0, 0, facing), mat=RAVEN)
ellipsoid("raven_throat", RV(0.065, 0.0, 0.2), (0.045, 0.04, 0.05), rot=(0, math.radians(20), facing), mat=RAVEN)
ellipsoid("raven_tailtip", RV(-0.22, 0.0, 0.045), (0.07, 0.035, 0.012), rot=(0, math.radians(-25), facing), mat=RAVEN)

# ---------------------------------------------------------------- the bench (long diagonal workbench)
P0, P1 = Vector((-0.85, -4.15)), Vector((0.6, -0.75))
BD = (P1 - P0).normalized()
BR = Vector((BD.y, -BD.x))                             # to the right of the bench axis
BL = (P1 - P0).length
HALF = 0.42
bang = math.atan2(BD.y, BD.x)


def bench_pt(u, v):
    p = P0 + BD * u + BR * v
    return p.x, p.y


bm_ = K.MB()
for k in range(5):                                     # top made of five planks with small gaps
    v0 = -HALF + k * 2 * HALF / 5 + 0.004
    v1 = -HALF + (k + 1) * 2 * HALF / 5 - 0.004
    c = P0 + BD * (BL / 2) + BR * ((v0 + v1) / 2)
    bm_.box((c.x, c.y, BT - 0.04 + rnd.uniform(-0.003, 0.003)), (BL + rnd.uniform(-0.04, 0.0), v1 - v0, 0.08), 0,
            rz=bang)
for u in (0.25, BL - 0.25):
    for v in (-HALF + 0.08, HALF - 0.08):
        x, y = bench_pt(u, v)
        bm_.box((x, y, (BT - 0.08) / 2), (0.1, 0.1, BT - 0.08), 0, rz=bang)
    x, y = bench_pt(u, 0)
    bm_.box((x, y, 0.2), (0.07, 2 * HALF - 0.1, 0.07), 0, rz=bang)
c = P0 + BD * (BL / 2)
bm_.box((c.x, c.y, 0.22), (BL - 0.5, 0.07, 0.07), 0, rz=bang)
bm_.build("bench", [BENCH])


def on_bench(x, y):
    d = Vector((x, y)) - P0
    return 0.05 < d.dot(BD) < BL - 0.05 and abs(d.dot(BR)) < HALF - 0.03


# ---------------------------------------------------------------- pumpkins: four lit jacks and a half-carved one
faces = {k: f"{WORK}/face_{k}.png" for k in ("classic", "grin", "scary")}
faces["sly"] = f"{WORK}/30-face-sly.png"
faces["half"] = f"{WORK}/30-face-half.png"
for p in faces.values():
    K.face_texture(p)
JW = float(A.get("jackw", 4.0))
JACKS = [  # name, (x, y), r, face, light W, rz offset
    ("jack_big", (-0.40, -2.75), 0.205, "sly", JW * 1.3, 0.0),
    ("jack_grin", (-0.122, -1.7), 0.125, "grin", JW * 0.6, 0.15),
    ("jack_tall", (0.427, -1.0), 0.19, "scary", JW * 0.9, -0.1),
    ("jack_small", (0.184, -2.2), 0.085, "classic", JW * 0.4, 0.3),
]
cam_xy = Vector((float(A.get("camx", -0.35)), float(A.get("camy", -4.4))))
jack_objs = []
for name, (x, y), r, face, w, rzo in JACKS:
    assert on_bench(x, y), name
    to_cam = cam_xy - Vector((x, y))
    rz_ = math.atan2(to_cam.x, -to_cam.y) + rzo       # local -Y faces the camera
    ob = K.jack(name, (x, y, BT + r * 0.6 + 0.002), r, faces[face], rz=rz_, seed=sum(map(ord, name)) % 97,
                light_w=w, strength=float(A.get("jstr", 30.0)))
    jack_objs.append(ob)
    L = bpy.data.objects[name + "_candle"]
    L.visible_volume_scatter = False
    L.location = (x, y, BT + r * 0.55)
    if A.get("shadowlink", "1") == "1":
        # the shell shadows the moon and the other candles, but not its own candle (shadow linking: exclude it)
        ob.visible_shadow = True
        coll = bpy.data.collections.new(name + "_self")
        coll.objects.link(ob)
        coll.collection_objects[0].light_linking.link_state = "EXCLUDE"
        L.light_linking.blocker_collection = coll
        L.light_linking.receiver_collection = coll       # ...and does not light its own shell (no red SSS blotch)
# the small jack goes up on the right shelf by the window instead: a warm spark in the cold upper right
js = bpy.data.objects["jack_small"]
jsx, jsy, jsr = 1.15, -0.15, 0.085
js.location = (jsx, jsy, 1.36 + jsr * 0.6 + 0.002)
to_cam = cam_xy - Vector((jsx, jsy))
js.rotation_euler = (0, 0, math.atan2(to_cam.x, -to_cam.y) + 0.25)
bpy.data.objects["jack_small_candle"].location = (jsx, jsy, 1.36 + jsr * 0.55)
for mat_ in {s.material for ob in jack_objs for s in ob.material_slots if s.material and s.material.name.startswith("jack")}:
    B.aov_mask(mat_, "actor")
    for nd in mat_.node_tree.nodes:                    # no subsurface: the candle inside made a red blotch through it
        if nd.type == "BSDF_PRINCIPLED":
            nd.inputs["Subsurface Weight"].default_value = 0.0

# the half-carved pumpkin: eyes and nose cut, mouth only scratched, lid off, a knife stuck in its crown
HX, HY, HR = -0.229, -3.25, 0.165
assert on_bench(HX, HY)
hmat = K.pumpkin_material("half_skin", faces["half"], glow=(1.0, 0.5, 0.16), strength=float(A.get("hglow", 0.35)))
B.aov_mask(hmat, "actor")
mbh = K.MB()
K.pumpkin_parts(mbh, (0, 0, 0), HR, 0, 1, squash=0.74, seed=5, stem=False)
hob = mbh.build("half_pumpkin", [hmat, RIND], smooth=True)
hob["face_scale"] = HR
to_cam = cam_xy - Vector((HX, HY))
hob.location = (HX, HY, BT + HR * 0.62 + 0.002)
hob.rotation_euler = (0, 0, math.atan2(to_cam.x, -to_cam.y) + 0.35)
# the cut opening: a ring of pale flesh and a dark hollow, seen just above eye line
ztop = BT + HR * 0.62 + HR * 0.74 - HR * 0.2
op = K.MB()
op.cyl((HX, HY, ztop - 0.01), (HX, HY, ztop + 0.012), HR * 0.42, HR * 0.40, n=24)
op.build("half_cut", [FLESH])
op2 = K.MB()
op2.cyl((HX, HY, ztop - 0.005), (HX, HY, ztop + 0.014), HR * 0.34, HR * 0.33, n=24)
op2.build("half_hole", [B.principled("hollow", (0.12, 0.04, 0.01), rough=0.8)])
# knife: blade buried in the crown at an angle, wooden handle up and toward the camera-right
kn = K.MB()
kb0 = Vector((HX + 0.03, HY + 0.02, ztop - 0.05))
kdir = Vector((0.35, -0.45, 0.82)).normalized()
kb1 = kb0 + kdir * 0.13
side = kdir.cross(Vector((0, 0, 1))).normalized()
w0, w1 = 0.016, 0.004
blade = [kb0 + side * w0, kb1 + side * w1 * 0.3, kb1 - side * w1, kb0 - side * w0]
nrm_ = side.cross(kdir).normalized() * 0.0018
kn.add([tuple(p + nrm_) for p in blade] + [tuple(p - nrm_) for p in blade], f6, 0)
kn.cyl(kb1 - kdir * 0.005, kb1 + kdir * 0.02, 0.017, 0.017, n=10, mat=1)            # bolster
kn.cyl(kb1 + kdir * 0.02, kb1 + kdir * 0.095, 0.014, 0.016, n=10, mat=2)            # handle
kn.build("knife", [STEEL, BRASS, HANDLE], smooth=False)
for m_ in (STEEL, HANDLE):
    B.aov_mask(m_, "actor")
# the lid with its stem, tipped over beside the pumpkin, showing the pale cut underside
lid = K.MB()
nu, nv, phm = 24, 6, 0.62
lv = []
for j in range(nv + 1):
    ph = phm * j / nv
    for i in range(nu):
        th = 2 * math.pi * i / nu
        rib = 1 - 0.07 * (1 - abs(math.cos(5 * th)) ** 0.6)
        lv.append((HR * math.sin(ph) * rib * math.cos(th), HR * math.sin(ph) * rib * math.sin(th),
                   HR * 0.74 * math.cos(ph) - HR * 0.22 * math.exp(-(ph / 0.32) ** 2)))
lf = [(j * nu + i, j * nu + (i + 1) % nu, (j + 1) * nu + (i + 1) % nu, (j + 1) * nu + i) for j in range(nv) for i in range(nu)]
zc_ = HR * 0.74 * (1 + math.cos(phm)) / 2              # build the lid around its own centre
lv = [(x, y, z - zc_) for (x, y, z) in lv]
lid.add(lv, lf, 0)
lid.add([lv[nv * nu + i] for i in range(nu)], [tuple(range(nu - 1, -1, -1))], 1)      # cut face
lid.cyl((0, 0, HR * 0.74 - HR * 0.22 - zc_), (0.01, 0.0, HR * 0.74 + HR * 0.12 - zc_), HR * 0.1, HR * 0.07, n=7, mat=2)
lob = lid.build("lid", [RIND, FLESH, B.principled("stem30", (0.14, 0.11, 0.04), rough=0.8)], smooth=True)
lx, ly = -0.533, -3.15
assert on_bench(lx, ly)
lob.location = (lx, ly, BT + 0.012)
lob.rotation_euler = (math.radians(-14), math.radians(9), math.radians(-35))

# rind curls and seeds around the half-carved pumpkin
cm = K.MB()
for k in range(6):
    cx, cy = HX + rnd.uniform(-0.3, 0.05), HY + rnd.uniform(-0.25, 0.1)
    if not on_bench(cx, cy) or math.hypot(cx - HX, cy - HY) < HR + 0.03:
        continue
    ax = rnd.uniform(0, math.pi)
    rr = rnd.uniform(0.012, 0.022)
    turns = rnd.uniform(1.2, 2.2)
    ln = rnd.uniform(0.05, 0.09)
    pts = []
    N_ = 40
    for i in range(N_ + 1):
        t = i / N_
        a = 2 * math.pi * turns * t
        along = (t - 0.5) * ln
        pts.append((along, rr * math.cos(a), rr * math.sin(a) + rr))
    vs = []
    for (u, v, w) in pts:
        for e in (-0.006, 0.006):
            x = cx + (u + e) * math.cos(ax) - v * math.sin(ax)
            y = cy + (u + e) * math.sin(ax) + v * math.cos(ax)
            vs.append((x, y, BT + w))
    fs = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(N_)]
    cm.add(vs, fs, 0)
cm.build("curls", [RIND])
sd = K.MB()
for k in range(46):
    a = rnd.uniform(0, 2 * math.pi)
    d = HR + 0.03 + abs(rnd.gauss(0, 0.11))
    sx, sy = HX - 0.06 + d * math.cos(a) * 1.2, HY + d * math.sin(a) * 0.8
    if not on_bench(sx, sy):
        continue
    th = rnd.uniform(0, math.pi)
    vs, fs = [], []
    for i in range(10):
        b = 2 * math.pi * i / 10
        vs.append((sx + 0.011 * math.cos(b) * math.cos(th) - 0.0065 * math.sin(b) * math.sin(th),
                   sy + 0.011 * math.cos(b) * math.sin(th) + 0.0065 * math.sin(b) * math.cos(th), BT + 0.0022))
    vs.append((sx, sy, BT + 0.004))
    fs = [(i, (i + 1) % 10, 10) for i in range(10)]
    sd.add(vs, fs, 0)
sd.build("seeds", [SEED])

# ---------------------------------------------------------------- the tool roll in the foreground
TRX, TRY = float(A.get("roll_x", -0.42)), float(A.get("roll_y", -3.47))
tr = K.MB()
RA = bang + math.radians(float(A.get("roll_rot", 12)))
tr.box((TRX, TRY, BT + 0.003), (0.48, 0.3, 0.006), 0, rz=RA)
ca_, sa_ = math.cos(RA), math.sin(RA)


def TR(u, v, w=0.0):
    return (TRX + u * ca_ - v * sa_, TRY + u * sa_ + v * ca_, BT + 0.006 + w)


for k in range(6):                                     # pocket flap stitching ridges
    u = -0.2 + k * 0.08
    tr.box(TR(u, -0.08, 0.002)[:2] + (BT + 0.009,), (0.006, 0.12, 0.005), 0, rz=RA)
tr.build("toolroll", [LEATHER])
tl = K.MB()
for k in range(6):
    u = -0.22 + k * 0.08 + 0.04
    tl.cyl(TR(u, -0.12, 0.01), TR(u, -0.01, 0.01), 0.0085, 0.0105, n=10, mat=1)       # handle
    kind = k % 3
    if kind == 0:                                      # flat blade
        tl.box(TR(u, 0.06, 0.006)[:2] + (BT + 0.012,), (0.016, 0.14, 0.003), 0, rz=RA)
    elif kind == 1:                                    # gouge
        tl.cyl(TR(u, -0.01, 0.012), TR(u, 0.12, 0.008), 0.006, 0.008, n=8, mat=0)
    else:                                              # awl / scoop
        tl.cyl(TR(u, -0.01, 0.012), TR(u, 0.1, 0.01), 0.004, 0.002, n=6, mat=0)
        tl.cyl(TR(u, 0.1, 0.01), TR(u, 0.125, 0.012), 0.012, 0.012, n=8, mat=0)
tl.build("tools", [STEEL, HANDLE])

# ---------------------------------------------------------------- candle in a dish, jar of candles, oil lamp


def flame(name, p, h=0.022, w=0.0075, light_w=1.5):
    fl = ellipsoid(name, (p[0], p[1], p[2] + h * 0.55), (w, w, h * 0.55), mat=FLAME, nu=10, nv=8)
    fl.visible_shadow = False
    L = B.light("POINT", (p[0], p[1], p[2] + h * 0.6), light_w, color=(1.0, 0.56, 0.22), size=0.01, name=name + "_l")
    L.visible_volume_scatter = False
    return fl


CDX, CDY = -0.066, -2.4
assert on_bench(CDX, CDY)
cd = K.MB()
cd.cyl((CDX, CDY, BT), (CDX, CDY, BT + 0.012), 0.065, 0.07, n=20, mat=0)
cd.cyl((CDX, CDY, BT + 0.012), (CDX, CDY, BT + 0.055), 0.024, 0.023, n=14, mat=1)
cd.cyl((CDX, CDY, BT + 0.055), (CDX, CDY, BT + 0.064), 0.002, 0.002, n=4, mat=2)
cd.build("candle_dish", [BRASS, WAX, IRON])
flame("candle_flame", (CDX, CDY, BT + 0.064), light_w=float(A.get("candlew", 1.4)))
JRX, JRY = -0.25, -1.9
assert on_bench(JRX, JRY)
jr = K.MB()
jr.cyl((JRX, JRY, BT), (JRX, JRY, BT + 0.16), 0.055, 0.055, n=20, mat=0, caps=False)
jr.cyl((JRX, JRY, BT), (JRX, JRY, BT + 0.004), 0.055, 0.055, n=20, mat=0)
for k in range(4):
    a = k * math.pi / 2 + 0.4
    x, y = JRX + 0.025 * math.cos(a), JRY + 0.025 * math.sin(a)
    jr.cyl((x, y, BT + 0.004), (x + 0.004 * math.cos(a), y + 0.004 * math.sin(a), BT + 0.17 + k * 0.012), 0.009,
           0.009, n=8, mat=1)
jr.build("candle_jar", [GLASS, WAX])
OLX, OLY = 0.55, -1.45
assert on_bench(OLX, OLY)
ol = K.MB()
ol.cyl((OLX, OLY, BT), (OLX, OLY, BT + 0.012), 0.05, 0.05, n=16)
olb = ellipsoid("oil_body", (OLX, OLY, BT + 0.045), (0.065, 0.065, 0.036), mat=BRASS, nu=20, nv=10)
sp0 = Vector((OLX, OLY, BT + 0.05))
spd = (Vector((CDX, CDY, 0)) - Vector((OLX, OLY, 0))).normalized()
sp1 = sp0 + spd * 0.1 + Vector((0, 0, 0.02))
ol.cyl(sp0, sp1, 0.018, 0.011, n=12)
ol.cyl(sp0 - spd * 0.05 + Vector((0, 0, 0.02)), sp0 - spd * 0.1 + Vector((0, 0, 0.04)), 0.008, 0.008, n=6)
ol.cyl(sp0 - spd * 0.1 + Vector((0, 0, 0.04)), sp0 - spd * 0.08 + Vector((0, 0, -0.01)), 0.008, 0.008, n=6)
ol.build("oil_lamp", [BRASS])
flame("oil_flame", (sp1.x, sp1.y, sp1.z + 0.006), h=0.03, w=0.009, light_w=float(A.get("oilw", 2.2)))

# ---------------------------------------------------------------- shelves beside the window: jars, gourds, books
sh = K.MB()
SHELVES = [(-1.75, -0.32, 1.42), (-1.75, -0.32, 1.98), (1.0, 1.65, 1.36)]
for (x0, x1, z) in SHELVES:
    sh.boxr(x0, x1, -0.3, 0.0, z - 0.035, z, 0)
    for xb in (x0 + 0.12, x1 - 0.12):
        sh.boxr(xb - 0.02, xb + 0.02, -0.06, 0.0, z - 0.22, z - 0.035, 1)
sh.build("shelves", [BEAM, IRON])
jars = K.MB()
contents = K.MB()
JARS = [(-1.6, 1.42, 0.06, 0.2, 1), (-1.42, 1.42, 0.045, 0.13, 2), (-0.9, 1.42, 0.05, 0.16, 0), 
        (-1.33, 1.98, 0.04, 0.12, 1), (-0.5, 1.42, 0.04, 0.11, 2)]
POTION_AT = None
for (x, z, r, h, kind) in JARS:
    y = -0.15 + rnd.uniform(-0.04, 0.04)
    jars.cyl((x, y, z), (x, y, z + h), r, r, n=18, caps=False)
    jars.cyl((x, y, z), (x, y, z + 0.004), r, r, n=18)
    jars.cyl((x, y, z + h), (x, y, z + h + 0.025), r * 0.75, r * 0.75, n=14, mat=1)    # cork / lid
    contents.cyl((x, y, z + 0.005), (x, y, z + h * rnd.uniform(0.45, 0.8)), r * 0.9, r * 0.9, n=16, mat=kind)
jars.build("jars", [GLASS, B.principled("cork", (0.3, 0.2, 0.1), rough=0.8)])
contents.build("jar_contents", [AMBER, GREENL, SEED, POTION])
fx, fy, fz = -0.56, -0.13, 1.98
POTION_AT = (fx, fy, fz + 0.07)
fl_ = K.MB()
fl_.cyl((fx, fy, fz + 0.12), (fx, fy, fz + 0.22), 0.018, 0.016, n=12, caps=False)
fl_.cyl((fx, fy, fz + 0.22), (fx, fy, fz + 0.24), 0.02, 0.02, n=12, mat=1)
fl_.build("flask_neck", [GLASS, B.principled("cork2", (0.3, 0.2, 0.1), rough=0.8)])
ellipsoid("flask", (fx, fy, fz + 0.07), (0.072, 0.072, 0.07), mat=GLASS, nu=20, nv=12)
ellipsoid("flask_potion", (fx, fy, fz + 0.06), (0.062, 0.062, 0.052), mat=POTION, nu=20, nv=12)
pl = B.light("POINT", POTION_AT, float(A.get("potionw", 2.0)), color=(1.0, 0.25, 0.65), size=0.04, name="potion_l")
pl.visible_volume_scatter = False
gm = K.MB()
skin_plain = K.pumpkin_material("plain_skin", None, skin=(0.5, 0.15, 0.025))
gourd = K.pumpkin_material("gourd_skin", None, skin=(0.42, 0.36, 0.12))
for (x, z, r, sq, mat_i) in [(-1.15, 1.42, 0.09, 0.7, 0), (-0.83, 1.98, 0.075, 0.75, 1), (-1.57, 1.98, 0.1, 0.68, 0),
                             (1.45, 1.36, 0.11, 0.7, 0)]:
    K.pumpkin_parts(gm, (x, -0.14, z + r * sq * 0.9), r, mat_i, 2, squash=sq, seed=int(x * 100), nu=32, nv=16)
gm.build("shelf_pumpkins", [skin_plain, gourd, B.principled("stem_s", (0.12, 0.1, 0.04), rough=0.8)], smooth=True)
bk = K.MB()
x = -1.2
for k in range(5):
    w_, h_ = rnd.uniform(0.035, 0.055), rnd.uniform(0.18, 0.26)
    bk.box((x, -0.14, 1.98 + h_ / 2), (w_, 0.17, h_), k % 4, rz=0.0)
    x += w_ + 0.004
bk.build("books", BOOKS)

# ---------------------------------------------------------------- dried herbs hanging from the collar beam
hb = K.MB()
st = K.MB()
for (hx, yc, ln, seed) in [(-0.95, COLLARS[0], 0.42, 1), (-0.45, COLLARS[0], 0.34, 2), (0.08, COLLARS[0], 0.3, 3),
                           (0.7, COLLARS[0], 0.4, 4), (-0.55, COLLARS[1], 0.5, 5), (0.25, COLLARS[1], 0.45, 6)]:
    r_ = random.Random(seed)
    top = Vector((hx, yc, 2.54))
    tie = top - Vector((0, 0, r_.uniform(0.12, 0.2)))
    st.cyl(top, tie, 0.003, 0.003, n=4)
    for k in range(22):
        a = r_.uniform(0, 2 * math.pi)
        spread = r_.uniform(0.04, 0.16)
        L_ = ln * r_.uniform(0.7, 1.0)
        p0 = tie + Vector((0, 0, 0.02))
        p1 = tie + Vector((math.cos(a) * spread * 0.4, math.sin(a) * spread * 0.4, -L_ * 0.5))
        p2 = tie + Vector((math.cos(a) * spread, math.sin(a) * spread, -L_))
        hb.cyl(p0, p1, 0.0035, 0.003, n=4, caps=False)
        hb.cyl(p1, p2, 0.003, 0.0015, n=4, caps=False)
        for j in range(4):                               # little dry leaves along the stem
            t = 0.3 + 0.18 * j
            q = p1 + (p2 - p1) * (t - 0.3) * 1.4 if t > 0.5 else p0 + (p1 - p0) * (t / 0.5)
            b = r_.uniform(0, 2 * math.pi)
            dvec = Vector((math.cos(b), math.sin(b), -0.6)).normalized() * 0.035
            sidev = dvec.cross(Vector((0, 0, 1))).normalized() * 0.008
            hb.add([tuple(q), tuple(q + dvec * 0.5 + sidev), tuple(q + dvec), tuple(q + dvec * 0.5 - sidev)],
                   [(0, 1, 2, 3)], 0)
hb.build("herbs", [HERB])
st.build("strings", [STRING])

# ---------------------------------------------------------------- the town outside: steep roofs, a spire, lit windows
GROUND = float(A.get("ground", -14.0))
tw = K.MB()
cam_ = Vector((float(A.get("camx", -0.35)), float(A.get("camy", -4.4))))
for row, yr in enumerate((9.0, 14.0, 20.0, 28.0, 38.0, 52.0, 70.0)):
    D = yr - cam_.y
    xa, xb = cam_.x + D * math.tan(math.radians(-6)), cam_.x + D * math.tan(math.radians(24))
    x = xa + rnd.uniform(-2, 0)
    while x < xb:
        w_ = rnd.uniform(3.0, 5.5) * (1 + row * 0.08)
        eave = GROUND + rnd.uniform(7.0, 11.0) + row * 0.6
        pitch = rnd.uniform(0.9, 1.5)
        ridge = eave + w_ / 2 * pitch
        dep = 7.0
        y0, y1 = yr, yr + dep
        if rnd.random() < 0.5:   # gable facing us
            poly = [(x, GROUND), (x + w_, GROUND), (x + w_, eave), (x + w_ / 2, ridge), (x, eave)]
            vs = [(px, yy, pz) for yy in (y0, y1) for (px, pz) in poly]
            n5 = 5
            fs = [tuple(range(n5)), tuple(range(2 * n5 - 1, n5 - 1, -1))] + [(i, n5 + i, n5 + (i + 1) % n5, (i + 1) % n5) for i in range(n5)]
            tw.add(vs, fs, 0)
            for (wx_, wz_) in ((x + w_ * 0.3, eave - 1.6), (x + w_ * 0.7, eave - 1.6), (x + w_ * 0.5, eave + 1.0),
                               (x + w_ * 0.3, eave - 4.2), (x + w_ * 0.7, eave - 4.2)):
                if rnd.random() < 0.38:
                    tw.boxr(wx_ - 0.35, wx_ + 0.35, y0 - 0.05, y0 + 0.02, wz_ - 0.5, wz_ + 0.5, 2)
        else:                    # ridge along x, roof slopes toward us
            tw.boxr(x, x + w_, y0, y1, GROUND, eave, 0)
            poly = [(y0 - 0.3, eave), (y1 + 0.3, eave), ((y0 + y1) / 2, eave + dep / 2 * pitch * 0.8)]
            vs = [(xx, py, pz) for xx in (x - 0.2, x + w_ + 0.2) for (py, pz) in poly]
            fs = [(0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
            tw.add(vs, fs, 1)
            for k in range(int(w_ / 1.6)):
                if rnd.random() < 0.33:
                    wx_ = x + 0.8 + k * 1.6
                    tw.boxr(wx_ - 0.3, wx_ + 0.3, y0 - 0.05, y0 + 0.02, eave - 1.8, eave - 0.9, 2)
        if rnd.random() < 0.6:
            cx_ = x + w_ * rnd.uniform(0.2, 0.8)
            tw.boxr(cx_ - 0.35, cx_ + 0.35, y0 + 2, y0 + 2.7, eave, ridge + 1.2, 0)
        x += w_ + rnd.uniform(0.0, 1.0)
# the spire: a tower with a tall octagonal needle and a lit belfry
SPX, SPY = float(A.get("spire_x", 4.4)), float(A.get("spire_y", 52.0))
tw.boxr(SPX - 1.25, SPX + 1.25, SPY - 1.25, SPY + 1.25, GROUND, GROUND + 15, 0)
tw.boxr(SPX - 0.3, SPX + 0.3, SPY - 1.3, SPY - 1.15, GROUND + 11.5, GROUND + 13.5, 2)
for s_ in (-1, 1):                                   # corner pinnacles
    for t_ in (-1, 1):
        tw.cyl((SPX + s_ * 1.15, SPY + t_ * 1.15, GROUND + 15), (SPX + s_ * 1.15, SPY + t_ * 1.15, GROUND + 17.5), 0.28, 0.02, n=6)
oc = [(SPX + 1.25 * math.cos(math.pi / 8 + k * math.pi / 4), SPY + 1.25 * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
apex = (SPX, SPY, GROUND + 27.6)
vs = [(x, y, GROUND + 15) for (x, y) in oc] + [apex]
tw.add(vs, [(k, (k + 1) % 8, 8) for k in range(8)] + [tuple(range(7, -1, -1))], 1)
tw.build("town", [TOWN, TOWNROOF, TOWNWIN])

# ---------------------------------------------------------------- moon, sky, moonlight, dusty air
MAZ, MEL = math.radians(float(A.get("moon_az", 12.2))), math.radians(float(A.get("moon_el", 7.9)))
mdir = Vector((math.sin(MAZ) * math.cos(MEL), math.cos(MAZ) * math.cos(MEL), math.sin(MEL)))
LAZ, LEL = math.radians(float(A.get("light_az", 3.0))), math.radians(float(A.get("light_el", 55.0)))
ldir = Vector((math.sin(LAZ) * math.cos(LEL), math.cos(LAZ) * math.cos(LEL), math.sin(LEL)))
moon = B.light("SUN", tuple(ldir * 10), float(A.get("moonlight", 18.0)), color=(0.55, 0.78, 1.0), size=math.radians(1.2),
               target=(0, 0, 0), name="moon")
w = sc.world
w.use_nodes = True
wt = w.node_tree
wt.nodes.clear()
tc = wt.nodes.new("ShaderNodeTexCoord")
nrmv = wt.nodes.new("ShaderNodeVectorMath")
nrmv.operation = "NORMALIZE"
wt.links.new(tc.outputs["Generated"], nrmv.inputs[0])
D_ = nrmv.outputs[0]


def wmath(op, a, b=0.0, clamp=False):
    nd = wt.nodes.new("ShaderNodeMath")
    nd.operation = op
    nd.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if hasattr(v, "is_output") or isinstance(v, bpy.types.NodeSocket):
            wt.links.new(v, nd.inputs[i])
        else:
            nd.inputs[i].default_value = v
    return nd.outputs[0]


def wmix(fac, a, b, blend="MIX"):
    nd = wt.nodes.new("ShaderNodeMix")
    nd.data_type = "RGBA"
    nd.blend_type = blend
    for sock, v in ((nd.inputs[0], fac), (nd.inputs[6], a), (nd.inputs[7], b)):
        if hasattr(v, "is_output") or isinstance(v, bpy.types.NodeSocket):
            wt.links.new(v, sock)
        else:
            sock.default_value = (*v, 1) if isinstance(v, tuple) else v
    return nd.outputs[2]


sep = wt.nodes.new("ShaderNodeSeparateXYZ")
wt.links.new(D_, sep.inputs[0])
el = sep.outputs["Z"]
ramp = wt.nodes.new("ShaderNodeValToRGB")
wt.links.new(el, ramp.inputs[0])
E = ramp.color_ramp.elements
SK = float(A.get("sky", 2.4))
E[0].position, E[0].color = 0.0, (0.30 * SK, 0.10 * SK, 0.22 * SK, 1)          # pink-magenta town glow at the horizon
E[1].position, E[1].color = 0.55, (0.006, 0.014, 0.03, 1)
e = E.new(0.07)
e.color = (0.07 * SK, 0.17 * SK, 0.22 * SK, 1)                                    # teal band
e = E.new(0.25)
e.color = (0.022 * SK, 0.06 * SK, 0.10 * SK, 1)
dot = wt.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
wt.links.new(D_, dot.inputs[0])
dot.inputs[1].default_value = tuple(mdir)
cosd = dot.outputs["Value"]
MR = math.radians(float(A.get("moon_r", 2.7)))
disc = wmath("MULTIPLY", wmath("SUBTRACT", cosd, math.cos(MR)), 1.0 / (math.cos(MR * 0.93) - math.cos(MR)), clamp=True)
halo1 = wmath("POWER", wmath("MAXIMUM", cosd, 0.0), 900.0)
halo2 = wmath("POWER", wmath("MAXIMUM", cosd, 0.0), 60.0)
mtex = wt.nodes.new("ShaderNodeTexNoise")
mtex.inputs["Scale"].default_value = 60.0
mtex.inputs["Detail"].default_value = 6.0
wt.links.new(D_, mtex.inputs["Vector"])
maria = wmath("MULTIPLY", wmath("SUBTRACT", mtex.outputs["Fac"], 0.42), 2.2, clamp=True)
mcol = wmix(maria, (1.0, 0.95, 0.84), (0.62, 0.66, 0.72))
# streaky clouds (stretched noise) drifting across the lower sky, lit at their edges near the moon
cn = wt.nodes.new("ShaderNodeTexNoise")
cn.inputs["Scale"].default_value = 2.2
cn.inputs["Detail"].default_value = 8.0
mp = wt.nodes.new("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (1.0, 1.0, 6.0)
wt.links.new(D_, mp.inputs["Vector"])
wt.links.new(mp.outputs[0], cn.inputs["Vector"])
cloud = wmath("MULTIPLY", wmath("SUBTRACT", cn.outputs["Fac"], 0.52), 6.0, clamp=True)
band = wmath("MULTIPLY", wmath("SUBTRACT", el, 0.04), 12.0, clamp=True)
cloud = wmath("MULTIPLY", cloud, band)
sky = wmix(wmath("MULTIPLY", cloud, 0.75), ramp.outputs[0], (0.012, 0.02, 0.035))
lit_edge = wmath("MULTIPLY", cloud, wmath("MULTIPLY", halo2, 3.0))
sky = wmix(lit_edge, sky, (0.45, 0.62, 0.7), "ADD")
sky = wmix(wmath("MULTIPLY", halo2, 0.35), sky, (0.25, 0.42, 0.48), "ADD")
sky = wmix(wmath("MULTIPLY", halo1, 1.5), sky, (0.6, 0.75, 0.8), "ADD")
MS = float(A.get("moonk", 3.0))
mdisc = wmix(wmath("MULTIPLY", disc, 1.0), (0, 0, 0), mcol)
mdisc = wmix(1.0, mdisc, (MS, MS, MS), "MULTIPLY")
sky = wmix(wmath("MULTIPLY", disc, wmath("SUBTRACT", 1.0, wmath("MULTIPLY", cloud, 0.7))), sky, mdisc)
bg = wt.nodes.new("ShaderNodeBackground")
wt.links.new(sky, bg.inputs[0])
wo = wt.nodes.new("ShaderNodeOutputWorld")
wt.links.new(bg.outputs[0], wo.inputs[0])

# cold fill from the window over the room's surfaces (not the dust: the dust shows the moonbeams alone)
fill = B.light("AREA", (WX, -0.12, 2.0), float(A.get("fill", 45.0)), color=(0.5, 0.74, 1.0), size=1.2,
               target=(WX - 0.8, -4.0, 1.0), name="windowfill")
fill.data.shape = "RECTANGLE"
fill.data.size_y = 2.0
fill.visible_glossy = False
fill.visible_volume_scatter = False
fill.visible_camera = False
warmfill = B.light("AREA", (-0.6, -3.2, 2.3), float(A.get("warmfill", 5.0)), color=(1.0, 0.6, 0.3), size=1.5,
                   target=(-0.1, -2.4, 0.84), name="warmfill")
warmfill.visible_volume_scatter = False
warmfill.visible_glossy = False
nearc = B.light("POINT", (-0.75, -3.6, 1.0), float(A.get("nearw", 2.5)), color=(1.0, 0.55, 0.22), size=0.02,
                name="offscreen_candle")                # a candle just outside the left edge: lights the near props
nearc.visible_volume_scatter = False
nearc.visible_glossy = False                            # no hot specular blotch on the big jack's face

dust, t = K.mat("dust")
pv = t.n("ShaderNodeVolumePrincipled", Color=(0.85, 0.93, 1.0), Density=float(A.get("dust", 0.09)),
         Anisotropy=float(A.get("aniso", 0.55)))
o = t.n("ShaderNodeOutputMaterial")
t.link(pv.outputs[0], o.inputs["Volume"])
dust_ob = box("dustbox", -HW + 0.02, HW - 0.02, YF + 0.02, -0.01, 0.01, RIDGE - 0.05, dust)

# ---------------------------------------------------------------- camera: bench height, 28 mm, toward the window
cam_loc = Vector((cam_.x, cam_.y, float(A.get("camz", 1.45))))
yaw = math.radians(float(A.get("yaw", 4.0)))
tgt = cam_loc + Vector((math.sin(yaw), math.cos(yaw), 0.0))
jb = jack_objs[0].location
focus = (Vector(jb) - cam_loc).length
B.camera(tuple(cam_loc), tuple(tgt), lens=float(A.get("lens", 28)), focus=float(A.get("focus", focus)),
         fstop=float(A.get("fstop", 5.6)), shift_y=float(A.get("shift", -0.03)))
B.passes(mist=(0.0, 12.0, "LINEAR"), extra=("z", "normal"))

cam = sc.camera
bpy.context.view_layer.update()
probe = {"window apex": (WX, 0.0, arch_z(WX)), "sill L": (WX - WW, 0.0, WS), "sill R": (WX + WW, 0.0, WS),
         "raven": RV(0.05, 0, 0.18), "half pumpkin": tuple(hob.location), "tool roll": (TRX, TRY, BT),
         "candle": (CDX, CDY, BT + 0.06), "oil lamp": (OLX, OLY, BT + 0.05), "spire tip": apex,
         "moon": tuple(cam_loc + mdir * 100)}
for ob in jack_objs:
    probe[ob.name] = tuple(ob.location)
for k, p in probe.items():
    v = world_to_camera_view(sc, cam, Vector(p))
    print(f"probe {k:13s} x={v.x:5.2f} y_from_top={1 - v.y:5.2f} depth={v.z:6.2f}")
if CHECK:
    print("check ok")
else:
    bpy.context.view_layer.cycles.use_pass_volume_direct = True
    B.render(TAG, outputs=("Image", "VolumeDir"))
    B.render_data(TAG, hide=[dust_ob])            # depth/normal/actor without the dust (it speckles them)
