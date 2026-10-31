"""33 arcane-greenhouse: an overgrown Victorian glasshouse at night, the base render for the Arcane-like painterly
finish (src/33_arcane_greenhouse_post.py).

Giant pumpkins among big leaves line a narrow brick path, a few carved and glowing. Moonlight falls through the
dirty glass roof (some panes broken) in teal shafts, cyan will-o'-wisps drift, magenta fuchsias trail from hanging
baskets. At the far end, an open door to a misty garden and an old gardener with a lantern and a watering can.
    python src/gpu_run.py blender/33_arcane_greenhouse.py -- tag=33-test pct=35 samples=16
    blender -b --factory-startup --python blender/33_arcane_greenhouse.py -- check=1   (build only, no GPU)
Passes: Image, then (volumes hidden) Depth, Normal, AOV "actor" (gardener + jacks: fine strokes + ink outline) and AOV "glow" (wisps,
lantern glass: fine strokes, no outline). work/<tag>/fx.json has the wisps' and the lantern's screen positions."""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/blender")
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import bl_build as K  # noqa: E402
import bl_common as B  # noqa: E402

A = B.args()
PCT = int(A.get("pct", 35))
SPP = int(A.get("samples", 16))
TAG = A.get("tag", "33-test")
CHECK = "check" in A
WORK = ROOT + "/work"
rnd = random.Random(33)

sc = B.empty_scene()
if not CHECK:
    B.cycles(samples=SPP, pct=PCT, bounces=4, clamp_indirect=3.0, vol_bounces=0, transparent_bounces=24)
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1440, 2560, PCT

HW, EAVE, APEX, PLINTH = 3.6, 3.1, 7.0, 0.7          # half width, top of the side glazing, roof apex, brick plinth
Y0, Y1, BAY = -3.0, 27.0, 2.0                          # nave from Y0 to the far gable at Y1, a rib every BAY
RISE = APEX - EAVE
PATH_W, PATH_Z, BED_Z = 0.62, 0.03, 0.06
CAM = Vector((float(A.get("camx", -0.2)), -0.6, float(A.get("camz", 1.2))))


def arch(t):
    """Roof profile in x-z: t=0 at the right eave, t=1 at the left eave."""
    a = math.pi * t
    return HW * math.cos(a), EAVE + RISE * math.sin(a)


def roof_z(x):
    return EAVE + RISE * math.sqrt(max(0.0, 1 - (x / HW) ** 2))


# ---------------------------------------------------------------- materials

def iron_mat():
    m, t = K.mat("iron")
    tc = t.coords("Object")
    nz = t.n("ShaderNodeTexNoise", Scale=3.5, Detail=6.0)
    t.link(tc, nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.45, 0.7), (0.016, 0.03, 0.028), (0.05, 0.032, 0.022))   # verdigris-black, rust
    t.out(t.bsdf(col, t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.32, 0.65), metal=0.35, spec=0.6))
    return m


def glass_mat(name, dirt, tint=(0.8, 0.95, 0.85), block=False):
    """Transparent pane with a grimy green film: `dirt` is how much of it is film. With `block`, the pane stops the
    light completely (shadow rays see it opaque), so the moon only comes in through clean or broken panes."""
    m, t = K.mat(name)
    tc = t.coords("Object")
    nz = t.n("ShaderNodeTexNoise", Scale=1.2, Detail=5.0)
    t.link(tc, nz.inputs["Vector"])
    tr = t.n("ShaderNodeBsdfTransparent", Color=tint)
    film = t.bsdf((0.035, 0.05, 0.03), 0.4, spec=float(A.get("glass_spec", 0.25)))
    mix = t.n("ShaderNodeMixShader")
    t.link(t.mapr(nz.outputs["Fac"], 0.3, 0.75, dirt * 0.4, dirt * 1.4), mix.inputs[0])
    t.link(tr.outputs[0], mix.inputs[1])
    t.link(film, mix.inputs[2])
    if not block:
        t.out(mix.outputs[0])
        return m
    lp = t.n("ShaderNodeLightPath")
    sh = t.n("ShaderNodeMixShader")
    t.link(lp.outputs["Is Shadow Ray"], sh.inputs[0])
    t.link(mix.outputs[0], sh.inputs[1])
    t.link(t.bsdf((0.02, 0.03, 0.02), 0.9), sh.inputs[2])
    t.out(sh.outputs[0])
    return m


def brick_mat(name, c1, c2, wall=False, wet=0.0):
    m, t = K.mat(name)
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    comb = t.n("ShaderNodeCombineXYZ")
    if wall:
        t.link(sep.outputs["Y"], comb.inputs["X"])
        t.link(sep.outputs["Z"], comb.inputs["Y"])
    else:
        t.link(sep.outputs["X"], comb.inputs["X"])
        t.link(sep.outputs["Y"], comb.inputs["Y"])
    br = t.n("ShaderNodeTexBrick", Color1=c1, Color2=c2, Mortar=(0.03, 0.035, 0.025), Scale=1.0, **{
        "Mortar Size": 0.012, "Brick Width": 0.215 if not wall else 0.23, "Row Height": 0.105 if not wall else 0.075,
        "Bias": 0.0})
    br.offset = 0.5
    t.link(comb.outputs[0], br.inputs["Vector"])
    moss = t.n("ShaderNodeTexNoise", Scale=2.2, Detail=8.0)
    t.link(tc, moss.inputs["Vector"])
    col = t.mixc(t.mapr(moss.outputs["Fac"], 0.5, 0.7, 0.0, 0.8), br.outputs["Color"], (0.02, 0.04, 0.012))
    rough = t.mapr(br.outputs["Fac"], 0.0, 1.0, 0.5, 0.85)
    if wet:                                             # puddles: glossy patches that mirror the door and the lantern
        pud = t.n("ShaderNodeTexNoise", Scale=0.7, Detail=3.0)
        t.link(tc, pud.inputs["Vector"])
        rough = t.math("SUBTRACT", rough, t.mapr(pud.outputs["Fac"], 0.42, 0.6, 0.0, wet), clamp=True)
    nb = t.bump(t.math("SUBTRACT", 1.0, br.outputs["Fac"]), 0.4, 0.01)
    t.out(t.bsdf(col, rough, normal=nb, spec=0.5))
    return m


def leaf_mat(name, col, trans=0.35, vary=1.7):
    m, t = K.mat(name)
    tc = t.coords("Object")
    nz = t.n("ShaderNodeTexNoise", Scale=2.5, Detail=4.0)
    t.link(tc, nz.inputs["Vector"])
    c = t.mixc(t.mapr(nz.outputs["Fac"], 0.35, 0.7), col, tuple(v * vary for v in col))
    sh = t.bsdf(c, 0.5, spec=0.35)
    tl = t.n("ShaderNodeBsdfTranslucent", Color=tuple(min(1.0, v * 3.0) for v in col))
    mix = t.n("ShaderNodeMixShader", Fac=trans)
    t.link(sh, mix.inputs[1])
    t.link(tl.outputs[0], mix.inputs[2])
    t.out(mix.outputs[0])
    return m


def soil_mat():
    m, t = K.mat("soil")
    tc = t.coords("Object")
    nz = t.n("ShaderNodeTexNoise", Scale=6.0, Detail=8.0)
    t.link(tc, nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.35, 0.7), (0.012, 0.009, 0.006), (0.03, 0.022, 0.014))
    t.out(t.bsdf(col, 0.9, normal=t.bump(nz.outputs["Fac"], 0.5, 0.03)))
    return m


IRON = iron_mat()
GLASS = [glass_mat("glass_clean", 0.06), glass_mat("glass_dirty", float(A.get("dirty", 0.25)), block=True),
         glass_mat("glass_grime", float(A.get("grime", 0.5)), block=True)]
GW = [float(v) for v in A.get("glassw", "0.0,0.35,0.65").split(",")]   # clean panes only in the HOLES
PATH = brick_mat("path_brick", (0.13, 0.055, 0.035), (0.085, 0.04, 0.03), wet=0.45)
WALLB = brick_mat("wall_brick", (0.12, 0.05, 0.035), (0.07, 0.035, 0.028), wall=True)
SOIL = soil_mat()
LEAF = [leaf_mat("leaf_a", (0.03, 0.07, 0.022), trans=0.45), leaf_mat("leaf_b", (0.045, 0.085, 0.03), trans=0.45),
        leaf_mat("leaf_c", (0.06, 0.075, 0.022), trans=0.4)]
IVY = leaf_mat("ivy", (0.016, 0.035, 0.014), trans=0.25)
STEM = B.principled("vine", (0.03, 0.045, 0.015), rough=0.7)
GRASS = B.principled("grass", (0.01, 0.018, 0.009), rough=0.95)

# ---------------------------------------------------------------- the glasshouse: plinth, ribs, glazing bars, panes
fr = K.MB()          # mats: 0 iron
gl = K.MB()          # mats: 0 clean, 1 dirty, 2 grime glass
bw = K.MB()          # mats: 0 wall brick


def sweep_xz(mb, pts, y, wy, wr, mat=0):
    """Rectangular bar along a polyline in the x-z plane at fixed y: wy thick along y, wr deep across the line."""
    n = len(pts)
    vs = []
    for i, (x, z) in enumerate(pts):
        xa, za = pts[max(i - 1, 0)]
        xb, zb = pts[min(i + 1, n - 1)]
        tx, tz = xb - xa, zb - za
        ln = math.hypot(tx, tz) or 1.0
        nx, nz = -tz / ln, tx / ln
        for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vs.append((x + nx * wr / 2 * b, y + wy / 2 * a, z + nz * wr / 2 * b))
    fs = [(i * 4 + k, i * 4 + (k + 1) % 4, (i + 1) * 4 + (k + 1) % 4, (i + 1) * 4 + k) for i in range(n - 1) for k in range(4)]
    fs += [(3, 2, 1, 0), ((n - 1) * 4, (n - 1) * 4 + 1, (n - 1) * 4 + 2, (n - 1) * 4 + 3)]
    mb.add(vs, fs, mat)


def rib_profile(inset=0.0, steps=40):
    pts = [(-(HW - inset), 0.0), (-(HW - inset), EAVE)]
    for k in range(1, steps):
        x, z = arch(1 - k / steps)
        pts.append((x * (HW - inset) / HW, EAVE + (z - EAVE) * (RISE - inset) / RISE))
    pts += [(HW - inset, EAVE), (HW - inset, 0.0)]
    return pts


# plinth walls
for s in (-1, 1):
    x0, x1 = sorted((s * HW, s * (HW + 0.32)))
    bw.boxr(x0, x1, Y0, Y1 + 0.3, -0.2, PLINTH, 0)
    bw.boxr(x0 - 0.04, x1 + 0.04, Y0, Y1 + 0.3, PLINTH, PLINTH + 0.07, 0)        # coping
# ribs: columns running on into the curved roof, every BAY, with a ring ornament at the eave
ys_rib = [Y0 + k * BAY for k in range(int((Y1 - Y0) / BAY) + 1)]
for y in ys_rib:
    sweep_xz(fr, rib_profile(0.0), y, 0.07, 0.12)
    for s in (-1, 1):
        ring = [(s * (HW - 0.32) + 0.17 * math.cos(a), EAVE + 0.1 + 0.17 * math.sin(a)) for a in
                [2 * math.pi * i / 20 for i in range(21)]]
        sweep_xz(fr, ring, y, 0.04, 0.035)
        fr.cyl((s * (HW - 0.06), y, EAVE + 0.1), (s * (HW - 0.15), y, EAVE + 0.1), 0.018, n=5)
# glazing bars on the roof (arcs every 0.5 m) and in the side walls (verticals every 0.5 m)
yb = Y0
while yb < Y1 - 0.01:
    if min(abs(yb - r) for r in ys_rib) > 0.1:
        sweep_xz(fr, rib_profile(0.02, 30)[1:-1], yb, 0.028, 0.04)
        for s in (-1, 1):
            fr.boxr(s * HW - 0.02, s * HW + 0.02, yb - 0.018, yb + 0.018, PLINTH, EAVE)
    yb += 0.5
# purlins along the roof and the side transoms
NP = 9
for k in range(1, NP):
    x, z = arch(k / NP)
    fr.cyl((x, Y0, z - 0.05), (x, Y1, z - 0.05), 0.035, n=6)
for s in (-1, 1):
    for z in (PLINTH + 0.06, 1.95, EAVE):
        fr.boxr(s * HW - 0.04, s * HW + 0.04, Y0, Y1, z - 0.035, z + 0.035)
fr.cyl((0, Y0, APEX + 0.02), (0, Y1, APEX + 0.02), 0.06, n=8)                    # ridge
# roof ridge cresting: little iron finials along the ridge
yy = Y0 + 0.25
while yy < Y1:
    fr.cyl((0, yy, APEX + 0.05), (0, yy, APEX + 0.28), 0.012, n=5)
    fr.cyl((0, yy, APEX + 0.28), (0, yy, APEX + 0.34), 0.03, 0.0, n=6)
    yy += 0.5

# panes: roof strips between purlins and glazing bars, side panes between transoms; some broken, some grimy
broken = float(A.get("broken", 0.0))
# where the roof is clean or broken: a few big patches on the moon side, so the light comes in as distinct shafts
# (y centre, arc position t, half sizes); elsewhere the glass is grimy and the vault stays dark
HOLES = [(8.25, 0.8, 0.24, 0.05), (9.75, 0.7, 0.24, 0.045), (10.25, 0.45, 0.24, 0.05), (12.25, 0.7, 0.24, 0.05), (14.25, 0.3, 0.24, 0.05), (16.25, 0.8, 0.24, 0.05),
         (18.75, 0.55, 0.24, 0.05), (22.25, 0.65, 0.24, 0.05), (24.75, 0.38, 0.24, 0.05)]     if A.get("holes", "1") == "1" else []      # one bay wide, spread across the vault (the 12.25 one lands on the gardener)


def pane_state(y, t):
    """0 clean, 1 dirty, 2 grime, None broken."""
    for (yc, tc, hy, ht) in HOLES:
        if abs(y - yc) < hy and abs(t - tc) < ht:
            return None if rnd.random() < 0.35 else 0
    if rnd.random() < broken:
        return None
    return rnd.choices((0, 1, 2), GW)[0]


def pane_quad(p, state):
    if state is not None:
        gl.add([tuple(v) for v in p], [(0, 1, 2, 3)], state)


ysub = [Y0 + 0.5 * k for k in range(int((Y1 - Y0) / 0.5))]
for y in ysub:
    for k in range(NP):
        ts = [k / NP + (1 / NP) * j / 3 for j in range(4)]
        for j in range(3):
            xa, za = arch(ts[j])
            xb, zb = arch(ts[j + 1])
            pane_quad([(xa, y, za), (xb, y, zb), (xb, y + 0.5, zb), (xa, y + 0.5, za)],         # no light leaks
                      pane_state(y + 0.25, (ts[j] + ts[j + 1]) / 2))
    for s in (-1, 1):
        for (z0, z1) in ((PLINTH + 0.06, 1.95), (1.95, EAVE)):
            pane_quad([(s * HW, y, z0), (s * HW, y + 0.5, z0), (s * HW, y + 0.5, z1), (s * HW, y, z1)],
                      None if rnd.random() < broken else rnd.choice((1, 2, 2)))

# ---------------------------------------------------------------- far gable with the open double door
DOOR_W, DOOR_H = 0.95, 2.75
FAN_R = DOOR_W
FAN_TOP = DOOR_H + FAN_R
cols = [-HW + (HW - DOOR_W) * k / 5 for k in range(6)] + [-DOOR_W + 2 * DOOR_W * k / 4 for k in range(1, 4)] + \
       [DOOR_W + (HW - DOOR_W) * k / 5 for k in range(6)]
for gx in cols:                                      # gable mullions (over the door they start above the fanlight)
    z0 = PLINTH if abs(gx) > DOOR_W - 0.01 else FAN_TOP
    fr.boxr(gx - 0.025, gx + 0.025, Y1 - 0.03, Y1 + 0.03, z0, roof_z(gx))
for s in (-1, 1):                                    # door posts, and the transom either side of the door
    fr.boxr(s * DOOR_W - 0.05, s * DOOR_W + 0.05, Y1 - 0.05, Y1 + 0.05, 0.0, roof_z(s * DOOR_W))
    x0, x1 = sorted((s * DOOR_W, s * HW))
    fr.boxr(x0, x1, Y1 - 0.04, Y1 + 0.04, EAVE - 0.04, EAVE + 0.04)
fan = [(DOOR_W * math.cos(a), DOOR_H + FAN_R * math.sin(a)) for a in [math.pi * i / 24 for i in range(25)]]
sweep_xz(fr, fan, Y1, 0.08, 0.08)
fr.boxr(-DOOR_W, DOOR_W, Y1 - 0.04, Y1 + 0.04, FAN_TOP - 0.03, FAN_TOP + 0.03)
fr.boxr(-DOOR_W, DOOR_W, Y1 - 0.04, Y1 + 0.04, DOOR_H - 0.04, DOOR_H + 0.04)
for a in (math.pi * 0.25, math.pi * 0.5, math.pi * 0.75):            # fanlight spokes
    fr.cyl((0, Y1, DOOR_H), (DOOR_W * math.cos(a), Y1, DOOR_H + FAN_R * math.sin(a)), 0.018, n=5)
# gable panes (not in the door opening)
for xa, xb in zip(cols[:-1], cols[1:]):
    if rnd.random() < broken:
        continue
    z0 = PLINTH + 0.05 if (xb <= -DOOR_W + 0.01 or xa >= DOOR_W - 0.01) else FAN_TOP + 0.05
    za, zb = roof_z(xa) - 0.05, roof_z(xb) - 0.05
    if min(za, zb) <= z0 + 0.05:
        continue
    if z0 < EAVE:                                    # a pane below the transom, then one above it
        gl.add([(xa, Y1, z0), (xb, Y1, z0), (xb, Y1, EAVE - 0.04), (xa, Y1, EAVE - 0.04)], [(0, 1, 2, 3)],
               rnd.choice((1, 2, 2)))
        z0 = EAVE + 0.04
    gl.add([(xa, Y1, z0), (xb, Y1, z0), (xb, Y1, zb), (xa, Y1, za)], [(0, 1, 2, 3)], rnd.choice((1, 2, 2)))
# fanlight glass (half disc)
gl.add([(0, Y1 + 0.01, DOOR_H)] + [(x, Y1 + 0.01, z) for x, z in fan], [(0, i + 1, i + 2) for i in range(len(fan) - 1)], 1)
# gable plinth either side of the door
for s in (-1, 1):
    x0, x1 = sorted((s * (DOOR_W + 0.05), s * HW))
    bw.boxr(x0, x1, Y1, Y1 + 0.3, -0.2, PLINTH)
# door leaves swung outward into the garden
for s, ang in ((-1, 118.0), (1, 72.0)):
    hinge = Vector((s * DOOR_W, Y1 + 0.05, 0.0))
    M = Matrix.Translation(hinge) @ Matrix.Rotation(math.radians(ang) * (-s), 4, "Z")

    def dp(u, z):
        return tuple(M @ Vector((-s * u, 0.0, z)))
    for (u0, u1, z0, z1) in ((0.0, 0.06, 0.0, DOOR_H), (DOOR_W - 0.06, DOOR_W, 0.0, DOOR_H), (0.0, DOOR_W, 0.0, 0.07),
                             (0.0, DOOR_W, DOOR_H - 0.07, DOOR_H), (0.0, DOOR_W, 0.9, 0.96), (0.0, DOOR_W, 1.85, 1.9)):
        fr.add([dp(u0, z0), dp(u1, z0), dp(u1, z1), dp(u0, z1)], [(0, 1, 2, 3)], 0)
    fr.add([dp(DOOR_W / 2 - 0.02, 0.96), dp(DOOR_W / 2 + 0.02, 0.96), dp(DOOR_W / 2 + 0.02, DOOR_H - 0.07),
            dp(DOOR_W / 2 - 0.02, DOOR_H - 0.07)], [(0, 1, 2, 3)], 0)
    for (z0, z1) in ((0.96, 1.85), (1.9, DOOR_H - 0.07)):
        gl.add([dp(0.06, z0), dp(DOOR_W - 0.06, z0), dp(DOOR_W - 0.06, z1), dp(0.06, z1)], [(0, 1, 2, 3)], 0)

# near gable, behind the camera: closed (grimy, light-tight) so no sky light floods the haze from behind
ng = rib_profile(0.0, 48)[1:-1]
gl.add([(-HW, Y0, 0.0)] + [(x, Y0, z) for (x, z) in ng] + [(HW, Y0, 0.0)], [tuple(range(len(ng) + 2))], 2)
fr.build("frame", [IRON])
gl.build("glass", GLASS)
bw.build("plinth", [WALLB])

# ---------------------------------------------------------------- path, kerbs, beds, outside ground
pb = K.MB()
pb.boxr(-PATH_W, PATH_W, Y0, Y1 + 0.1, -0.1, PATH_Z)
pb.boxr(-DOOR_W, DOOR_W, Y1, Y1 + 6.0, -0.12, PATH_Z - 0.02)                    # step-out beyond the door
for s in (-1, 1):
    x0, x1 = sorted((s * PATH_W, s * (PATH_W + 0.09)))
    pb.boxr(x0, x1, Y0, Y1, -0.1, PATH_Z + 0.09)
pb.build("path", [PATH])
sb = K.MB()
for s in (-1, 1):
    x0, x1 = sorted((s * (PATH_W + 0.09), s * HW))
    sb.boxr(x0, x1, Y0, Y1, -0.2, BED_Z)
sb.build("beds", [SOIL])
B.mesh("garden", [(-80, Y1 - 1, -0.04), (80, Y1 - 1, -0.04), (80, 160, -0.04), (-80, 160, -0.04)], [(0, 1, 2, 3)], GRASS)
B.mesh("ground_out", [(-80, -30, -0.05), (80, -30, -0.05), (80, Y1 - 1, -0.05), (-80, Y1 - 1, -0.05)], [(0, 1, 2, 3)], GRASS)

# ---------------------------------------------------------------- pumpkins: giant ones and carved jacks
FACE = [f"{WORK}/face_{k}.png" for k in ("classic", "grin", "scary")]
SKINS = [K.pumpkin_material("giant_orange"), K.pumpkin_material("giant_pale", skin=(0.42, 0.38, 0.28)),
         K.pumpkin_material("giant_green", skin=(0.1, 0.14, 0.07)), K.pumpkin_material("giant_deep", skin=(0.45, 0.1, 0.015))]
PSTEM = B.principled("pstem", (0.09, 0.08, 0.04), rough=0.8)
pk = K.MB()
pumpkins = []      # (x, y, r) for leaf avoidance
GIANTS = [(-1.55, 2.6, 0.62, 0), (1.75, 3.6, 0.42, 1), (-2.6, 6.2, 0.75, 3), (2.7, 9.6, 0.68, 0), (-2.4, 12.5, 0.55, 2),
          (2.3, 15.5, 0.5, 3), (-2.2, 18.8, 0.6, 0), (2.4, 21.5, 0.45, 1), (-1.6, 23.5, 0.4, 3), (1.4, 5.6, 0.28, 3),
          (-1.25, 10.4, 0.3, 1), (1.3, 13.0, 0.25, 0), (-1.2, 16.4, 0.26, 3), (1.25, 19.2, 0.3, 2), (-3.0, 15.0, 0.35, 1),
          (3.1, 6.5, 0.38, 2), (3.0, 18.0, 0.4, 0), (-1.1, 4.6, 0.22, 0), (1.15, 24.6, 0.3, 3)]
for i, (x, y, r, sk) in enumerate(GIANTS):
    K.pumpkin_parts(pk, (x, y, BED_Z + r * 0.6), r, sk, 4, squash=rnd.uniform(0.62, 0.74), ribs=rnd.choice((8, 10, 12)),
                    groove=0.09, rz=rnd.uniform(0, 6), seed=100 + i)
    pumpkins.append((x, y, r))
pk.build("giants", SKINS + [PSTEM], smooth=True)

JACKS = [(1.05, 2.1, 0.24, 0, 14.0), (-1.0, 3.4, 0.27, 2, 14.0), (-0.95, 6.6, 0.3, 1, 18.0), (1.6, 7.9, 0.58, 2, 55.0), (-1.0, 9.6, 0.2, 0, 9.0),
         (0.95, 11.6, 0.22, 1, 9.0), (-1.75, 14.6, 0.4, 0, 22.0), (1.0, 17.4, 0.25, 2, 9.0), (-1.0, 20.8, 0.24, 1, 8.0),
         (1.0, 23.2, 0.2, 0, 6.0)]
for i, (x, y, r, f, w) in enumerate(JACKS):
    d = Vector((CAM.x - x, CAM.y - y))
    rz = math.atan2(d.x, -d.y) + math.radians(rnd.uniform(-22, 22))
    K.jack(f"jack{i}", (x, y, BED_Z + r * 0.58), r, FACE[f], rz=rz, seed=200 + i, light_w=w * float(A.get("jackw", 1.0)),
           strength=float(A.get("jacks", 30.0)))
    pumpkins.append((x, y, r))
for key, m in list(K._CACHE.items()):
    if key != "stem":
        B.aov_mask(m, "actor")

# ---------------------------------------------------------------- the gardener (local: facing -y, feet at origin)
GX, GY = float(A.get("gx", 0.38)), float(A.get("gy", 7.6))
GRZ = math.radians(float(A.get("grz", -25)))
GM = Matrix.Translation((GX, GY, PATH_Z)) @ Matrix.Rotation(GRZ, 4, "Z")
COAT = B.principled("coat", (0.03, 0.036, 0.036), rough=0.75)
SKIN = B.principled("skin", (0.42, 0.25, 0.17), rough=0.55, subsurface=0.15)
HAT = B.principled("hat", (0.035, 0.024, 0.016), rough=0.7)
LEATHER = B.principled("leather", (0.035, 0.02, 0.012), rough=0.45)
ZINC = B.principled("zinc", (0.42, 0.44, 0.43), rough=0.32, metal=0.9)
LIRON = B.principled("lantern_iron", (0.02, 0.02, 0.02), rough=0.4, metal=0.6)
LGLASS = B.emission("lantern_glass", (1.0, 0.62, 0.26), float(A.get("lglass", 16.0)))
BEARD = B.principled("beard", (0.5, 0.48, 0.44), rough=0.8)
SCARF = B.principled("scarf", (0.22, 0.05, 0.025), rough=0.8)
GMATS = [COAT, SKIN, HAT, LEATHER, ZINC, LIRON, LGLASS, BEARD, SCARF]
for m in GMATS:
    if m is LGLASS:
        B.aov_mask(m, "glow")
    else:
        B.aov_mask(m, "actor")
gm = K.MB()


def W(p, M=None):
    return tuple((GM if M is None else M) @ Vector(p))


def lathe(mb, prof, n=20, mat=0, sx=1.0, sy=1.0, M=None, wave=None, cap0=True, cap1=True, off=(0.0, 0.0)):
    """Surface of revolution about a local vertical axis through `off`: prof = [(z, r), ...] ascending in z."""
    vs, fs = [], []
    for (z, r) in prof:
        for i in range(n):
            a = 2 * math.pi * i / n
            rr = r * (1 + (wave(a, z) if wave else 0.0))
            vs.append(W((off[0] + rr * sx * math.cos(a), off[1] + rr * sy * math.sin(a), z), M))
    for j in range(len(prof) - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            fs.append((a, b, b + n, a + n))
    if cap0:
        fs.append(tuple(range(n - 1, -1, -1)))
    if cap1:
        k = (len(prof) - 1) * n
        fs.append(tuple(range(k, k + n)))
    mb.add(vs, fs, mat)


def limb(mb, pts, r0, r1, mat, n=10, M=None):
    m = len(pts)
    for i in range(m - 1):
        ra = r0 + (r1 - r0) * i / (m - 1)
        rb = r0 + (r1 - r0) * (i + 1) / (m - 1)
        mb.cyl(W(pts[i], M), W(pts[i + 1], M), ra, rb, n=n, mat=mat)
    for i in range(1, m - 1):                      # round the joints
        ra = r0 + (r1 - r0) * i / (m - 1)
        x, y, z = pts[i]
        lathe(mb, [(z - ra, ra * 0.3), (z - ra * 0.7, ra * 0.75), (z, ra), (z + ra * 0.7, ra * 0.75), (z + ra, ra * 0.3)],
              n=n, mat=mat, M=M, off=(x, y))


def folds(a, z):
    return 0.06 * math.sin(5 * a + 1.3) * max(0.0, min(1.0, (0.95 - z) / 0.5)) + 0.02 * math.sin(11 * a)


lathe(gm, [(0.36, 0.3), (0.4, 0.315), (0.6, 0.285), (0.85, 0.235), (1.02, 0.205), (1.2, 0.215), (1.33, 0.23),
           (1.42, 0.22), (1.48, 0.16), (1.52, 0.08)], n=28, mat=0, sx=1.08, sy=0.68, wave=folds)
lathe(gm, [(1.1, 0.33), (1.18, 0.315), (1.3, 0.29), (1.4, 0.25), (1.47, 0.17), (1.52, 0.09)], n=28, mat=0, sx=1.12,
      sy=0.8, wave=lambda a, z: 0.05 * math.sin(6 * a) * max(0.0, (1.3 - z) / 0.2), cap0=False)     # shoulder cape
lathe(gm, [(1.46, 0.085), (1.5, 0.11), (1.54, 0.1), (1.57, 0.075)], n=18, mat=8, sy=0.9)                # scarf
limb(gm, [(0.06, -0.1, 1.5), (0.09, -0.14, 1.35), (0.1, -0.15, 1.22)], 0.035, 0.03, 8)
lathe(gm, [(1.52, 0.035), (1.55, 0.075), (1.6, 0.092), (1.66, 0.094), (1.71, 0.084), (1.75, 0.055), (1.77, 0.02)],
      n=18, mat=1, sx=0.95, sy=1.05)                                                                   # head
gm.cyl(W((0, -0.088, 1.665)), W((0, -0.125, 1.635)), 0.017, 0.008, n=8, mat=1)                         # nose
gm.cyl(W((0, -0.06, 1.61)), W((0, -0.095, 1.46)), 0.062, 0.01, n=12, mat=7)                            # pointed beard
HM = GM @ Matrix.Translation((0, 0, 1.72)) @ Matrix.Rotation(math.radians(-7), 4, "X") @ Matrix.Translation((0, 0, -1.72))
lathe(gm, [(1.715, 0.29), (1.728, 0.26), (1.737, 0.2), (1.74, 0.12)], n=32, mat=2, M=HM, cap0=False, cap1=False)  # brim
lathe(gm, [(1.73, 0.118), (1.8, 0.115), (1.86, 0.104), (1.885, 0.07), (1.892, 0.02)], n=24, mat=2, M=HM)       # crown
# right arm raised forward with the lantern, left arm down with the watering can
limb(gm, [(0.21, 0.0, 1.38), (0.31, -0.17, 1.18), (0.32, -0.41, 1.27)], 0.062, 0.046, 0)
limb(gm, [(-0.21, 0.0, 1.38), (-0.28, 0.03, 1.1), (-0.3, 0.0, 0.88)], 0.062, 0.046, 0)
for p in ((0.325, -0.44, 1.275), (-0.3, -0.005, 0.85)):          # gloves
    lathe(gm, [(p[2] - 0.04, 0.012), (p[2] - 0.025, 0.035), (p[2], 0.042), (p[2] + 0.025, 0.035), (p[2] + 0.04, 0.012)],
          n=12, mat=3, off=(p[0], p[1]))
LX, LY = 0.33, -0.45
gm.cyl(W((LX, LY, 1.24)), W((LX, LY, 1.215)), 0.008, n=5, mat=5)                                       # bail
lathe(gm, [(1.175, 0.07), (1.2, 0.045), (1.22, 0.012)], n=12, mat=5, off=(LX, LY))                    # cap
lathe(gm, [(0.99, 0.065), (1.01, 0.065)], n=12, mat=5, off=(LX, LY))                                    # base
lathe(gm, [(1.01, 0.05), (1.175, 0.05)], n=12, mat=6, off=(LX, LY))                                     # glass
for k in range(4):
    a = math.pi / 4 + k * math.pi / 2
    gm.cyl(W((LX + 0.058 * math.cos(a), LY + 0.058 * math.sin(a), 1.0)),
           W((LX + 0.058 * math.cos(a), LY + 0.058 * math.sin(a), 1.18)), 0.006, n=4, mat=5)
CX, CY = -0.32, 0.0                                                                                      # watering can
lathe(gm, [(0.5, 0.105), (0.74, 0.1), (0.76, 0.06)], n=20, mat=4, off=(CX, CY))
limb(gm, [(CX, 0.085, 0.74), (CX, 0.06, 0.84), (CX, -0.005, 0.875), (CX, -0.07, 0.83), (CX, -0.09, 0.74)], 0.012, 0.012, 4, n=6)
gm.cyl(W((CX - 0.09, CY - 0.02, 0.56)), W((CX - 0.36, CY - 0.06, 0.8)), 0.022, 0.012, n=8, mat=4)        # spout
gm.cyl(W((CX - 0.35, CY - 0.06, 0.79)), W((CX - 0.4, CY - 0.07, 0.84)), 0.012, 0.035, n=10, mat=4)        # rose
for s in (-1, 1):                                                                                         # legs, boots
    limb(gm, [(s * 0.09, 0.0, 0.1), (s * 0.09, 0.0, 0.45)], 0.055, 0.06, 0, n=10)
    gm.cyl(W((s * 0.09, 0.05, 0.055)), W((s * 0.095, -0.14, 0.05)), 0.055, 0.05, n=10, mat=3)
    gm.cyl(W((s * 0.09, 0.0, 0.0)), W((s * 0.09, 0.0, 0.16)), 0.062, 0.058, n=10, mat=3)
gm.build("gardener", GMATS, smooth=True)
LANTERN = Vector(W((LX, LY, 1.09)))
B.light("POINT", tuple(LANTERN), float(A.get("lantern", 40.0)), color=(1.0, 0.58, 0.25), size=0.04, name="lantern")

# ---------------------------------------------------------------- leaves: pumpkin leaves in the beds, elephant ears
lv = K.MB()          # mats 0-2 leaf, 3 vine stem, 4 ivy


def frame_from(normal, along):
    n = Vector(normal).normalized()
    u = (Vector(along) - n * Vector(along).dot(n)).normalized()
    return u, n.cross(u), n


def leaf_fan(mb, c, normal, along, outline, mat, droop=0.0):
    u, v, n = frame_from(normal, along)
    vs = [tuple(c)]
    for (a, b) in outline:
        d2 = a * a + b * b
        vs.append(tuple(Vector(c) + u * a + v * b - n * droop * d2))
    m = len(outline)
    mb.add(vs, [(0, 1 + i, 1 + (i + 1) % m) for i in range(m)], mat)


def palmate(size, lobes=5, n=40):
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = size * (0.6 + 0.4 * abs(math.cos(lobes / 2 * th)) ** 0.8)
        r *= 1 - 0.35 * math.exp(-((th - math.pi) / 0.35) ** 2)          # the sinus where the stalk joins
        pts.append((r * math.cos(th) + size * 0.25, r * math.sin(th)))
    return pts


def heart(size, n=28, long=1.25):
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = size * (0.55 + 0.45 * math.cos(th) ** 2) * (1 - 0.55 * math.exp(-((th - math.pi) / 0.4) ** 2))
        pts.append((r * math.cos(th) * long + size * 0.45, r * math.sin(th)))
    return pts


def clear_of(x, y, pad):
    if abs(x) < PATH_W + 0.12:
        return False
    if abs(x - GX) < 0.9 + pad and abs(y - GY) < 0.9:
        return False
    return all(math.hypot(x - px, y - py) > pr + pad for (px, py, pr) in pumpkins)


def blade(mb, top, az, size, mat, lift=0.3, bend=1.0, cup=0.5, shape=None):
    """A leaf blade whose notch sits on the stalk top: it runs out along azimuth `az`, lifted by `lift` (rad),
    bending down toward its tip and cupped across."""
    u = Vector((math.cos(az) * math.cos(lift), math.sin(az) * math.cos(lift), math.sin(lift)))
    v = Vector((-math.sin(az), math.cos(az), 0.0))
    n = u.cross(v)
    pts = shape or palmate(size)
    sx = -min(a for a, b in pts)                          # put the notch (leftmost point) on the stalk
    c0 = Vector(top) + u * (sx + size * 0.25 if shape is None else sx)
    vs = [tuple(c0)]
    for (a, b) in pts:
        a2 = a + sx
        vs.append(tuple(Vector(top) + u * a2 + v * b - n * (bend * a2 * a2 / (2 * size) + cup * b * b / size)))
    m = len(pts)
    mb.add(vs, [(0, 1 + i, 1 + (i + 1) % m) for i in range(m)], mat)


placed = 0
tries = 0
while placed < int(A.get("leaves", 520)) and tries < 9000:
    tries += 1
    y = Y0 + 2.8 + (Y1 - Y0 - 3.3) * rnd.random() ** 1.3                   # denser near the camera
    x = rnd.choice((-1, 1)) * rnd.uniform(PATH_W + 0.12, HW - 0.1)
    near = y < 5.0
    size = rnd.uniform(0.14, 0.32) * (1.1 if near else 1.0)
    if not clear_of(x, y, 0.05) or (y < 2.6 and abs(x) < 1.5):
        continue
    h = rnd.uniform(0.08, 0.38)
    if abs(x) < PATH_W + 0.45 and rnd.random() < 0.7:
        az = math.atan2(rnd.uniform(-0.6, 0.3), -x) + rnd.uniform(-0.5, 0.5)    # edge leaves spill over the kerb
    else:
        az = rnd.uniform(0, 2 * math.pi)
    lift = rnd.uniform(0.1, 0.6)
    base = Vector((x, y, BED_Z))
    top = Vector((x + math.cos(az) * 0.03, y + math.sin(az) * 0.03, BED_Z + h))
    lv.cyl(tuple(base), tuple(top), 0.011, 0.008, n=5, mat=3, caps=False)
    blade(lv, top, az, size, rnd.randrange(3), lift=lift, bend=rnd.uniform(0.5, 1.6), cup=rnd.uniform(-0.3, 0.7))
    placed += 1
# elephant ears along the side walls: big drooping hearts on tall stalks, silhouettes against the glass
for i in range(int(A.get("ears", 16))):
    s = 1 if i % 2 else -1
    y = rnd.uniform(Y0 + 4.0, Y1 - 2.0)
    x = s * rnd.uniform(HW - 0.9, HW - 0.35)
    if not clear_of(x, y, 0.05):
        continue
    for k in range(rnd.randint(3, 5)):
        h = rnd.uniform(1.1, 2.3)
        az = math.atan2(0, -s) + rnd.uniform(-1.2, 1.2)
        along = Vector((math.cos(az), math.sin(az), -0.25)).normalized()
        top = Vector((x + along.x * 0.35, y + along.y * 0.35, BED_Z + h))
        lv.cyl((x, y, BED_Z), tuple((Vector((x, y, BED_Z)) + top) / 2 + Vector((0, 0, 0.1))), 0.02, 0.016, n=6, mat=3, caps=False)
        lv.cyl(tuple((Vector((x, y, BED_Z)) + top) / 2 + Vector((0, 0, 0.1))), tuple(top), 0.016, 0.01, n=6, mat=3, caps=False)
        sz = rnd.uniform(0.3, 0.5)
        blade(lv, top, az, sz, rnd.randrange(3), lift=rnd.uniform(-0.2, 0.3), bend=rnd.uniform(0.8, 1.5), cup=0.4,
              shape=heart(sz))
# vines: hanging strands from the roof with ivy leaves, and a few wrapped up the ribs
for i in range(int(A.get("strands", 46))):
    y = rnd.uniform(Y0 + 1.0, Y1 - 1.0)
    t = rnd.choice((rnd.uniform(0.04, 0.3), rnd.uniform(0.7, 0.96), rnd.uniform(0.3, 0.7)))
    x, z = arch(t)
    x *= 0.97
    z -= 0.08
    ln = rnd.uniform(0.5, 2.6) * (1.0 if 0.25 < t < 0.75 else 1.3)
    nseg = max(3, int(ln / 0.12))
    sway = (rnd.uniform(-0.25, 0.25), rnd.uniform(-0.2, 0.2))
    prev = Vector((x, y, z))
    for k in range(1, nseg + 1):
        f = k / nseg
        p = Vector((x + sway[0] * f * f, y + sway[1] * f * f + 0.04 * math.sin(k * 0.9), z - ln * f))
        lv.cyl(tuple(prev), tuple(p), 0.006, n=4, mat=3, caps=False)
        side = Vector((math.cos(k * 2.4 + i), math.sin(k * 2.4 + i), 0.0))
        c = p + side * 0.025
        normal = (side + Vector((0, -0.6, 0.2))).normalized()
        leaf_fan(lv, tuple(c), normal, (side + Vector((0, 0, -0.9))).normalized(), heart(rnd.uniform(0.035, 0.07) * (1.2 - 0.4 * f), n=14), 4,
                 droop=2.0)
        prev = p
for y in rnd.sample(ys_rib[2:-1], 6):                       # ivy climbing the ribs from one side
    s = rnd.choice((-1, 1))
    prof = rib_profile(0.12, 40)
    prof = prof[:len(prof) // 2] if s < 0 else prof[::-1][:len(prof) // 2]
    upto = rnd.randint(8, len(prof) - 1)
    for k in range(1, upto):
        (xa, za), (xb, zb) = prof[k - 1], prof[k]
        for j in range(2):
            f = j / 2
            px, pz = xa + (xb - xa) * f, za + (zb - za) * f
            ang = k * 1.7 + j
            c = Vector((px, y + 0.09 * math.cos(ang), pz)) + Vector((0, 0, 0.02))
            normal = Vector((-px * 0.2, -0.8 + 0.3 * math.sin(ang), 0.3)).normalized()
            leaf_fan(lv, tuple(c), normal, Vector((math.cos(ang), 0, math.sin(ang))), heart(rnd.uniform(0.04, 0.08), n=14), 4, droop=2.0)
lv.build("leaves", LEAF + [STEM, IVY], smooth=False)

# ---------------------------------------------------------------- hanging baskets with magenta fuchsias
FUCHSIA = B.principled("fuchsia", (0.72, 0.03, 0.3), rough=0.45, emission=(0.9, 0.08, 0.45),
                       strength=float(A.get("fuchsia", 0.35)))
SEPAL = B.principled("sepal", (0.5, 0.02, 0.1), rough=0.5, emission=(0.8, 0.05, 0.2), strength=float(A.get("fuchsia", 0.35)) * 0.6)
BASKET = B.principled("basket", (0.03, 0.04, 0.02), rough=0.85)
bk = K.MB()          # 0 basket/moss, 1 chain iron, 2 petals, 3 sepals, 4 ivy leaves
for (bx, by, bz) in ((-1.35, 3.8, 3.45), (1.7, 7.0, 3.6), (-1.6, 13.0, 3.7), (1.3, 18.5, 3.9)):
    lathe(bk, [(bz - 0.2, 0.05), (bz - 0.16, 0.16), (bz - 0.08, 0.24), (bz, 0.27)], n=16, mat=0, M=Matrix.Identity(4),
          off=(bx, by), cap1=False)
    lathe(bk, [(bz - 0.02, 0.26), (bz + 0.04, 0.2), (bz + 0.07, 0.05)], n=16, mat=0, M=Matrix.Identity(4), off=(bx, by), cap0=False)
    top = Vector((bx * 0.92, by, roof_z(bx * 0.92) - 0.12))
    for k in range(3):
        a = 2 * math.pi * k / 3
        bk.cyl((bx + 0.25 * math.cos(a), by + 0.25 * math.sin(a), bz), tuple(top), 0.006, n=4, mat=1)
    for j in range(int(A.get("trails", 10))):
        a = 2 * math.pi * j / 10 + rnd.uniform(-0.2, 0.2)
        p = Vector((bx + 0.25 * math.cos(a), by + 0.25 * math.sin(a), bz - 0.02))
        ln = rnd.uniform(0.35, 1.0)
        out = Vector((math.cos(a), math.sin(a), 0.0))
        prev = p
        for k in range(1, 8):
            f = k / 7
            q = p + out * (0.12 * f) + Vector((0, 0, -ln * f * f - 0.05 * f))
            bk.cyl(tuple(prev), tuple(q), 0.005, n=4, mat=4)
            if k % 2 == 0:
                leaf_fan(bk, tuple(q), (out + Vector((0, -0.5, 0.4))).normalized(), Vector((0, 0, -1)), heart(0.035, n=12), 4, droop=3.0)
            if k >= 3 and rnd.random() < 0.6:                            # a fuchsia: sepal tube and a bell
                fl = q + Vector((rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), -0.03))
                bk.cyl(tuple(q), tuple(fl), 0.004, n=4, mat=3)
                lathe(bk, [(fl.z - 0.045, 0.009), (fl.z - 0.03, 0.015), (fl.z - 0.01, 0.012), (fl.z, 0.005)], n=8, mat=2,
                      M=Matrix.Identity(4), off=(fl.x, fl.y))
                for m in range(4):                                         # flared sepals
                    a2 = m * math.pi / 2 + 0.4
                    bk.cyl(tuple(fl), (fl.x + 0.03 * math.cos(a2), fl.y + 0.03 * math.sin(a2), fl.z - 0.012), 0.005, 0.002, n=4, mat=3)
            prev = q
bk.build("baskets", [BASKET, IRON, FUCHSIA, SEPAL, IVY], smooth=True)

# ---------------------------------------------------------------- will-o'-wisps
WISP = B.emission("wisp", (0.2, 1.0, 0.7), float(A.get("wisp", 16.0)))
B.aov_mask(WISP, "glow")
wm = K.MB()
WISPS = [(-0.9, 4.2, 1.15, 0.035), (0.55, 5.6, 1.75, 0.03), (-1.55, 6.8, 0.95, 0.04), (1.35, 10.6, 1.4, 0.035),
         (-0.45, 12.2, 2.05, 0.03), (0.9, 13.6, 0.85, 0.03), (-1.7, 9.0, 1.9, 0.03), (2.2, 12.0, 2.3, 0.035),
         (-2.0, 16.5, 1.3, 0.035), (0.35, 17.5, 2.4, 0.03), (1.8, 19.6, 1.1, 0.03), (-0.9, 21.0, 1.7, 0.03),
         (0.6, 3.1, 0.75, 0.025), (-1.25, 3.0, 1.85, 0.03), (2.6, 4.8, 1.6, 0.04)]
for (x, y, z, r) in WISPS:
    lathe(wm, [(z - r, r * 0.15), (z - r * 0.7, r * 0.72), (z, r), (z + r * 0.7, r * 0.72), (z + r, r * 0.15)], n=10, mat=0,
          M=Matrix.Identity(4), off=(x, y))
wo = wm.build("wisps", [WISP], smooth=True)
wo.visible_shadow = False

# ---------------------------------------------------------------- outside: misty garden with bare trees beyond the door
tm = K.MB()
for i, (x, y, h) in enumerate(((-3.5, 34.0, 9.0), (4.0, 37.0, 10.0), (-1.0, 44.0, 12.0), (6.5, 47.0, 11.0), (-7.5, 41.0, 10.0),
                              (2.0, 55.0, 13.0), (-5.0, 58.0, 12.0), (9.0, 33.0, 9.0))):
    K.tree_parts(tm, (x, y, -0.2), h, 900 + i, depth=6, spread=0.62)
tm.build("trees", [B.principled("bark", (0.02, 0.02, 0.018), rough=0.85)], smooth=True)
# hedges and shrubs right outside the side glazing: dark shapes in the windows instead of a bright horizon
hg = K.MB()
for s_ in (-1, 1):
    yy = Y0
    while yy < Y1 + 2:
        ln = rnd.uniform(1.2, 3.0)
        h = rnd.uniform(1.4, 2.9)
        x0 = s_ * (HW + 0.6 + rnd.uniform(0, 0.8))
        x1 = x0 + s_ * rnd.uniform(1.2, 2.5)
        lo, hi = sorted((x0, x1))
        for k in range(3):                                 # a lumpy bush: three squashed blobs
            r = ln * rnd.uniform(0.35, 0.55)
            K.pumpkin_parts(hg, ((lo + hi) / 2 + rnd.uniform(-0.4, 0.4), yy + ln * (0.25 + 0.25 * k), h * rnd.uniform(0.45, 0.7)),
                            r, 0, 0, squash=h / (2 * r) * rnd.uniform(0.8, 1.1), ribs=5, groove=0.25, seed=int(yy * 10) + k,
                            nu=14, nv=8, stem=False)
        yy += ln * rnd.uniform(0.8, 1.1)
hg.build("hedges", [B.principled("hedge", (0.008, 0.016, 0.01), rough=0.95)])
# a stone urn on a plinth on the garden path, as a middle-distance silhouette
um = K.MB()
lathe(um, [(0.0, 0.32), (0.7, 0.3), (0.75, 0.36), (0.8, 0.2), (0.95, 0.12), (1.15, 0.3), (1.35, 0.38), (1.4, 0.3)],
      n=20, mat=0, M=Matrix.Identity(4), off=(0.4, 33.5))
um.build("urn", [B.principled("urnstone", (0.08, 0.085, 0.08), rough=0.8)], smooth=True)

mist, t = K.mat("garden_mist")
tc = t.coords("Object")
sep = t.n("ShaderNodeSeparateXYZ")
t.link(tc, sep.inputs[0])
nz = t.n("ShaderNodeTexNoise", Scale=0.08, Detail=3.0)
t.link(tc, nz.inputs["Vector"])
hfall = t.math("EXPONENT", t.math("MULTIPLY", sep.outputs["Z"], -1.0 / float(A.get("misth", 2.5))))
dens = t.math("MULTIPLY", t.math("MULTIPLY", hfall, float(A.get("mist", 0.07))), t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.4, 1.5))
pv = t.n("ShaderNodeVolumePrincipled", Color=(0.75, 0.92, 0.95), Anisotropy=0.5, **{"Emission Color": (0.5, 0.9, 0.95)})
t.link(dens, pv.inputs["Density"])
t.link(t.math("MULTIPLY", dens, float(A.get("mistglow", 1.6))), pv.inputs["Emission Strength"])   # self-lit, bounded
o = t.n("ShaderNodeOutputMaterial")
t.link(pv.outputs[0], o.inputs["Volume"])
mist.cycles.volume_step_rate = 3.0
# low and only beyond the far gable: if it rose higher, the camera would see it through the roof glass as a flat glow
mv = [(x, y, z) for z in (-0.5, float(A.get("misttop", 5.0))) for (x, y) in ((-45, Y1 + 0.4), (45, Y1 + 0.4), (45, 120), (-45, 120))]
mf = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
B.mesh("mistbox", mv, mf, mist)

# interior air: homogeneous haze, so the moonlight through the roof shows as shafts
air, t = K.mat("air")
pv = t.n("ShaderNodeVolumePrincipled", Color=(0.85, 1.0, 0.95), Density=float(A.get("air", 0.06)),
         Anisotropy=float(A.get("aniso", 0.6)))
o = t.n("ShaderNodeOutputMaterial")
t.link(pv.outputs[0], o.inputs["Volume"])
# the haze fills exactly the inside of the arch: a box would poke out above the curved roof, where the moon lights
# it unshadowed and the camera sees it through the glass
prof = [p for p in rib_profile(0.03, 48)]
prof[0] = (prof[0][0], 0.0)
prof[-1] = (prof[-1][0], 0.0)
npf = len(prof)
av = [(x, Y0, z) for (x, z) in prof] + [(x, Y1 - 0.05, z) for (x, z) in prof]
af = [(i, i + 1, npf + i + 1, npf + i) for i in range(npf - 1)] + [(npf - 1, 0, npf, 2 * npf - 1)]
af += [tuple(range(npf - 1, -1, -1)), tuple(range(npf, 2 * npf))]
B.mesh("airbox", av, af, air)

# ---------------------------------------------------------------- sky and moon
w = sc.world
w.use_nodes = True
wt = w.node_tree
wt.nodes.clear()
tc = wt.nodes.new("ShaderNodeTexCoord")
sep = wt.nodes.new("ShaderNodeSeparateXYZ")
wt.links.new(tc.outputs["Generated"], sep.inputs[0])
ramp = wt.nodes.new("ShaderNodeValToRGB")
el = ramp.color_ramp.elements
el[0].position, el[0].color = 0.0, (0.045, 0.085, 0.09, 1)
el[1].position, el[1].color = 0.8, (0.006, 0.014, 0.03, 1)
e = el.new(0.3)
e.color = (0.016, 0.038, 0.055, 1)
wt.links.new(sep.outputs["Z"], ramp.inputs[0])
cl = wt.nodes.new("ShaderNodeTexNoise")
cl.inputs["Scale"].default_value = 2.5
cl.inputs["Detail"].default_value = 6.0
wt.links.new(tc.outputs["Generated"], cl.inputs["Vector"])
cmap = wt.nodes.new("ShaderNodeMapRange")
cmap.inputs["From Min"].default_value, cmap.inputs["From Max"].default_value = 0.5, 0.75
mixn = wt.nodes.new("ShaderNodeMix")
mixn.data_type = "RGBA"
wt.links.new(cl.outputs["Fac"], cmap.inputs["Value"])
wt.links.new(cmap.outputs[0], mixn.inputs[0])
wt.links.new(ramp.outputs[0], mixn.inputs[6])
mixn.inputs[7].default_value = (0.03, 0.06, 0.07, 1)
MEL, MAZ = math.radians(float(A.get("moon_el", 50.0))), math.radians(float(A.get("moon_az", -25.0)))
mdir = Vector((math.sin(MAZ) * math.cos(MEL), math.cos(MAZ) * math.cos(MEL), math.sin(MEL)))
nrm = wt.nodes.new("ShaderNodeVectorMath")
nrm.operation = "NORMALIZE"
wt.links.new(tc.outputs["Generated"], nrm.inputs[0])
dot = wt.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
wt.links.new(nrm.outputs[0], dot.inputs[0])
dot.inputs[1].default_value = tuple(mdir)
halo = []
for ex, k in ((60.0, 1.2), (8.0, 0.08)):                    # moon glow on the clouds: tight halo + broad wash
    mx = wt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    wt.links.new(dot.outputs["Value"], mx.inputs[0])
    mx.inputs[1].default_value = 0.0
    pw = wt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    wt.links.new(mx.outputs[0], pw.inputs[0])
    pw.inputs[1].default_value = ex
    ml = wt.nodes.new("ShaderNodeMath")
    ml.operation = "MULTIPLY"
    wt.links.new(pw.outputs[0], ml.inputs[0])
    ml.inputs[1].default_value = k
    halo.append(ml)
hs = wt.nodes.new("ShaderNodeMath")
hs.operation = "ADD"
wt.links.new(halo[0].outputs[0], hs.inputs[0])
wt.links.new(halo[1].outputs[0], hs.inputs[1])
addc = wt.nodes.new("ShaderNodeMix")
addc.data_type = "RGBA"
addc.blend_type = "ADD"
wt.links.new(hs.outputs[0], addc.inputs[0])
wt.links.new(mixn.outputs[2], addc.inputs[6])
addc.inputs[7].default_value = (0.55, 0.85, 0.88, 1)
bg = wt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = float(A.get("sky", 2.2))
wt.links.new(addc.outputs[2], bg.inputs[0])
wout = wt.nodes.new("ShaderNodeOutputWorld")
wt.links.new(bg.outputs[0], wout.inputs[0])

B.light("SUN", tuple(CAM + mdir * 100), float(A.get("moonlight", 55.0)), color=(0.45, 0.88, 0.85),
        size=math.radians(0.5), target=tuple(CAM), name="moon")

# ---------------------------------------------------------------- camera, passes, fx positions, render
tgt = CAM + Vector((0.0, 30.0, 0.0))
cam = B.camera(tuple(CAM), tuple(tgt), lens=float(A.get("lens", 20.0)), shift_y=float(A.get("shift", 0.08)), clip_end=500)
B.passes(mist=(1.0, 60.0, "LINEAR"), extra=("z", "normal"))


bpy.context.view_layer.update()


def screen(p, r=0.0):
    co = world_to_camera_view(sc, cam, Vector(p))
    rv = r * (cam.data.lens / cam.data.sensor_width) / max(co.z, 1e-3)      # radius as a fraction of the image height
    return [co.x, 1.0 - co.y, co.z, rv]


fx = {"wisps": [screen((x, y, z), r) for (x, y, z, r) in WISPS], "lantern": screen(LANTERN, 0.06),
      "door": screen((0.0, Y1, 1.6)), "gardener_feet": screen((GX, GY, 0.0)), "gardener_head": screen((GX, GY, 1.85))}
if CHECK:
    nv = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == "MESH")
    print("CHECK objects", len(bpy.data.objects), "verts", nv, "lights", len(bpy.data.lights), "leaves", placed)
    print("CHECK fx", json.dumps(fx))
    if "save" in A:
        bpy.ops.wm.save_as_mainfile(filepath=f"{WORK}/33_check.blend")
    sys.exit(0)
os.makedirs(f"{WORK}/{TAG}", exist_ok=True)
with open(f"{WORK}/{TAG}/fx.json", "w") as f:
    json.dump(fx, f)
bpy.context.view_layer.cycles.use_pass_volume_direct = True     # the shafts on their own, so the post can lift them
B.render(TAG, outputs=("Image", "VolumeDir"))
# data passes again without the volumes: a camera ray that scatters in the haze writes no surface depth/normal/AOV
B.render_data(TAG, hide=[bpy.data.objects["mistbox"], bpy.data.objects["airbox"]],
              outputs=("Depth", "Normal", "actor", "glow"))
