"""31 arcane-rooftop: two kids in costumes (a little witch and a sheet ghost) sit on a steep roof ridge beside a
brick chimney, a lit jack-o'-lantern between them, and watch hundreds of paper pumpkin lanterns rise from an old
city of steep roofs, spires and a river toward a big low moon. Base render for the Arcane-style painter
(src/31_arcane_rooftop_post.py). The sky is painted in post (film is transparent; the world only lights and
reflects).
python src/gpu_run.py blender/31_arcane_rooftop.py -- tag=31-test pct=35 samples=16 [check=1]"""
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
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

import bl_build as K  # noqa: E402
import bl_common as B  # noqa: E402

A = B.args()
PCT = int(A.get("pct", 25))
SPP = int(A.get("samples", 16))
TAG = A.get("tag", "31-test")
WORK = ROOT + "/work"
rnd = random.Random(31)
nrng = np.random.default_rng(31)

sc = B.empty_scene()
B.cycles(samples=SPP, pct=PCT, bounces=4, clamp_indirect=3.0)
sc.render.film_transparent = True

ZR = float(A.get("zr", 70.0))              # ridge height of the kids' roof (a house on a hill) above the city
PITCH = math.radians(50)                    # roof pitch of the kids' roof
CAM = Vector((float(A.get("camx", 0.15)), -float(A.get("camd", 3.6)), ZR + float(A.get("camh", 1.9))))
YAW = math.radians(float(A.get("yaw", 0.0)))
TILT = math.radians(float(A.get("tilt", -7.5)))
LENS = float(A.get("lens", 32))
MOON_AZ = YAW + math.radians(float(A.get("moon_az", -4.5)))
MOON_EL = math.radians(float(A.get("moon_el", 4.5)))
MOON_R = math.radians(float(A.get("moon_r", 3.0)))
MDIR = Vector((math.sin(MOON_AZ) * math.cos(MOON_EL), math.cos(MOON_AZ) * math.cos(MOON_EL), math.sin(MOON_EL)))
WIN_STR = float(A.get("win", 1.6))
LAN_STR = float(A.get("lan", 5.0))


def aov(m, name):
    B.aov_mask(m, name)
    return m


def quiet(m):
    """Emissive surfaces that should glow but not be sampled as lights (hundreds of them: noise for nothing)."""
    m.cycles.emission_sampling = "NONE"
    return m


# ---------------------------------------------------------------- a mesh builder with UVs and a per-face tint

class UMB:
    def __init__(self):
        self.v, self.f, self.uv, self.m, self.col = [], [], [], [], []

    def face(self, pts, uvs, mat=0, col=(1, 1, 1)):
        o = len(self.v)
        self.v.extend(tuple(p) for p in pts)
        self.f.append(tuple(range(o, o + len(pts))))
        self.uv.extend(uvs)
        self.m.append(mat)
        self.col.extend([(*col, 1.0)] * len(pts))

    def build(self, name, mats):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(self.uv, np.float32).ravel())
        ca = me.color_attributes.new("tint", "FLOAT_COLOR", "CORNER")
        ca.data.foreach_set("color", np.asarray(self.col, np.float32).ravel())
        for m in mats:
            me.materials.append(m)
        if len(mats) > 1:
            me.polygons.foreach_set("material_index", self.m)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def lathe_parts(mb, cx, cy, prof, mat, nu=40, sx=1.0, sy=1.0, fold=None, close_top=True):
    """Surface of revolution: prof = [(z, r), ...] bottom to top; fold(theta, t) adds a relative radius offset."""
    vs, fs = [], []
    rows = len(prof)
    for j, (z, r) in enumerate(prof):
        t = j / (rows - 1)
        for i in range(nu):
            th = 2 * math.pi * i / nu
            rr = r * (1 + (fold(th, t) if fold else 0.0))
            vs.append((cx + rr * math.cos(th) * sx, cy + rr * math.sin(th) * sy, z))
    for j in range(rows - 1):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            fs.append((a, b, b + nu, a + nu))
    if close_top:
        vs.append((cx, cy, prof[-1][0]))
        c = len(vs) - 1
        top = (rows - 1) * nu
        fs.extend((top + i, top + (i + 1) % nu, c) for i in range(nu))
    mb.add(vs, fs, mat)


def ellipsoid_parts(mb, c, r, mat, nu=28, nv=16):
    vs, fs = [], []
    for j in range(nv + 1):
        ph = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            vs.append((c[0] + r[0] * math.sin(ph) * math.cos(th), c[1] + r[1] * math.sin(ph) * math.sin(th),
                       c[2] + r[2] * math.cos(ph)))
    for j in range(nv):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            fs.append((a, a + nu, b + nu, b))
    mb.add(vs, fs, mat)


# ---------------------------------------------------------------- materials

def city_wall_mat():
    """Plaster walls; windows from a Brick texture laid out in metres on the UVs (u along the wall, v up), the
    brick's random grey decides which windows are lit and how warm."""
    m, t = K.mat("city_wall")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(t.coords("UV"), sep.inputs[0])
    vec = t.n("ShaderNodeCombineXYZ")
    t.link(t.math("MULTIPLY", sep.outputs["X"], 0.1), vec.inputs["X"])
    t.link(t.math("MULTIPLY", sep.outputs["Y"], 0.1), vec.inputs["Y"])
    br = t.n("ShaderNodeTexBrick", Color1=(0, 0, 0), Color2=(1, 1, 1), Mortar=(0, 0, 0), Scale=1.0,
             **{"Mortar Size": 0.05, "Mortar Smooth": 0.15, "Brick Width": 0.25, "Row Height": 0.31, "Bias": 0.0})
    br.offset = 0.0
    t.link(vec.outputs[0], br.inputs["Vector"])
    win = t.math("SUBTRACT", 1.0, br.outputs["Fac"])
    rv = t.math("ADD", br.outputs["Color"], 0.0)
    lit = t.math("MULTIPLY", t.mapr(rv, 0.56, 0.6), win)
    warm = t.mixc(t.math("FRACT", t.math("MULTIPLY", rv, 7.31)), (1.0, 0.36, 0.09), (1.0, 0.62, 0.28))
    tint = t.n("ShaderNodeAttribute", _attribute_type="GEOMETRY", _attribute_name="tint").outputs["Color"]
    nz = t.n("ShaderNodeTexNoise", Scale=0.6, Detail=4.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    plaster = t.mixc(t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.0, 0.5), tint, (0.03, 0.03, 0.035))
    col = t.mixc(win, plaster, (0.008, 0.01, 0.016))
    strength = t.math("MULTIPLY", lit, t.math("MULTIPLY", t.mapr(rv, 0.6, 1.0, 0.25, 1.4), WIN_STR))
    t.out(t.bsdf(col, t.mixf(win, 0.85, 0.2), emission=warm, strength=strength))
    return quiet(m)


def city_roof_mat():
    m, t = K.mat("city_roof")
    tint = t.n("ShaderNodeAttribute", _attribute_type="GEOMETRY", _attribute_name="tint").outputs["Color"]
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(t.coords("UV"), sep.inputs[0])
    rows = t.math("FRACT", t.math("MULTIPLY", sep.outputs["Y"], 3.2))
    nz = t.n("ShaderNodeTexNoise", Scale=1.5, Detail=5.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.0, 0.6), tint, (0.02, 0.018, 0.02))
    col = t.mixc(t.mapr(rows, 0.0, 0.25, 0.45, 0.0), col, (0.005, 0.005, 0.007))
    t.out(t.bsdf(col, 0.42, spec=0.6))
    return m


def stone_mat(name, base):
    m, t = K.mat(name)
    nz = t.n("ShaderNodeTexNoise", Scale=0.8, Detail=8.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.3, 0.7), base, tuple(c * 0.55 for c in base))
    t.out(t.bsdf(col, 0.8))
    return m


def ground_mat(squares):
    """Dark ground with a faint warm street glow; bright squares where the lanterns start; beyond the houses a
    carpet of tiny lights reaching for the horizon."""
    m, t = K.mat("ground")
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    glow = t.math("ADD", 0.12, 0.0)
    for (sx, sy, sr, sk) in squares:
        d = t.n("ShaderNodeVectorMath", _operation="DISTANCE", Vector=tc)
        d.inputs[1].default_value = (sx, sy, 0.0)
        glow = t.math("ADD", glow, t.mapr(d.outputs["Value"], sr * 0.3, sr * 1.6, sk, 0.0))
    vor = t.n("ShaderNodeTexVoronoi", Scale=0.05)
    t.link(tc, vor.inputs["Vector"])
    dots = t.math("MULTIPLY", t.mapr(vor.outputs["Distance"], 0.16, 0.05),
                  t.mapr(t.math("ADD", vor.outputs["Color"], 0.0), 0.35, 0.45))
    far = t.mapr(sep.outputs["Y"], 2100.0, 2600.0)
    glow = t.math("ADD", glow, t.math("MULTIPLY", t.math("MULTIPLY", dots, far), 9.0))
    t.out(t.bsdf((0.025, 0.022, 0.02), 0.9, emission=(1.0, 0.45, 0.15), strength=glow))
    return quiet(m)


def water_mat():
    m, t = K.mat("water")
    tc = t.coords("Object")
    mp = t.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.25, 2.5, 1.0)              # ripples stretched across the river
    t.link(tc, mp.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=0.9, Detail=6.0, Roughness=0.6)
    t.link(mp.outputs[0], nz.inputs["Vector"])
    t.out(t.bsdf((0.01, 0.015, 0.02), 0.07, spec=0.6, normal=t.bump(nz.outputs["Fac"], 0.22, 0.4)))
    return aov(m, "water")


def tile_mat():
    """The kids' roof: clay pantiles. UV u runs along the ridge, v down the slope: half-round channels run down
    the slope (|sin| across u) and every row's lower edge overlaps the next one with a dark shadow line."""
    m, t = K.mat("tiles")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(t.coords("UV"), sep.inputs[0])
    u, v = sep.outputs["X"], sep.outputs["Y"]
    ch = t.math("ABSOLUTE", t.math("SINE", t.math("MULTIPLY", u, math.pi / 0.21)))       # tile channels
    row = t.math("FRACT", t.math("DIVIDE", v, 0.3))                                        # 0 at a row's top
    edge = t.mapr(row, 0.86, 1.0, 0.0, 1.0)                                                # the overlapping lip
    rid = t.math("FLOOR", t.math("DIVIDE", v, 0.3))
    cid = t.math("FLOOR", t.math("DIVIDE", u, 0.21))
    wn = t.n("ShaderNodeTexWhiteNoise", _noise_dimensions="2D")
    cmb = t.n("ShaderNodeCombineXYZ")
    t.link(cid, cmb.inputs["X"])
    t.link(rid, cmb.inputs["Y"])
    t.link(cmb.outputs[0], wn.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=7.0, Detail=6.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    col = t.mixc(wn.outputs["Value"], (0.2, 0.075, 0.045), (0.12, 0.05, 0.035))
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.45, 0.75, 0.0, 0.5), col, (0.04, 0.045, 0.03))     # lichen
    col = t.mixc(t.math("MULTIPLY", edge, 0.8), col, (0.01, 0.006, 0.005))
    h = t.math("ADD", t.math("MULTIPLY", ch, 0.6), t.math("MULTIPLY", row, -0.5))
    h = t.math("ADD", h, t.math("MULTIPLY", nz.outputs["Fac"], 0.15))
    t.out(t.bsdf(col, t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.4, 0.7), spec=0.55, normal=t.bump(h, 0.7, 0.04)))
    return m


def brick_mat():
    m, t = K.mat("chimney_brick")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(t.coords("UV"), sep.inputs[0])
    vec = t.n("ShaderNodeCombineXYZ")
    t.link(sep.outputs["X"], vec.inputs["X"])
    t.link(sep.outputs["Y"], vec.inputs["Y"])
    br = t.n("ShaderNodeTexBrick", Color1=(0.24, 0.085, 0.05), Color2=(0.13, 0.055, 0.04), Mortar=(0.09, 0.08, 0.07),
             Scale=1.0, **{"Mortar Size": 0.01, "Brick Width": 0.22, "Row Height": 0.072, "Bias": 0.0})
    br.offset = 0.5
    t.link(vec.outputs[0], br.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", Scale=3.0, Detail=6.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.4, 0.8, 0.0, 0.7), br.outputs["Color"], (0.025, 0.02, 0.02))   # soot
    t.out(t.bsdf(col, 0.8, normal=t.bump(t.math("SUBTRACT", 1.0, br.outputs["Fac"]), 0.4, 0.01)))
    return m


def cloth_mat(name, color, sheen=0.6, rough=0.75, sss=0.0):
    m, t = K.mat(name)
    nz = t.n("ShaderNodeTexNoise", Scale=14.0, Detail=4.0)
    t.link(t.coords("Object"), nz.inputs["Vector"])
    col = t.mixc(t.mapr(nz.outputs["Fac"], 0.3, 0.7, 0.0, 0.3), color, tuple(c * 0.7 for c in color))
    p = t.n("ShaderNodeBsdfPrincipled")
    t.set(p.inputs["Base Color"], col)
    t.set(p.inputs["Roughness"], rough)
    t.set(p.inputs["Sheen Weight"], sheen)
    t.set(p.inputs["Sheen Roughness"], 0.35)
    if sss:
        t.set(p.inputs["Subsurface Weight"], sss)
        p.inputs["Subsurface Radius"].default_value = (0.05, 0.04, 0.03)
    t.out(p.outputs[0])
    return aov(m, "actor")


def lantern_mat(name, face_png=None):
    """Paper pumpkin lantern lit from inside: brightest and yellowest where the paper faces you and at the open
    bottom where the flame is, deeper orange toward the rim; a dark painted face on the side facing local -Y."""
    m, t = K.mat(name)
    lw = t.n("ShaderNodeLayerWeight", Blend=0.45)
    tc = t.coords("Object")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc, sep.inputs[0])
    facing = t.math("SUBTRACT", 1.0, lw.outputs["Facing"])
    bottom = t.mapr(sep.outputs["Z"], 0.4, -0.75, 0.0, 1.0)
    k = t.math("ADD", t.math("MULTIPLY", facing, 0.8), t.math("MULTIPLY", bottom, 0.6))
    col = t.mixc(t.mapr(k, 0.2, 1.1), (0.9, 0.22, 0.03), (1.0, 0.62, 0.22))
    stren = t.math("MULTIPLY", t.mapr(k, 0.1, 1.2, 0.35, 1.25),
                   t.n("ShaderNodeAttribute", _attribute_type="OBJECT", _attribute_name="glow").outputs["Fac"])
    if face_png:
        comb = t.n("ShaderNodeCombineXYZ")
        t.link(t.math("ADD", t.math("MULTIPLY", sep.outputs["X"], 0.5), 0.5), comb.inputs["X"])
        t.link(t.math("ADD", t.math("MULTIPLY", t.math("ADD", sep.outputs["Z"], 0.08), 0.62), 0.5), comb.inputs["Y"])
        img = t.n("ShaderNodeTexImage", _extension="CLIP")
        img.image = bpy.data.images.load(face_png, check_existing=True)
        img.image.colorspace_settings.name = "Non-Color"
        t.link(comb.outputs[0], img.inputs["Vector"])
        front = t.mapr(sep.outputs["Y"], -0.3, -0.7)
        face = t.math("MULTIPLY", t.math("ADD", img.outputs["Color"], 0.0), front, clamp=True)
        stren = t.math("MULTIPLY", stren, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", face, 0.88)))
        col = t.mixc(face, col, (0.35, 0.05, 0.0))
    e = t.n("ShaderNodeEmission")
    t.link(col, e.inputs["Color"])
    t.link(stren, e.inputs["Strength"])
    t.out(e.outputs[0])
    return quiet(aov(m, "lamp"))


WALL, ROOF = city_wall_mat(), city_roof_mat()
STONE = stone_mat("stone", (0.09, 0.085, 0.08))
SPIRE = B.principled("spire_roof", (0.03, 0.045, 0.05), rough=0.4, metal=0.3)
CLOCKF = quiet(B.emission("clockface", (1.0, 0.72, 0.4), 6.0))
LAMPS = quiet(B.emission("bridge_lamps", (1.0, 0.6, 0.25), 25.0))
TILES, BRICK = tile_mat(), brick_mat()
HAT = cloth_mat("hat", (0.012, 0.01, 0.016), sheen=0.5, rough=0.6)
BAND = cloth_mat("hatband", (0.55, 0.14, 0.02), sheen=0.3)
CAPE = cloth_mat("cape", (0.035, 0.012, 0.05), sheen=0.3)
HAIR = cloth_mat("hair", (0.28, 0.065, 0.02), sheen=0.6, rough=0.55)
SKIN = cloth_mat("skin", (0.55, 0.32, 0.24), sheen=0.0, rough=0.5, sss=0.3)
SHEET = cloth_mat("sheet", (0.72, 0.72, 0.76), sheen=0.5, rough=0.85, sss=0.15)
FACES = [f"{WORK}/face_{k}.png" for k in ("classic", "grin", "scary")]
LAN_FACE = [lantern_mat(f"lantern_face{k}", FACES[k]) for k in range(3)]
LAN_PLAIN = lantern_mat("lantern_plain")
LAN_STEM = quiet(B.emission("lantern_stem", (0.25, 0.3, 0.05), 0.6))

# ---------------------------------------------------------------- the river (centre line x(y), width w(y))
RY = np.array([-200, 0, 150, 300, 430, 600, 850, 1100, 1400, 1800, 2300, 3000, 9000], float)
RX = np.array([200, 175, 110, 35, -30, -75, -65, -115, -200, -285, -330, -340, -340], float)
_yy = np.linspace(-200, 9000, 4000)
_xx = np.interp(_yy, RY, RX)
_k = np.ones(81) / 81
_xx = np.convolve(np.pad(_xx, 40, mode="edge"), _k, mode="valid")


def river_x(y):
    return float(np.interp(y, _yy, _xx))


def river_w(y):
    return 46.0 + 0.012 * max(y, 0.0)


# ---------------------------------------------------------------- ground, embankments, water, far hills
g = UMB()
ys = np.concatenate([np.arange(-60, 600, 3.0), np.arange(600, 2600, 10.0), np.arange(2600, 9001, 40.0)])
XL, XR = -4000.0, 4000.0
for y0, y1 in zip(ys[:-1], ys[1:]):
    l0, l1 = river_x(y0) - river_w(y0) / 2, river_x(y1) - river_w(y1) / 2
    r0, r1 = river_x(y0) + river_w(y0) / 2, river_x(y1) + river_w(y1) / 2
    g.face([(XL, y0, 0), (l0, y0, 0), (l1, y1, 0), (XL, y1, 0)], [(0, 0)] * 4, 0)
    g.face([(r0, y0, 0), (XR, y0, 0), (XR, y1, 0), (r1, y1, 0)], [(0, 0)] * 4, 0)
    g.face([(l0, y0, -1.6), (l1, y1, -1.6), (l1, y1, 0), (l0, y0, 0)], [(0, 0)] * 4, 1)      # embankment walls
    g.face([(r1, y1, -1.6), (r0, y0, -1.6), (r0, y0, 0), (r1, y1, 0)], [(0, 0)] * 4, 1)
SQUARES = [(70.0, 480.0, 34.0, 1.6), (-150.0, 760.0, 26.0, 0.9), (130.0, 1050.0, 30.0, 0.9),
           (-60.0, 1350.0, 28.0, 0.8), (230.0, 1500.0, 30.0, 0.6)]
g.build("ground", [ground_mat(SQUARES), STONE])
B.mesh("water", [(XL, -200, -1.2), (XR, -200, -1.2), (XR, 9000, -1.2), (XL, 9000, -1.2)], [(0, 1, 2, 3)], water_mat())
hv, hf = [], []
hx = np.linspace(-9000, 9000, 220)
hz = 60 + 120 * np.abs(np.sin(hx / 1300 + 1.0)) + 70 * np.sin(hx / 480 + 2.0) ** 2 + 30 * nrng.random(len(hx))
for i, (x, z) in enumerate(zip(hx, hz)):
    hv += [(x, 6800.0 + 400 * math.sin(x / 2000), -5.0), (x, 6800.0 + 400 * math.sin(x / 2000), float(z))]
for i in range(len(hx) - 1):
    hf.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
B.mesh("hills", hv, hf, B.principled("hills", (0.01, 0.012, 0.016), rough=0.9))

# ---------------------------------------------------------------- the city
city = UMB()
WALL_T = [(0.18, 0.15, 0.12), (0.15, 0.15, 0.15), (0.19, 0.135, 0.11), (0.13, 0.15, 0.16), (0.2, 0.17, 0.125),
          (0.16, 0.12, 0.13)]
ROOF_T = [(0.12, 0.05, 0.035), (0.09, 0.045, 0.035), (0.05, 0.055, 0.065), (0.06, 0.06, 0.07), (0.14, 0.065, 0.04)]


def house(cx, cy, w, d, h, rz, wt, rt, chimney=True):
    """Gabled house, ridge along its local x. Walls carry windows (UV in metres), roofs tile rows."""
    ca, sa = math.cos(rz), math.sin(rz)

    def P(x, y, z):
        return (cx + x * ca - y * sa, cy + x * sa + y * ca, z)

    hw, hd = w / 2, d / 2
    rh = hd * math.tan(math.radians(rnd.uniform(48, 58)))
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    u = rnd.uniform(0, 500)
    for i in range(4):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        L = math.hypot(x1 - x0, y1 - y0)
        city.face([P(x0, y0, 0), P(x1, y1, 0), P(x1, y1, h), P(x0, y0, h)],
                  [(u, 0), (u + L, 0), (u + L, h), (u, h)], 0, wt)
        u += L
    for sx in (-1, 1):                                                     # gables
        pts = [P(sx * hw, -sx * hd, h), P(sx * hw, sx * hd, h), P(sx * hw, 0, h + rh)]
        city.face(pts, [(u, h), (u + d, h), (u + hd, h + rh)], 0, wt)
        u += d
    ov = 0.5
    slope = math.hypot(hd + ov, rh + ov * rh / hd)
    for sy in (-1, 1):                                                     # roof planes
        eave = (-sy * 0 + sy * (hd + ov))
        pts = [P(-hw - ov, eave, h - ov * rh / hd), P(hw + ov, eave, h - ov * rh / hd), P(hw + ov, 0, h + rh),
               P(-hw - ov, 0, h + rh)]
        if sy > 0:
            pts = [pts[1], pts[0], pts[3], pts[2]]
        city.face(pts, [(0, 0), (w, 0), (w, slope), (0, slope)], 1, rt)
    if chimney and rnd.random() < 0.6:
        x = rnd.uniform(-hw * 0.7, hw * 0.7)
        y = rnd.uniform(-hd * 0.6, hd * 0.6)
        z0 = h + rh * (1 - abs(y) / hd) - 0.3
        cw = rnd.uniform(0.6, 1.0)
        zt = h + rh + rnd.uniform(0.6, 1.8)
        cs = [(x - cw, y - cw * 0.6), (x + cw, y - cw * 0.6), (x + cw, y + cw * 0.6), (x - cw, y + cw * 0.6)]
        for i in range(4):
            (x0, y0), (x1, y1) = cs[i], cs[(i + 1) % 4]
            city.face([P(x0, y0, z0), P(x1, y1, z0), P(x1, y1, zt), P(x0, y0, zt)], [(0, 0)] * 4, 1, (0.08, 0.04, 0.03))
        city.face([P(*cs[0], zt), P(*cs[1], zt), P(*cs[2], zt), P(*cs[3], zt)], [(0, 0)] * 4, 1, (0.02, 0.02, 0.02))


def tower(cx, cy, w, h, spire_h, rz=0.0, clock=False):
    ca, sa = math.cos(rz), math.sin(rz)
    mb = K.MB()
    mb.box((cx, cy, h / 2), (w, w, h), 0, rz)
    mb.box((cx, cy, h + 0.4), (w * 1.12, w * 1.12, 0.8), 0, rz)
    hw = w * 0.56
    apex = (cx, cy, h + 0.8 + spire_h)
    base = [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]
    bp = [(cx + x * ca - y * sa, cy + x * sa + y * ca, h + 0.8) for x, y in base]
    mb.add(bp + [apex], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], 1)
    for (x, y) in base:                                                    # four corner pinnacles
        px, py = cx + x * 0.92 * ca - y * 0.92 * sa, cy + x * 0.92 * sa + y * 0.92 * ca
        mb.cyl((px, py, h + 0.8), (px, py, h + 0.8 + spire_h * 0.25), w * 0.06, w * 0.005, n=6, mat=1)
    ob = mb.build(f"tower{len(bpy.data.objects)}", [STONE, SPIRE])
    if clock:
        r = w * 0.3
        cz = h * 0.82
        vs = [(cx + r * math.cos(a), cy - w / 2 - 0.15, cz + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 33)[:-1]]
        B.mesh("clockface", vs + [(cx, cy - w / 2 - 0.15, cz)], [(i, (i + 1) % 32, 32) for i in range(32)], CLOCKF)
    return ob


def bridge(y, deck_z=7.0, width=12.0):
    x0 = river_x(y) - river_w(y) / 2 - 8
    x1 = river_x(y) + river_w(y) / 2 + 8
    mb = K.MB()
    mb.boxr(x0, x1, y - width / 2, y + width / 2, deck_z - 1.0, deck_z, 0)
    for sx in (-1, 1):
        mb.boxr(x0, x1, y + sx * width / 2 - 0.4 * (sx > 0), y + sx * width / 2 + 0.4 * (sx < 0), deck_z, deck_z + 1.1, 0)
    n = 4
    for i in range(1, n):
        x = x0 + (x1 - x0) * i / n
        mb.boxr(x - 2.2, x + 2.2, y - width / 2, y + width / 2, -1.5, deck_z - 1.0, 0)
    mb.build(f"bridge{int(y)}", [STONE])
    lm = K.MB()
    for i in range(9):
        x = x0 + (x1 - x0) * (i + 0.5) / 9
        for sx in (-1, 1):
            lm.box((x, y + sx * width / 2, deck_z + 2.2), (0.5, 0.5, 0.7), 0)
    lm.build(f"bridgelamps{int(y)}", [LAMPS])


def in_view(x, y, margin=40.0):
    return y > 40 and abs(x - CAM.x - (y - CAM.y) * math.tan(YAW)) < (y - CAM.y) * 0.4 + margin


TOWERS = [(-120, 380, 9, 30, 20), (95, 690, 10, 34, 24), (-270, 930, 9, 28, 18), (170, 1150, 12, 40, 26),
          (-20, 1560, 10, 36, 22), (-430, 1700, 12, 44, 30), (300, 1950, 11, 38, 24), (-640, 2300, 12, 46, 30)]
CLOCK_D = float(A.get("clock_d", 700.0))
caz = MOON_AZ + math.radians(float(A.get("clock_az", 2.3)))
CLOCK = (CAM.x + math.sin(caz) * CLOCK_D, CAM.y + math.cos(caz) * CLOCK_D)
keepout = [(x, y, w * 0.9 + 6) for (x, y, w, h, s) in TOWERS] + [(CLOCK[0], CLOCK[1], 16)]
keepout += [(sx, sy, sr) for (sx, sy, sr, sk) in SQUARES]
nh = 0
y = 52.0
row = 0
while y < 2350:
    cell = 13.0 + 0.0075 * y
    street_y = 9.0 if row % 5 == 0 else 3.0
    x = -1400.0 + rnd.uniform(0, cell)
    col_i = 0
    while x < 1400:
        cw = cell * rnd.uniform(0.8, 1.25)
        street_x = 9.0 if col_i % 6 == 0 else rnd.uniform(1.5, 4.0)
        cx, cy = x + cw / 2, y + cell / 2
        x += cw + street_x
        col_i += 1
        if not in_view(cx, cy):
            continue
        if abs(cx - river_x(cy)) < river_w(cy) / 2 + 7 + cw / 2:
            continue
        if any(math.hypot(cx - kx, cy - ky) < kr + cw / 2 for (kx, ky, kr) in keepout):
            continue
        w = cw * rnd.uniform(0.85, 1.0)
        d = (cell - street_y) * rnd.uniform(0.85, 1.0)
        floors = rnd.choice((2, 3, 3, 4, 4, 5)) if cy < 1400 else rnd.choice((3, 4, 5, 6))
        h = floors * 3.1 + rnd.uniform(-0.3, 0.4)
        rz = rnd.uniform(-0.06, 0.06)
        if rnd.random() < 0.35:
            w, d, rz = d, w, rz + math.pi / 2
        k = rnd.random()
        wt = tuple(c * rnd.uniform(0.8, 1.15) for c in WALL_T[rnd.randrange(len(WALL_T))])
        rt = ROOF_T[rnd.randrange(len(ROOF_T))]
        house(cx, cy, w, d, h, rz, wt, rt)
        nh += 1
    y += cell + street_y
    row += 1
city.build("city", [WALL, ROOF])
for (x, y, w, h, s) in TOWERS:
    tower(x, y, w, h, s, rz=rnd.uniform(-0.1, 0.1))
CLOCK_H = float(A.get("clock_h", 100.0))
tower(CLOCK[0], CLOCK[1], 14, CLOCK_H, 30.0, clock=True)
for y in (230.0, 520.0, 800.0, 1240.0, 1700.0):
    bridge(y)
print("houses", nh)

# ---------------------------------------------------------------- the kids' roof, chimney and the kids
roof = UMB()
X0, X1 = -7.0, 7.0
run = 7.0
slope_len = run / math.cos(PITCH)
for sy in (-1, 1):
    eave = (sy * run, ZR - run * math.tan(PITCH))
    pts = [(X0, eave[0], eave[1]), (X1, eave[0], eave[1]), (X1, 0.0, ZR), (X0, 0.0, ZR)]
    uvs = [(0, slope_len), (X1 - X0, slope_len), (X1 - X0, 0), (0, 0)]
    if sy > 0:
        pts, uvs = [pts[1], pts[0], pts[3], pts[2]], [uvs[1], uvs[0], uvs[3], uvs[2]]
    roof.face(pts, uvs, 0)
for sx in (-1, 1):                                                         # gable walls, eave walls
    x = X0 if sx < 0 else X1
    zb = ZR - run * math.tan(PITCH)
    pts = [(x, -run, zb - 25), (x, run, zb - 25), (x, run, zb), (x, 0, ZR), (x, -run, zb)]
    if sx < 0:
        pts = pts[::-1]
    roof.face(pts, [(0, 0)] * 5, 1)
roof_ob = roof.build("kids_roof", [TILES, B.principled("house_dark", (0.04, 0.035, 0.035), rough=0.9)])
mb = K.MB()
mb.cyl((X0, 0, ZR - 0.005), (X1, 0, ZR - 0.005), 0.05, n=12)                   # half-round ridge tiles
ridge_ob = mb.build("ridge", [B.principled("ridge_tile", (0.07, 0.03, 0.022), rough=0.7)], smooth=True)

CHX, CHY = float(A.get("chx", 1.12)), float(A.get("chy", -0.6))
ch = UMB()
cw, cd = 0.26, 0.2
zb, zt = ZR - abs(CHY) * math.tan(PITCH) - 0.5, ZR + float(A.get("chh", 0.95))
cs = [(CHX - cw, CHY - cd), (CHX + cw, CHY - cd), (CHX + cw, CHY + cd), (CHX - cw, CHY + cd)]
u = 0.0
for i in range(4):
    (x0, y0), (x1, y1) = cs[i], cs[(i + 1) % 4]
    L = math.hypot(x1 - x0, y1 - y0)
    ch.face([(x0, y0, zb), (x1, y1, zb), (x1, y1, zt), (x0, y0, zt)], [(u, 0), (u + L, 0), (u + L, zt - zb), (u, zt - zb)], 0)
    u += L
chim_ob = ch.build("chimney", [BRICK])
mb = K.MB()
mb.box((CHX, CHY, zt + 0.06), (cw * 2 + 0.12, cd * 2 + 0.12, 0.12), 0)
for k, dx in enumerate((-0.11, 0.12)):
    mb.cyl((CHX + dx, CHY, zt + 0.1), (CHX + dx, CHY, zt + 0.42 + 0.06 * k), 0.07, 0.062, n=16, mat=1)
    mb.cyl((CHX + dx, CHY, zt + 0.4 + 0.06 * k), (CHX + dx, CHY, zt + 0.46 + 0.06 * k), 0.082, 0.082, n=16, mat=1)
chim_top = mb.build("chimney_top", [stone_mat("cap", (0.06, 0.035, 0.03)), B.principled("pots", (0.25, 0.1, 0.05), rough=0.7)],
                    smooth=False)

# the little witch (left): cape, head, braids, pointed hat with a bent tip, one arm pointing at the lanterns
WX, WY = float(A.get("wx", -0.5)), 0.04
kid = K.MB()
cape_prof = [(ZR - 0.32, 0.23), (ZR - 0.05, 0.235), (ZR + 0.12, 0.225), (ZR + 0.27, 0.19), (ZR + 0.36, 0.165),
             (ZR + 0.41, 0.12), (ZR + 0.445, 0.065), (ZR + 0.46, 0.045)]
lathe_parts(kid, WX, WY, cape_prof, 0, nu=48, sx=1.0, sy=0.78,
            fold=lambda th, t: (1 - t) ** 2 * (0.05 * math.sin(7 * th + 0.5) + 0.03 * math.sin(13 * th)))
ellipsoid_parts(kid, (WX, WY, ZR + 0.555), (0.098, 0.092, 0.104), 1)                       # head (skin)
ellipsoid_parts(kid, (WX, WY - 0.012, ZR + 0.565), (0.106, 0.1, 0.1), 2)                   # hair over the back
for s in (-1, 1):                                                                           # braids
    p = Vector((WX + s * 0.075, WY - 0.03, ZR + 0.53))
    for j in range(5):
        q = p + Vector((s * 0.012, -0.022 + 0.004 * j, -0.055))
        kid.cyl(tuple(p), tuple(q), 0.026 - 0.002 * j, 0.024 - 0.002 * j, n=10, mat=2)
        ellipsoid_parts(kid, tuple(q), (0.03, 0.028, 0.028), 2, nu=12, nv=8)
        p = q
    ellipsoid_parts(kid, tuple(p + Vector((0, 0, -0.03))), (0.022, 0.022, 0.03), 4, nu=10, nv=6)   # ribbon tip
hb = ZR + 0.615
kid.cyl((WX, WY, hb), (WX, WY, hb + 0.012), 0.205, 0.2, n=48, mat=3)                         # brim
kid.cyl((WX, WY, hb + 0.012), (WX, WY, hb + 0.06), 0.112, 0.104, n=32, mat=4)                # hat band
p, r = Vector((WX, WY, hb + 0.012)), 0.11
lean = Vector((0.0, 0.0, 1.0))
for j in range(10):                                                                          # cone, tip bends back-left
    t = (j + 1) / 10
    lean = (lean + Vector((-0.045 * t * t, -0.07 * t * t, 0))).normalized()
    q = p + lean * 0.037
    r1 = 0.11 * (1 - t) ** 0.95 + 0.004
    kid.cyl(tuple(p), tuple(q), r, r1, n=24, mat=3, caps=(j == 9))
    p, r = q, r1
# the pointing arm: a soft witch sleeve swept along a gently bent curve, flaring to ~2.5x at the wrist, its lower
# side hanging down under gravity into a drooping cuff; a small mitten hand with one finger poking out
P0 = Vector((WX + 0.12, WY + 0.02, ZR + 0.38))
P2 = P0 + Vector((float(A.get("armx", 0.1)), float(A.get("army", 0.27)), float(A.get("armz", 0.17))))
C = P0 + Vector((0.13, 0.08, 0.03))                         # pulls the curve outward and down: the elbow


def arm_at(t):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * C + t * t * P2, (2 * (1 - t) * (C - P0) + 2 * t * (P2 - C)).normalized()


NT_, NR_ = 18, 22
vs, fs = [], []
for i in range(NT_ + 1):
    t = i / NT_
    c, T = arm_at(t)
    u = T.cross(Vector((0, 0, 1))).normalized()
    w = u.cross(T).normalized()
    r = 0.042 + 0.052 * t ** 1.6
    droop = 0.11 * t ** 2.2
    for j in range(NR_):
        th = 2 * math.pi * j / NR_
        dvec = u * math.cos(th) + w * math.sin(th)
        q = c + dvec * r * (1 + 0.06 * math.sin(3 * th + 1.0) * t)
        low = max(0.0, -dvec.z)
        q = q + Vector((0, 0, -1)) * droop * low ** 1.3 - T * 0.03 * t ** 3 * low     # the hem hangs and trails
        vs.append(tuple(q))
for i in range(NT_):
    for j in range(NR_):
        a, b = i * NR_ + j, i * NR_ + (j + 1) % NR_
        fs.append((a, b, b + NR_, a + NR_))
kid.add(vs, fs, 0)
cw_, Tw = arm_at(1.0)
hand = cw_ + Tw * 0.03 + Vector((0, 0, 0.012))
ellipsoid_parts(kid, tuple(hand), (0.031, 0.034, 0.03), 1, nu=16, nv=10)                # mitten hand
f0 = hand + Tw * 0.024 + Vector((0, 0, 0.008))
f1 = f0 + (Tw + Vector((0.0, 0.0, 0.35))).normalized() * 0.05
kid.cyl(tuple(f0), tuple(f1), 0.0095, 0.0075, n=10, mat=1)                                # the pointing finger
ellipsoid_parts(kid, tuple(f1), (0.008, 0.008, 0.008), 1, nu=10, nv=6)
witch_ob = kid.build("witch", [CAPE, SKIN, HAIR, HAT, BAND], smooth=True)

# the sheet ghost (right)
GX, GY = float(A.get("gx", 0.52)), 0.03
gh = K.MB()
ghost_prof = [(ZR - 0.34, 0.25), (ZR - 0.08, 0.262), (ZR + 0.08, 0.255), (ZR + 0.22, 0.215), (ZR + 0.34, 0.185),
              (ZR + 0.42, 0.15), (ZR + 0.47, 0.112), (ZR + 0.52, 0.112), (ZR + 0.58, 0.112), (ZR + 0.63, 0.098),
              (ZR + 0.665, 0.07), (ZR + 0.685, 0.035)]
gh_fold = lambda th, t: max(0.0, 1 - t * 1.6) ** 1.5 * (0.07 * math.sin(6 * th + 1.3) + 0.035 * math.sin(11 * th + 0.4))
lathe_parts(gh, GX, GY, ghost_prof, 0, nu=64, sx=1.0, sy=0.85, fold=gh_fold)
ghost_ob = gh.build("ghost", [SHEET], smooth=True)

# the jack-o'-lantern between them, turned three-quarters toward us
JR = 0.15
jack = K.jack("jack", (float(A.get("jx", 0.0)), -0.1, ZR + 0.015), JR, FACES[1], rz=math.radians(-28), seed=3,
              light_w=float(A.get("jackw", 7.0)), strength=float(A.get("jacks", 16.0)))
for slot in jack.material_slots:
    aov(slot.material, "actor")
# rim and fill lights that only the kids see (light linking): cool moon-blue from the front left, warm lantern
# amber from the front right, and a dim cool fill on their backs so the dark cape keeps its folds
kids = bpy.data.collections.new("kids_recv")
for ob in (witch_ob, ghost_ob, jack):
    kids.objects.link(ob)
lit_by_candle = bpy.data.collections.new("candle_recv")
for ob in (witch_ob, ghost_ob, jack, chim_ob, chim_top):
    lit_by_candle.objects.link(ob)
# the candle lights the kids and the chimney; the tiles get a softer pool from higher up, otherwise the patch
# right under the pumpkin turns into a glowing plinth
cand = bpy.data.objects["jack_candle"]
cand.light_linking.receiver_collection = lit_by_candle
tiles_c = bpy.data.collections.new("tiles_recv")
for ob in (roof_ob,):
    tiles_c.objects.link(ob)
pool = B.light("POINT", (0.0, -0.1, ZR + 0.6), float(A.get("poolw", 0.6)), color=(1.0, 0.5, 0.18), size=0.2, name="tile_pool")
pool.data.use_shadow = False                # no wedge of light between the kids' shadows
pool.light_linking.receiver_collection = tiles_c
for (name, az, el, k, col) in (("rim_cool", -42.0, 26.0, float(A.get("rimc", 3.0)), (0.7, 0.82, 1.0)),
                               ("rim_warm", 38.0, 14.0, float(A.get("rimw", 1.6)), (1.0, 0.58, 0.28))):
    a_, e_ = YAW + math.radians(az), math.radians(el)
    dv = Vector((math.sin(a_) * math.cos(e_), math.cos(a_) * math.cos(e_), math.sin(e_)))
    rl = B.light("SUN", tuple(Vector((0, 0, ZR)) + dv * 100), k, color=col, size=math.radians(2.5), target=(0, 0, ZR), name=name)
    rl.light_linking.receiver_collection = kids
fill = B.light("AREA", tuple(CAM + Vector((0.0, -1.5, 1.2))), float(A.get("fillw", 25.0)), color=(0.5, 0.62, 1.0), size=3.0,
               target=(0, 0, ZR + 0.3), name="kids_fill")
fill.light_linking.receiver_collection = kids

# ---------------------------------------------------------------- paper pumpkin lanterns
def lantern_mesh(name, nu, nv, stem=True):
    mb = K.MB()
    K.pumpkin_parts(mb, (0, 0, 0), 1.0, 0, 1, squash=0.8, ribs=8, groove=0.09, seed=1, nu=nu, nv=nv, stem=stem)
    me_ob = mb.build(name, [LAN_PLAIN, LAN_STEM], smooth=True)
    me = me_ob.data
    bpy.data.objects.remove(me_ob)
    return me


ME_HI = lantern_mesh("lan_hi", 32, 16)
ME_LO = lantern_mesh("lan_lo", 10, 6, stem=False)
lanterns = []


def add_lantern(p, r, face=None, glow=None):
    p = Vector(p)
    dist = (p - CAM).length
    me = ME_HI if dist < 150 else ME_LO
    ob = bpy.data.objects.new(f"lan{len(lanterns)}", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = p
    ob.scale = (r, r, r)
    to_cam = CAM - p
    ob.rotation_euler = (rnd.uniform(-0.12, 0.12), rnd.uniform(-0.12, 0.12), math.atan2(to_cam.x, -to_cam.y))
    ob["glow"] = glow if glow is not None else LAN_STR * rnd.uniform(0.75, 1.25)
    if face is not None:
        ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = LAN_FACE[face]
    ob.visible_shadow = False
    lanterns.append((p, r))
    return ob


# the stream: rises from the main square, drifts left and away as it climbs and spreads out
S0 = Vector((SQUARES[0][0], SQUARES[0][1], 0.0))
for i in range(int(A.get("n_stream", 520))):
    hh = 700.0 * nrng.random() ** 1.5 + 8.0
    c = S0 + Vector((-0.5 * hh, 0.25 * hh, hh))
    sp = 5.0 + 0.12 * hh
    p = c + Vector((nrng.normal(0, sp), nrng.normal(0, sp * 0.8), nrng.normal(0, sp * 0.25)))
    if p.z < 10:
        continue
    add_lantern(p, rnd.uniform(0.45, 0.9))
# lanterns just let go of all over town
for i in range(int(A.get("n_town", 180))):
    y = 140 + 2100 * nrng.random() ** 1.4
    x = CAM.x + (y - CAM.y) * nrng.uniform(-0.45, 0.45)
    z = 14 + (rnd.random() ** 2.5) * (50 + 0.12 * y)
    add_lantern((x, y, z), rnd.uniform(0.45, 0.8))
# high and far ones dotting the sky
for i in range(int(A.get("n_high", 25))):
    y = nrng.uniform(250, 2600)
    x = CAM.x + (y - CAM.y) * nrng.uniform(-0.5, 0.5)
    z = CAM.z + (y - CAM.y) * math.tan(math.radians(nrng.uniform(1.0, 22.0)))
    add_lantern((x, y, z), rnd.uniform(0.7, 1.0))
# a few close ones drifting past the roof, faces toward us; the nearest light the kids
NEAR = [(-2.6, 9.0, ZR + 3.6, 0.3, 0, 500.0), (3.3, 16.0, ZR + 7.5, 0.32, 1, 300.0), (-6.5, 26.0, ZR + 12.0, 0.34, 2, 0.0),
        (1.6, 34.0, ZR + 2.2, 0.32, 0, 0.0), (6.8, 48.0, ZR + 16.0, 0.35, 1, 0.0), (-3.2, 60.0, ZR + 24.0, 0.36, 2, 0.0),
        (-10.0, 75.0, ZR + 6.0, 0.38, 0, 0.0), (9.5, 90.0, ZR + 30.0, 0.4, 2, 0.0)]
for (x, y, z, r, f, lw) in NEAR:
    add_lantern((x, y, z), r, face=f, glow=LAN_STR * 0.85)
    if lw > 0:
        L = B.light("POINT", (x, y, z - r * 0.4), lw, color=(1.0, 0.55, 0.22), size=r * 0.6, name=f"nearlan{x}")
        L.data.use_shadow = True
print("lanterns", len(lanterns))

# ---------------------------------------------------------------- moonlight and world (lighting/reflections only)
moonlight = B.light("SUN", tuple(CAM + MDIR * 1000), float(A.get("moonlight", 1.1)), color=(0.6, 0.72, 1.0),
                    size=math.radians(1.5), target=tuple(CAM), name="moonlight")
moonlight.visible_glossy = False          # its tiny disc reflected in the river is a blinding blob; post paints a glitter path
w = sc.world
w.use_nodes = True
wt = w.node_tree
wt.nodes.clear()
tc = wt.nodes.new("ShaderNodeTexCoord")
sep = wt.nodes.new("ShaderNodeSeparateXYZ")
wt.links.new(tc.outputs["Generated"], sep.inputs[0])
ramp = wt.nodes.new("ShaderNodeValToRGB")
el_ = ramp.color_ramp.elements
el_[0].position, el_[0].color = 0.5, (0.03, 0.05, 0.08, 1)
el_[1].position, el_[1].color = 1.0, (0.004, 0.008, 0.025, 1)
mid = el_.new(0.56)
mid.color = (0.03, 0.06, 0.1, 1)
mr = wt.nodes.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -1.0, 1.0
wt.links.new(sep.outputs["Z"], mr.inputs["Value"])
wt.links.new(mr.outputs[0], ramp.inputs[0])
dot = wt.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
wt.links.new(tc.outputs["Generated"], dot.inputs[0])
dot.inputs[1].default_value = tuple(MDIR)
halo = wt.nodes.new("ShaderNodeMapRange")
halo.inputs["From Min"].default_value, halo.inputs["From Max"].default_value = 0.94, 1.0
halo.inputs["To Min"].default_value, halo.inputs["To Max"].default_value = 0.0, 0.35
wt.links.new(dot.outputs["Value"], halo.inputs["Value"])
disc = wt.nodes.new("ShaderNodeMapRange")
disc.inputs["From Min"].default_value = math.cos(MOON_R * 1.02)
disc.inputs["From Max"].default_value = math.cos(MOON_R * 0.96)
disc.inputs["To Min"].default_value, disc.inputs["To Max"].default_value = 0.0, float(A.get("wdisc", 0.0))
wt.links.new(dot.outputs["Value"], disc.inputs["Value"])
add1 = wt.nodes.new("ShaderNodeMath")
add1.operation = "ADD"
wt.links.new(halo.outputs[0], add1.inputs[0])
wt.links.new(disc.outputs[0], add1.inputs[1])
mixc = wt.nodes.new("ShaderNodeMix")
mixc.data_type = "RGBA"
mixc.blend_type = "ADD"
mixc.inputs[0].default_value = 1.0
wt.links.new(ramp.outputs[0], mixc.inputs[6])
moonc = wt.nodes.new("ShaderNodeMix")
moonc.data_type = "RGBA"
moonc.blend_type = "MULTIPLY"
moonc.inputs[0].default_value = 1.0
wt.links.new(add1.outputs[0], moonc.inputs[6])
moonc.inputs[7].default_value = (0.95, 0.8, 0.85, 1)
wt.links.new(moonc.outputs[2], mixc.inputs[7])
bg = wt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = float(A.get("sky", 1.0))
wt.links.new(mixc.outputs[2], bg.inputs[0])
wo = wt.nodes.new("ShaderNodeOutputWorld")
wt.links.new(bg.outputs[0], wo.inputs[0])

# ---------------------------------------------------------------- camera, passes, projection data for the post
tgt = CAM + Vector((math.sin(YAW) * math.cos(TILT), math.cos(YAW) * math.cos(TILT), math.sin(TILT))) * 50
cam = B.camera(tuple(CAM), tuple(tgt), lens=LENS, clip_end=20000)
B.passes(extra=("z", "normal"))
bpy.context.view_layer.update()
W_, H_ = sc.render.resolution_x, sc.render.resolution_y


def proj(p):
    v = world_to_camera_view(sc, cam, Vector(p))
    return [v.x * W_, (1 - v.y) * H_, v.z]


moon_c = proj(CAM + MDIR * 5000)
mside = Vector((math.sin(MOON_AZ + MOON_R), math.cos(MOON_AZ + MOON_R), 0)) * math.cos(MOON_EL) + Vector((0, 0, math.sin(MOON_EL)))
moon_e = proj(CAM + mside * 5000)
info = {"W": W_, "H": H_, "moon": moon_c[:2], "moon_r": abs(moon_e[0] - moon_c[0]),
        "horizon": proj(CAM + Vector((math.sin(YAW), math.cos(YAW), 0)) * 20000)[1],
        "lanterns": [proj(p) + [r] for (p, r) in lanterns], "clock": proj((CLOCK[0], CLOCK[1] - 7, 0.82 * CLOCK_H)),
        "jack": proj(jack.location), "witch_hat": proj((WX, WY, ZR + 1.0)), "ghost_top": proj((GX, GY, ZR + 0.69)),
        "chimney_top": proj((CHX - 0.11, CHY, zt + 0.46)), "lens": LENS}
os.makedirs(f"{WORK}/{TAG}", exist_ok=True)
with open(f"{WORK}/{TAG}/scene.json", "w") as fh:
    json.dump(info, fh)
print("moon px", moon_c, "r", info["moon_r"], "horizon", info["horizon"])
if "check" in A:
    nv = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == "MESH")
    print("CHECK objects", len(bpy.data.objects), "verts", nv, "lights", len(bpy.data.lights))
    sys.exit(0)
B.render(TAG, outputs=("Image", "Alpha", "Depth", "Normal", "actor", "lamp", "water"))
