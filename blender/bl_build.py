"""Procedural geometry and shader-node helpers for the main session's Cycles scenes (01-04).

Imported inside Blender after bl_common:  sys.path.insert(0, "<repo>/blender"); import bl_build as K
"""
import math
import random

import bpy
import numpy as np
from mathutils import Vector

import bl_common as B


# ---------------------------------------------------------------- mesh accumulation

class MB:
    """Accumulates many simple parts into one mesh object (with per-face material slots)."""

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def add(self, verts, faces, mat=0):
        o = len(self.v)
        self.v.extend(verts)
        self.f.extend([tuple(i + o for i in fc) for fc in faces])
        self.m.extend([mat] * len(faces))

    def box(self, c, s, mat=0, rz=0.0):
        """Axis box centred at c with full size s, optionally rotated about Z by rz radians."""
        cx, cy, cz = c
        hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
        cs, sn = math.cos(rz), math.sin(rz)
        vs = []
        for z in (-hz, hz):
            for (x, y) in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                vs.append((cx + x * cs - y * sn, cy + x * sn + y * cs, cz + z))
        fs = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self.add(vs, fs, mat)

    def boxr(self, x0, x1, y0, y1, z0, z1, mat=0):
        """Axis box from ranges."""
        self.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), mat)

    def cyl(self, p0, p1, r0, r1=None, n=8, mat=0, caps=True):
        """Tapered cylinder between two points."""
        r1 = r0 if r1 is None else r1
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        if d.length < 1e-6:
            return
        a = d.normalized()
        t = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
        u = a.cross(t).normalized()
        w = a.cross(u)
        vs = []
        for p, r in ((p0, r0), (p1, r1)):
            for i in range(n):
                th = 2 * math.pi * i / n
                vs.append(tuple(p + (u * math.cos(th) + w * math.sin(th)) * r))
        fs = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        if caps:
            fs.append(tuple(range(n - 1, -1, -1)))
            fs.append(tuple(range(n, 2 * n)))
        self.add(vs, fs, mat)

    def prism(self, poly, z0, z1, mat=0):
        """Extrude a convex-or-not polygon [(x, y), ...] (counter-clockwise) between z0 and z1."""
        n = len(poly)
        vs = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
        fs = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        fs += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        self.add(vs, fs, mat)

    def build(self, name, mats, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.validate()
        for m in mats:
            me.materials.append(m)
        if len(mats) > 1:
            me.polygons.foreach_set("material_index", self.m)
        if smooth:
            me.shade_smooth()
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


# ---------------------------------------------------------------- shader node helper

class NT:
    """Tiny wrapper around a material node tree."""

    def __init__(self, mat):
        self.mat = mat
        mat.use_nodes = True
        self.t = mat.node_tree
        self.t.nodes.clear()

    def n(self, typ, **inputs):
        node = self.t.nodes.new(typ)
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(node, k[1:], v)
            else:
                self.set(node.inputs[k], v)
        return node

    def set(self, sock, v):
        if hasattr(v, "is_output") or isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        elif isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
            sock.default_value = (*v, 1)
        else:
            sock.default_value = v

    def link(self, a, b):
        self.t.links.new(a, b)

    def math(self, op, a, b=0.0, clamp=False):
        node = self.t.nodes.new("ShaderNodeMath")
        node.operation = op
        node.use_clamp = clamp
        self.set(node.inputs[0], a)
        self.set(node.inputs[1], b)
        return node.outputs[0]

    def mixc(self, fac, a, b):
        node = self.t.nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        self.set(node.inputs[0], fac)
        self.set(node.inputs[6], a)
        self.set(node.inputs[7], b)
        return node.outputs[2]

    def mixf(self, fac, a, b):
        node = self.t.nodes.new("ShaderNodeMix")
        node.data_type = "FLOAT"
        self.set(node.inputs[0], fac)
        self.set(node.inputs[2], a)
        self.set(node.inputs[3], b)
        return node.outputs[0]

    def mapr(self, x, a, b, c=0.0, d=1.0, clamp=True):
        node = self.t.nodes.new("ShaderNodeMapRange")
        node.clamp = clamp
        self.set(node.inputs["Value"], x)
        node.inputs["From Min"].default_value = a
        node.inputs["From Max"].default_value = b
        node.inputs["To Min"].default_value = c
        node.inputs["To Max"].default_value = d
        return node.outputs[0]

    def ramp(self, x, stops):
        """ColorRamp from [(pos, (r, g, b)), ...] -> colour output."""
        node = self.t.nodes.new("ShaderNodeValToRGB")
        self.set(node.inputs[0], x)
        el = node.color_ramp.elements
        el[0].position, el[0].color = stops[0][0], (*stops[0][1], 1)
        el[1].position, el[1].color = stops[-1][0], (*stops[-1][1], 1)
        for p, c in stops[1:-1]:
            e = el.new(p)
            e.color = (*c, 1)
        return node.outputs[0]

    def out(self, shader, volume=None, displacement=None):
        o = self.t.nodes.new("ShaderNodeOutputMaterial")
        self.link(shader, o.inputs["Surface"])
        if volume is not None:
            self.link(volume, o.inputs["Volume"])
        if displacement is not None:
            self.link(displacement, o.inputs["Displacement"])
        return o

    def bsdf(self, color=(0.8, 0.8, 0.8), rough=0.5, metal=0.0, normal=None, spec=0.5, emission=None,
             strength=0.0, subsurface=0.0, ss_radius=(0.1, 0.05, 0.02), coat=0.0, alpha=1.0):
        p = self.n("ShaderNodeBsdfPrincipled")
        self.set(p.inputs["Base Color"], color)
        self.set(p.inputs["Roughness"], rough)
        self.set(p.inputs["Metallic"], metal)
        self.set(p.inputs["Specular IOR Level"], spec)
        self.set(p.inputs["Coat Weight"], coat)
        self.set(p.inputs["Alpha"], alpha)
        if subsurface:
            self.set(p.inputs["Subsurface Weight"], subsurface)
            p.inputs["Subsurface Radius"].default_value = ss_radius
        if normal is not None:
            self.link(normal, p.inputs["Normal"])
        if emission is not None:
            self.set(p.inputs["Emission Color"], emission)
            self.set(p.inputs["Emission Strength"], strength)
        return p.outputs[0]

    def coords(self, kind="Object"):
        return self.n("ShaderNodeTexCoord").outputs[kind]

    def bump(self, height, strength=0.5, distance=0.02, normal=None):
        b = self.n("ShaderNodeBump", Strength=strength, Distance=distance)
        self.set(b.inputs["Height"], height)
        if normal is not None:
            self.link(normal, b.inputs["Normal"])
        return b.outputs[0]


def mat(name):
    m = bpy.data.materials.new(name)
    return m, NT(m)


# ---------------------------------------------------------------- organic parts

def pumpkin_parts(mb, c, r, mat_skin, mat_stem, squash=0.72, ribs=10, groove=0.07, rz=0.0, seed=0,
                  nu=48, nv=24, stem=True):
    """Ribbed pumpkin centred at c (bottom resting at c.z - r*squash), radius r."""
    rnd = random.Random(seed)
    cx, cy, cz = c
    vs, fs = [], []
    lean = (rnd.uniform(-0.05, 0.05), rnd.uniform(-0.05, 0.05))
    for j in range(nv + 1):
        phi = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu + rz
            rib = 1 - groove * (1 - abs(math.cos(ribs * (th - rz) / 2)) ** 0.6)
            rad = r * math.sin(phi) * rib * (1 + 0.04 * math.sin(3 * th + seed))
            z = r * squash * math.cos(phi)
            z -= r * 0.22 * math.exp(-(phi / 0.32) ** 2)      # dimple where the stem sits
            z += r * 0.12 * math.exp(-((math.pi - phi) / 0.35) ** 2)  # flat-ish base
            vs.append((cx + rad * math.cos(th) + lean[0] * z, cy + rad * math.sin(th) + lean[1] * z, cz + z))
    for j in range(nv):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            fs.append((a, b, b + nu, a + nu))
    mb.add(vs, fs, mat_skin)
    if stem:
        top = Vector((cx + lean[0] * r * 0.5, cy + lean[1] * r * 0.5, cz + r * squash - r * 0.22))
        bend = Vector((rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3), 1)).normalized()
        p1 = top + bend * r * 0.32
        mb.cyl(top, p1, r * 0.09, r * 0.06, n=7, mat=mat_stem)
        p2 = p1 + Vector((bend.x * 1.6, bend.y * 1.6, 0.4)).normalized() * r * 0.12
        mb.cyl(p1, p2, r * 0.06, r * 0.045, n=7, mat=mat_stem)


def tree_parts(mb, base, height, seed, mat=0, depth=7, spread=0.55, trunk_r=None, lean=(0.0, 0.0), n=6,
               up_bias=0.35, twig=True, tips=None):
    """Bare gnarled tree: a leaning trunk that forks into limbs, pipe-model radii, bent segments, fine twigs."""
    rnd = random.Random(seed)
    trunk_r = trunk_r or height * 0.024

    def seg_chain(p, d, length, r0, r1, nseg, sides, wob):
        q, dd = Vector(p), Vector(d)
        for s in range(nseg):
            dd = (dd + Vector((rnd.gauss(0, wob), rnd.gauss(0, wob), rnd.gauss(0, wob * 0.5)))).normalized()
            nq = q + dd * (length / nseg)
            ra = r0 + (r1 - r0) * s / nseg
            rb = r0 + (r1 - r0) * (s + 1) / nseg
            mb.cyl(q, nq, ra, rb, n=sides, mat=mat, caps=False)
            q = nq
        return q, dd

    def branch(p, d, length, r, lvl):
        sides = max(n, 10) if lvl < 2 else (7 if lvl < 4 else (5 if lvl < 6 else 3))
        r_end = r * 0.72
        q, dd = seg_chain(p, d, length, r, r_end, 5 if lvl < 3 else (3 if lvl < 5 else 2), sides,
                          0.1 if lvl < 2 else 0.2)
        if lvl >= depth or r_end < 0.0035:
            if tips is not None:
                tips.append((Vector(q), Vector(dd), length))
            if twig:
                for _ in range(rnd.randint(2, 4)):
                    td = (dd + Vector((rnd.gauss(0, 0.7), rnd.gauss(0, 0.7), rnd.gauss(0.2, 0.5)))).normalized()
                    seg_chain(q, td, length * rnd.uniform(0.25, 0.5), r_end * 0.6, r_end * 0.2, 2, 3, 0.3)
            return
        k = rnd.choice((2, 2, 2, 3))
        for i in range(k):
            ang = spread * rnd.uniform(0.5, 1.25)
            az = 2 * math.pi * (i / k) + rnd.uniform(-0.6, 0.6)
            t = Vector((0, 0, 1)) if abs(dd.z) < 0.95 else Vector((1, 0, 0))
            u = dd.cross(t).normalized()
            w = dd.cross(u)
            side = (u * math.cos(az) + w * math.sin(az))
            nd = (dd * math.cos(ang) + side * math.sin(ang)).normalized()
            droop = -0.12 * lvl / depth
            nd = (nd + Vector((0, 0, up_bias * (1 - lvl / (depth + 1)) + droop))).normalized()
            rr = r_end * (1.0 / k) ** (1 / 2.3) * rnd.uniform(0.9, 1.05)
            branch(q, nd, length * rnd.uniform(0.66, 0.82), rr, lvl + 1)

    d0 = Vector((lean[0], lean[1], 1)).normalized()
    # trunk, then 3-4 main limbs that spread into a broad crown
    q, dd = seg_chain(Vector(base), d0, height * 0.28, trunk_r, trunk_r * 0.82, 8, max(n, 12), 0.04)
    k = rnd.choice((3, 3, 4))
    for i in range(k):
        az = 2 * math.pi * i / k + rnd.uniform(-0.5, 0.5)
        ang = rnd.uniform(0.45, 0.8)
        t = Vector((1, 0, 0)) if abs(dd.z) > 0.95 else Vector((0, 0, 1))
        u = dd.cross(t).normalized()
        w = dd.cross(u)
        nd = (dd * math.cos(ang) + (u * math.cos(az) + w * math.sin(az)) * math.sin(ang)).normalized()
        branch(q, nd, height * rnd.uniform(0.3, 0.4), trunk_r * 0.82 * (1.0 / k) ** (1 / 2.3), 2)


def face_texture(path, size=512, seed=0, kind="classic"):
    """Face masks are drawn by src/faces.py with system Python (Blender has no PIL); this just checks the file."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path + " - run: python src/faces.py")
    return path


def pumpkin_material(name, face_png=None, glow=(1.0, 0.45, 0.08), strength=30.0, skin=(0.55, 0.16, 0.02)):
    """Orange skin with subsurface; with face_png, holes facing local -Y glow (decal projected in object space).
    The face is centred on the object origin: pass objects whose origin is the pumpkin centre."""
    m, t = mat(name)
    tc = t.n("ShaderNodeTexCoord")
    noise = t.n("ShaderNodeTexNoise", Scale=18.0, Detail=6.0)
    t.link(tc.outputs["Object"], noise.inputs["Vector"])
    col = t.mixc(t.mapr(noise.outputs["Fac"], 0.35, 0.7), skin, (skin[0] * 1.25, skin[1] * 1.3, skin[2]))
    rough = t.mapr(noise.outputs["Fac"], 0.3, 0.7, 0.35, 0.55)
    nb = t.bump(noise.outputs["Fac"], 0.15, 0.01)
    skin_sh = t.bsdf(col, rough, normal=nb, subsurface=0.25, ss_radius=(0.08, 0.03, 0.01), spec=0.4)
    if not face_png:
        t.out(skin_sh)
        return m
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(tc.outputs["Object"], sep.inputs[0])
    comb = t.n("ShaderNodeCombineXYZ")
    # object x in [-r, r] -> u in [0, 1]; object z -> v. The face texture is authored for r = 1 units,
    # so faces are scaled through an object-level 'face_scale' custom property via an Attribute node.
    attr = t.n("ShaderNodeAttribute", _attribute_type="OBJECT", _attribute_name="face_scale")
    inv = t.math("DIVIDE", 0.5, attr.outputs["Fac"])
    u = t.math("ADD", t.math("MULTIPLY", sep.outputs["X"], inv), 0.5)
    v = t.math("ADD", t.math("MULTIPLY", sep.outputs["Z"], inv), 0.5)
    t.link(u, comb.inputs["X"])
    t.link(v, comb.inputs["Y"])
    img = t.n("ShaderNodeTexImage", _extension="CLIP")
    img.image = bpy.data.images.load(face_png, check_existing=True)
    img.image.colorspace_settings.name = "Non-Color"
    t.link(comb.outputs[0], img.inputs["Vector"])
    yn = t.math("MULTIPLY", t.math("MULTIPLY", sep.outputs["Y"], inv), 2.0)   # y / r
    front = t.mapr(yn, -0.25, -0.6, 0.0, 1.0)                                  # only the side facing local -Y
    hole = t.math("MULTIPLY", img.outputs["Color"], front, clamp=True)
    emit = t.n("ShaderNodeEmission", Color=glow, Strength=strength)
    sh = t.n("ShaderNodeMixShader")
    t.link(hole, sh.inputs[0])
    t.link(skin_sh, sh.inputs[1])
    t.link(emit.outputs[0], sh.inputs[2])
    t.out(sh.outputs[0])
    return m


_CACHE = {}


def jack(name, c, r, face_png, rz=0.0, seed=0, light_w=8.0, glow=(1.0, 0.42, 0.08), strength=30.0,
         stem_mat=None, squash=0.72, candle_color=(1.0, 0.55, 0.2)):
    """A carved, lit jack-o'-lantern object (origin at its centre, facing local -Y, rotated by rz)."""
    mb = MB()
    pumpkin_parts(mb, (0, 0, 0), r, 0, 1, squash=squash, seed=seed)
    key = (face_png, tuple(glow), strength)
    if key not in _CACHE:
        _CACHE[key] = pumpkin_material("jack_skin", face_png, glow=glow, strength=strength)
    m = _CACHE[key]
    if "stem" not in _CACHE:
        _CACHE["stem"] = B.principled("stem", (0.12, 0.09, 0.03), rough=0.8)
    sm = stem_mat or _CACHE["stem"]
    ob = mb.build(name, [m, sm], smooth=True)
    ob["face_scale"] = r
    ob.location = c
    ob.rotation_euler = (0, 0, rz)
    ob.visible_shadow = False           # let the candle inside light the ground through the shell
    if light_w > 0:
        L = B.light("POINT", (c[0], c[1], c[2] - r * 0.1), light_w, color=candle_color, size=r * 0.25,
                    name=name + "_candle")
        L.data.use_shadow = True
    return ob
