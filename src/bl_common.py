"""Helpers imported INSIDE Blender scripts (Blender 4.3, run via gpu_run.py).

    import sys; sys.path.insert(0, "<repo>/src"); import bl_common as B

Renders go to work/<tag>/ as float16 .npy passes (scene-linear), which post.py turns into the final
picture (tone mapping, fog from the mist pass, bloom, grading) without re-rendering.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root


def args():
    """Script arguments after '--' as a dict: key=value pairs (values stay strings)."""
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {}
    for s in a:
        if s == "--":
            continue
        k, _, v = s.partition("=")
        out[k] = v if _ else "1"
    return out


def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    return sc


def cycles(samples=128, w=1440, h=2560, pct=100, denoise=True, bounces=6, clamp_indirect=6.0,
           adaptive=0.02, vol_step=1.0, vol_bounces=0, transparent_bounces=12):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    c = sc.cycles
    c.device = "GPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = adaptive
    c.use_denoising = denoise
    c.denoiser = "OPTIX"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.use_auto_tile = True
    c.tile_size = 1024                      # keeps VRAM moderate at 1440x2560
    c.max_bounces = bounces
    c.diffuse_bounces = min(bounces, 4)
    c.glossy_bounces = min(bounces, 4)
    c.transmission_bounces = bounces
    c.volume_bounces = vol_bounces
    c.transparent_max_bounces = transparent_bounces
    c.sample_clamp_indirect = clamp_indirect
    c.volume_step_rate = vol_step
    c.caustics_reflective = False
    c.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = pct
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    return sc


def look_at(obj, target, roll=0.0):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if roll:
        obj.rotation_euler.rotate_axis("Z", math.radians(roll))


def camera(loc, target, lens=35, sensor=36, focus=None, fstop=2.8, shift_x=0.0, shift_y=0.0, clip_end=5000):
    cam = bpy.data.cameras.new("Cam")
    cam.lens = lens
    cam.sensor_fit = "AUTO"
    cam.sensor_width = sensor
    cam.shift_x, cam.shift_y = shift_x, shift_y
    cam.clip_start, cam.clip_end = 0.05, clip_end
    if focus is not None:
        cam.dof.use_dof = True
        cam.dof.focus_distance = focus
        cam.dof.aperture_fstop = fstop
    ob = bpy.data.objects.new("Cam", cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    bpy.context.scene.camera = ob
    return ob


def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh(name, verts, faces, mat=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    me.update()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    if mat:
        ob.data.materials.append(mat)
    return link(ob)


def principled(name, color=(0.8, 0.8, 0.8), rough=0.5, metal=0.0, emission=None, strength=0.0,
               spec=0.5, transmission=0.0, ior=1.45, alpha=1.0, coat=0.0, subsurface=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Specular IOR Level"].default_value = spec
    p.inputs["Transmission Weight"].default_value = transmission
    p.inputs["IOR"].default_value = ior
    p.inputs["Alpha"].default_value = alpha
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Subsurface Weight"].default_value = subsurface
    if emission is not None:
        p.inputs["Emission Color"].default_value = (*emission, 1)
        p.inputs["Emission Strength"].default_value = strength
    return m


def emission(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def light(kind, loc, energy, color=(1, 1, 1), size=0.1, target=None, spot=None, blend=0.3, name=None):
    ld = bpy.data.lights.new(name or kind, kind)
    ld.energy = energy
    ld.color = color
    if kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = size
    elif kind == "AREA":
        ld.size = size
    elif kind == "SUN":
        ld.angle = size
    if kind == "SPOT" and spot:
        ld.spot_size = math.radians(spot)
        ld.spot_blend = blend
    ob = bpy.data.objects.new(name or kind, ld)
    link(ob)
    ob.location = loc
    if target is not None:
        look_at(ob, target)
    return ob


def passes(mist=(0.0, 100.0, "LINEAR"), extra=()):
    """Enable the mist pass (start, depth, falloff) plus extra view-layer passes, e.g. ('emit', 'ao')."""
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    vl.use_pass_mist = True
    w = sc.world
    w.mist_settings.start, w.mist_settings.depth, w.mist_settings.falloff = mist
    for e in extra:
        setattr(vl, f"use_pass_{e}", True)


def aov_mask(mat, name="actor", value=1.0):
    """Write `value` into the shader AOV `name` wherever this material is seen: an anti-aliased object mask
    (Render Layers output `name`, dumped as work/<tag>/<name>.npy when listed in render(outputs=...))."""
    vl = bpy.context.view_layer
    if name not in [a.name for a in vl.aovs]:
        a = vl.aovs.add()
        a.name = name
        a.type = "VALUE"
    nt = mat.node_tree
    n = nt.nodes.new("ShaderNodeOutputAOV")
    n.aov_name = name
    n.inputs["Value"].default_value = value
    return mat


def render_data(tag, hide=(), outputs=("Depth", "Normal", "actor"), samples=8):
    """A second, cheap render of the data passes with the volumes in `hide` switched off and no depth of field.

    Cycles writes no surface depth/normal/AOV for a camera ray that scatters in a volume, so in fog those passes
    come out speckled (depth jumps to 1e10); call this after render(tag, ("Image",)). Keeps the first preview."""
    sc = bpy.context.scene
    for ob in hide:
        ob.hide_render = True
    if sc.camera and sc.camera.data.dof.use_dof:
        sc.camera.data.dof.use_dof = False
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    keep = f"{ROOT}/work/{tag}/preview_agx.png"
    if os.path.exists(keep):
        os.replace(keep, keep + ".keep")
    render(tag, outputs)
    if os.path.exists(keep + ".keep"):
        os.replace(keep + ".keep", keep)


def render(tag, outputs=("Image", "Mist")):
    """Render the scene camera and dump the chosen Render Layers outputs to work/<tag>/<name>.npy (float16)."""
    sc = bpy.context.scene
    out = f"{ROOT}/work/{tag}"
    os.makedirs(out, exist_ok=True)
    sc.use_nodes = True
    tree = sc.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = out
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_slots.clear()
    single = {name for name in outputs if rl.outputs[name].type == "VALUE"}
    for name in outputs:
        fo.file_slots.new(name.lower() + "_")
        tree.links.new(rl.outputs[name], fo.inputs[-1])
    sc.frame_set(1)
    sc.render.filepath = f"{out}/preview_agx.png"
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    for name in outputs:
        p = f"{out}/{name.lower()}_0001.exr"
        img = bpy.data.images.load(p)
        w, h = img.size
        buf = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(buf)
        a = buf.reshape(h, w, 4)[::-1]
        a = a[..., 0] if name in ("Mist", "Depth", "AO") or name in single else a[..., :3]
        np.save(f"{out}/{name.lower()}.npy", a.astype(np.float16))
        bpy.data.images.remove(img)
        os.remove(p)
        print("dumped", name, a.shape)
