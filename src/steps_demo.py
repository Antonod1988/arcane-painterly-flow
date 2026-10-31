"""Show the painter's flow on one rendered scene, stage by stage, for the README.

    python src/steps_demo.py <tag>      # reads work/<tag>/*.npy from bl_common.render / render_data

Writes gallery/steps/<tag>-passes.jpg (what Blender hands over) and gallery/steps/<tag>-steps.jpg
(render -> brush strokes -> ink -> colour design), plus each stage as its own JPG.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import arcane as A
import hp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
OUT = ROOT + "/gallery/steps"
TW, TH = 360, 640


def panel(img, label):
    im = hp.to_pil(np.clip(img, 0, 1)).resize((TW, TH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = hp.font(18, True)
    d.rectangle((0, TH - 34, TW, TH), fill=(0, 0, 0))
    d.text((10, TH - 30), label, font=f, fill=(240, 220, 190))
    return im


def strip(panels, path):
    sheet = Image.new("RGB", (len(panels) * TW + (len(panels) - 1) * 8, TH), (12, 12, 14))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (TW + 8), 0))
    sheet.save(path, quality=90)
    print("wrote", path)


tag = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
P = A.load(tag)
depth, actor = P["depth"], P.get("actor")

# what Blender hands over: the linear render plus clean data passes
logd = np.log(np.clip(depth, 0.5, 300))
dv = 1 - (logd - logd.min()) / (logd.max() - logd.min())
nv = P["normal"] * 0.5 + 0.5
strip([panel(np.repeat(dv[..., None], 3, 2), "depth"), panel(nv, "normals"),
       panel(np.repeat(actor[..., None], 3, 2) if actor is not None else nv * 0, "actor mask (AOV)")],
      f"{OUT}/{tag}-passes.jpg")

stages = []
img = A.tonemap(P["image"], exposure=1.0)
stages.append((img, "1  render + bloom + filmic"))
with A.paint_slot():
    img = A.paint(img, depth=depth, detail=actor, seed=29)
stages.append((img, "2  brush strokes"))
outline = np.clip((actor - 0.65) * 4, 0, 1) if actor is not None else None
img = A.ink(img, depth, P.get("normal"), near=2.0, far=24.0, keep_out=actor, outline=outline, amount=0.55)
stages.append((img, "3  ink from depth / normals"))
img = A.finish(img)
stages.append((img, "4  colour design + canvas"))
for k, (im, label) in enumerate(stages, 1):
    hp.to_pil(np.clip(im, 0, 1)).resize((720, 1280), Image.LANCZOS).save(f"{OUT}/{tag}-{k}.jpg", quality=88)
strip([panel(im, label) for im, label in stages], f"{OUT}/{tag}-steps.jpg")
