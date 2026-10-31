"""Contact sheet of the Arcane-style paintings (20 plus 29-33) -> captioned/00-arcane-sheet.png (+ .jpg)."""
import glob
import os

from PIL import Image, ImageDraw

import hp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
LABELS = {20: "STAIR ALLEY", 29: "PUMPKIN FERRY", 30: "CARVER'S WORKSHOP", 31: "LANTERNS OVER THE CITY",
          32: "PUMPKIN LIGHTHOUSE", 33: "OVERGROWN GLASSHOUSE"}

files = [f for f in sorted(glob.glob(f"{ROOT}/out/*.png")) if int(os.path.basename(f)[:2] or 0) in LABELS]
n = len(files)
cols = 3 if n > 4 else n
rows = (n + cols - 1) // cols
tw, th = 460, 818
gap, top, lab = 26, 270, 56
W = cols * tw + (cols + 1) * gap
H = top + rows * (th + lab + gap) + gap
sheet = Image.new("RGB", (W, H), (9, 9, 12))
d = ImageDraw.Draw(sheet)
ft = hp.font(50, True)
fl = hp.font(19, True)
title = f"HALLOWEEN 2026  ·  ARCANE-STYLE PAINTINGS"
d.text(((W - d.textlength(title, font=ft)) / 2, 160), title, font=ft, fill=(235, 225, 210))
for i, f in enumerate(files):
    num = int(os.path.basename(f)[:2])
    r, c = divmod(i, cols)
    x = gap + c * (tw + gap)
    y = top + r * (th + lab + gap)
    sheet.paste(Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
    d.text((x + 2, y + th + 13), f"{num:02d}  {LABELS[num]}", font=fl, fill=(232, 150, 60))
sheet = hp.captioned(sheet)
sheet.save(f"{ROOT}/captioned/00-arcane-sheet.png", optimize=True)
sheet.save(f"{ROOT}/captioned/00-arcane-sheet.jpg", quality=92)
print("sheet", sheet.size, n, "pictures")
