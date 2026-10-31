"""Two extra jack-o'-lantern face masks for picture 30 (system Python; Blender has no PIL).
work/30-face-half.png: a half-carved face (eyes and nose cut, the mouth only scratched in outline);
work/30-face-sly.png : slanted crescent eyes and a crooked one-tooth grin."""
from PIL import Image, ImageDraw, ImageFilter
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root

S = 512


def half(path):
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    for sx in (-1, 1):
        cx = 0.5 + sx * 0.17
        d.polygon([(S * (cx - 0.08), S * 0.47), (S * (cx + 0.08), S * 0.47), (S * (cx + sx * 0.03), S * 0.31)], fill=255)
    d.polygon([(S * 0.5, S * 0.51), (S * 0.455, S * 0.58), (S * 0.545, S * 0.58)], fill=255)
    pts = [(0.26, 0.66), (0.36, 0.71), (0.5, 0.73), (0.64, 0.71), (0.74, 0.66), (0.66, 0.79), (0.5, 0.83),
           (0.34, 0.79)]
    d.line([(S * x, S * y) for x, y in pts + pts[:1]], fill=150, width=5)      # knife scratch, not cut through
    im.filter(ImageFilter.GaussianBlur(S * 0.004)).save(path)


def sly(path):
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    for sx in (-1, 1):
        cx = 0.5 + sx * 0.18
        box = [S * (cx - 0.1), S * 0.30, S * (cx + 0.1), S * 0.48]
        d.chord(box, 200 if sx < 0 else 200, 340, fill=255)           # upper crescents, tilted by the cut below
        d.chord([S * (cx - 0.1), S * 0.36, S * (cx + 0.1), S * 0.52], 200, 340, fill=0)
    pts = [(0.2, 0.58), (0.35, 0.68), (0.5, 0.70), (0.68, 0.66), (0.82, 0.54), (0.74, 0.74), (0.6, 0.83),
           (0.44, 0.84), (0.3, 0.78)]
    d.polygon([(S * x, S * y) for x, y in pts], fill=255)
    d.rectangle([S * 0.52, S * 0.69, S * 0.58, S * 0.76], fill=0)                  # the one tooth
    im.filter(ImageFilter.GaussianBlur(S * 0.004)).save(path)


if __name__ == "__main__":
    half(ROOT + "/work/30-face-half.png")
    sly(ROOT + "/work/30-face-sly.png")
    a = Image.open(ROOT + "/work/30-face-half.png")
    b = Image.open(ROOT + "/work/30-face-sly.png")
    sheet = Image.new("L", (2 * S, S))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (S, 0))
    sheet.resize((512, 256)).save(ROOT + "/work/30-faces.png")
    print("faces written")
