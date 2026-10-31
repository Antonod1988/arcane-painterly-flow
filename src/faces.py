"""Jack-o'-lantern face masks for the Blender scenes (run with system Python: PIL is not in Blender)."""
import math
import random
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root


def face_texture(path, size=512, seed=0, kind="classic"):
    """Jack-o'-lantern face mask (white = hole), drawn with PIL, saved as PNG for an Image Texture node."""
    from PIL import Image, ImageDraw, ImageFilter
    S = size
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    rnd = random.Random(seed)
    if kind == "classic":
        ey = 0.40
        for sx in (-1, 1):
            cx = 0.5 + sx * 0.17
            d.polygon([(S * (cx - 0.09), S * (ey + 0.07)), (S * (cx + 0.09), S * (ey + 0.07)),
                       (S * (cx + sx * 0.02), S * (ey - 0.10))], fill=255)
        d.polygon([(S * 0.5, S * 0.50), (S * 0.45, S * 0.58), (S * 0.55, S * 0.58)], fill=255)
        pts = []
        n = 7
        for i in range(n + 1):
            x = 0.24 + 0.52 * i / n
            y = 0.66 + 0.06 * math.sin(math.pi * i / n)
            pts.append((S * x, S * (y - (0.05 if i % 2 else 0.0))))
        bot = [(S * (0.76 - 0.52 * i / n), S * (0.70 + 0.11 * math.sin(math.pi * i / n) + (0.04 if i % 2 else 0)))
               for i in range(n + 1)]
        d.polygon(pts + bot, fill=255)
    elif kind == "grin":
        for sx in (-1, 1):
            cx = 0.5 + sx * 0.17
            d.ellipse([S * (cx - 0.07), S * 0.30, S * (cx + 0.07), S * 0.44], fill=255)
        d.chord([S * 0.22, S * 0.42, S * 0.78, S * 0.84], 10, 170, fill=255)
        for i in range(4):
            x = 0.32 + 0.12 * i
            d.rectangle([S * x, S * 0.62, S * (x + 0.05), S * 0.70], fill=0)
    else:  # "scary"
        for sx in (-1, 1):
            cx = 0.5 + sx * 0.18
            d.polygon([(S * (cx - 0.1 * sx), S * 0.32), (S * (cx + 0.09 * sx), S * 0.40), (S * (cx - 0.06 * sx), S * 0.47)], fill=255)
        pts = [(0.2, 0.62), (0.28, 0.68), (0.33, 0.6), (0.4, 0.7), (0.46, 0.6), (0.54, 0.7), (0.6, 0.6), (0.67, 0.68),
               (0.72, 0.6), (0.8, 0.62), (0.72, 0.8), (0.66, 0.74), (0.58, 0.83), (0.5, 0.75), (0.42, 0.83), (0.34, 0.74), (0.28, 0.8)]
        d.polygon([(S * x, S * y) for x, y in pts], fill=255)
    im = im.filter(ImageFilter.GaussianBlur(S * 0.004))
    im.save(path)
    return path


if __name__ == "__main__":
    for i, k in enumerate(("classic", "grin", "scary")):
        print(face_texture(f"{ROOT}/work/face_{k}.png", seed=i, kind=k))
