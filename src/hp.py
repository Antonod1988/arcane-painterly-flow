"""Shared helpers for the Halloween picture series (1440x2560 portrait stills).

Every picture script builds a float RGB image in [0, 1] (numpy, shape H x W x 3) or a PIL image
and calls save(img, num, slug). That writes:
    out/NN-slug.png        the clean picture
    captioned/NN-slug.png  the same picture with the series credit at the top
    work/NN-slug-small.jpg a 720x1280 preview for quick looks
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")   # repository root
W, H = 1440, 2560


# ---------------------------------------------------------------- series credit

def font(size, bold=False):
    """Segoe UI as in the series credit; elsewhere DejaVu Sans, else Pillow's built-in font."""
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        for folder in ("C:/Windows/Fonts/", ""):
            try:
                return ImageFont.truetype(folder + name, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


def caption_overlay(w=W, h=H):
    """The series credit as an RGBA overlay: letter-spaced caps, white at 75%, soft shadow.
    Font size follows min(h/1080, w/1100), so in portrait it scales with the image width."""
    size = round(22 * min(h / 1080, w / 1100))
    fonts = {"r": font(size), "b": font(size, True)}
    parts = [("r", "MADE BY "), ("b", "CLAUDE OPUS 5.5"), ("r", " (MAX REASONING)")]
    track = round(size * 0.3)
    widths = [sum(fonts[f].getlength(ch) + track for ch in s) for f, s in parts]
    x = (w - sum(widths)) / 2
    top = round(30 * h / 1080)
    text = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(text)
    for (f, s), wd in zip(parts, widths):
        for ch in s:
            d.text((x, top), ch, font=fonts[f], fill=255)
            x += fonts[f].getlength(ch) + track
    shadow = text.filter(ImageFilter.GaussianBlur(round(5 * h / 1080)))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (w, h), (0, 0, 0, 255)), (0, 0), shadow.point(lambda v: int(v * 0.5)))
    img.paste(Image.new("RGBA", (w, h), (255, 255, 255, 255)), (0, 0), text.point(lambda v: int(v * 0.75)))
    return img


def captioned(img):
    """PIL RGB image -> copy with the credit composited on top (straight alpha, like ffmpeg's overlay)."""
    base = img.convert("RGBA")
    return Image.alpha_composite(base, caption_overlay(*base.size)).convert("RGB")


# ---------------------------------------------------------------- output

def to_pil(img, dither=True, seed=0):
    """float RGB [0,1] array (or PIL image) -> PIL RGB. A half-LSB of noise hides banding in gradients."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3] * 255.0
    if dither:
        a = a + (np.random.default_rng(seed).random(a.shape[:2], dtype=np.float32)[..., None] - 0.5)
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGB")


def save(img, num, slug):
    """Write the clean PNG, the captioned PNG and a small preview JPG. Returns the three paths."""
    im = to_pil(img)
    assert im.size == (W, H), f"picture must be {W}x{H}, got {im.size}"
    name = f"{num:02d}-{slug}"
    paths = (f"{ROOT}/out/{name}.png", f"{ROOT}/captioned/{name}.png", f"{ROOT}/work/{name}-small.jpg")
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    im.save(paths[0], optimize=True)
    cap = captioned(im)
    cap.save(paths[1], optimize=True)
    cap.resize((720, 1280), Image.LANCZOS).save(paths[2], quality=90)
    for p in paths[:2]:
        mb = os.path.getsize(p) / 2**20
        print(f"wrote {p}  {mb:.1f} MiB")
        if mb > 29:   # delivery limit is 30 MiB; send a JPG instead
            jp = p[:-4] + ".jpg"
            Image.open(p).save(jp, quality=92)
            print(f"  too large to send, also wrote {jp}")
    return paths


# ---------------------------------------------------------------- small numeric toolkit

def grid(w=W, h=H):
    """Pixel-centre coordinates: x in [0, w), y in [0, h), each float32 (h, w)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x + 0.5, y + 0.5


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    """Lerp that broadcasts a (h,w) weight over colour channels."""
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1.0 - t) + b * t


def rgb(hexstr):
    """'#ff8800' -> float32 array [1, 0.53, 0]."""
    h = hexstr.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def vgrad(stops, h=H, w=W):
    """Vertical gradient from [(pos 0..1, '#hex'), ...] -> (h, w, 3)."""
    pos = np.array([p for p, _ in stops], dtype=np.float32)
    cols = np.stack([rgb(c) for _, c in stops])
    t = (np.arange(h, dtype=np.float32) + 0.5) / h
    col = np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], axis=1)
    return np.repeat(col[:, None, :], w, axis=1).astype(np.float32)


def noise(shape, cell, seed=0, octaves=1, persistence=0.5):
    """Smooth value noise in [0,1] at the given cell size (pixels); octaves > 1 makes fBm."""
    h, w = shape
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = int(h / cell) + 3, int(w / cell) + 3
        g = rng.random((gh, gw), dtype=np.float32)
        big = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
        oy, ox = rng.integers(0, max(1, int(cell))), rng.integers(0, max(1, int(cell)))
        total += amp * big[oy:oy + h, ox:ox + w]
        norm += amp
        amp *= persistence
        cell = max(1.0, cell / 2.0)
    return np.clip(total / norm, 0.0, 1.0)


def blur(a, sigma):
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


def glow(img, threshold=0.7, sigmas=(6, 24, 80), gains=(0.6, 0.4, 0.3)):
    """Additive bloom from everything brighter than threshold."""
    lum = img.max(axis=2, keepdims=True)
    hot = img * np.clip((lum - threshold) / max(1e-3, 1 - threshold), 0, 1)
    out = img.copy()
    for s, g in zip(sigmas, gains):
        out += g * blur(hot, s)
    return out


def vignette(img, strength=0.35, power=2.2):
    x, y = grid(img.shape[1], img.shape[0])
    dx = (x / img.shape[1] - 0.5) * 2
    dy = (y / img.shape[0] - 0.5) * 2
    r = np.sqrt(dx * dx * 0.8 + dy * dy * 0.6)
    return img * (1.0 - strength * np.clip(r, 0, 1.5) ** power)[..., None]


def grain(img, amount=0.03, seed=1, cell=1.0):
    n = np.random.default_rng(seed).standard_normal(img.shape[:2]).astype(np.float32)
    if cell > 1:
        n = blur(n, cell * 0.5) * cell
    return img + amount * n[..., None]


def poly_mask(polys, w=W, h=H, ss=4):
    """Anti-aliased coverage mask (float32 h x w) of one or more polygons [[(x, y), ...], ...] in pixels."""
    shift = 4
    m = np.zeros((h * ss, w * ss), np.uint8)
    pts = [np.round(np.asarray(p, np.float64) * ss * (1 << shift)).astype(np.int32) for p in polys]
    cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_8, shift=shift)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def over(dst, color, mask):
    """Paint a flat colour (or an image) through a coverage mask."""
    return mix(dst, np.broadcast_to(color, dst.shape), mask)
