#!/usr/bin/env python3
"""
sciart.py - tiny procedural 2D art library for science videos (Pillow + NumPy only)

Everything is drawn in code: no downloads, no licences, no network.
Objects:  draw_earth, draw_moon, draw_rocket, draw_satellite, draw_atom, draw_dna,
          draw_trophy, draw_tortoise, draw_hare
Text:     get_font, draw_text, draw_caption, draw_bubble
CLI:      run_cli (shared --still / --start / --end / --frames-dir runner)
Helpers:  vgrad, draw_stars, paste_center, scaled, rocket_layer, easing (clamp/lerp/smooth)

  python sciart.py --preview out/library.png     # contact sheet of every object
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1080, 1920


# ----------------------------------------------------------------- utilities
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def vgrad(top, bot, h=H, w=W):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t
    arr = np.broadcast_to(arr, (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def paste_center(img, layer, x, y):
    img.paste(layer, (int(x - layer.width / 2), int(y - layer.height / 2)), layer)


def scaled(sprite, k):
    return sprite.resize((max(1, int(sprite.width * k)), max(1, int(sprite.height * k))), Image.BILINEAR)


# --------------------------------------------------------------------- stars
_rs = random.Random(7)
STARS = [(_rs.random() * W, _rs.random() * H, _rs.choice([1, 1, 1, 2, 2, 3]), _rs.random() * 6.28)
         for _ in range(260)]


def draw_stars(img, alpha, t, dx=0.0, dy=0.0, ymax=H):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img, "RGBA")
    for x, y, r, ph in STARS:
        sx, sy = (x + dx) % W, (y + dy) % H
        if sy > ymax:
            continue
        a = int(255 * alpha * (0.65 + 0.35 * math.sin(t * 2.2 + ph)))
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 255, 255, a))


# ------------------------------------------------------------------- sprites
def shade_sphere(base, n, R, light, floor, tint=None):
    """Lambert-shade a flat texture onto a sphere, return RGBA sprite."""
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    nx, ny = (xx - c) / R, (yy - c) / R
    r2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    arr = np.asarray(base.convert("RGB"), np.float32)
    arr = arr * (floor + (1.05 - floor) * lam)[..., None]
    if tint is not None:
        rim = (np.clip((r2 - 0.80) / 0.20, 0, 1) ** 2 * 0.65)[..., None]
        glow = np.array(tint, np.float32) * np.clip(lam * 1.2 + 0.25, 0, 1)[..., None]
        arr = arr * (1 - rim) + glow * rim
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    alpha = (np.clip((1.0 - np.sqrt(r2)) * R, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB").convert("RGBA")
    im.putalpha(Image.fromarray(alpha, "L"))
    return im


def make_earth():
    n, R = 880, 380
    c = n // 2
    rng = random.Random(3)
    base = Image.new("RGB", (n, n), (22, 78, 170))
    d = ImageDraw.Draw(base)
    for _ in range(14):
        cx_, cy_ = rng.randint(c - 300, c + 300), rng.randint(c - 300, c + 300)
        for _k in range(5):
            ox, oy = rng.randint(-60, 60), rng.randint(-50, 50)
            rw, rh = rng.randint(40, 120), rng.randint(30, 90)
            col = rng.choice([(58, 134, 66), (74, 148, 72), (150, 138, 84)])
            d.ellipse([cx_ + ox - rw, cy_ + oy - rh, cx_ + ox + rw, cy_ + oy + rh], fill=col)
    clouds = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    cd = ImageDraw.Draw(clouds)
    for _ in range(40):
        x, y = rng.randint(c - 380, c + 380), rng.randint(c - 380, c + 380)
        w, h = rng.randint(60, 200), rng.randint(10, 30)
        cd.ellipse([x - w, y - h, x + w, y + h], fill=(255, 255, 255, rng.randint(90, 170)))
    clouds = clouds.filter(ImageFilter.GaussianBlur(6))
    base = base.convert("RGBA")
    base.alpha_composite(clouds)
    earth = shade_sphere(base, n, R, (0.55, -0.45, 0.70), 0.14, tint=(120, 190, 255))
    glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([c - R - 10, c - R - 10, c + R + 10, c + R + 10], fill=(90, 165, 255, 120))
    glow = glow.filter(ImageFilter.GaussianBlur(14))
    out = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    out.alpha_composite(glow)
    out.alpha_composite(earth)
    return out


def make_moon():
    n, R = 700, 300
    c = n // 2
    rng = random.Random(11)
    base = Image.new("RGB", (n, n), (176, 176, 182))
    d = ImageDraw.Draw(base, "RGBA")
    for _ in range(9):
        x, y = rng.randint(c - 200, c + 200), rng.randint(c - 200, c + 200)
        w, h = rng.randint(40, 110), rng.randint(30, 90)
        d.ellipse([x - w, y - h, x + w, y + h], fill=(120, 120, 128, 200))
    base = base.filter(ImageFilter.GaussianBlur(5))
    d = ImageDraw.Draw(base, "RGBA")
    for _ in range(70):
        x, y, r = rng.randint(c - 260, c + 260), rng.randint(c - 260, c + 260), rng.randint(6, 38)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(104, 104, 112, 170))
        d.ellipse([x - r + 3, y - r + 3, x + r - 1, y + r - 1], fill=(88, 88, 96, 150))
        d.arc([x - r, y - r, x + r, y + r], 200, 340, fill=(218, 218, 224, 210), width=2)
    return shade_sphere(base, n, R, (-0.55, -0.45, 0.70), 0.10)


def make_rocket(legs=False):
    S = 3
    w, h = 360, 560
    cx = 180
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    def R_(x0, y0, x1, y1):
        return [x0 * S, y0 * S, x1 * S, y1 * S]

    red, dred = (214, 48, 48, 255), (170, 34, 40, 255)
    for sgn in (-1, 1):  # fins
        d.polygon(P([(cx + sgn * 65, 380), (cx + sgn * 122, 505), (cx + sgn * 65, 470)]), fill=dred)
    if legs:
        for sgn in (-1, 1):
            d.line(P([(cx + sgn * 55, 400), (cx + sgn * 135, 528)]), fill=(110, 116, 132, 255), width=9 * S)
            d.line(P([(cx + sgn * 62, 452), (cx + sgn * 118, 512)]), fill=(90, 96, 112, 255), width=5 * S)
            d.ellipse(R_(cx + sgn * 135 - 24, 523, cx + sgn * 135 + 24, 537), fill=(150, 155, 170, 255))
    d.polygon(P([(cx - 26, 486), (cx + 26, 486), (cx + 40, 520), (cx - 40, 520)]), fill=(70, 74, 88, 255))
    d.rectangle(R_(cx - 65, 140, cx + 65, 490), fill=(236, 239, 246, 255))
    d.rectangle(R_(cx + 30, 140, cx + 65, 490), fill=(205, 210, 224, 255))
    d.rectangle(R_(cx - 56, 140, cx - 46, 490), fill=(251, 252, 255, 255))
    left, right = [], []
    for i in range(41):
        y = 140 - 135 * i / 40
        hw = 65 * math.sqrt(max(0.0, 1 - ((140 - y) / 135) ** 2))
        left.append((cx - hw, y))
        right.append((cx + hw, y))
    d.polygon(P(left + right[::-1]), fill=red)
    d.rectangle(R_(cx - 65, 137, cx + 65, 143), fill=dred)
    d.rectangle(R_(cx - 65, 300, cx + 65, 326), fill=red)
    d.rectangle(R_(cx - 65, 334, cx + 65, 340), fill=(60, 70, 90, 255))
    d.ellipse(R_(cx - 30, 185, cx + 30, 245), fill=(70, 80, 100, 255))
    d.ellipse(R_(cx - 23, 192, cx + 23, 238), fill=(96, 176, 236, 255))
    d.ellipse(R_(cx - 14, 199, cx - 2, 213), fill=(200, 235, 255, 255))
    d.rounded_rectangle(R_(cx - 26, 360, cx + 26, 452), radius=8 * S, fill=(222, 226, 236, 255),
                        outline=(140, 148, 166, 255), width=2 * S)
    d.ellipse(R_(cx + 12, 402, cx + 18, 408), fill=(90, 98, 116, 255))
    d.rectangle(R_(cx - 30, 482, cx + 30, 492), fill=(50, 54, 66, 255))
    return im.resize((max(w,1), max(1, h)), Image.LANCZOS)


EARTH = make_earth()
MOON = make_moon()
ROCKET = make_rocket(False)
ROCKET_LEGS = make_rocket(True)
EARTH_SMALL = scaled(EARTH, 0.25)


def make_cap():
    n, R = 880, 380
    c = n // 2
    k = 2000 / R
    x0, x1 = c - 540 / k, c + 540 / k
    y0 = (c - R) - 60 / k
    y1 = y0 + 720 / k
    return EARTH.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1)).resize((max(1,W), 720), Image.BICUBIC)


EARTH_CAP, CAP_TOP = make_cap(), 60


def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx / W - 0.5) * 2, (yy / H - 0.5) * 2
    r = np.sqrt(nx * nx * 0.8 + ny * ny * 0.9)
    v = 1 - 0.38 * np.clip((r - 0.55) / 0.8, 0, 1) ** 1.6
    g = (v * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([g, g, g]), "RGB")


VIGNETTE = vignette()


# -------------------------------------------------------------------- rocket
def draw_flame(fd, L, wscale, rng):
    ox, oy = 180, 985
    for col, lf, wf in (((255, 96, 24, 205), 1.0, 1.0), ((255, 170, 40, 235), 0.74, 0.70),
                        ((255, 240, 190, 255), 0.46, 0.38)):
        ln = L * lf * (0.93 + 0.14 * rng.random())
        wd = 40 * wf * wscale
        left, right = [], []
        for i in range(13):
            u = i / 12
            half = wd * (1 - u ** 1.6) * (1 + 0.10 * math.sin(u * 9 + rng.random()))
            sway = (rng.random() - 0.5) * 6 * u
            y = oy + ln * u
            left.append((ox - half + sway, y))
            right.append((ox + half + sway, y))
        fd.polygon(left + right[::-1], fill=col)


def rocket_layer(sprite, flame_len, wscale, rng, angle, scale):
    """Rocket + flame in a rig whose centre is the rocket centre, so it rotates about it."""
    rig = Image.new("RGBA", (360, 1500), (0, 0, 0, 0))
    if flame_len > 2:
        fl = Image.new("RGBA", (360, 1500), (0, 0, 0, 0))
        draw_flame(ImageDraw.Draw(fl), flame_len, wscale, rng)
        rig.alpha_composite(fl.filter(ImageFilter.GaussianBlur(2.5)))
    rig.alpha_composite(sprite, (0, 470))
    if scale != 1:
        rig = rig.resize((max(1,int(360 * scale)), max(1,int(1500 * scale))), Image.BILINEAR)
    if abs(angle) > 0.05:
        rig = rig.rotate(-angle, resample=Image.BICUBIC, expand=True)
    return rig


# =========================================================== library API ===
def rot(v, deg):
    """Rotate a 2D vector clockwise on screen by deg degrees."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def draw_earth(img, cx, cy, radius):
    paste_center(img, scaled(EARTH, radius / 380), cx, cy)


def draw_moon(img, cx, cy, radius):
    paste_center(img, scaled(MOON, radius / 300), cx, cy)


def draw_rocket(img, cx, cy, scale=1.0, angle=0.0, flame=0.0, legs=False, wscale=1.4, seed=0):
    """angle: degrees clockwise from nose-up. flame: 0..1 throttle."""
    layer = rocket_layer(ROCKET_LEGS if legs else ROCKET, 420 * flame, wscale, random.Random(seed), angle, scale)
    paste_center(img, layer, cx, cy)


def make_satellite():
    S = 3
    w, h = 520, 300
    cx, cy = 260, 150
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def R_(x0, y0, x1, y1):
        return [x0 * S, y0 * S, x1 * S, y1 * S]

    d.rectangle(R_(cx - 150, cy - 4, cx + 150, cy + 4), fill=(150, 156, 170, 255))
    for sgn in (-1, 1):
        xa, xb = sorted((cx + sgn * 60, cx + sgn * 250))
        d.rectangle(R_(xa, cy - 62, xb, cy + 62), fill=(24, 52, 120, 255), outline=(170, 180, 200, 255), width=2 * S)
        for i in range(1, 6):
            x = (xa + (xb - xa) * i / 6) * S
            d.line([(x, (cy - 62) * S), (x, (cy + 62) * S)], fill=(90, 130, 210, 255), width=S)
        for j in range(1, 4):
            y = (cy - 62 + 124 * j / 4) * S
            d.line([(xa * S, y), (xb * S, y)], fill=(90, 130, 210, 255), width=S)
    d.rounded_rectangle(R_(cx - 60, cy - 55, cx + 60, cy + 55), radius=8 * S, fill=(212, 175, 70, 255),
                        outline=(150, 120, 40, 255), width=2 * S)
    for k in range(4):
        d.line([((cx - 60) * S, (cy - 35 + k * 22) * S), ((cx + 60) * S, (cy - 35 + k * 22) * S)],
               fill=(236, 205, 110, 255), width=S)
    d.rectangle(R_(cx - 30, cy + 18, cx + 30, cy + 44), fill=(50, 56, 72, 255))
    for i, col in enumerate(((230, 70, 70, 255), (90, 210, 120, 255), (250, 210, 80, 255))):
        d.ellipse(R_(cx - 22 + i * 18, cy + 27, cx - 12 + i * 18, cy + 37), fill=col)
    d.line([(cx * S, (cy - 55) * S), (cx * S, (cy - 78) * S)], fill=(190, 196, 208, 255), width=4 * S)
    d.pieslice(R_(cx - 44, cy - 112, cx + 44, cy - 56), 180, 360, fill=(226, 229, 237, 255),
               outline=(150, 156, 170, 255), width=2 * S)
    d.ellipse(R_(cx - 5, cy - 90, cx + 5, cy - 80), fill=(230, 70, 70, 255))
    return im.resize((max(1,w), max(1,h)), Image.LANCZOS)


SATELLITE = make_satellite()


def draw_satellite(img, cx, cy, scale=1.0, angle=0.0):
    sp = SATELLITE if scale == 1 else scaled(SATELLITE, scale)
    if abs(angle) > 0.05:
        sp = sp.rotate(-angle, resample=Image.BICUBIC, expand=True)
    paste_center(img, sp, cx, cy)


def draw_atom(img, cx, cy, r, t):
    """Nucleus plus three tilted electron orbits; electrons pass behind and in front."""
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(3):
        pts = []
        for i in range(73):
            a = 2 * math.pi * i / 72
            x, y = rot((r * math.cos(a), r * 0.38 * math.sin(a)), k * 60)
            pts.append((cx + x, cy + y))
        d.line(pts, fill=(120, 190, 255, 110), width=3)
    electrons = []
    for k in range(3):
        th = t * (1.3 + 0.35 * k) + k * 2.1
        x, y = rot((r * math.cos(th), r * 0.38 * math.sin(th)), k * 60)
        electrons.append((math.sin(th), cx + x, cy + y))

    def electron(dp, x, y):
        er = r * 0.055 * (1 + 0.3 * dp)
        for m, a in ((3.2, 40), (2.0, 90), (1.0, 255)):
            d.ellipse([x - er * m, y - er * m, x + er * m, y + er * m],
                      fill=(150, 220, 255, a) if m > 1 else (255, 255, 255, a))

    for dp, x, y in electrons:
        if dp < 0:
            electron(dp, x, y)
    nr = r * 0.16
    for m, a in ((2.6, 14), (2.0, 20), (1.5, 30), (1.15, 45)):
        d.ellipse([cx - nr * m, cy - nr * m, cx + nr * m, cy + nr * m], fill=(255, 200, 120, a))
    rng = random.Random(4)
    for i in range(8):
        a, rr = rng.uniform(0, 6.28), rng.uniform(0, nr * 0.9)
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        pr = nr * 0.55
        proton = i % 2 == 0
        d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=(228, 72, 72, 255) if proton else (158, 168, 190, 255))
        d.ellipse([x - pr * 0.6, y - pr * 0.7, x - pr * 0.1, y - pr * 0.2], fill=(255, 255, 255, 110))
    for dp, x, y in electrons:
        if dp >= 0:
            electron(dp, x, y)


def draw_dna(img, cx, cy, length, t, amp=90, turns=2.6, n=40):
    """Vertical double helix with base-pair rungs, rotating with t."""
    d = ImageDraw.Draw(img, "RGBA")
    s1, s2, items = [], [], []
    for i in range(n):
        y = cy - length / 2 + length * i / (n - 1)
        ph = 2 * math.pi * turns * i / (n - 1) + t * 1.6
        s1.append((cx + amp * math.sin(ph), y, math.cos(ph)))
        s2.append((cx - amp * math.sin(ph), y, -math.cos(ph)))
    c1, c2 = (80, 170, 255), (255, 120, 170)
    for i in range(n - 1):
        for strand, col in ((s1, c1), (s2, c2)):
            p, q = strand[i], strand[i + 1]
            dp = (p[2] + q[2]) / 2
            items.append((dp, "seg", (p, q, col)))
    for i in range(n):
        p, q = s1[i], s2[i]
        items.append((0.0, "rung", (p, q, i)))
        items.append((p[2], "node", (p, c1)))
        items.append((q[2], "node", (q, c2)))
    items.sort(key=lambda it: it[0])
    for dp, kind, data in items:
        shade = 0.55 + 0.45 * (dp + 1) / 2
        if kind == "seg":
            p, q, col = data
            d.line([p[:2], q[:2]], fill=tuple(int(c * shade) for c in col) + (255,), width=int(9 + 5 * dp))
        elif kind == "rung" and data[2] % 2 == 0:
            p, q, i = data
            mx = (p[0] + q[0]) / 2
            a = (240, 200, 80) if (i // 2) % 2 == 0 else (100, 220, 150)
            b = (100, 220, 150) if (i // 2) % 2 == 0 else (240, 200, 80)
            d.line([p[:2], (mx, p[1])], fill=a + (200,), width=6)
            d.line([(mx, p[1]), q[:2]], fill=b + (200,), width=6)
        elif kind == "node":
            p, col = data
            r = 11 + 5 * dp
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=tuple(int(c * shade) for c in col) + (255,))


# ------------------------------------------------------------------- text
_FONTS = {}


def get_font(size, bold=True):
    key = (size, bold)
    if key in _FONTS:
        return _FONTS[key]
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size)
            break
        except OSError:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:
            f = ImageFont.load_default()
    _FONTS[key] = f
    return f


def wrap_text(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if font.getlength(test) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def draw_text(img, x, y, text, size=48, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), alpha=1.0):
    d = ImageDraw.Draw(img, "RGBA")
    d.multiline_text((x, y), text, font=get_font(size), fill=tuple(fill[:3]) + (int(255 * alpha),), anchor="mm",
                     align="center", stroke_width=stroke, stroke_fill=tuple(stroke_fill[:3]) + (int(255 * alpha),),
                     spacing=int(size * 0.2))


def draw_caption(img, text, y, size=46, alpha=1.0, max_w=900):
    if alpha <= 0.01:
        return
    f = get_font(size)
    wrapped = wrap_text(text, f, max_w)
    d = ImageDraw.Draw(img, "RGBA")
    bb = d.multiline_textbbox((0, 0), wrapped, font=f, align="center", spacing=int(size * 0.2))
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 26
    d.rounded_rectangle([W / 2 - tw / 2 - pad, y - th / 2 - pad, W / 2 + tw / 2 + pad, y + th / 2 + pad],
                        radius=34, fill=(10, 14, 30, int(165 * alpha)))
    draw_text(img, W / 2, y, wrapped, size, alpha=alpha)


def draw_bubble(img, x, y, text, size=46, tail=None, alpha=1.0):
    """Speech bubble centred at (x, y); tail=(tx, ty) is the point it touches."""
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img, "RGBA")
    f = get_font(size)
    bb = d.multiline_textbbox((0, 0), text, font=f, align="center", spacing=int(size * 0.2))
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 28
    x0, y0, x1, y1 = x - tw / 2 - pad, y - th / 2 - pad, x + tw / 2 + pad, y + th / 2 + pad
    a = int(255 * alpha)
    ink = (40, 44, 60, a)
    if tail:
        tx, ty = tail
        bx = min(max(tx, x0 + 46), x1 - 46)
        d.polygon([(bx - 20, y1 - 3), (bx + 20, y1 - 3), (tx, ty)], fill=(255, 255, 255, a))
        d.line([(bx - 20, y1), (tx, ty), (bx + 20, y1)], fill=ink, width=4)
    d.rounded_rectangle([x0, y0, x1, y1], radius=32, fill=(255, 255, 255, a), outline=ink, width=4)
    if tail:
        d.line([(bx - 17, y1 - 1), (bx + 17, y1 - 1)], fill=(255, 255, 255, a), width=6)
    d.multiline_text((x, y), text, font=f, fill=(30, 34, 50, a), anchor="mm", align="center",
                     spacing=int(size * 0.2))


# -------------------------------------------------------------- characters
def _blit_char(img, layer, ax, ay, x, y, scale, facing):
    if scale != 1:
        layer = layer.resize((max(1,int(layer.width * scale)), max(1,int(layer.height * scale))), Image.LANCZOS)
        ax, ay = ax * scale, ay * scale
    if facing < 0:
        layer = ImageOps.mirror(layer)
        ax = layer.width - ax
    img.paste(layer, (int(x - ax), int(y - ay)), layer)


def _tortoise_layer(phase, mood, blink, bob):
    S = 2
    w, h = 420, 340
    ox, oy = 200, 300
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def E(x0, y0, x1, y1):
        return [x0 * S, y0 * S, x1 * S, y1 * S]

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    skin, skin_d = (143, 186, 86, 255), (104, 148, 64, 255)
    line = (34, 78, 40, 255)
    amp = {"walk": 1.0, "cheer": 0.25}.get(mood, 0.0)
    d.polygon(P([(ox - 100, oy - 64), (ox - 134, oy - 50), (ox - 100, oy - 44)]), fill=skin_d)

    def leg(x, off, col):
        lift = amp * max(0.0, math.sin(2 * math.pi * (phase + off))) * 16
        sw = amp * math.cos(2 * math.pi * (phase + off)) * 12
        d.rounded_rectangle(E(x + sw - 18, oy - 56 - lift, x + sw + 18, oy - lift), radius=12 * S, fill=col)
        d.ellipse(E(x + sw + 2, oy - 16 - lift, x + sw + 16, oy - 2 - lift), fill=(232, 226, 190, 255))

    leg(ox - 74, 0.0, skin_d)
    leg(ox + 40, 0.5, skin_d)
    leg(ox - 36, 0.5, skin)
    leg(ox + 76, 0.0, skin)
    d.pieslice(E(ox - 112, oy - 205, ox + 112, oy + 65), 180, 360, fill=(76, 136, 66, 255), outline=line, width=3 * S)
    d.rounded_rectangle(E(ox - 118, oy - 78, ox + 118, oy - 50), radius=12 * S, fill=(198, 166, 98, 255),
                        outline=(120, 96, 52, 255), width=3 * S)

    def hexa(cx, cy, r):
        d.polygon(P([(cx + r * math.cos(math.radians(30 + 60 * k)), cy + r * 0.85 * math.sin(math.radians(30 + 60 * k)))
                     for k in range(6)]), fill=(98, 160, 80, 255), outline=(40, 96, 46, 255), width=3 * S)

    hexa(ox, oy - 135, 38)
    hexa(ox - 72, oy - 112, 30)
    hexa(ox + 72, oy - 112, 30)
    hexa(ox - 36, oy - 180, 22)
    hexa(ox + 36, oy - 180, 22)
    d.arc(E(ox - 96, oy - 196, ox + 6, oy - 100), 200, 262, fill=(255, 255, 255, 130), width=5 * S)

    lift = 18 if mood == "cheer" else 0
    d.ellipse(E(ox + 78, oy - 104 - lift + bob, ox + 134, oy - 58), fill=skin)
    hx, hy = ox + 132, oy - 98 - lift + bob
    d.ellipse(E(hx - 36, hy - 29, hx + 36, hy + 29), fill=skin, outline=line, width=2 * S)
    if mood == "cheer" or blink:
        d.arc(E(hx + 2, hy - 16, hx + 24, hy + 4), 200, 340, fill=(40, 30, 30, 255), width=4 * S)
    else:
        d.ellipse(E(hx + 3, hy - 16, hx + 22, hy + 3), fill=(255, 255, 255, 255))
        d.ellipse(E(hx + 10, hy - 12, hx + 19, hy - 3), fill=(30, 24, 24, 255))
        d.ellipse(E(hx + 13, hy - 11, hx + 16, hy - 8), fill=(255, 255, 255, 255))
    d.ellipse(E(hx + 12, hy + 4, hx + 30, hy + 16), fill=(240, 150, 140, 110))
    if mood == "cheer":
        d.pieslice(E(hx - 4, hy - 2, hx + 32, hy + 28), 0, 180, fill=(150, 40, 50, 255))
        d.pieslice(E(hx + 6, hy + 12, hx + 24, hy + 28), 0, 180, fill=(240, 120, 130, 255))
    else:
        d.arc(E(hx - 4, hy - 4, hx + 30, hy + 18), 20, 160, fill=(60, 40, 30, 255), width=3 * S)
    return im.resize((max(1,w), max(1,h)), Image.LANCZOS)


def draw_tortoise(img, x, y, scale=1.0, phase=0.0, mood="walk", facing=1, t=0.0):
    """(x, y) = ground contact point. mood: walk | stand | cheer."""
    blink = int(t * 3) % 11 == 0
    bob = 2.0 * math.sin(phase * 4 * math.pi) if mood == "walk" else 0.0
    _blit_char(img, _tortoise_layer(phase, mood, blink, bob), 200, 300, x, y, scale, facing)


def _hare_layer(pose, phase, t, crouch):
    S = 2
    w, h = 470, 400
    ox, oy = 235, 350
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def E(x0, y0, x1, y1):
        return [x0 * S, y0 * S, x1 * S, y1 * S]

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    fur, fur_d = (214, 176, 132, 255), (172, 134, 96, 255)
    belly, pink, dark = (248, 236, 220, 255), (238, 150, 160, 255), (40, 30, 30, 255)

    def LN(pts, col, wd):
        d.line(P(pts), fill=col, width=int(wd * S))

    def ear(bx, by, ang, L=108, wd=26):
        outer, inner = [], []
        for k in range(28):
            a = 2 * math.pi * k / 28
            outer.append(tuple(c + o for c, o in zip((bx, by), rot((wd / 2 * math.cos(a), -L / 2 + L / 2 * math.sin(a)), ang))))
            inner.append(tuple(c + o for c, o in zip((bx, by), rot((wd * 0.28 * math.cos(a), -L * 0.5 + L * 0.38 * math.sin(a)), ang))))
        d.polygon(P(outer), fill=fur, outline=fur_d)
        d.polygon(P(inner), fill=pink)

    if pose == "sleep":
        ry = 34 + 2.5 * math.sin(t * 2.2)
        bx, by = ox - 14, oy - 4 - ry
        d.ellipse(E(bx - 86, by - ry, bx + 82, by + ry), fill=fur, outline=fur_d)
        d.ellipse(E(bx - 55, by + ry * 0.1, bx + 70, by + ry), fill=belly)
        d.ellipse(E(bx - 112, by - 22, bx - 76, by + 14), fill=(252, 248, 240, 255))
        d.ellipse(E(bx - 100, oy - 22, bx - 34, oy - 3), fill=fur_d)
        hx, hy = bx + 104, oy - 38
        d.ellipse(E(bx + 60, oy - 22, bx + 136, oy - 2), fill=fur)
        d.ellipse(E(hx - 38, hy - 30, hx + 38, hy + 30), fill=fur, outline=fur_d)
        d.ellipse(E(hx + 8, hy, hx + 50, hy + 26), fill=belly)
        d.ellipse(E(hx + 40, hy - 1, hx + 54, hy + 9), fill=(210, 110, 120, 255))
        d.arc(E(hx + 2, hy - 14, hx + 26, hy + 4), 20, 160, fill=dark, width=3 * S)
        ear(hx - 14, hy - 24, -100)
        ear(hx + 4, hy - 28, -94)
        return im.resize((max(1,w), max(1,h)), Image.LANCZOS)

    th = 2 * math.pi * phase
    run, walk = pose == "run", pose == "walk"
    hop = 34 * max(0.0, math.sin(th + 0.3)) if run else (7 * max(0.0, math.sin(th)) if walk else 0.0)
    if pose == "surprised":
        hop = 38 * abs(math.sin(t * 13))
    jit = 3 * math.sin(t * 38) if pose == "laugh" else 0.0
    lower = 22 * crouch
    bx, by = ox - 10 + jit, oy - 102 - hop + lower

    def foot(kind, off):
        a = th + off
        if run:
            if kind == "hind":
                return bx - 40 + 52 * math.cos(a), oy - 12 - 42 * max(0.0, math.sin(a))
            return bx + 62 + 42 * math.cos(a + 1.0), oy - 12 - 46 * max(0.0, math.sin(a + 1.0))
        if walk:
            if kind == "hind":
                return bx - 36 + 22 * math.cos(a), oy - 12 - 12 * max(0.0, math.sin(a))
            return bx + 58 + 18 * math.cos(a + 1.0), oy - 12 - 12 * max(0.0, math.sin(a + 1.0))
        return (bx - 34, oy - 12) if kind == "hind" else (bx + 58, oy - 12)

    def leg(hip, ft, col, thigh=False):
        LN([hip, ft], col, 22)
        d.ellipse(E(ft[0] - 12, ft[1] - 11, ft[0] + 34, ft[1] + 5), fill=col)
        if thigh:
            d.ellipse(E(hip[0] - 32, hip[1] - 30, hip[0] + 30, hip[1] + 34), fill=col)

    hip_h, hip_f = (bx - 48, by + 20), (bx + 52, by + 24)
    leg(hip_h, foot("hind", math.pi), fur_d, True)
    leg(hip_f, foot("front", math.pi), fur_d)
    d.ellipse(E(bx - 100, by - 26, bx - 62, by + 12), fill=(252, 248, 240, 255), outline=(228, 220, 208, 255))
    d.ellipse(E(bx - 82, by - 50, bx + 78, by + 48), fill=fur, outline=fur_d)
    d.ellipse(E(bx - 50, by + 6, bx + 62, by + 48), fill=belly)
    leg(hip_h, foot("hind", 0.0), fur, True)
    leg(hip_f, foot("front", 0.0), fur)

    hx = bx + 84
    hy = by - 30 - (8 if pose == "laugh" else 0)
    d.ellipse(E(hx - 40, hy - 34, hx + 38, hy + 32), fill=fur, outline=fur_d)
    d.ellipse(E(hx + 8, hy - 2, hx + 52, hy + 28), fill=belly)
    d.ellipse(E(hx + 42, hy - 2, hx + 57, hy + 9), fill=(210, 110, 120, 255))
    d.ellipse(E(hx + 12, hy + 4, hx + 30, hy + 16), fill=(240, 150, 150, 120))

    if pose == "laugh":
        d.arc(E(hx + 4, hy - 18, hx + 28, hy + 4), 200, 340, fill=dark, width=4 * S)
        d.pieslice(E(hx + 16, hy + 2, hx + 54, hy + 34), 0, 180, fill=(150, 40, 50, 255))
        d.pieslice(E(hx + 26, hy + 18, hx + 46, hy + 34), 0, 180, fill=(240, 120, 130, 255))
    elif pose == "surprised":
        d.ellipse(E(hx + 3, hy - 22, hx + 29, hy + 4), fill=(255, 255, 255, 255), outline=dark)
        d.ellipse(E(hx + 13, hy - 13, hx + 20, hy - 6), fill=dark)
        d.ellipse(E(hx + 26, hy + 8, hx + 40, hy + 24), fill=(150, 40, 50, 255))
    elif pose == "sad":
        d.ellipse(E(hx + 6, hy - 16, hx + 25, hy + 3), fill=(255, 255, 255, 255), outline=dark)
        d.ellipse(E(hx + 12, hy - 8, hx + 20, hy), fill=dark)
        LN([(hx + 2, hy - 24), (hx + 26, hy - 18)], dark, 3)
        d.arc(E(hx + 20, hy + 12, hx + 48, hy + 30), 200, 340, fill=dark, width=3 * S)
        d.ellipse(E(hx + 24, hy + 2, hx + 31, hy + 16), fill=(120, 190, 255, 220))
    else:
        d.ellipse(E(hx + 6, hy - 18, hx + 25, hy + 1), fill=(255, 255, 255, 255), outline=dark)
        d.ellipse(E(hx + 13, hy - 12, hx + 22, hy - 3), fill=dark)
        d.ellipse(E(hx + 16, hy - 11, hx + 19, hy - 8), fill=(255, 255, 255, 255))
        LN([(hx + 46, hy + 8), (hx + 46, hy + 14)], dark, 2)
        d.arc(E(hx + 30, hy + 8, hx + 50, hy + 22), 20, 160, fill=dark, width=2 * S)
    for dy, dx in ((-2, 40), (8, 42), (18, 38)):
        LN([(hx + 40, hy + 8), (hx + 40 + dx, hy + 8 + dy)], (90, 70, 60, 150), 1)

    flutter = 6 * math.sin(th * 2) if run else 3 * math.sin(t * 2)
    if run:
        a1, a2 = -78 + flutter, -68 - flutter
    elif pose == "surprised":
        a1, a2 = -2, 8
    elif pose == "sad":
        a1, a2 = -142, -126
    elif pose == "laugh":
        a1, a2 = -22 + jit * 2, -6 - jit
    else:
        a1 = -10 * (1 - crouch) - 62 * crouch + flutter
        a2 = 4 * (1 - crouch) - 54 * crouch - flutter
    ear(hx - 14, hy - 28, a1)
    ear(hx + 6, hy - 32, a2)
    return im.resize((max(1,w), max(1,h)), Image.LANCZOS)


def draw_hare(img, x, y, scale=1.0, pose="stand", phase=0.0, facing=1, t=0.0, crouch=0.0):
    """(x, y) = ground contact. pose: stand | run | walk | laugh | sleep | surprised | sad."""
    _blit_char(img, _hare_layer(pose, phase, t, crouch), 235, 350, x, y, scale, facing)


def draw_trophy(img, x, y, scale=1.0, t=0.0):
    S = 3
    w, h = 200, 240
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def E(x0, y0, x1, y1):
        return [x0 * S, y0 * S, x1 * S, y1 * S]

    gold, gold_d = (245, 196, 40, 255), (200, 142, 20, 255)
    d.arc(E(6, 30, 62, 104), 90, 270, fill=gold_d, width=10 * S)
    d.arc(E(138, 30, 194, 104), 270, 90, fill=gold_d, width=10 * S)
    d.polygon([(40 * S, 30 * S), (160 * S, 30 * S), (150 * S, 100 * S), (125 * S, 130 * S), (75 * S, 130 * S),
               (50 * S, 100 * S)], fill=gold, outline=gold_d)
    d.ellipse(E(40, 22, 160, 40), fill=(255, 226, 112, 255), outline=gold_d)
    d.rectangle(E(90, 130, 110, 172), fill=gold_d)
    d.rounded_rectangle(E(60, 170, 140, 204), radius=6 * S, fill=gold_d)
    d.rectangle(E(78, 180, 122, 194), fill=(255, 226, 112, 255))
    pts = []
    for k in range(10):
        r = 20 if k % 2 == 0 else 8
        a = math.radians(-90 + 36 * k)
        pts.append(((100 + r * math.cos(a)) * S, (76 + r * math.sin(a)) * S))
    d.polygon(pts, fill=(255, 250, 220, 255))
    im = im.resize((max(1,w), max(1,h)), Image.LANCZOS)
    if scale != 1:
        im = im.resize((max(1,int(w * scale)), max(1,int(h * scale))), Image.LANCZOS)
    paste_center(img, im, x, y)
    d2 = ImageDraw.Draw(img, "RGBA")
    for k, (sx_, sy_) in enumerate(((-70, -60), (76, -40), (-50, 50), (68, 62))):
        a = 0.5 + 0.5 * math.sin(t * 5 + k * 1.7)
        r = (10 + 8 * a) * scale
        px, py = x + sx_ * scale, y + sy_ * scale
        c = (255, 255, 230, int(220 * a))
        d2.polygon([(px, py - r), (px + r * 0.25, py - r * 0.25), (px + r, py), (px + r * 0.25, py + r * 0.25),
                    (px, py + r), (px - r * 0.25, py + r * 0.25), (px - r, py), (px - r * 0.25, py - r * 0.25)], fill=c)


def run_cli(render_frame, total, default_out="out/video.mp4", description="2D animation"):
    """Shared command line: full video, --start/--end, --still, --frames-dir."""
    import shutil
    import subprocess
    import sys
    import time

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=total)
    ap.add_argument("--still", default="", help="comma list of times; saves PNG stills and exits")
    ap.add_argument("--frames-dir", default="")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if args.still:
        for s in args.still.split(","):
            ts = float(s)
            p = out.parent / f"still_{ts:05.1f}.png"
            render_frame(ts, int(ts * args.fps)).save(p)
            print("saved", p)
        return
    n = int(round((args.end - args.start) * args.fps))
    proc = None
    if args.frames_dir:
        Path(args.frames_dir).mkdir(parents=True, exist_ok=True)
    else:
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found. Install it (apt install ffmpeg) or use --frames-dir.")
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(args.crf), "-preset", "medium", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
    for i in range(n):
        t = args.start + i / args.fps
        frame = render_frame(t, i + int(args.start * args.fps))
        if proc:
            proc.stdin.write(frame.tobytes())
        else:
            frame.save(Path(args.frames_dir) / f"frame_{i:05d}.png")
        if i % 60 == 0:
            print(f"frame {i + 1}/{n}  t={t:5.1f}s", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()
    total_s = time.time() - t_all
    print(f"Total: {total_s:.1f} s for {n} frames ({n / args.fps:.1f} s of video, {total_s / max(n, 1):.3f} s/frame)")
    print("Output:", args.frames_dir or out)


# ----------------------------------------------------------------- preview
def preview(path, t=1.0):
    """Contact sheet of every object in the library (check it from a phone)."""
    sheet = vgrad((6, 8, 20), (14, 18, 36))
    d = ImageDraw.Draw(sheet, "RGBA")
    f = get_font(28)
    cw, ch = W // 2, H // 5

    def cell(i, label, fn):
        x0, y0 = (i % 2) * cw, (i // 2) * ch
        d.rectangle([x0 + 8, y0 + 8, x0 + cw - 8, y0 + ch - 8], outline=(255, 255, 255, 40), width=2)
        d.text((x0 + 22, y0 + 16), label, fill=(255, 255, 255, 200), font=f)
        fn(x0 + cw / 2, y0 + ch / 2 + 22)

    def ground(x, y):
        d.rectangle([x - 200, y, x + 200, y + 10], fill=(120, 150, 90, 255))

    cell(0, "draw_earth", lambda x, y: draw_earth(sheet, x, y, 120))
    cell(1, "draw_moon", lambda x, y: draw_moon(sheet, x, y, 110))
    cell(2, "draw_rocket (flame)", lambda x, y: draw_rocket(sheet, x, y - 40, 0.34, 0, 1.0))
    cell(3, "draw_rocket (legs)", lambda x, y: draw_rocket(sheet, x, y, 0.34, 0, 0.0, legs=True))
    cell(4, "draw_satellite", lambda x, y: draw_satellite(sheet, x, y, 0.62, -15))
    cell(5, "draw_atom", lambda x, y: draw_atom(sheet, x, y, 115, t))
    cell(6, "draw_dna", lambda x, y: draw_dna(sheet, x, y, 270, t, amp=55, turns=2.0, n=30))
    cell(7, "draw_trophy", lambda x, y: draw_trophy(sheet, x, y, 0.9, t))
    cell(8, "draw_tortoise", lambda x, y: (ground(x, y + 70), draw_tortoise(sheet, x, y + 78, 0.8, 0.2, "walk", 1, t)))
    cell(9, "draw_hare", lambda x, y: (ground(x, y + 70), draw_hare(sheet, x, y + 78, 0.72, "stand", 0.0, 1, t)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sciart library tools")
    ap.add_argument("--preview", default="out/library.png", help="write a contact sheet of all objects")
    preview(ap.parse_args().preview)
