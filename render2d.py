#!/usr/bin/env python3
"""
render2d.py - 2D rocket mission animation (Pillow + NumPy + ffmpeg only)

Story (42.5 s, 1080x1920 vertical, 30 fps):
  1. LAUNCH       0.0 -  9.0  ignition, lift-off, climb through the atmosphere
  2. EARTH ORBIT  9.0 - 15.0  one full revolution around Earth (behind/in front)
  3. TRANSIT     15.0 - 23.0  trans-lunar burn, rocket shrinks as it leaves
                              Earth, Moon grows ahead
  4. MOON ORBIT  23.0 - 29.0  lunar orbit insertion, one revolution, de-orbit burn
  5. LANDING     29.0 - 35.5  retro-burn, legs out, dust, soft touchdown
  6. HATCH       35.5 - 42.5  hatch opens, light spills out, ladder, astronaut waves

No posting, no network, no API keys. Pure local rendering.

Usage:
  python render2d.py                          # full video -> out/rocket_mission.mp4
  python render2d.py --still 2,6,12,20,26     # save PNG stills (fast preview)
  python render2d.py --start 22 --end 30      # render just one section
  python render2d.py --fps 15                 # quicker test render
  python render2d.py --frames-dir out/frames  # PNG sequence instead of mp4
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from sciart import *  # noqa: F401,F403  (shared objects and helpers live in sciart.py)
from sciart import W, H

FPS = 30
XFADE = 0.6  # seconds of dissolve between phases

PHASES = [
    ("launch", 0.0, 9.0),
    ("orbit_earth", 9.0, 15.0),
    ("transit", 15.0, 23.0),
    ("orbit_moon", 23.0, 29.0),
    ("landing", 29.0, 35.5),
    ("hatch", 35.5, 42.5),
    ("moonwalk", 42.5, 56.5),
]
TOTAL = PHASES[-1][2]


# ------------------------------------------------------------- phase 1: launch
ALT_MAX = 15000.0
_cr = random.Random(21)
CLOUDS = [(_cr.uniform(-100, W + 100), _cr.uniform(600, 9500), _cr.uniform(0.8, 2.0)) for _ in range(28)]
FG_CLOUDS = [(_cr.uniform(-100, W + 100), _cr.uniform(900, 6500), _cr.uniform(1.6, 3.0)) for _ in range(7)]
MOUNTS = [(x, _cr.uniform(80, 260)) for x in range(-60, W + 120, 110)]
SMOKE = [(1.0 + i * 0.04, _cr.choice([-1, 1]), 540 + _cr.uniform(-30, 30), _cr.uniform(120, 380),
          _cr.uniform(0, 14), _cr.uniform(26, 50)) for i in range(90)]


def alt_at(t):
    if t <= 2.0:
        return 0.0
    return ALT_MAX * clamp((t - 2.0) / 7.0) ** 2.2


def base_y_at(t):
    return 1480 - 340 * smooth(alt_at(t) / 700)


def throttle_launch(t):
    if t < 1.0:
        return 0.0
    if t < 2.0:
        return lerp(0.15, 0.55, smooth(t - 1.0))
    return lerp(0.55, 1.0, smooth((t - 2.0) / 2.5))


def draw_cloud(d, x, y, s, alpha):
    for dx, dy, rw, rh in ((0, 0, 110, 42), (-85, 10, 75, 32), (80, 12, 85, 34), (-30, -22, 70, 38), (40, -18, 60, 30)):
        d.ellipse([x + (dx - rw) * s * 0.7, y + (dy - rh) * s * 0.7,
                   x + (dx + rw) * s * 0.7, y + (dy + rh) * s * 0.7], fill=(255, 255, 255, alpha))


def phase_launch(t, fi):
    rng = random.Random(fi)
    alt = alt_at(t)
    frac = alt / ALT_MAX
    by = base_y_at(t)
    gy = by + alt
    img = vgrad(mix((38, 100, 215), (3, 5, 18), clamp(frac * 1.15) ** 0.9),
                mix((155, 208, 255), (12, 22, 60), clamp(frac * 1.1) ** 0.8))
    draw_stars(img, clamp((frac - 0.4) / 0.35), t)
    e = 330 * smooth((frac - 0.5) / 0.5)
    if e > 1:
        img.paste(EARTH_CAP, (0, int(H - e - CAP_TOP)), EARTH_CAP)
    d = ImageDraw.Draw(img, "RGBA")

    for x, ac, s in CLOUDS:
        y = gy - ac
        if -150 < y < H + 150:
            draw_cloud(d, x, y, s, int(235 * clamp(1 - (ac / ALT_MAX) * 1.6)))

    thr = throttle_launch(t)
    if gy < H + 60:
        for x, hgt in MOUNTS:
            d.polygon([(x - 90, gy), (x, gy - hgt), (x + 90, gy)], fill=(88, 130, 150, 255))
        d.rectangle([0, gy, W, max(H, gy)], fill=(72, 112, 62, 255))
        d.rectangle([0, gy, W, gy + 14], fill=(88, 130, 74, 255))
        tx = 250
        d.rectangle([tx, gy - 560, tx + 46, gy], outline=(90, 96, 108, 255), width=5)
        for k in range(10):
            d.line([(tx, gy - 56 * k), (tx + 46, gy - 56 * (k + 1))], fill=(90, 96, 108, 255), width=3)
        arm = 179 * (1 - smooth((t - 1.5) / 0.8))
        d.rectangle([tx + 46, gy - 330, tx + 46 + arm, gy - 318], fill=(120, 126, 138, 255))
        d.rectangle([380, gy, 700, gy + 26], fill=(150, 152, 158, 255))
        ga = int(90 * thr * clamp(1 - alt / 600))
        if ga > 0:
            d.ellipse([340, gy - 30, 740, gy + 50], fill=(255, 190, 90, ga))
        for ts, side, x0, sp, sy, sz in SMOKE:
            age = t - ts
            if age < 0 or age > 4.5:
                continue
            px = x0 + side * sp * (1 - math.exp(-age * 1.1)) / 1.1
            py = gy - 8 - sy - age * 18
            r = sz + age * 55
            d.ellipse([px - r, py - r * 0.62, px + r, py + r * 0.62],
                      fill=(226, 226, 232, int(170 * clamp(1 - age / 4.5))))

    for k in range(1, 16):  # exhaust trail
        ta = t - k * 0.07
        if ta < 2.0:
            continue
        y = gy - alt_at(ta) + 12
        x = 540 + math.sin(k * 12.9898) * 20
        r = 22 + k * 5
        a = int(150 * (1 - k / 16) * clamp(1 - frac * 1.3))
        if a > 0:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 235, 240, a))

    speed = (alt_at(t + 0.03) - alt_at(t - 0.03)) / 0.06
    vs = clamp(speed / 4500) if frac > 0.12 else 0.0
    if vs > 0.02:
        for i in range(14):
            x = (i * 79 + 31) % W
            y = (i * 397 + t * speed * 0.9) % (H + 400) - 200
            d.line([(x, y), (x, y + 100 + vs * 300)], fill=(255, 255, 255, int(70 * vs)), width=2)

    amp = 4.0 * thr * (1.0 if alt < 300 else 0.3)
    sx, sy = (rng.random() - 0.5) * 2 * amp, (rng.random() - 0.5) * 2 * amp
    layer = rocket_layer(ROCKET, 360 * thr * (1 + 0.15 * frac), (0.7 + 0.3 * thr) * (1 + 0.9 * frac), rng, 0, 1.0)
    paste_center(img, layer, 540 + sx, by - 240 + sy)

    d = ImageDraw.Draw(img, "RGBA")
    for x, ac, s in FG_CLOUDS:
        y = (by + alt * 1.25) - ac
        if -200 < y < H + 200:
            draw_cloud(d, x, y, s, int(120 * clamp(1 - (ac / 6500) * 0.8) * clamp(1 - frac * 1.8)))
    return img


# ------------------------------------------------- phases 2-4: orbit / transit
def draw_orbit(img, t, dur, planet, native_r, pr, center, burn_len, rng):
    """One full revolution around a planet. Rocket passes behind it, then in front,
    and looks bigger when nearer the camera."""
    cx, cy = center
    rx, ry = 1.3 * pr, 0.42 * pr

    def pt(a):
        return (cx + rx * math.cos(a), cy + ry * math.sin(a))

    th = math.pi + 2 * math.pi * clamp(t / dur)  # starts at the left, heading up
    s = math.sin(th)
    pos = pt(th)
    scale = 0.36 * (1 + 0.28 * s)
    ang = math.degrees(math.atan2(-rx * math.sin(th), -(ry * math.cos(th))))
    layer = rocket_layer(ROCKET, burn_len, 1.9, rng, ang, scale)
    d = ImageDraw.Draw(img, "RGBA")

    def dots(front):
        for i in range(90):
            a = 2 * math.pi * i / 90
            if (math.sin(a) >= 0) == front:
                x, y = pt(a)
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255, 70))

    dots(False)
    if s < 0:
        paste_center(img, layer, *pos)
    paste_center(img, scaled(planet, pr / native_r), cx, cy)
    dots(True)
    if s >= 0:
        paste_center(img, layer, *pos)


def phase_orbit_earth(t, fi):
    rng = random.Random(fi)
    img = vgrad((2, 3, 12), (10, 14, 34))
    draw_stars(img, 1.0, t, dx=-t * 6, dy=t * 4)
    paste_center(img, scaled(MOON, 0.07), 900, 250)  # the distant Moon
    burn = 300 * clamp(1 - t / 0.8)  # orbit-insertion burn
    draw_orbit(img, t, 6.0, EARTH, 380, 330, (540, 1050), burn, rng)
    return img


BEZ = [(128, 1050), (128, 760), (330, 640), (520, 560)]


def bez(p):
    q = 1 - p
    return tuple(q ** 3 * BEZ[0][i] + 3 * q * q * p * BEZ[1][i] + 3 * q * p * p * BEZ[2][i] + p ** 3 * BEZ[3][i]
                 for i in (0, 1))


def bez_d(p):
    q = 1 - p
    return tuple(3 * (q * q * (BEZ[1][i] - BEZ[0][i]) + 2 * q * p * (BEZ[2][i] - BEZ[1][i])
                      + p * p * (BEZ[3][i] - BEZ[2][i])) for i in (0, 1))


def phase_transit(t, fi):
    rng = random.Random(fi)
    u = clamp(t / 8.0)
    su = smooth(u)
    img = vgrad((2, 3, 12), (10, 14, 34))
    draw_stars(img, 1.0, t, dx=-t * 9, dy=t * 14)

    er = lerp(330, 130, su)
    paste_center(img, scaled(EARTH, er / 380), lerp(540, 190, su), lerp(1050, 1660, su))
    mr = lerp(24, 260, u ** 2.0)
    paste_center(img, scaled(MOON, mr / 300), lerp(880, 790, su), lerp(300, 470, su))

    p = smooth(clamp((t - 0.3) / 7.4))
    px, py = bez(p)
    dx, dy = bez_d(p)
    ang = math.degrees(math.atan2(dx, -dy))
    scale = lerp(0.36, 0.14, p)  # shrinks as it gets farther from Earth

    d = ImageDraw.Draw(img, "RGBA")
    for i in range(61):
        q = i / 60
        x, y = bez(q)
        if q < p:
            d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 210, 140, 110))
        else:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255, 45))

    burn = 300 * clamp(1 - (t - 0.6) / 1.0)  # trans-lunar injection burn
    paste_center(img, rocket_layer(ROCKET, burn, 1.9, rng, ang, scale), px, py)
    return img


def phase_orbit_moon(t, fi):
    rng = random.Random(fi)
    img = vgrad((1, 2, 8), (10, 12, 26))
    draw_stars(img, 1.0, t, dx=-t * 5, dy=t * 3)
    paste_center(img, scaled(EARTH, 0.17), 160, 330)  # Earth, far behind
    if t < 0.9:
        burn = 300 * clamp(1 - t / 0.9)  # lunar orbit insertion burn
    elif t > 5.0:
        burn = 240 * smooth((t - 5.0) / 0.3)  # de-orbit burn
    else:
        burn = 0.0
    draw_orbit(img, t, 6.0, MOON, 300, 330, (540, 1000), burn, rng)
    return img


# ------------------------------------------------------------- lunar surface
_gr = random.Random(5)
HILLS = [(x, 40 + 28 * math.sin(x * 0.012) + _gr.uniform(-10, 10)) for x in range(-100, W + 200, 90)]
CRATERS = [(_gr.uniform(0, W), _gr.uniform(0.05, 0.95), _gr.uniform(40, 170)) for _ in range(16)]
ROCKS = [(_gr.uniform(0, W), _gr.uniform(0.1, 0.95), _gr.uniform(6, 20)) for _ in range(22)]


def lunar_ground(img, horizon):
    d = ImageDraw.Draw(img, "RGBA")
    pts = [(-100, horizon + 2)] + [(x, horizon - h) for x, h in HILLS] + [(W + 200, horizon + 2)]
    d.polygon(pts, fill=(62, 62, 70, 255))
    gh = H - horizon
    img.paste(vgrad((124, 124, 132), (38, 38, 46), h=gh), (0, horizon))
    d = ImageDraw.Draw(img, "RGBA")
    for fx, fy, w in CRATERS:
        y = horizon + fy * gh
        ww = w * (0.25 + fy * 1.1)
        hh = ww * 0.26
        d.ellipse([fx - ww, y - hh, fx + ww, y + hh], fill=(52, 52, 60, 120))
        d.arc([fx - ww, y - hh, fx + ww, y + hh], 200, 340, fill=(176, 176, 184, 150), width=3)
    for fx, fy, r in ROCKS:
        y = horizon + fy * gh
        rr = r * (0.4 + fy)
        d.ellipse([fx - rr, y - rr * 0.6, fx + rr, y + rr * 0.6], fill=(84, 84, 92, 255))
        d.ellipse([fx - rr, y - rr * 0.6, fx + rr * 0.4, y + rr * 0.1], fill=(120, 120, 128, 255))


# -------------------------------------------------------------- phase 3: landing
HORIZON_C, CONTACT_C = 1380, 1560
SC, TD = 0.95, 5.0
FEET_OFF = 250 * SC


def landing_y(t):
    u = clamp(t / TD)
    y_land = CONTACT_C + 6 - FEET_OFF
    return lerp(-320, y_land, 1 - (1 - u) ** 2.2)


def _dust_t0():
    for i in range(500):
        tt = i * 0.01
        if CONTACT_C - (landing_y(tt) + FEET_OFF) < 380:
            return tt
    return 3.0


DUST_T0 = _dust_t0()
_dr = random.Random(9)
DUST = [(DUST_T0 + (TD + 0.6 - DUST_T0) * i / 80, _dr.choice([-1, 1]), 540 + _dr.uniform(-40, 40),
         _dr.uniform(220, 560), _dr.uniform(-6, 22), _dr.uniform(30, 60)) for i in range(80)]


def phase_landing(t, fi):
    rng = random.Random(fi)
    img = vgrad((1, 2, 8), (18, 18, 26))
    draw_stars(img, 1.0, t, ymax=HORIZON_C - 40)
    paste_center(img, EARTH_SMALL, 190, 330)
    lunar_ground(img, HORIZON_C)
    d = ImageDraw.Draw(img, "RGBA")

    y = landing_y(t)
    if t > TD:
        y += 7 * math.exp(-(t - TD) * 6) * math.sin((t - TD) * 28)
    height = CONTACT_C - (y + FEET_OFF)
    sa = int(120 * clamp(1 - height / 1500))
    sw = lerp(110, 160, clamp(1 - height / 900))
    d.ellipse([540 - sw, CONTACT_C + 4, 540 + sw, CONTACT_C + 34], fill=(10, 10, 14, sa))

    speed = (landing_y(t + 0.03) - landing_y(t - 0.03)) / 0.06
    vmax = (CONTACT_C + 6 - FEET_OFF + 320) / TD * 2.2
    if t < TD:
        thr = (0.25 + 0.75 * clamp(speed / vmax) ** 0.7) * clamp((TD - t) / 0.2)
    else:
        thr = 0.0
    sprite = ROCKET_LEGS if t / TD > 0.25 else ROCKET
    x = 540 + 25 * math.sin(clamp(t / TD) * 5) * (1 - clamp(t / TD))

    for ts, side, x0, sp, y0, sz in DUST:
        age = t - ts
        if age < 0 or age > 3.4:
            continue
        px = x0 + side * sp * (1 - math.exp(-age * 1.3)) / 1.3
        py = CONTACT_C + y0 - age * 6
        w = sz + age * 90
        d.ellipse([px - w, py - w * 0.28, px + w, py + w * 0.28], fill=(170, 170, 178, int(150 * clamp(1 - age / 3.4))))
        # (dust is drawn first so the rocket stands in front of it)

    layer = rocket_layer(sprite, 400 * thr, 1.4, rng, 0, SC)
    paste_center(img, layer, x, y)
    return img


# ---------------------------------------------------------------- phase 4: hatch
HORIZON_D, CONTACT_D = 1400, 1700
HX0, HX1, HY0, HY1 = 380, 700, 740, 1200


def make_hull():
    x0, x1 = 150, 930
    w = x1 - x0
    s = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    prof = 0.80 + 0.20 * np.clip(np.sin(np.pi * s), 0, 1) ** 0.6
    arr = np.broadcast_to(np.array([238, 241, 248], np.float32) * prof, (CONTACT_D, w, 3))
    im = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    for sx in (130, 650):
        d.line([(sx, 0), (sx, CONTACT_D)], fill=(180, 186, 202, 255), width=3)
    for sy in (300, 1330):
        d.line([(0, sy), (w, sy)], fill=(150, 156, 172, 255), width=4)
        for x in range(20, w, 46):
            for off in (-16, 16):
                d.ellipse([x - 4, sy + off - 4, x + 4, sy + off + 4], fill=(170, 176, 190, 255))
    d.rectangle([0, 1336, w, 1404], fill=(214, 48, 48, 255))
    d.rectangle([0, 1414, w, 1420], fill=(60, 70, 90, 255))
    d.rectangle([0, CONTACT_D - 34, w, CONTACT_D], fill=(120, 126, 142, 255))
    cx, cy = 390, 520  # mission roundel
    d.ellipse([cx - 80, cy - 80, cx + 80, cy + 80], fill=(214, 48, 48, 255))
    d.ellipse([cx - 68, cy - 68, cx + 68, cy + 68], fill=(255, 255, 255, 255))
    d.ellipse([cx - 58, cy - 58, cx + 58, cy + 58], fill=(40, 70, 140, 255))
    d.ellipse([cx - 4, cy - 40, cx + 32, cy - 4], fill=(230, 230, 236, 255))
    d.arc([cx - 40, cy - 10, cx + 40, cy + 40], 10, 170, fill=(255, 255, 255, 255), width=4)
    return im


HULL = make_hull()


def door_geometry(theta, pop):
    th = math.radians(theta)
    wid = (HX1 - HX0) * pop
    xh = HX0 - (pop - 1) * (HX1 - HX0) / 2
    xf = xh + wid * math.cos(th)
    cy, hh = (HY0 + HY1) / 2, (HY1 - HY0) / 2 * pop
    hf = hh * (1 + 0.16 * math.sin(th))

    def uv(u, v):
        x = xh + (xf - xh) * u
        top = (cy - hh) + ((cy - hf) - (cy - hh)) * u
        bot = (cy + hh) + ((cy + hf) - (cy + hh)) * u
        return (x, top + (bot - top) * v)

    return uv


def draw_astronaut_doorway(layer, t, ay):
    d = ImageDraw.Draw(layer, "RGBA")
    ox = (HX1 - HX0) // 2
    white, shade = (244, 246, 250, 255), (200, 204, 216, 255)
    d.rounded_rectangle([ox - 95, 205 + ay, ox + 95, 355 + ay], radius=26, fill=shade)
    d.rounded_rectangle([ox - 80, 215 + ay, ox + 80, 470 + ay], radius=40, fill=white)
    d.rectangle([ox - 34, 262 + ay, ox + 34, 312 + ay], fill=(60, 70, 90, 255))
    for i, col in enumerate(((230, 60, 60, 255), (240, 200, 60, 255), (80, 200, 110, 255))):
        d.ellipse([ox - 24 + i * 20, 322 + ay, ox - 14 + i * 20, 332 + ay], fill=col)
    d.rounded_rectangle([ox - 118, 235 + ay, ox - 70, 390 + ay], radius=22, fill=(236, 239, 246, 255))
    ang = math.radians(20 + 20 * math.sin(t * 6)) if t > 5.2 else math.radians(14)
    sh = (ox + 82, 250 + ay)
    tip = (sh[0] + 92 * math.sin(ang), sh[1] - 92 * math.cos(ang))
    d.line([sh, tip], fill=(236, 239, 246, 255), width=46)
    d.ellipse([tip[0] - 27, tip[1] - 27, tip[0] + 27, tip[1] + 27], fill=(120, 126, 140, 255))
    d.ellipse([ox - 78, 90 + ay, ox + 78, 244 + ay], fill=white)
    d.ellipse([ox - 56, 120 + ay, ox + 56, 218 + ay], fill=(28, 36, 64, 255))
    d.ellipse([ox - 42, 128 + ay, ox - 8, 160 + ay], fill=(120, 170, 230, 200))
    d.ellipse([ox + 10, 162 + ay, ox + 36, 186 + ay], fill=(60, 120, 210, 230))


def phase_hatch(t, fi):
    rng = random.Random(fi)
    img = vgrad((1, 2, 8), (14, 14, 22))
    draw_stars(img, 1.0, t, ymax=HORIZON_D - 40)
    paste_center(img, EARTH_SMALL, 860, 250)
    lunar_ground(img, HORIZON_D)
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([540 - 460, CONTACT_D - 20, 540 + 460, CONTACT_D + 40], fill=(8, 8, 12, 150))
    img.paste(HULL, (150, 0), HULL)
    d = ImageDraw.Draw(img, "RGBA")

    # state
    theta = 108 * smooth((t - 2.0) / 2.4)
    pop = 1 + 0.035 * smooth((t - 1.2) / 0.8) if theta < 0.5 else 1.0
    li = smooth(theta / 60)
    ext = smooth((t - 4.3) / 1.0)
    ay = lerp(200, 0, smooth((t - 4.0) / 1.2))

    # status LED
    lx, ly = 820, 960
    d.rounded_rectangle([lx - 24, ly - 24, lx + 24, ly + 24], radius=8, fill=(50, 56, 70, 255))
    on = t > 2.0 or int(t * 2) % 2 == 0
    col = (80, 220, 120, 255) if t > 2.0 else (230, 60, 60, 255)
    d.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=col if on else (70, 40, 40, 255))

    # frame + interior
    d.rounded_rectangle([HX0 - 16, HY0 - 16, HX1 + 16, HY1 + 16], radius=40, fill=(130, 138, 156, 255))
    ow, oh = HX1 - HX0, HY1 - HY0
    if theta > 0.5:
        inner = vgrad((255, 238, 190), (235, 170, 90), h=oh).crop((0, 0, ow, oh))
        idr = ImageDraw.Draw(inner, "RGBA")
        for k, a in enumerate((60, 70, 80)):
            idr.ellipse([ow / 2 - 150 + k * 40, oh / 2 - 190 + k * 50, ow / 2 + 150 - k * 40, oh / 2 + 190 - k * 50],
                        fill=(255, 255, 235, a))
        idr.rectangle([70, 40, ow - 70, 110], fill=(150, 110, 70, 255))
        for i, c in enumerate(((230, 80, 70, 255), (90, 210, 120, 255), (250, 210, 80, 255))):
            idr.ellipse([100 + i * 50, 66, 118 + i * 50, 84], fill=c)
        if t > 4.0:
            draw_astronaut_doorway(inner, t, ay)
        mask = Image.new("L", (ow, oh), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, ow, oh], radius=28, fill=255)
        img.paste(inner, (HX0, HY0), mask)
        d = ImageDraw.Draw(img, "RGBA")

    # light spill
    if li > 0.01:
        d.polygon([(HX0 + 10, HY1), (HX1 - 10, HY1), (HX1 + 90, CONTACT_D), (HX0 - 90, CONTACT_D)],
                  fill=(255, 225, 150, int(50 * li)))
        d.polygon([(HX0 - 90, CONTACT_D), (HX1 + 90, CONTACT_D), (HX1 + 300, H), (HX0 - 300, H)],
                  fill=(255, 222, 140, int(80 * li)))

    # door
    uv = door_geometry(theta, pop)
    quad = [uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1)]
    front = theta < 90
    d.polygon(quad, fill=(214, 219, 232, 255) if front else (168, 174, 190, 255), outline=(110, 118, 136, 255))
    if front:
        d.polygon([uv(0.10, 0.06), uv(0.90, 0.06), uv(0.90, 0.94), uv(0.10, 0.94)], fill=(196, 202, 218, 255))
        for rr, colr in ((1.0, (70, 80, 100, 255)), (0.72, (96, 176, 236, 255))):
            ring = [uv(0.5 + 0.18 * rr * math.cos(a / 12 * math.pi), 0.30 + 0.125 * rr * math.sin(a / 12 * math.pi))
                    for a in range(24)]
            d.polygon(ring, fill=colr)
        d.polygon([uv(0.80, 0.47), uv(0.90, 0.47), uv(0.90, 0.55), uv(0.80, 0.55)], fill=(80, 88, 106, 255))

    # ladder
    if ext > 0.01:
        yb = lerp(HY1, CONTACT_D, ext)
        xl_t, xr_t = HX0 + 60, HX1 - 60
        f = (yb - HY1) / (CONTACT_D - HY1)
        xl_b, xr_b = xl_t - 40 * f, xr_t + 40 * f
        d.line([(xl_t, HY1), (xl_b, yb)], fill=(176, 182, 196, 255), width=10)
        d.line([(xr_t, HY1), (xr_b, yb)], fill=(176, 182, 196, 255), width=10)
        n = int((yb - HY1) / 44)
        for i in range(1, n + 1):
            yy = HY1 + i * 44
            ff = (yy - HY1) / (CONTACT_D - HY1)
            d.line([(xl_t - 40 * ff, yy), (xr_t + 40 * ff, yy)], fill=(196, 202, 214, 255), width=6)
        if ext > 0.98:
            for x in (xl_b, xr_b):
                d.ellipse([x - 20, CONTACT_D - 6, x + 20, CONTACT_D + 8], fill=(150, 156, 170, 255))

    # vapour at the seal
    for i in range(26):
        ts = 1.2 + i * 0.05
        age = t - ts
        if age < 0 or age > 2.2:
            continue
        vr = random.Random(i)
        side = vr.choice([-1, 1])
        x0 = HX0 if side < 0 else HX1
        y0 = vr.uniform(HY0, HY1)
        px = x0 + side * (20 + 90 * age * vr.uniform(0.5, 1.2))
        py = y0 - 30 * age + vr.uniform(-10, 10)
        r = 14 + 34 * age
        d.ellipse([px - r, py - r, px + r, py + r], fill=(240, 244, 255, int(130 * clamp(1 - age / 2.2))))
    return img


# ---------------------------------------------------------------- phase: moonwalk
LX = 300.0                       # lander x
SC_L, SC_A = 1.6, 0.72           # lander / astronaut scale (cartoon proportions)
HATCH_BOTTOM_Y = 1441.0
FLAG_X = 880.0
AX_FLAG = FLAG_X - 80.0
EARTH_M = scaled(EARTH, 175 / 380)


def astro_state(t):
    """x, feet-y, pose, facing, gait phase for the moonwalk."""
    if t < 4.2:
        u = smooth(clamp((t - 1.0) / 3.2))
        return lerp(LX - 6, LX + 96, u), lerp(HATCH_BOTTOM_Y, CONTACT_C, u), "descend", -1, u * 3.0
    if t < 5.4:
        return lerp(LX + 96, LX + 150, smooth((t - 4.2) / 1.2)), CONTACT_C, "stand", 1, 0.0
    if t < 9.0:
        x = lerp(LX + 150, AX_FLAG, smooth(clamp((t - 5.4) / 3.6)))
        return x, CONTACT_C, "walk", 1, x / 130.0
    if t < 11.4:
        return AX_FLAG, CONTACT_C, "plant", 1, 0.0
    return AX_FLAG, CONTACT_C, "wave", 1, 0.0


def _print_times():
    out = [(LX + 96, 4.25)]
    for k in range(8):
        xk = LX + 150 + 58 * k
        for i in range(0, 900):
            tt = 4.2 + i * 0.01
            if astro_state(tt)[0] >= xk:
                out.append((xk, tt))
                break
    return out


FOOTPRINTS = _print_times()
MOON_CAPTIONS = [
    (0.4, 4.0, "The astronaut climbs down the ladder."),
    (4.3, 7.6, "One small step onto the Moon."),
    (7.8, 9.1, "The first footprints."),
    (9.3, 11.4, "Planting the flag."),
]


def phase_moonwalk(t, fi):
    rng = random.Random(fi)
    img = vgrad((1, 2, 8), (18, 18, 26))
    draw_stars(img, 1.0, t, ymax=HORIZON_C - 40)
    paste_center(img, EARTH_M, 830, 900)
    lunar_ground(img, HORIZON_C)
    d = ImageDraw.Draw(img, "RGBA")

    # lander with the hatch open
    d.ellipse([LX - 260, CONTACT_C - 6, LX + 260, CONTACT_C + 36], fill=(10, 10, 14, 110))
    paste_center(img, rocket_layer(ROCKET_LEGS, 0, 1.4, rng, 0, SC_L), LX, CONTACT_C + 6 - 250 * SC_L)
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([LX - 42, 1294, LX + 42, HATCH_BOTTOM_Y], radius=10, fill=(255, 224, 150, 255),
                        outline=(120, 128, 146, 255), width=5)
    d.rectangle([LX - 36, 1300, LX + 36, 1330], fill=(210, 160, 90, 255))

    # ladder
    for ax_, bx_ in (((LX - 28, 1441), (LX + 74, CONTACT_C)), ((LX + 16, 1441), (LX + 118, CONTACT_C))):
        d.line([ax_, bx_], fill=(176, 182, 196, 255), width=9)
    for k in range(1, 5):
        f = k / 5
        d.line([(LX - 28 + 102 * f, 1441 + (CONTACT_C - 1441) * f), (LX + 16 + 102 * f, 1441 + (CONTACT_C - 1441) * f)],
               fill=(196, 202, 214, 255), width=7)

    # footprints
    for k, (xk, tk) in enumerate(FOOTPRINTS):
        if t >= tk:
            y = CONTACT_C + 6 + (9 if k % 2 else 0)
            d.ellipse([xk - 22, y - 7, xk + 22, y + 7], fill=(56, 56, 66, 170))
            d.ellipse([xk - 14, y - 4, xk + 10, y + 3], fill=(92, 92, 102, 120))
            d.arc([xk - 22, y - 7, xk + 22, y + 7], 180, 330, fill=(170, 170, 180, 150), width=2)

    # flag
    if t > 9.2:
        draw_flag(img, FLAG_X, CONTACT_C, 0.9, smooth((t - 9.2) / 1.6))

    # astronaut
    ax, ay, pose, face, ph = astro_state(t)
    k = smooth(t / 0.8)
    if k > 0.01:
        if ay >= CONTACT_C - 2:
            d.ellipse([ax - 60, CONTACT_C - 4, ax + 60, CONTACT_C + 24], fill=(10, 10, 14, 120))
        draw_astronaut(img, ax, ay, SC_A * k, pose, ph, face, t)

    for t0, t1, txt in MOON_CAPTIONS:
        draw_caption(img, txt, 250, 48, clamp(min((t - t0) / 0.35, (t1 - t) / 0.35)))
    a = clamp((t - 12.0) / 0.6)
    if a > 0:
        draw_text(img, 540, 470, "MISSION\nCOMPLETE", 120, (255, 244, 200), 7, (60, 50, 30), a)
        draw_trophy(img, 470, 735, 0.8 * smooth(a), t)
    return img


PHASE_FUNCS = [phase_launch, phase_orbit_earth, phase_transit, phase_orbit_moon, phase_landing, phase_hatch, phase_moonwalk]


# ------------------------------------------------------------------ main loop
def render_frame(t, fi):
    idx = len(PHASES) - 1
    for i, (_, t0, t1) in enumerate(PHASES):
        if t < t1:
            idx = i
            break
    _, t0, _ = PHASES[idx]
    img = PHASE_FUNCS[idx](t - t0, fi)
    if idx > 0 and t - t0 < XFADE:
        pn, pa, pb = PHASES[idx - 1]
        prev = PHASE_FUNCS[idx - 1](pb - pa, fi)
        img = Image.blend(prev, img, smooth((t - t0) / XFADE))
    img = ImageChops.multiply(img, VIGNETTE)
    if t < 0.5:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, smooth(t / 0.5))
    return img


def phase_name(t):
    for name, _, t1 in PHASES:
        if t < t1:
            return name
    return PHASES[-1][0]


def main():
    ap = argparse.ArgumentParser(description="2D rocket mission animation")
    ap.add_argument("--out", default="out/rocket_mission.mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=TOTAL, help="end time in seconds")
    ap.add_argument("--still", default="", help="comma list of times; saves PNG stills and exits")
    ap.add_argument("--frames-dir", default="", help="write a PNG sequence here instead of an mp4")
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

    spent = {}
    for i in range(n):
        t = args.start + i / args.fps
        t0 = time.time()
        frame = render_frame(t, i + int(args.start * args.fps))
        if proc:
            proc.stdin.write(frame.tobytes())
        else:
            frame.save(Path(args.frames_dir) / f"frame_{i:05d}.png")
        spent[phase_name(t)] = spent.get(phase_name(t), 0.0) + time.time() - t0
        if i % 30 == 0:
            print(f"frame {i + 1}/{n}  t={t:5.1f}s  [{phase_name(t)}]", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()

    total = time.time() - t_all
    print("\nRender time per phase:")
    for name, secs in spent.items():
        print(f"  {name:8s} {secs:7.1f} s")
    print(f"Total: {total:.1f} s for {n} frames ({n / args.fps:.1f} s of video, {total / max(n, 1):.3f} s/frame)")
    print("Output:", args.frames_dir or out)


if __name__ == "__main__":
    main()
