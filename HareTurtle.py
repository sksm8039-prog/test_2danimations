#!/usr/bin/env python3
"""
render_tortoise_hare.py - "The Tortoise and the Hare" in 2D, built on sciart.py

A side-scrolling race in a vertical 1080x1920 frame (36 s, 30 fps):
  0.0- 3.5  the boastful hare mocks the slow tortoise
  3.5- 5.5  Ready... Set... Go!
  5.5- 8.8  the hare sprints far ahead
  8.8-12.2  he laughs, then settles under a tree
 12.2-25.4  he naps while the sun crosses the sky; the tortoise plods past
 25.4-28.0  the hare wakes in a panic and sprints, too late
 27.5-32.0  the tortoise breaks the ribbon, confetti, trophy
 32.0-36.0  moral card

No network, no API keys. Pure local rendering.

  python render_tortoise_hare.py --still 3,9,18,27,34   # PNG stills
  python render_tortoise_hare.py                        # full video -> out/tortoise_hare.mp4
"""
from __future__ import annotations

import math
import random

from PIL import Image, ImageChops, ImageDraw

from sciart import (H, VIGNETTE, W, clamp, draw_bubble, draw_caption, draw_hare, draw_text, draw_tortoise,
                    draw_trophy, lerp, mix, rot, run_cli, smooth, vgrad)

TOTAL = 36.0

# ------------------------------------------------------------- story timing
T_GO = 5.5
X_T0, X_H0 = 300.0, 420.0
X_TREE, X_NAP = 1860.0, 1810.0
X_TAPE = 3150.0
T_FINISH = 27.3                      # tortoise head touches the tape
V_T = (3000.0 - X_T0) / (27.5 - T_GO)
LANE_H, LANE_T = 1535.0, 1615.0      # ground contact y (hare runs behind the tortoise's lane)
SC_H, SC_T = 1.08, 1.22


def tortoise_x(t):
    if t < T_GO:
        return X_T0
    if t < 27.5:
        return X_T0 + V_T * (t - T_GO)
    return 3000.0 + 60.0 * (1 - math.exp(-(t - 27.5) * 2.0))


def hare_x(t):
    if t < T_GO:
        return X_H0
    if t < 8.8:
        s = (t - T_GO) / 3.3
        return lerp(X_H0, 1750.0, s * s * (3 - 2 * s))
    if t < 10.8:
        return 1750.0
    if t < 12.2:
        return lerp(1750.0, X_NAP, smooth((t - 10.8) / 1.4))
    if t < 26.0:
        return X_NAP
    if t < 28.0:
        return lerp(X_NAP, 3000.0, ((t - 26.0) / 2.0) ** 1.15)
    return 3000.0 + 90.0 * (1 - math.exp(-(t - 28.0) * 3.0))


def hare_pose(t):
    """pose, facing, gait stride (px per step cycle), crouch"""
    if t < 4.0:
        return "stand", 1, 260, 0.0
    if t < T_GO:
        return "stand", 1, 260, smooth((t - 4.0) / 0.6) * 0.9
    if t < 8.8:
        return "run", 1, 260, 0.0
    if t < 10.8:
        return "laugh", -1, 260, 0.0
    if t < 12.2:
        return "walk", 1, 70, 0.0
    if t < 25.4:
        return "sleep", 1, 260, 0.0
    if t < 26.0:
        return "surprised", 1, 260, 0.0
    if t < 28.1:
        return "run", 1, 260, 0.0
    if t < 28.5:
        return "stand", 1, 260, 0.0
    return "sad", 1, 260, 0.0


def _build_cam():
    n, cam, out, k = int(TOTAL * 120) + 2, 360.0, [], 2.4
    for i in range(n):
        t = i / 120
        target = 360.0 if t < 5.0 else min(max(hare_x(t), tortoise_x(t)) + 170, 3040.0)
        cam += (target - cam) * (1 - math.exp(-k / 120))
        out.append(cam)
    return out


CAM = _build_cam()


def cam_at(t):
    f = clamp(t, 0, TOTAL) * 120
    i = int(f)
    return lerp(CAM[i], CAM[min(i + 1, len(CAM) - 1)], f - i)


# ------------------------------------------------------------------ scenery
_r = random.Random(12)
TREES = [(x + _r.uniform(-80, 80), _r.uniform(0.8, 1.3)) for x in range(-200, 3900, 330)]
BUSHES = [(_r.uniform(-200, 3900), _r.uniform(0.7, 1.2)) for _ in range(34)]
FLOWERS = [(_r.uniform(-300, 3900), _r.uniform(1700, 1890), _r.choice([(255, 90, 120), (255, 220, 70),
                                                                        (255, 255, 255), (170, 120, 255)]))
           for _ in range(150)]
BLADES = [(x + _r.uniform(-20, 20), _r.uniform(60, 120)) for x in range(-300, 3900, 55)]
CLOUDS = [(_r.uniform(-400, 3800), _r.uniform(380, 900), _r.uniform(0.8, 1.6)) for _ in range(12)]
CONFETTI = [(_r.uniform(-1.0, 1.0), _r.uniform(-1500, -600), _r.choice([(255, 90, 120), (255, 220, 70), (90, 200, 255),
                                                                        (140, 255, 140), (255, 140, 60)]),
             _r.uniform(0, 6.28), _r.uniform(8, 16)) for _ in range(140)]


def sky_colors(t):
    keys = [(0.0, (120, 190, 250), (205, 234, 255)), (14.0, (70, 150, 240), (170, 220, 255)),
            (26.0, (120, 140, 215), (255, 196, 140)), (36.0, (70, 60, 120), (250, 150, 110))]
    for (t0, a0, b0), (t1, a1, b1) in zip(keys, keys[1:]):
        if t <= t1:
            u = smooth((t - t0) / (t1 - t0))
            return mix(a0, a1, u), mix(b0, b1, u)
    return keys[-1][1], keys[-1][2]


def hills(d, cam, p, base, amp, freq, ph, color):
    pts = [(-40, H)]
    for sxp in range(-40, W + 81, 40):
        wx = cam + (sxp - 540) / p
        y = base - amp * (0.55 + 0.45 * math.sin(wx * freq + ph)) - amp * 0.35 * math.sin(wx * freq * 2.7 + ph * 1.7)
        pts.append((sxp, y))
    pts.append((W + 80, H))
    d.polygon(pts, fill=color)


def draw_tree(d, sxp, base, s, color=(58, 130, 62)):
    d.rounded_rectangle([sxp - 14 * s, base - 130 * s, sxp + 14 * s, base], radius=6, fill=(110, 76, 48))
    for dx, dy, r in ((0, -170, 74), (-48, -140, 56), (50, -142, 58), (0, -215, 52)):
        d.ellipse([sxp + (dx - r) * s, base + (dy - r) * s, sxp + (dx + r) * s, base + (dy + r) * s], fill=color)
    d.ellipse([sxp - 36 * s, base - 215 * s, sxp + 10 * s, base - 180 * s], fill=(86, 160, 84))


def draw_sign(d, sxp, base, text, col):
    d.rectangle([sxp - 8, base - 300, sxp + 8, base], fill=(120, 90, 60))
    d.rounded_rectangle([sxp - 150, base - 330, sxp + 150, base - 240], radius=16, fill=col, outline=(255, 255, 255), width=5)
    from sciart import get_font
    d.text((sxp, base - 285), text, font=get_font(54), fill=(255, 255, 255), anchor="mm")


def scenery(img, t, cam):
    d = ImageDraw.Draw(img, "RGBA")
    ang = math.radians(lerp(14, 166, clamp((t - 1.0) / 29.0)))
    sx_, sy_ = 540 - 470 * math.cos(ang), 980 - 760 * math.sin(ang)
    warm = clamp((t - 18) / 12)
    for m, a in ((3.4, 22), (2.5, 34), (1.8, 55), (1.3, 90)):
        r = 80 * m
        d.ellipse([sx_ - r, sy_ - r, sx_ + r, sy_ + r], fill=(255, int(240 - 70 * warm), int(170 - 60 * warm), a))
    d.ellipse([sx_ - 80, sy_ - 80, sx_ + 80, sy_ + 80], fill=(255, int(246 - 40 * warm), int(200 - 80 * warm), 255))
    for bx, by, s in CLOUDS:
        cx = 540 + ((bx - t * 26 - cam * 0.08) % 4200) - 1000
        for dx, dy, rw, rh in ((0, 0, 120, 44), (-90, 10, 80, 34), (90, 12, 90, 36), (-30, -26, 76, 40), (44, -20, 64, 32)):
            d.ellipse([cx + (dx - rw) * s, by + (dy - rh) * s, cx + (dx + rw) * s, by + (dy + rh) * s],
                      fill=(255, 255, 255, 215))
    for k in range(3):  # a few birds
        bx = 100 + ((t * 70 + k * 400) % 1400) - 150
        by = 520 + 90 * k + 18 * math.sin(t * 2.2 + k)
        w = 22 * math.sin(t * 9 + k)
        d.line([(bx - 24, by + w), (bx, by), (bx + 24, by + w)], fill=(60, 60, 80, 220), width=4)
    hills(d, cam, 0.15, 1250, 230, 0.004, 0.0, mix((150, 190, 215), (190, 170, 190), clamp((t - 18) / 12)) + (255,))
    hills(d, cam, 0.35, 1345, 150, 0.006, 1.7, (118, 176, 112, 255))
    hills(d, cam, 0.6, 1445, 66, 0.010, 3.1, (104, 170, 84, 255))
    for wx, s in TREES:
        sxp = 540 + (wx - cam) * 0.6
        if -150 < sxp < W + 150:
            draw_tree(d, sxp, 1452, s * 0.9, (50, 120, 58))
    d.rectangle([0, 1445, W, H], fill=(112, 178, 86))
    for wx, s in BUSHES:
        sxp = 540 + (wx - cam) * 0.85
        if -120 < sxp < W + 120:
            for dx, r in ((-30, 30), (0, 38), (32, 30)):
                d.ellipse([sxp + (dx - r) * s, 1478 - 2 * r * s * 0.9, sxp + (dx + r) * s, 1478], fill=(66, 140, 66))
    # signs + nap tree stand just behind the road
    sxp = 540 + (250 - cam)
    if -200 < sxp < W + 200:
        draw_sign(d, sxp, 1482, "START", (200, 60, 60))
    sxp = 540 + (X_TAPE + 60 - cam)
    if -200 < sxp < W + 200:
        draw_sign(d, sxp, 1482, "FINISH", (50, 100, 200))
    sxp = 540 + (X_TREE - cam)
    if -400 < sxp < W + 400:
        d.rounded_rectangle([sxp - 48, 1040, sxp + 48, 1496], radius=14, fill=(112, 78, 50))
        for k in range(6):
            d.line([(sxp - 30 + k * 12, 1100 + k * 30), (sxp - 24 + k * 12, 1180 + k * 30)], fill=(86, 58, 38, 160), width=4)
        for dx, dy, r in ((0, 0, 150), (-130, 50, 120), (130, 50, 120), (-60, -100, 110), (70, -90, 110), (0, -170, 90)):
            d.ellipse([sxp + dx - r, 980 + dy - r, sxp + dx + r, 980 + dy + r], fill=(50, 122, 58))
        for dx, dy, r in ((-60, -40, 60), (50, -110, 46), (-120, 40, 40)):
            d.ellipse([sxp + dx - r, 980 + dy - r, sxp + dx + r, 980 + dy + r], fill=(76, 150, 78))
    # road
    d.rectangle([0, 1480, W, 1672], fill=(208, 178, 128))
    d.rectangle([0, 1480, W, 1488], fill=(184, 154, 106))
    d.rectangle([0, 1664, W, 1672], fill=(184, 154, 106))
    for k in range(-2, 40):
        sxp = 540 + (k * 150 - cam)
        if -150 < sxp < W + 150:
            d.rounded_rectangle([sxp, 1572, sxp + 70, 1580], radius=4, fill=(226, 200, 156, 200))
    sxp = 540 + (240 - cam)
    d.rectangle([sxp, 1488, sxp + 12, 1664], fill=(255, 255, 255, 230))
    sxp = 540 + (X_TAPE - cam)
    for r in range(6):
        for c in range(2):
            col = (30, 30, 36, 255) if (r + c) % 2 == 0 else (250, 250, 250, 255)
            d.rectangle([sxp - 28 + c * 28, 1488 + r * 29.3, sxp + c * 28, 1488 + (r + 1) * 29.3], fill=col)
    # shade under the nap tree
    sxp = 540 + (X_TREE - cam)
    d.ellipse([sxp - 260, 1488, sxp + 200, 1600], fill=(20, 40, 20, 70))


def dust(img, t, cam, xfun, lane, windows):
    d = ImageDraw.Draw(img, "RGBA")
    for t0, t1 in windows:
        ts = t0 + math.ceil(max(0.0, t - 1.4 - t0) / 0.04) * 0.04
        while ts < min(t1, t):
            age = t - ts
            sxp = 540 + (xfun(ts) - 30 - cam)
            r = 16 + age * 46
            y = lane - 12 - age * 38
            d.ellipse([sxp - r, y - r * 0.7, sxp + r, y + r * 0.7], fill=(228, 206, 170, int(120 * (1 - age / 1.4))))
            ts += 0.04


def ribbon_and_posts(img, t, cam):
    d = ImageDraw.Draw(img, "RGBA")
    sxp = 540 + (X_TAPE - cam)
    for base, h, w in ((1490, 130, 12), (1676, 150, 14)):
        d.rounded_rectangle([sxp - w / 2, base - h, sxp + w / 2, base], radius=4, fill=(250, 250, 250, 255))
        d.ellipse([sxp - w, base - h - w, sxp + w, base - h + w], fill=(230, 60, 60, 255))
    top_b, top_f = (sxp, 1490 - 125), (sxp, 1676 - 145)
    if t < T_FINISH:
        d.line([top_b, top_f], fill=(235, 60, 70, 255), width=9)
    else:
        age = clamp(t - T_FINISH)
        for (px, py), side in ((top_b, -1), (top_f, 1)):
            mid = (px + side * 16 * age, py + 40 + 70 * age)
            end = (px + side * 30 * age, py + 90 + 140 * age)
            d.line([(px, py), mid, end], fill=(235, 60, 70, int(255 * (1 - 0.6 * age))), width=9)


def confetti(img, t):
    if t < T_FINISH:
        return
    d = ImageDraw.Draw(img, "RGBA")
    age = t - T_FINISH
    for vx, vy, col, ph, sz in CONFETTI:
        if age > 4.6:
            return
        x = 560 + vx * 420 * (1 - math.exp(-age * 1.3)) + 22 * math.sin(age * 5 + ph)
        y = 1250 + vy * (1 - math.exp(-age * 1.6)) * 0.45 + 420 * age * age * 0.6
        a = int(255 * clamp(1.0 - (age - 3.2) / 1.4))
        pts = [rot(p, math.degrees(ph + age * 7)) for p in ((-sz, -sz * 0.5), (sz, -sz * 0.5), (sz, sz * 0.5), (-sz, sz * 0.5))]
        d.polygon([(x + px, y + py) for px, py in pts], fill=col + (a,))


def foreground(img, t, cam):
    d = ImageDraw.Draw(img, "RGBA")
    for wx, y, col in FLOWERS:
        sxp = 540 + (wx - cam) * 1.1
        if -30 < sxp < W + 30:
            d.line([(sxp, y), (sxp, y + 34)], fill=(60, 130, 60, 255), width=4)
            for k in range(5):
                a = k * 1.2566
                d.ellipse([sxp + 9 * math.cos(a) - 6, y + 9 * math.sin(a) - 6, sxp + 9 * math.cos(a) + 6, y + 9 * math.sin(a) + 6],
                          fill=col + (255,))
            d.ellipse([sxp - 5, y - 5, sxp + 5, y + 5], fill=(255, 190, 40, 255))
    for wx, hgt in BLADES:
        sxp = 540 + (wx - cam) * 1.5
        if -60 < sxp < W + 60:
            sway = 8 * math.sin(t * 2 + wx * 0.05)
            d.polygon([(sxp - 16, H), (sxp + 16, H), (sxp + sway, H - hgt - 40)], fill=(54, 120, 52, 255))


def zzz(img, t, cam, head):
    d = ImageDraw.Draw(img, "RGBA")
    from sciart import get_font
    for k in range(6):
        ts = 12.6 + k * 0.9
        age = (t - ts) % 5.4 if 12.6 <= t < 25.4 else None
        if age is None or age > 2.4:
            continue
        x = head[0] + 30 + age * 26 + 8 * math.sin(age * 4)
        y = head[1] - 40 - age * 70
        size = int(40 + age * 18)
        a = int(255 * clamp(1 - age / 2.4))
        d.text((x, y), "Z", font=get_font(size), fill=(255, 255, 255, a), anchor="mm", stroke_width=3, stroke_fill=(70, 90, 160, a))


CAPTIONS = [
    (0.4, 3.4, "A boastful hare meets a slow, steady tortoise."),
    (3.4, 5.5, "They agree to race."),
    (5.9, 10.6, "The hare sprints far ahead..."),
    (10.8, 17.0, "...and takes a nap, sure he will win."),
    (17.0, 25.2, "The tortoise never stops."),
    (25.4, 27.4, "The hare wakes up too late!"),
    (28.0, 31.8, "Slow and steady wins the race!"),
]


def overlay_text(img, t, cam, hx, tx):
    sh, st = 540 + (hx - cam), 540 + (tx - cam)
    hpose, hface, _, _ = hare_pose(t)
    hhead = (sh + 74 * hface * SC_H, LANE_H - 132 * SC_H)
    thead = (st + 125 * SC_T, LANE_T - 98 * SC_T)

    def pop(t0, t1, fade=0.25):
        return clamp(min((t - t0) / fade, (t1 - t) / fade))

    a = pop(0.8, 3.3)
    draw_bubble(img, hhead[0] + 40, hhead[1] - 260, "I'm the fastest\nanimal alive!", 44, (hhead[0], hhead[1] - 60), a)
    a = pop(2.4, 4.0)
    draw_bubble(img, thead[0] - 40, thead[1] - 300, "Let's race, then.", 44, (thead[0], thead[1] - 60), a)
    a = pop(8.9, 10.7)
    draw_bubble(img, hhead[0] - 60, hhead[1] - 260, "Ha ha ha!\nSo slow!", 44, (hhead[0], hhead[1] - 60), a)
    a = pop(25.5, 26.7)
    draw_bubble(img, hhead[0] - 40, hhead[1] - 250, "Oh no!!", 52, (hhead[0], hhead[1] - 60), a)
    a = pop(28.6, 31.6)
    draw_bubble(img, thead[0] - 140, thead[1] - 380, "Slow and steady!", 46, (thead[0] - 30, thead[1] - 220), a)

    for t0, t1, txt in CAPTIONS:
        draw_caption(img, txt, 250, 48, pop(t0, t1, 0.35))
    if t < 3.4:
        draw_text(img, 540, 520, "The Tortoise\nand the Hare", 104, (255, 244, 200), 6, (60, 50, 30), clamp(1 - (t - 2.6) / 0.7) * clamp(t / 0.4))
    for t0, label, size in ((3.6, "Ready...", 120), (4.35, "Set...", 130), (5.0, "GO!", 190)):
        age = t - t0
        if 0 <= age < 0.75:
            k = 0.8 + 0.3 * smooth(age / 0.2)
            draw_text(img, 540, 760, label, int(size * k), (255, 255, 255), 8, (200, 60, 60), clamp(1 - (age - 0.5) / 0.25))


def moral_card(img, t):
    a = smooth((t - 32.0) / 0.9)
    if a <= 0:
        return
    ov = Image.new("RGBA", (W, H), (12, 16, 40, int(215 * a)))
    img.paste(ov, (0, 0), ov)
    draw_trophy(img, 540, 620, 1.35, t)
    draw_text(img, 540, 930, "MORAL", 56, (255, 210, 90), 0, alpha=a)
    draw_text(img, 540, 1130, "Slow and steady\nwins the race.", 92, (255, 255, 255), 4, (30, 40, 90), a)
    draw_text(img, 540, 1480, "The Tortoise and the Hare", 44, (190, 205, 255), 0, alpha=clamp((t - 32.8) / 0.6))


def render_frame(t, fi):
    cam = cam_at(t)
    c_top, c_bot = sky_colors(t)
    img = vgrad(c_top, c_bot, h=1500).crop((0, 0, W, 1500))
    full = vgrad(c_bot, c_bot)
    full.paste(img, (0, 0))
    img = full
    scenery(img, t, cam)
    hx, tx = hare_x(t), tortoise_x(t)
    dust(img, t, cam, hare_x, LANE_H, ((T_GO, 8.8), (26.0, 28.0)))
    ribbon_and_posts(img, t, cam)

    d = ImageDraw.Draw(img, "RGBA")
    pose, face, stride, crouch = hare_pose(t)
    sh = 540 + (hx - cam)
    st = 540 + (tx - cam)
    for sxp, lane, w in ((sh, LANE_H, 150), (st, LANE_T, 190)):
        if pose == "sleep" and lane == LANE_H:
            w = 200
        d.ellipse([sxp - w, lane - 12, sxp + w, lane + 18], fill=(20, 40, 20, 80))
    draw_hare(img, sh, LANE_H, SC_H, pose, hx / stride, face, t, crouch)
    if pose == "sleep":
        zzz(img, t, cam, (sh + 90 * SC_H, LANE_H - 40 * SC_H))
    mood = "cheer" if t >= 27.5 else "walk"
    draw_tortoise(img, st, LANE_T, SC_T, tx / 150.0, mood, 1, t)
    if t >= 28.0:
        k = clamp((t - 28.0) / 0.5)
        pop = 1 + 0.25 * math.sin(k * math.pi) if k < 1 else 1.0
        draw_trophy(img, st + 20, LANE_T - 300 * SC_T / 1.22 - 60 + 8 * math.sin(t * 3), 0.55 * pop * smooth(k), t)

    confetti(img, t)
    foreground(img, t, cam)
    overlay_text(img, t, cam, hx, tx)
    img = ImageChops.multiply(img, VIGNETTE)
    moral_card(img, t)
    if t < 0.4:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, smooth(t / 0.4))
    return img


if __name__ == "__main__":
    run_cli(render_frame, TOTAL, "out/tortoise_hare.mp4", "The Tortoise and the Hare (2D)")
