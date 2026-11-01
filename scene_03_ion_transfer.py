#!/usr/bin/env python3
"""
scene_03_ion_transfer.py - Earth orbit to lunar orbit with an ion-drive tug, in ONE continuous shot.

Every position, speed, flame and clock reading comes from mission_sim.build_mission(): real two-body gravity with
the actual thrust, so nothing stops before it fires and nothing cuts.

  0 LEO lap        one lap of low Earth orbit (92 min)
  1 chemical burn  raises the orbit (flame)
  2 coast          up the transfer ellipse
  3 chemical burn  circularises at high orbit, solar panels unfold
  4 ion spiral     weeks of tiny thrust spiral the ship out toward the Moon (speeded clock)
  5 ion spiral in  ship flips, the Moon captures it, spirals down to a 100 km orbit

  python scene_03_ion_transfer.py --still 1,3,6,9,14,20,22,26,31
  python scene_03_ion_transfer.py
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw

import mission_sim as ms
import sciart as sa
import sciart_world as sw
from sciart import H, VIGNETTE, W, clamp, draw_text, lerp, smooth

# ----------------------------------------------------------------- the flight
M = ms.build_mission()
T, X, Y, VX, VY, THR, STG, TH = (M[k] for k in ("T", "X", "Y", "VX", "VY", "THR", "STG", "TH"))
SEG, MOON, DAY = M["seg"], M["M"], ms.DAY
T_ARR, PHI_ARR, W_M = M["t_arrive"], M["phi_arr"], 2 * math.pi / (27.32 * DAY)

DUR = {0: 2.4, 1: 1.1, 2: 4.4, 3: 0.9, 4: 12.0, 5: 10.0}
HOLD = 1.5
S, _acc = {}, 0.0
for _st in range(6):
    S[_st] = (_acc, _acc + DUR[_st])
    _acc += DUR[_st]
TOTAL = _acc + HOLD
S_SW = S[4][1] - 1.2                       # camera starts moving over to the Moon


def t_of_s(s):
    """Screen seconds -> real flight seconds. Spirals show equal time per lap, so the clock speeds up."""
    for st in range(6):
        a, b = S[st]
        if s < b or st == 5:
            u = clamp((s - a) / (b - a))
            i0, i1 = SEG[st]
            if st in (4, 5):
                return float(np.interp(lerp(TH[i0], TH[i1], u), TH[i0:i1 + 1], T[i0:i1 + 1]))
            return lerp(T[i0], T[i1], u)
    return float(T[-1])


def state(t):
    i = min(int(np.searchsorted(T, t)), len(T) - 1)
    return (float(np.interp(t, T, X)), float(np.interp(t, T, Y)), float(np.interp(t, T, VX)),
            float(np.interp(t, T, VY)), float(np.interp(t, T, THR)), int(STG[i]), float(np.interp(t, T, TH)))


def moon_pos(t):
    if t >= T_ARR:
        return MOON
    phi = PHI_ARR - W_M * (T_ARR - t)
    return (ms.D_MOON * math.cos(phi), ms.D_MOON * math.sin(phi))


def _build_camera(fps=120):
    out, lnR = [], math.log(M["r1"])
    k = 1 - math.exp(-1.0 / fps / 0.35)
    for i in range(int(TOTAL * fps) + 2):
        s = i / fps
        x, y, *_ = state(t_of_s(s))
        if s < S[1][0]:
            ref = M["r1"]
        elif s < S[4][0]:
            ref = M["r_apo"]
        else:
            ref = max(M["r_apo"], math.hypot(x, y))
        w = smooth((s - S_SW) / 2.6)
        if s >= S_SW:
            rm = max(math.hypot(x - MOON[0], y - MOON[1]), 1e6)
            ref = math.exp((1 - w) * math.log(M["r_end"]) + w * math.log(rm))
        c = (w * MOON[0], w * MOON[1])
        tgt = max(ref, math.hypot(x - c[0], y - c[1]))
        lnR += (math.log(tgt) - lnR) * k
        out.append((c[0], c[1], math.exp(lnR)))
    return out


CAM = _build_camera()
TR_HZ = 240
_tr = [state(t_of_s(i / TR_HZ))[:2] for i in range(int(TOTAL * TR_HZ) + 2)]
TR_X, TR_Y = np.array([a for a, b in _tr]), np.array([b for a, b in _tr])


def cam_at(s):
    f = clamp(s, 0, TOTAL) * 120
    i = int(f)
    a, b = CAM[i], CAM[min(i + 1, len(CAM) - 1)]
    return tuple(lerp(a[j], b[j], f - i) for j in range(3))


# ---------------------------------------------------------------- the picture
EARTH, MOON_P = sw.Planet("earth"), sw.Planet("moon")
LIGHT = (-0.62, -0.30, 0.72)
CY = 960
STARS_IMG = None


def body(img, planet, radius_m, wx, wy, cx, cy, Z, rot, tilt, color):
    sx, sy = 540 + (wx - cx) * Z, CY - (wy - cy) * Z
    r = radius_m * Z
    if sx + r < -20 or sx - r > W + 20 or sy + r < -20 or sy - r > H + 20 or r > 1800:
        return
    if r < 9:
        d = ImageDraw.Draw(img, "RGBA")
        rr = max(r, 1.6)
        d.ellipse([sx - rr * 2.2, sy - rr * 2.2, sx + rr * 2.2, sy + rr * 2.2], fill=color + (60,))
        d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=color + (255,))
    else:
        sa.paste_center(img, planet.render(r, rot, tilt, LIGHT), sx, sy)


def label(img, x, y, text, size, fill=(255, 255, 255), a=1.0):
    draw_text(img, x, y, text, size, fill, 3, (0, 0, 0), a)


CAPTIONS = [
    (0.3, 2.3, "Low Earth orbit: one lap takes 92 minutes"),
    (2.6, 8.6, "A chemical burn raises the orbit"),
    (9.3, 19.0, "Ion drive: a tiny push, weeks of patience"),
    (19.5, 22.6, "The Moon captures the ship"),
    (22.9, 30.4, "Spiralling down to lunar orbit"),
    (30.9, TOTAL - 0.1, f"Day {int(T[-1] // DAY)}: lunar orbit reached"),
]


def render_frame(s, fi):
    rng = random.Random(fi)
    t = t_of_s(s)
    x, y, vx, vy, thr, stg, th = state(t)
    cx, cy, Rc = cam_at(s)
    Z = 380.0 / Rc
    img = sa.vgrad((2, 3, 10), (8, 10, 22), h=H, w=W)
    sa.draw_stars(img, 1.0, s)

    mx, my = moon_pos(t)
    body(img, EARTH, ms.R_EARTH, 0, 0, cx, cy, Z, (t / DAY) * 360.0, 23, (90, 150, 255))
    body(img, MOON_P, ms.R_MOON, mx, my, cx, cy, Z, 0, 0, (200, 200, 205))

    # glowing trail: bright when recent, a faint history that draws the spiral (dense, precomputed path)
    d = ImageDraw.Draw(img, "RGBA")
    i_end = min(int(s * TR_HZ), len(TR_X) - 1)
    i_start = max(0, i_end - 16 * TR_HZ)
    idx = np.arange(i_start, i_end + 1, 2)
    if len(idx) > 2:
        px = 540 + (TR_X[idx] - cx) * Z
        py = CY - (TR_Y[idx] - cy) * Z
        age = (i_end - idx) / TR_HZ
        pts = list(zip(px.tolist(), py.tolist()))
        n_chunks = 48
        step = max(2, len(pts) // n_chunks)
        for a in range(0, len(pts) - 1, step):
            seg = pts[a:a + step + 1]
            ag = float(age[min(a + step // 2, len(age) - 1)])
            al = int(34 + 200 * max(0.0, 1 - ag / 3.0) ** 1.6)
            d.line(seg, fill=(120, 200, 255, al), width=3 if ag < 3.0 else 2, joint="curve")

    # ship
    ang = math.degrees(math.atan2(vx, vy))
    flip = smooth((s - (S_SW + 0.4)) / 1.2)
    ang += 180.0 * flip
    chem = 420.0 * clamp(thr / M["a_chem"]) if stg in (1, 3) else 0.0
    ion = 0.0
    if stg in (4, 5) and s >= S[4][0] + 0.3:
        ion = 1.0
        if s >= S_SW + 0.5:
            ion = clamp(smooth((S_SW + 0.9 - s) / 0.3) + smooth((s - (S_SW + 1.7)) / 0.3))
    deploy = smooth((s - (S[3][1] - 0.1)) / 1.0)
    sx, sy = 540 + (x - cx) * Z, CY - (y - cy) * Z
    d = ImageDraw.Draw(img, "RGBA")
    rr = 36 + 4 * math.sin(s * 6)
    d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(255, 255, 255, 110), width=2)
    sa.paste_center(img, sw.tug_layer(deploy, chem, ion, rng, ang, 0.24), sx, sy)

    # clock and readouts
    day, rem = int(t // DAY), t % DAY
    label(img, 540, 170, f"DAY {day}", 112, (255, 236, 170))
    label(img, 540, 270, f"{int(rem // 3600):02d}:{int(rem % 3600 // 60):02d}:{int(rem % 60):02d}", 58)
    turning = stg in (4, 5) and S_SW + 0.4 < s < S_SW + 1.8
    status = {0: ("PARKING ORBIT", (200, 220, 255)), 1: ("CHEMICAL BURN", (255, 170, 90)), 2: ("COASTING", (200, 220, 255)),
              3: ("CHEMICAL BURN", (255, 170, 90))}.get(stg, ("ION DRIVE  0.5 mm/s\u00b2", (140, 200, 255)))
    if turning:
        status = ("TURNING AROUND", (255, 220, 140))
    label(img, 540, 350, status[0], 44, status[1])
    sp = math.hypot(vx, vy) / 1e3
    if stg <= 1:
        dist = ("ALTITUDE", f"{(math.hypot(x, y) - ms.R_EARTH) / 1e3:,.0f} km")
    elif stg <= 4:
        dist = ("FROM EARTH", f"{math.hypot(x, y) / 1e3:,.0f} km")
    else:
        dist = ("MOON ALTITUDE", f"{(math.hypot(x - MOON[0], y - MOON[1]) - ms.R_MOON) / 1e3:,.0f} km")
    orb = f"{int(th // (2 * math.pi)) + 1}"
    for cxp, lab, val in ((190, "SPEED", f"{sp:.2f} km/s"), (540, dist[0], dist[1]), (890, "MOON ORBIT" if stg == 5 else "ORBIT", orb)):
        label(img, cxp, 1745, val, 44)
        label(img, cxp, 1690, lab, 28, (170, 190, 230))
    for t0, t1, txt in CAPTIONS:
        sa.draw_caption(img, txt, 1580, 42, clamp(min((s - t0) / 0.35, (t1 - s) / 0.35)))
    img = ImageChops.multiply(img, VIGNETTE)
    if s < 0.4:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, smooth(s / 0.4))
    return img


if __name__ == "__main__":
    sa.run_cli(render_frame, TOTAL, "out/scene03_ion_transfer.mp4", "Scene 3: ion tug transfer, Earth to Moon orbit")
