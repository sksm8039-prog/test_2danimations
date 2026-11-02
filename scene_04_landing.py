#!/usr/bin/env python3
"""
scene_04_landing.py - lunar orbit to touchdown in ONE continuous shot.

The camera starts far out looking at the whole Moon with the ship orbiting, then dives in and chases the ship down
to the dust. The Moon is ray-traced at every height (sciart_world.LunarView), and every number comes from
mission_sim.build_descent(): de-orbit burn, coast to the braking point, then a smooth powered descent whose thrust
is derived from the motion. The ship decelerates while the engine fires; nothing stops, nothing cuts.

  python scene_04_landing.py --still 1,5,12,16,22,30,40,50
  python scene_04_landing.py
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

D = ms.build_descent()
T, R_, TH, VR, VT, TR, TT, STG = (D[k] for k in ("T", "R", "TH", "VR", "VT", "TR", "TT", "STG"))
RM, G_M = ms.R_MOON, 1.62
TH_SITE, T_PDI, T_TD = D["th_site"], D["t_pdi"], float(T[-1])
T_END = T_TD + 8.0
A_MAX = 3.6

# --- time-lapse: real seconds per screen second, slow where the action is
_K = [(0, 3.0), (10, 3.0), (40, 420.0), (3300, 420.0), (T_PDI, 50.0), (T_PDI + 260, 50.0), (T_PDI + 540, 24.0),
      (T_PDI + 680, 10.0), (T_PDI + 730, 4.0), (T_TD, 2.5), (T_END, 1.5)]
_kt, _kk = zip(*_K)
_t = np.arange(0.0, T_END + 0.25, 0.25)
_s = np.concatenate([[0.0], np.cumsum(0.25 / np.interp(_t[:-1] + 0.125, _kt, _kk))])
TOTAL = float(_s[-1])


def t_of_s(s):
    return float(np.interp(s, _s, _t))


S_PDI = float(np.interp(T_PDI, _t, _s))


def state(t):
    f = lambda a: float(np.interp(t, T, a))
    return f(R_), f(TH), f(VR), f(VT), f(TR), f(TT)


def ship_site(r, th):
    phi = th - TH_SITE
    return np.array([r * math.sin(phi), r * math.cos(phi), 0.0]), phi


# --- ground-contact time for the dust (when the ship centre drops below 35 m)
_alt = R_ - RM
T_DUST0 = float(T[np.nonzero(_alt <= 35.0)[0][0]])
_dr = random.Random(21)
DUST = [(_dr.uniform(0, 1) ** 0.6, _dr.uniform(0, 2 * math.pi), _dr.uniform(14, 85), _dr.uniform(0.15, 1.6), _dr.uniform(0.5, 1.0))
        for _ in range(1800)]


# --- camera path (precomputed at 120 Hz so it is smooth and stateless)
F_PX, D_FAR = 900.0, 4.35e6


def _cam_params(s):
    t = t_of_s(s)
    r, th, vr, vt, tr, tt = state(t)
    P, phi = ship_site(r, th)
    alt = max(r - RM, 5.0)
    w = smooth((s - (S_PDI - 5.0)) / 6.0)
    Dc = max(36.0, 3.0 * alt)
    Dist = math.exp((1 - w) * math.log(D_FAR) + w * math.log(Dc))
    tgt = smooth(w * 4.0) * P            # slide over to the ship early so the camera never dips into the Moon
    radial = np.array([math.sin(phi), math.cos(phi), 0.0])
    up = (1 - w) * np.array([0.0, 1.0, 0.0]) + w * radial
    up /= np.linalg.norm(up)
    cam = tgt + np.array([0.0, 0.0, -Dist])
    return cam, up, lerp(960.0, 860.0, w), w, Dist, P, phi


def _build_attitude(fps=120):
    out, ang, k = [], -90.0, 1 - math.exp(-1.0 / fps / 0.6)
    for i in range(int(TOTAL * fps) + 2):
        t = t_of_s(i / fps)
        r, th, vr, vt, tr, tt = state(t)
        if math.hypot(tr, tt) > 0.15:
            tgt = math.degrees(math.atan2(tt, tr))
            while tgt - ang > 180:
                tgt -= 360
            while tgt - ang < -180:
                tgt += 360
            ang += (tgt - ang) * k
        out.append(ang)
    return out


ATT = _build_attitude()


def att_at(s):
    f = clamp(s, 0, TOTAL) * 120
    i = int(f)
    return lerp(ATT[i], ATT[min(i + 1, len(ATT) - 1)], f - i)


# --- the picture
VIEW = sw.LunarView()
EARTH_IMG = sw.Planet("earth").render(70, 40, 23, (-0.62, -0.30, 0.72))
SC_DEG = 12.0


def project(P, cam, fwd, up, cy):
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    up2 = np.cross(fwd, right)
    d = np.asarray(P) - cam
    z = d @ fwd
    if z < 1.0:
        return None
    return 540 + F_PX * (d @ right) / z, cy - F_PX * (d @ up2) / z, z


def label(img, x, y, text, size, fill=(255, 255, 255), a=1.0):
    draw_text(img, x, y, text, size, fill, 3, (0, 0, 0), a)


CAPTIONS = [
    (0.3, 3.4, "De-orbit burn: a gentle push drops the orbit"),
    (3.8, S_PDI - 1.2, "Half an orbit of coasting to the braking point"),
    (S_PDI - 0.6, S_PDI + 6.5, "Powered descent: the engine slows 1,700 m/s to zero"),
    (S_PDI + 8.0, S_PDI + 15.0, "Pitching upright for the final approach"),
    (S_PDI + 16.5, TOTAL - 7.0, "Hovering down to the dust"),
    (TOTAL - 4.5, TOTAL - 0.1, "Touchdown"),
]


def render_frame(s, fi):
    rng = random.Random(fi)
    t = t_of_s(s)
    r, th, vr, vt, tr, tt = state(t)
    cam, up, cy, w, Dist, P, phi = _cam_params(s)
    fwd = np.array([0.0, 0.0, 1.0])
    img = sa.vgrad((2, 3, 10), (6, 8, 18), h=H, w=W)
    sa.draw_stars(img, 1.0, s)
    sa.paste_center(img, EARTH_IMG, 920, 520)
    moon = VIEW.render(cam, fwd, up, F_PX, cy=cy, q=0.5)
    img.paste(moon, (0, 0), moon)
    d = ImageDraw.Draw(img, "RGBA")

    alt = r - RM
    # lunar dust: ballistic sheets (no air, so no billowing: every grain flies a clean parabola)
    if t >= T_DUST0:
        for u, ang, vh, vy, sz in DUST:
            ts = T_DUST0 + u * (T_END - T_DUST0 - 1.0)
            tau = t - ts
            if tau < 0:
                continue
            ground_y = RM + 0.0
            x = vh * math.cos(ang) * tau
            z = vh * math.sin(ang) * tau
            y = vy * tau - 0.5 * G_M * tau * tau
            if y < 0:
                y, x, z = 0.0, vh * math.cos(ang) * (2 * vy / G_M), vh * math.sin(ang) * (2 * vy / G_M)
            pt = np.array([x, ground_y + y, z])
            Pw = np.array([0.0, 0.0, 0.0]) + pt
            Pw[1] = RM + y
            pr = project(Pw, cam, fwd, up, cy)
            if pr and -50 < pr[0] < W + 50 and -50 < pr[1] < H + 50:
                rad = max(1.8, F_PX * 0.45 * sz / pr[2])
                age = clamp(1 - tau / 13.0)
                d.ellipse([pr[0] - rad, pr[1] - rad * 0.8, pr[0] + rad, pr[1] + rad * 0.8], fill=(206, 204, 200, int(160 * age)))

    # the ship (with its shadow on the ground once it is low)
    pr = project(P, cam, fwd, up, cy)
    if alt < 400:
        gp = project(P / np.linalg.norm(P) * RM, cam, fwd, up, cy)
        if gp:
            wd = F_PX * 9.0 / gp[2]
            sa_ = int(150 * clamp(1 - (alt - 6) / 400))
            d.ellipse([gp[0] - wd, gp[1] - wd * 0.16, gp[0] + wd, gp[1] + wd * 0.16], fill=(5, 5, 8, sa_))
    thr = math.hypot(tr, tt) if t < T_TD else 0.0
    chem = 420.0 * clamp(thr / A_MAX)
    ang = att_at(s) + (1 - w) * math.degrees(phi)
    px_h = clamp(F_PX * 12.0 / Dist, 90.0, 330.0)
    sc = px_h / 520.0
    deploy = 1 - smooth(s / 1.5)
    if pr:
        sx, sy = pr[0], pr[1]
        if px_h < 160:
            rr = 54 + 4 * math.sin(s * 6)
            d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(255, 255, 255, 110), width=2)
        layer = sw.tug_layer(deploy, chem, 0.0, rng, ang, sc, legs=s > 1.6)
        sa.paste_center(img, layer, sx, sy)

    # readouts
    stg = int(np.interp(t, T, STG.astype(float)) + 0.5)
    touch = t >= T_TD - 0.2
    status = ("TOUCHDOWN", (140, 255, 170)) if touch else {1: ("DE-ORBIT BURN", (255, 170, 90)), 2: ("COASTING", (200, 220, 255))}.get(
        stg, ("POWERED DESCENT", (255, 170, 90)))
    label(img, 540, 170, f"T+{int(t // 60):02d}:{int(t % 60):02d}", 96, (255, 236, 170))
    label(img, 540, 270, status[0], 50, status[1])
    alt_txt = f"{alt / 1e3:,.1f} km" if alt >= 1000 else f"{max(alt - 5.8, 0):,.0f} m"
    sp = math.hypot(vr, vt)
    for cxp, lab, val in ((190, "ALTITUDE", alt_txt), (540, "SPEED", f"{sp:,.0f} m/s"), (890, "THRUST", f"{thr:.1f} m/s\u00b2")):
        label(img, cxp, 1745, val, 44)
        label(img, cxp, 1690, lab, 28, (170, 190, 230))
    for t0, t1, txt in CAPTIONS:
        sa.draw_caption(img, txt, 1580, 42, clamp(min((s - t0) / 0.35, (t1 - s) / 0.35)))
    img = ImageChops.multiply(img, VIGNETTE)
    if s < 0.4:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, smooth(s / 0.4))
    return img


if __name__ == "__main__":
    sa.run_cli(render_frame, TOTAL, "out/scene04_landing.mp4", "Scene 4: lunar orbit to touchdown")
