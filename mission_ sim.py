#!/usr/bin/env python3
"""
mission_sim.py - continuous-thrust flight physics for the Earth -> Moon mission

Ion thrusters give a tiny, steady push (never a big burn), so the ship does not 'fire and coast'.
It spirals outward around Earth over months, then spirals inward around the Moon.
Every number here comes from integrating real two-body gravity with a small thrust along (or against) the velocity,
so the flame/beam, the orbit shape, the speed and the clock all agree with each other.

  python mission_sim.py                 # prints the mission report
  python mission_sim.py --plot out.png  # draws both spirals
"""
from __future__ import annotations

import argparse
import math

import numpy as np

MU_EARTH, MU_MOON = 3.986004e14, 4.9028e12
R_EARTH, R_MOON = 6.371e6, 1.7374e6
D_MOON = 3.844e8            # Earth-Moon distance, m
DAY = 86400.0


def _acc(mu, p, v, a_thr, sign):
    r = math.hypot(p[0], p[1])
    vn = math.hypot(v[0], v[1])
    g = -mu / r ** 3
    t = sign * a_thr / vn
    return (g * p[0] + t * v[0], g * p[1] + t * v[1])


def spiral(mu, r0, r1, a_thr, outward=True, steps_per_orbit=360):
    """Integrate a constant-acceleration low-thrust spiral between circular radii r0 and r1.
    Returns dict of arrays: t (s), r (m), theta (rad, unwrapped), v (m/s)."""
    sign = 1.0 if outward else -1.0
    p, v = [r0, 0.0], [0.0, math.sqrt(mu / r0)]
    t, th = 0.0, 0.0
    T, R, TH, V = [0.0], [r0], [0.0], [v[1]]
    prev = math.atan2(p[1], p[0])
    while True:
        r = math.hypot(*p)
        if (outward and r >= r1) or (not outward and r <= r1) or t > 900 * DAY:
            break
        dt = 2 * math.pi * math.sqrt(r ** 3 / mu) / steps_per_orbit
        k1p, k1v = v, _acc(mu, p, v, a_thr, sign)
        p2 = [p[i] + 0.5 * dt * k1p[i] for i in range(2)]
        v2 = [v[i] + 0.5 * dt * k1v[i] for i in range(2)]
        k2p, k2v = v2, _acc(mu, p2, v2, a_thr, sign)
        p3 = [p[i] + 0.5 * dt * k2p[i] for i in range(2)]
        v3 = [v[i] + 0.5 * dt * k2v[i] for i in range(2)]
        k3p, k3v = v3, _acc(mu, p3, v3, a_thr, sign)
        p4 = [p[i] + dt * k3p[i] for i in range(2)]
        v4 = [v[i] + dt * k3v[i] for i in range(2)]
        k4p, k4v = v4, _acc(mu, p4, v4, a_thr, sign)
        p = [p[i] + dt / 6 * (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i]) for i in range(2)]
        v = [v[i] + dt / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) for i in range(2)]
        t += dt
        ang = math.atan2(p[1], p[0])
        d = ang - prev
        if d < -math.pi:
            d += 2 * math.pi
        elif d > math.pi:
            d -= 2 * math.pi
        th += d
        prev = ang
        T.append(t)
        R.append(math.hypot(*p))
        TH.append(th)
        V.append(math.hypot(*v))
    return {"t": np.array(T), "r": np.array(R), "theta": np.array(TH), "v": np.array(V)}


def propagate(mu, p, v, t, accel, stop, steps_per_orbit=360, dt_cap=None):
    """RK4 two-body integrator. accel(t,p,v) -> (ax, ay, thrust_magnitude). Returns records and final state."""
    rec = []
    p, v = list(p), list(v)

    def rhs(p_, v_, t_):
        g = -mu / math.hypot(*p_) ** 3
        ax, ay, thr = accel(t_, p_, v_)
        return v_, (g * p_[0] + ax, g * p_[1] + ay), thr

    for _ in range(3000000):
        r = math.hypot(*p)
        dt = 2 * math.pi * math.sqrt(r ** 3 / mu) / steps_per_orbit
        if dt_cap:
            dt = min(dt, dt_cap)
        k1p, k1v, thr = rhs(p, v, t)
        rec.append((t, p[0], p[1], v[0], v[1], thr))
        if stop(t, p, v):
            break
        p2 = [p[i] + 0.5 * dt * k1p[i] for i in range(2)]
        v2 = [v[i] + 0.5 * dt * k1v[i] for i in range(2)]
        k2p, k2v, _ = rhs(p2, v2, t + dt / 2)
        p3 = [p[i] + 0.5 * dt * k2p[i] for i in range(2)]
        v3 = [v[i] + 0.5 * dt * k2v[i] for i in range(2)]
        k3p, k3v, _ = rhs(p3, v3, t + dt / 2)
        p4 = [p[i] + dt * k3p[i] for i in range(2)]
        v4 = [v[i] + dt * k3v[i] for i in range(2)]
        k4p, k4v, _ = rhs(p4, v4, t + dt)
        p = [p[i] + dt / 6 * (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i]) for i in range(2)]
        v = [v[i] + dt / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) for i in range(2)]
        t += dt
    return rec, (t, p, v)


def _prograde(a):
    def f(t, p, v):
        n = math.hypot(*v)
        return a * v[0] / n, a * v[1] / n, a
    return f


def _retro(a):
    def f(t, p, v):
        n = math.hypot(*v)
        return -a * v[0] / n, -a * v[1] / n, a
    return f


def _coast(t, p, v):
    return 0.0, 0.0, 0.0


def build_mission(a_ion=5e-4, a_chem=10.0, leo_alt=400e3, geo=42164e3, moon_gap=30000e3):
    """One continuous flight, Earth frame then Moon frame, as arrays.
    stages: 0 LEO lap, 1 chemical burn (raise), 2 coast to apogee, 3 chemical burn (circularise),
            4 ion spiral out, 5 ion spiral in around the Moon."""
    r1 = R_EARTH + leo_alt
    mu = MU_EARTH
    a_tr = (r1 + geo) / 2
    eps = lambda p, v: 0.5 * (v[0] ** 2 + v[1] ** 2) - mu / math.hypot(*p)
    T1 = 2 * math.pi * math.sqrt(r1 ** 3 / mu)
    pieces, state = [], ([r1, 0.0], [0.0, math.sqrt(mu / r1)], 0.0)

    def run(stage, mu_, accel, stop, **kw):
        nonlocal state
        p, v, t = state
        rec, (t, p, v) = propagate(mu_, p, v, t, accel, stop, **kw)
        pieces.append((stage, rec[1:] if pieces else rec))
        state = (p, v, t)

    run(0, mu, _coast, lambda t, p, v: t >= T1)
    run(1, mu, _prograde(a_chem), lambda t, p, v: eps(p, v) >= -mu / (2 * a_tr), dt_cap=4.0)
    run(2, mu, _coast, lambda t, p, v: p[0] * v[0] + p[1] * v[1] <= 0)
    run(3, mu, _prograde(a_chem), lambda t, p, v: eps(p, v) >= -mu / (2 * math.hypot(*p)), dt_cap=4.0)
    r_apo = max(max(math.hypot(r[1], r[2]) for r in rec) for st, rec in pieces)
    r_end = D_MOON - moon_gap
    run(4, mu, _prograde(a_ion), lambda t, p, v: math.hypot(*p) >= r_end)
    p_end, v_end, t_end = state
    k = D_MOON / math.hypot(*p_end)
    M = (p_end[0] * k, p_end[1] * k)                    # the Moon, radially outward of the ship
    rel = (p_end[0] - M[0], p_end[1] - M[1])
    rr = math.hypot(*rel)
    vc = math.sqrt(MU_MOON / rr)
    vrel = (-rel[1] / rr * vc, rel[0] / rr * vc)
    state = (list(rel), list(vrel), t_end)
    run(5, MU_MOON, _retro(a_ion), lambda t, p, v: math.hypot(*p) <= R_MOON + 100e3)
    T, X, Y, VX, VY, THR, STG = [], [], [], [], [], [], []
    for stage, rec in pieces:
        for (t, x, y, vx, vy, thr) in rec:
            T.append(t)
            X.append(x + (M[0] if stage == 5 else 0.0))
            Y.append(y + (M[1] if stage == 5 else 0.0))
            VX.append(vx)
            VY.append(vy)
            THR.append(thr)
            STG.append(stage)
    T, X, Y = np.array(T), np.array(X), np.array(Y)
    VX, VY, THR, STG = np.array(VX), np.array(VY), np.array(THR), np.array(STG)
    # cumulative orbit angle: Earth frame for stages 0-4, Moon frame for stage 5
    ang = np.where(STG <= 4, np.arctan2(Y, X), np.arctan2(Y - M[1], X - M[0]))
    TH = np.zeros_like(T)
    for i in range(1, len(T)):
        d = ang[i] - ang[i - 1]
        if STG[i] != STG[i - 1]:
            d = 0.0
        d = (d + math.pi) % (2 * math.pi) - math.pi
        TH[i] = TH[i - 1] + d
    TH[STG == 5] -= TH[STG == 5][0]
    seg = {}
    for st in range(6):
        idx = np.nonzero(STG == st)[0]
        seg[st] = (int(idx[0]), int(idx[-1]))
    return {"T": T, "X": X, "Y": Y, "VX": VX, "VY": VY, "THR": THR, "STG": STG, "TH": TH, "seg": seg,
            "M": M, "r1": r1, "r_apo": r_apo, "r_end": r_end, "t_arrive": float(T[seg[4][1]]),
            "phi_arr": math.atan2(M[1], M[0]), "a_ion": a_ion, "a_chem": a_chem}


def _quintic(x0, v0, a0, x1, v1, a1, T):
    """Quintic polynomial hitting position, velocity and acceleration at t=0 and t=T."""
    c0, c1, c2 = x0, v0, a0 / 2
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([x1 - (c0 + c1 * T + c2 * T ** 2), v1 - (c1 + 2 * c2 * T), a1 - 2 * c2])
    c3, c4, c5 = np.linalg.solve(A, b)
    return (c0, c1, c2, c3, c4, c5)


def _poly(c, t):
    t = np.asarray(t, float)
    x = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
    v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
    a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t ** 2 + 20 * c[5] * t ** 3
    return x, v, a


def build_descent(alt0=100e3, alt_p=15e3, a_dorb=2.0, h_td=5.8, v_td=0.8, dt3=0.5):
    """Lunar landing from a circular orbit, as one continuous flight in Moon-centred polar coordinates.
    1 de-orbit burn (retro) -> 2 coast to periapsis -> 3 powered descent along a smooth path to the site.
    The thrust is DERIVED from the path (accel - gravity), so the engine fires exactly as hard as the motion needs."""
    mu, R = MU_MOON, R_MOON
    r0, rp = R + alt0, R + alt_p
    v_c = math.sqrt(mu / r0)
    v_a = math.sqrt(2 * mu * rp / (r0 * (r0 + rp)))
    p, v, t = [r0, 0.0], [0.0, v_c], 0.0
    rec1, (t, p, v) = propagate(mu, p, v, t, _retro(a_dorb), lambda t, p, v: math.hypot(*v) <= v_a, dt_cap=0.5)
    neg = [False]

    def stop2(t, p, v):
        rv = p[0] * v[0] + p[1] * v[1]
        neg[0] = neg[0] or rv < 0
        return neg[0] and rv >= 0

    rec2, (t, p, v) = propagate(mu, p, v, t, _coast, stop2, steps_per_orbit=720)
    rows = []
    for stg, rec in ((0, [(0.0, r0, 0.0, 0.0, v_c, 0.0)]), (1, rec1[1:]), (2, rec2[1:])):
        for (tt, x, y, vx, vy, thr) in rec:
            r = math.hypot(x, y)
            vr, vth = (x * vx + y * vy) / r, (x * vy - y * vx) / r
            sp = math.hypot(vx, vy)
            sign = -1.0 if stg == 1 else 0.0
            rows.append((tt, r, math.atan2(y, x), vr, vth, sign * thr * vr / sp, sign * thr * vth / sp, stg))
    rows[0] = (0.0, r0, 0.0, 0.0, v_c, 0.0, 0.0, 1)
    t_p, r_p, th_p, vr_p, vt_p = rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4]
    om0 = vt_p / r_p
    ar0 = vt_p ** 2 / r_p - mu / r_p ** 2
    best = None
    for T in range(420, 761, 20):
        for k in np.linspace(0.20, 0.60, 41):
            cr = _quintic(r_p, vr_p, ar0, R + h_td, -v_td, 0.0, T)
            cth = _quintic(0.0, om0, 0.0, k * om0 * T, 0.0, 0.0, T)
            ts = np.linspace(0, T, 400)
            r, rd, rdd = _poly(cr, ts)
            th, thd, thdd = _poly(cth, ts)
            if (r < R + h_td - 0.5).any():
                continue
            tr, tth = rdd - r * thd ** 2 + mu / r ** 2, r * thdd + 2 * rd * thd
            if (tr < -0.1).any():
                continue
            peak = float(np.max(np.hypot(tr, tth)))
            if best is None or peak < best[0]:
                best = (peak, T, k, cr, cth)
    peak, T3, k3, cr, cth = best
    ts = np.arange(0.0, T3 + 1e-9, dt3)
    r, rd, rdd = _poly(cr, ts)
    th, thd, thdd = _poly(cth, ts)
    tr, tth = rdd - r * thd ** 2 + mu / r ** 2, r * thdd + 2 * rd * thd
    th_unw = np.unwrap([row[2] for row in rows])
    th_p_u = th_unw[-1]
    for i in range(len(ts) - 1 if rows else len(ts)):
        rows.append((t_p + ts[i], float(r[i]), th_p_u + float(th[i]), float(rd[i]), float(r[i] * thd[i]), float(tr[i]), float(tth[i]), 3))
    rows.append((t_p + ts[-1], float(r[-1]), th_p_u + float(th[-1]), float(rd[-1]), float(r[-1] * thd[-1]), float(tr[-1]), float(tth[-1]), 3))
    arr = np.array(rows)
    arr[:len(th_unw), 2] = th_unw
    out = {k_: arr[:, i] for i, k_ in enumerate(("T", "R", "TH", "VR", "VT", "TR", "TT", "STG"))}
    out["STG"] = out["STG"].astype(int)
    out["th_site"] = float(th_p_u + th[-1])
    out["peak_thrust"], out["T_descent"], out["k"], out["t_pdi"] = peak, T3, k3, t_p
    return out


def build_ascent(leo_alt=400e3, h0=35.0, twr0=1.25, dt=0.5):
    """Pad to low Earth orbit as one continuous flight in Earth-centred polar coordinates.
    The path is a smooth curve that starts at rest on the pad (thrust-to-weight twr0) and ends in a circular orbit
    with the engine throttled smoothly to zero; the thrust is DERIVED from the path (accel - gravity)."""
    mu, R, g0 = MU_EARTH, R_EARTH, 9.80665
    r0, r1 = R + h0, R + leo_alt
    vc = math.sqrt(mu / r1)
    om1 = vc / r1
    best = None
    for T in range(500, 621, 20):
        for th1 in np.linspace(0.10, 0.50, 41):
            cr = _quintic(r0, 0.0, (twr0 - 1.0) * g0, r1, 0.0, 0.0, T)
            cth = _quintic(0.0, 0.0, 0.0, th1, om1, 0.0, T)
            ts = np.linspace(0, T, 500)
            r, rd, rdd = _poly(cr, ts)
            th, thd, thdd = _poly(cth, ts)
            if (np.diff(r) < -1e-6).any() or (r < r0 - 1).any() or (rd < -1.0).any():
                continue
            tr, tth = rdd - r * thd ** 2 + mu / r ** 2, r * thdd + 2 * rd * thd
            if (tr < 0.3 * g0).any() and False:
                continue
            mag = np.hypot(tr, tth)
            if (mag[:int(0.88 * len(mag))] < 1.1 * g0).any():   # out-push gravity until the final throttle-down
                continue
            peak = float(mag.max())
            if best is None or peak < best[0]:
                best = (peak, T, th1, cr, cth)
    if best is None:
        raise RuntimeError("no feasible ascent found")
    peak, T, th1, cr, cth = best
    ts = np.arange(0.0, T + 1e-9, dt)
    r, rd, rdd = _poly(cr, ts)
    th, thd, thdd = _poly(cth, ts)
    tr, tth = rdd - r * thd ** 2 + mu / r ** 2, r * thdd + 2 * rd * thd
    return {"T": ts, "R": r, "TH": th, "VR": rd, "VT": r * thd, "TR": tr, "TT": tth, "peak": peak / g0,
            "T_asc": float(T), "th_ins": float(th1), "r1": r1}


def report(a_thr=5e-4, leo_alt=400e3, start="LEO"):
    r0 = R_EARTH + leo_alt if start == "LEO" else 42164e3
    up = spiral(MU_EARTH, r0, 0.985 * D_MOON, a_thr)
    down = spiral(MU_MOON, 30000e3, R_MOON + 100e3, a_thr, outward=False)
    return r0, up, down


def plot(path, up, down):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1080, 540), (6, 8, 20))
    d = ImageDraw.Draw(img)
    for ox, data, rmax, label in ((270, up, D_MOON, "Earth spiral-out"), (810, down, 30000e3, "Moon spiral-in")):
        pts = []
        for r, th in zip(data["r"][::40], data["theta"][::40]):
            pts.append((ox + 240 * r / rmax * math.cos(th), 270 + 240 * r / rmax * math.sin(th)))
        d.line(pts, fill=(120, 200, 255), width=1)
        d.text((ox - 120, 12), label, fill=(255, 255, 255))
    img.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    for a_thr, label in ((2e-4, "SMART-1 class (70 mN, 370 kg)"), (5e-4, "bigger solar-electric tug"), (1.5e-3, "high-power future tug")):
        for start in ("LEO", "GEO"):
            r0, up, down = report(a_thr, start=start)
            print(f"a={a_thr:.0e} m/s^2 [{label}] start {start}: "
                  f"Earth spiral {up['t'][-1] / DAY:6.1f} d, {up['theta'][-1] / (2 * math.pi):6.1f} orbits | "
                  f"Moon spiral {down['t'][-1] / DAY:5.1f} d, {abs(down['theta'][-1]) / (2 * math.pi):5.1f} orbits")
    r0, up, down = report(5e-4, start="GEO")
    # rate of orbit angle on screen if the whole Earth spiral is shown in 14 s with the radius growing on a log scale
    print("GEO start, a=5e-4: speed at start/end (km/s):", round(up['v'][0] / 1e3, 2), round(up['v'][-1] / 1e3, 2))
    if args.plot:
        plot(args.plot, up, down)
