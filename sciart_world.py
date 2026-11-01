#!/usr/bin/env python3
"""
sciart_world.py - realistic bodies, perspective view and physics timing for sciart

  Planet(kind).render(radius, rot, tilt, light)   rotating, lit, textured spheres
                                                  kinds: earth | moon | mars | jupiter
  GroundCamera                                    perspective ground plane (depth, foreshortening)
  physics helpers                                 fall_time, hop_time, orbit_period, Units ...

  python sciart_world.py --demo out/world      # writes planets.png and perspective.png
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import sciart as sa
from sciart import clamp, lerp, smooth

# =================================================================== physics
G_EARTH, G_MOON = 9.81, 1.62                 # m/s^2
MU_EARTH, MU_MOON = 3.986004e14, 4.9028e12   # gravitational parameters, m^3/s^2
R_EARTH, R_MOON = 6.371e6, 1.7374e6          # mean radii, m


def fall_time(height_m, g):
    """Seconds for something dropped from rest to fall height_m."""
    return math.sqrt(2 * height_m / g)


def fall_speed(height_m, g):
    return math.sqrt(2 * g * height_m)


def hop_time(v0, g):
    """Seconds in the air for a jump leaving the ground at v0 m/s."""
    return 2 * v0 / g


def orbit_period(r_m, mu):
    """Circular orbit period at distance r_m from the centre (Kepler)."""
    return 2 * math.pi * math.sqrt(r_m ** 3 / mu)


def circular_speed(r_m, mu):
    return math.sqrt(mu / r_m)


class Units:
    """Connects metres and seconds to pixels and screen seconds for one scene.

    px_per_m   : how many pixels one metre is at the scene's main distance
    time_scale : real seconds per screen second (1 = real time, 60 = a minute per second)
    """

    def __init__(self, px_per_m, time_scale=1.0):
        self.px_per_m, self.time_scale = px_per_m, time_scale

    def px(self, metres):
        return metres * self.px_per_m

    def accel_px(self, g):
        """Gravity in px per screen-second squared."""
        return g * self.px_per_m * self.time_scale ** 2

    def screen_seconds(self, real_seconds):
        return real_seconds / self.time_scale


# ===================================================================== noise
def _hash(ix, iy, iz, seed):
    h = (ix * np.uint32(374761393)) + (iy * np.uint32(668265263)) + (iz * np.uint32(2147483647)) \
        + np.uint32((seed * 1013904223) % 4294967296)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return (h & np.uint32(0xFFFF)).astype(np.float32) / 65535.0


def _vnoise(x, y, z, seed):
    xf, yf, zf = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - xf, y - yf, z - zf
    fx, fy, fz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    xi = (xf.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    yi = (yf.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    zi = (zf.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    one = np.uint32(1)

    def c(a, b, d):
        return _hash(xi + a, yi + b, zi + d, seed)

    x00 = c(0, 0, 0) + (c(one, 0, 0) - c(0, 0, 0)) * fx
    x10 = c(0, one, 0) + (c(one, one, 0) - c(0, one, 0)) * fx
    x01 = c(0, 0, one) + (c(one, 0, one) - c(0, 0, one)) * fx
    x11 = c(0, one, one) + (c(one, one, one) - c(0, one, one)) * fx
    y0 = x00 + (x10 - x00) * fy
    y1 = x01 + (x11 - x01) * fy
    return y0 + (y1 - y0) * fz


def fbm(x, y, z, octaves=6, seed=0, lac=2.0, gain=0.5):
    total, amp, norm, f = 0.0, 1.0, 0.0, 1.0
    for o in range(octaves):
        total = total + amp * _vnoise(x * f, y * f, z * f, seed + o * 17)
        norm += amp
        amp *= gain
        f *= lac
    return total / norm


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-6)


def _ss(a, e0, e1):
    t = np.clip((a - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _mixc(c0, c1, t):
    return np.array(c0, np.float32) / 255 * (1 - t[..., None]) + np.array(c1, np.float32) / 255 * t[..., None]


def _sphere_grid(tw, th):
    lon = (np.arange(tw, dtype=np.float32) + 0.5) / tw * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(th, dtype=np.float32) + 0.5) / th * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    cl = np.cos(LAT)
    return cl * np.cos(LON), np.sin(LAT), cl * np.sin(LON), LAT, LON


# ================================================================== textures
def _craters(X, Y, Z, g, n, seed, rim=0.22, depth=0.2, rmin=0.012, rmax=0.11):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        rho = rmin + (rmax - rmin) * rng.random() ** 3.5
        d = X * c[0] + Y * c[1] + Z * c[2]
        sel = d > math.cos(min(1.7 * rho, 3.0))
        if not sel.any():
            continue
        ang = np.arccos(np.clip(d[sel], -1, 1))
        v = g[sel]
        inside = ang < rho
        v = np.where(inside, v * ((1 - depth) + depth * 0.9 * (ang / rho)), v)
        v = v + rim * np.exp(-((ang - rho) / (0.14 * rho)) ** 2) + 0.04 * np.exp(-((ang - 1.4 * rho) / (0.45 * rho)) ** 2)
        g[sel] = v
    return g


def _tex_earth(tw, th, seed):
    X, Y, Z, LAT, LON = _sphere_grid(tw, th)
    h = _norm01(fbm(X * 1.9 + 11, Y * 1.9, Z * 1.9, 7, seed))
    sea = float(np.quantile(h, 0.62))
    land = _ss(h, sea - 0.004, sea + 0.004)
    elev = np.clip((h - sea) / (1 - sea), 0, 1)
    m = _norm01(fbm(X * 2.6 + 40, Y * 2.6, Z * 2.6 + 9, 5, seed + 7))
    depth = np.clip((sea - h) / sea, 0, 1)
    ocean = _mixc((34, 118, 178), (6, 26, 78), depth ** 0.6)
    shelf = np.exp(-((h - (sea - 0.025)) / 0.03) ** 2)
    ocean = ocean * (1 - 0.5 * shelf[..., None]) + np.array([52, 160, 176], np.float32) / 255 * 0.5 * shelf[..., None]
    ground = _mixc((46, 108, 52), (112, 98, 66), np.clip(elev * 2.2, 0, 1))
    forest = np.clip((m - 0.52) * 3, 0, 1) * (1 - elev)
    ground = ground * (1 - 0.5 * forest[..., None]) + np.array([26, 78, 38], np.float32) / 255 * 0.5 * forest[..., None]
    desert = _ss(0.46 - m, 0, 0.1) * (np.abs(LAT) < 0.75)
    ground = ground * (1 - desert[..., None] * 0.85) + np.array([196, 168, 106], np.float32) / 255 * desert[..., None] * 0.85
    peak = _ss(elev, 0.62, 0.9)
    ground = ground * (1 - peak[..., None]) + np.array([214, 214, 218], np.float32) / 255 * peak[..., None]
    alb = ocean * (1 - land[..., None]) + ground * land[..., None]
    ice = _ss(np.abs(LAT) / (np.pi / 2) + 0.07 * (fbm(X * 6, Y * 6, Z * 6, 4, seed + 3) - 0.5), 0.80, 0.9)
    alb = alb * (1 - ice[..., None]) + np.array([236, 243, 250], np.float32) / 255 * ice[..., None]
    clouds = fbm(X * 3.2 + 70, Y * 5.5, Z * 3.2, 7, seed + 21)
    clouds = np.clip((_norm01(clouds) - 0.52) * 3.2, 0, 1) ** 1.15
    lights = np.clip((fbm(X * 14, Y * 14, Z * 14, 3, seed + 5) - 0.6) * 6, 0, 1) * land * (1 - ice) * (m > 0.38)
    return {"albedo": alb.astype(np.float32), "ocean": ((1 - land) * (1 - ice)).astype(np.float32),
            "clouds": clouds.astype(np.float32), "lights": lights.astype(np.float32)}


def _tex_moon(tw, th, seed):
    X, Y, Z, LAT, LON = _sphere_grid(tw, th)
    g = 0.46 + 0.30 * _norm01(fbm(X * 3, Y * 3, Z * 3, 6, seed))
    maria = _ss(_norm01(fbm(X * 2.2 + 5, Y * 2.2, Z * 2.2, 4, seed + 4)), 0.64, 0.72)
    g = g * (1 - 0.30 * maria)
    g = _craters(X, Y, Z, g, 190, seed + 9)
    alb = np.stack([g * 1.0, g * 0.99, g * 0.96], -1)
    return {"albedo": np.clip(alb, 0, 1).astype(np.float32)}


def _tex_mars(tw, th, seed):
    X, Y, Z, LAT, LON = _sphere_grid(tw, th)
    base = _norm01(fbm(X * 2.2, Y * 2.2, Z * 2.2, 6, seed))
    dark = _ss(_norm01(fbm(X * 1.6 + 9, Y * 1.6, Z * 1.6, 4, seed + 5)), 0.52, 0.62)
    dust = _ss(_norm01(fbm(X * 4 + 3, Y * 4, Z * 4, 5, seed + 11)), 0.45, 0.75)
    alb = _mixc((168, 88, 50), (206, 140, 92), dust * 0.8 + base * 0.2)
    alb = alb * (1 - 0.55 * dark[..., None]) + np.array([74, 44, 36], np.float32) / 255 * 0.55 * dark[..., None]
    g = alb.mean(-1)
    g2 = _craters(X, Y, Z, g.copy(), 70, seed + 13, rim=0.10, depth=0.16, rmax=0.08)
    alb = alb * (g2 / np.maximum(g, 1e-3))[..., None]
    cap = _ss(np.abs(LAT) - 1.36 + 0.05 * (fbm(X * 8, Y * 8, Z * 8, 3, seed + 2) - 0.5), 0, 0.08)
    alb = alb * (1 - cap[..., None]) + np.array([240, 240, 245], np.float32) / 255 * cap[..., None]
    return {"albedo": np.clip(alb, 0, 1).astype(np.float32)}


def _tex_jupiter(tw, th, seed):
    X, Y, Z, LAT, LON = _sphere_grid(tw, th)
    warp = fbm(X * 3, Y * 1.2, Z * 3, 5, seed)
    b = np.sin(LAT * 13 + 4.2 * (warp - 0.5)) * 0.5 + 0.5
    b2 = np.sin(LAT * 31 + 6 * (warp - 0.5) + 1.3) * 0.5 + 0.5
    pal = [(232, 214, 178), (200, 160, 112), (150, 98, 64), (214, 182, 140), (178, 100, 70)]
    t = np.clip(b * 0.75 + b2 * 0.25, 0, 0.9999) * (len(pal) - 1)
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    pa = np.array(pal, np.float32) / 255
    alb = pa[i] * (1 - f) + pa[np.minimum(i + 1, len(pal) - 1)] * f
    streak = (fbm(X * 9, Y * 40, Z * 9, 4, seed + 6) - 0.5) * 0.35
    alb = alb * (1 + streak[..., None])
    dl = (((LON - 0.7) / 0.30) ** 2 + ((LAT + 0.42) / 0.13) ** 2)
    spot = np.clip(1 - dl, 0, 1) ** 0.6
    ring = np.exp(-((np.sqrt(dl) - 1.05) / 0.12) ** 2) * 0.35
    alb = alb * (1 - spot[..., None]) + np.array([190, 82, 58], np.float32) / 255 * spot[..., None]
    alb = alb + ring[..., None] * 0.35
    return {"albedo": np.clip(alb, 0, 1).astype(np.float32)}


_BUILDERS = {"earth": _tex_earth, "moon": _tex_moon, "mars": _tex_mars, "jupiter": _tex_jupiter}
# kind -> (atmosphere rim strength, rim colour, halo strength)
_ATMOS = {"earth": (1.0, (0.40, 0.62, 1.0), 0.9), "moon": (0.0, (1, 1, 1), 0.0),
          "mars": (0.35, (0.85, 0.55, 0.40), 0.35), "jupiter": (0.30, (0.95, 0.82, 0.60), 0.30)}


# =================================================================== planets
class Planet:
    """A textured, lit, rotating sphere rendered as an RGBA sprite."""

    def __init__(self, kind="earth", tex=(1280, 640), seed=3):
        self.kind, self.tw, self.th = kind, tex[0], tex[1]
        self.tex = _BUILDERS[kind](tex[0], tex[1], seed)
        self._geo = {}

    def _geom(self, R, tilt, light):
        key = (R, round(tilt, 2), tuple(round(v, 3) for v in light))
        if key in self._geo:
            return self._geo[key]
        if len(self._geo) > 3:
            self._geo.clear()
        pad = int(R * 1.18) + 2
        S = 2 * pad
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        xx -= pad - 0.5
        yy -= pad - 0.5
        nx, ny = xx / R, yy / R
        r2 = nx * nx + ny * ny
        r = np.sqrt(r2)
        inside = r2 <= 1.0
        nz = np.sqrt(np.clip(1 - r2, 0, 1))
        L = np.array(light, np.float32)
        L /= np.linalg.norm(L)
        lam = nx * L[0] + ny * L[1] + nz * L[2]
        Hv = L + np.array([0, 0, 1], np.float32)
        Hv /= np.linalg.norm(Hv)
        spec = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1)
        a = math.radians(tilt)
        c, s = math.cos(a), math.sin(a)
        tx, ty = nx * c + ny * s, -nx * s + ny * c
        idx = np.nonzero(inside.ravel())[0]
        halo = np.nonzero((~inside & (r < 1.18)).ravel())[0]
        rf, nxf, nyf = r.ravel(), nx.ravel(), ny.ravel()
        lxy = max(math.hypot(L[0], L[1]), 1e-6)
        hr = (rf[halo] - 1) / 0.18
        dirdot = (nxf[halo] * L[0] + nyf[halo] * L[1]) / np.maximum(rf[halo], 1e-6) / lxy
        g = {"S": S, "idx": idx, "halo": halo, "R": R,
             "lat": np.arcsin(np.clip(-ty.ravel()[idx], -1, 1)), "lon0": np.arctan2(nz.ravel()[idx], tx.ravel()[idx]),
             "nz": nz.ravel()[idx], "lam": lam.ravel()[idx], "spec": spec.ravel()[idx], "r": rf[idx],
             "hr": hr, "hlit": 0.15 + 0.85 * np.clip(dirdot * 0.5 + 0.5, 0, 1) ** 1.3}
        self._geo[key] = g
        return g

    def _sample(self, arr, u, v):
        tw, th = self.tw, self.th
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)
        u1 = (u0 + 1) % tw
        u0 = u0 % tw
        v0 = np.clip(v0, 0, th - 1)
        v1 = np.clip(v0 + 1, 0, th - 1)
        w00, w10, w01, w11 = (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv
        if arr.ndim == 3:
            w00, w10, w01, w11 = w00[:, None], w10[:, None], w01[:, None], w11[:, None]
        return arr[v0, u0] * w00 + arr[v0, u1] * w10 + arr[v1, u0] * w01 + arr[v1, u1] * w11

    def render(self, radius, rot=0.0, tilt=0.0, light=(-0.62, -0.30, 0.72), cloud_rot=None):
        R = int(round(radius))
        g = self._geom(R, tilt, light)
        S, idx = g["S"], g["idx"]
        cr = rot * 1.08 + 12 if cloud_rot is None else cloud_rot
        lon = g["lon0"] + math.radians(rot)
        u = (lon + np.pi) / (2 * np.pi) * self.tw - 0.5
        v = np.clip((np.pi / 2 - g["lat"]) / np.pi * self.th - 0.5, 0, self.th - 1.001)
        col = self._sample(self.tex["albedo"], u, v)
        lam, nz = g["lam"], g["nz"]
        lam_w = np.clip((lam + 0.08) / 1.08, 0, 1)
        amb = 0.035
        diff = amb + (1 - amb) * lam_w ** 0.9
        col = col * diff[:, None]
        if self.kind == "earth":
            u_c = u + math.radians(cr - rot) / (2 * np.pi) * self.tw
            cl = self._sample(self.tex["clouds"], u_c, v)
            cs = self._sample(self.tex["clouds"], u_c - 0.006 * self.tw, v - 0.004 * self.th)
            col = col * (1 - 0.30 * cs[:, None])
            cloud_col = (amb + (1 - amb) * lam_w ** 0.8)[:, None] * 0.97
            col = col * (1 - cl[:, None]) + cloud_col * cl[:, None]
            ocean = self._sample(self.tex["ocean"], u, v)
            sp = (g["spec"] ** 170) * ocean * (1 - cl) * 0.38 * (lam > 0)
            col = col + sp[:, None] * np.array([1.0, 0.97, 0.9], np.float32)
            ls = self._sample(self.tex["lights"], u, v) * (1 - np.clip(lam * 4 + 0.3, 0, 1)) * (1 - cl)
            col = col + ls[:, None] * np.array([1.0, 0.78, 0.42], np.float32) * 0.5
        if self.kind == "jupiter":
            col = col * (0.55 + 0.45 * np.sqrt(nz))[:, None]
        rim_amt, rim_col, halo_amt = _ATMOS[self.kind]
        if rim_amt > 0:
            atm = (1 - nz) ** 3.0 * (0.12 + 0.88 * np.clip(lam + 0.30, 0, 1)) * rim_amt
            col = 1 - (1 - np.clip(col, 0, 1)) * (1 - atm[:, None] * np.array(rim_col, np.float32) * 1.1)
        col = (np.clip(col, 0, 1) * 255).astype(np.uint8)
        disc = np.zeros((S * S, 4), np.uint8)
        disc[idx, :3] = col
        disc[idx, 3] = (np.clip((1 - g["r"]) * g["R"], 0, 1) * 255).astype(np.uint8)
        out = Image.fromarray(disc.reshape(S, S, 4), "RGBA")
        if halo_amt > 0:
            h = np.zeros((S * S, 4), np.uint8)
            hid = g["halo"]
            h[hid, 0], h[hid, 1], h[hid, 2] = [int(255 * c) for c in rim_col]
            h[hid, 3] = (np.clip(np.exp(-g["hr"] * 3.4) * 0.85 * g["hlit"] * halo_amt, 0, 1) * 255).astype(np.uint8)
            halo = Image.fromarray(h.reshape(S, S, 4), "RGBA")
            halo.alpha_composite(out)
            out = halo
        return out


# ============================================================== perspective
class GroundCamera:
    """Pinhole camera looking along a flat ground plane.

    A point on the ground x metres right and z metres ahead lands at
        screen_x = cx + f*x/z      screen_y = horizon + f*cam_h/z
    so things shrink as 1/z, climb toward the horizon, and circles flatten to ellipses.
    """

    def __init__(self, w=sa.W, h=sa.H, horizon=0.56, f=900.0, cam_h=2.0):
        self.w, self.h, self.f, self.cam_h = w, h, f, cam_h
        self.cx, self.hy = w / 2, h * horizon

    def project(self, x, z):
        s = self.f / z
        return self.cx + x * s, self.hy + self.cam_h * s, s     # s = pixels per metre at that depth

    def visible_half_width(self, z):
        return self.w / 2 / self.f * z


# =============================================================== surfaces
def _pnoise2(ux, vy, P, seed):
    """Tileable 2D value noise: lattice wraps every P cells."""
    x, y = ux * P, vy * P
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = (x - xf).astype(np.float32), (y - yf).astype(np.float32)
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    xi, xj = (xf.astype(np.int64) % P).astype(np.uint32), ((xf.astype(np.int64) + 1) % P).astype(np.uint32)
    yi, yj = (yf.astype(np.int64) % P).astype(np.uint32), ((yf.astype(np.int64) + 1) % P).astype(np.uint32)
    z0 = np.zeros((1, 1), np.uint32)
    c00, c10 = _hash(xi, yi, z0, seed), _hash(xj, yi, z0, seed)
    c01, c11 = _hash(xi, yj, z0, seed), _hash(xj, yj, z0, seed)
    a = c00 + (c10 - c00) * fx
    b = c01 + (c11 - c01) * fx
    return a + (b - a) * fy


def fbm_p(ux, vy, P0, octaves, seed, gain=0.5):
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * _pnoise2(ux, vy, P0 * (2 ** o), seed + o * 13)
        norm += amp
        amp *= gain
    return total / norm


def sun_vector(el_deg, az_deg):
    """Unit vector pointing TOWARD the sun. x right, y up, z away from the camera.
    az 0 = sun directly behind the camera, positive = swings round to the camera's left."""
    el, az = math.radians(el_deg), math.radians(az_deg)
    return (-math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))


class Terrain:
    """A tileable patch of ground: height map -> craters, pebbles, regolith grain -> baked sunlight
    (including cast shadows) -> mipmaps, seen through a GroundCamera in true perspective."""

    PRESETS = {
        "moon": dict(base=(0.30, 0.298, 0.288), grain=0.30, craters=380, rmin=0.22, rmax=8.0, pebbles=3200,
                     sun_el=19.0, sun_az=58.0, amb=0.12, exposure=4.6, roll=0.35, ripples=0.0, rock=0.46),
        "mars": dict(base=(0.50, 0.28, 0.17), grain=0.26, craters=110, rmin=0.25, rmax=9.0, pebbles=6000,
                     sun_el=28.0, sun_az=55.0, amb=0.22, exposure=2.3, roll=1.2, ripples=0.028, rock=0.40),
    }

    def __init__(self, kind="moon", size_m=80.0, tex=2048, seed=7, blast=None):
        import time
        t0 = time.time()
        P = self.PRESETS[kind]
        self.kind, self.P, self.n, self.size = kind, P, tex, size_m
        n, mpt = tex, size_m / tex
        self.mpt = mpt
        ux = (np.arange(n, dtype=np.float32) / n)[None, :]
        vy = (np.arange(n, dtype=np.float32) / n)[:, None]

        def F(p0, octs, sd):
            return _norm01(fbm_p(ux, vy, p0, octs, seed + sd)).astype(np.float32)

        h = (P["roll"] * (F(3, 4, 1) - 0.5) + 0.12 * (F(10, 4, 2) - 0.5) + 0.02 * (F(70, 4, 3) - 0.5)).astype(np.float32)
        if P["ripples"]:
            h += P["ripples"] * np.sin(2 * np.pi * (73 * ux + 2.5 * F(6, 3, 8)))
        albm = np.ones((n, n), np.float32)
        rng = np.random.default_rng(seed)
        for _ in range(P["craters"]):
            r = P["rmin"] * (P["rmax"] / P["rmin"]) ** (rng.random() ** 1.9)
            cy, cx = rng.integers(0, n, 2)
            rt = r / mpt
            w = int(2.6 * rt) + 2
            if w > n // 2 - 2:
                continue
            dy, dx = np.mgrid[-w:w + 1, -w:w + 1].astype(np.float32)
            d = np.sqrt(dx * dx + dy * dy) / rt
            fresh = rng.random() ** 0.7
            D = 0.10 * r * (0.6 + 0.5 * fresh)
            prof = np.where(d < 1, -D * (1 - d * d), 0.0) + D * 0.40 * (0.4 + fresh) * np.exp(-((d - 1.0) / 0.34) ** 2) \
                + D * 0.18 * np.exp(-((np.maximum(d, 1.0) - 1.0) / 0.9) ** 2)
            sub = np.ix_((cy + np.arange(-w, w + 1)) % n, (cx + np.arange(-w, w + 1)) % n)
            h[sub] += prof.astype(np.float32)
            albm[sub] *= (1 + 0.35 * fresh * np.exp(-((d - 1.25) / 0.55) ** 2) - 0.10 * np.clip(1 - d, 0, 1)).astype(np.float32)
        for _ in range(P["pebbles"]):
            r = 0.025 + 0.22 * rng.random() ** 2.5
            cy, cx = rng.integers(0, n, 2)
            rt = r / mpt
            w = int(rt) + 2
            dy, dx = np.mgrid[-w:w + 1, -w:w + 1].astype(np.float32)
            d2 = (dx * dx + dy * dy) / (rt * rt)
            sub = np.ix_((cy + np.arange(-w, w + 1)) % n, (cx + np.arange(-w, w + 1)) % n)
            h[sub] += (0.8 * r * (0.7 + 0.6 * rng.random()) * np.sqrt(np.clip(1 - d2, 0, 1))).astype(np.float32)
            albm[sub] *= np.where(d2 < 1, 1 + 0.25 * (rng.random() - 0.5), 1.0).astype(np.float32)
        if blast:                                              # scoured, streaked soil under a landing engine
            bx, bz, br = blast
            cy, cx = int(bz / mpt) % n, int(bx / mpt) % n
            w = int(br / mpt)
            dy, dx = np.mgrid[-w:w + 1, -w:w + 1].astype(np.float32)
            d = np.sqrt(dx * dx + dy * dy) / w
            ang = np.arctan2(dy, dx)
            streak = 0.5 + 0.5 * np.sin(ang * 23 + 4 * np.sin(ang * 7 + d * 6))
            m = np.clip(1 - d, 0, 1) ** 0.7
            sub = np.ix_((cy + np.arange(-w, w + 1)) % n, (cx + np.arange(-w, w + 1)) % n)
            albm[sub] *= (1 - 0.32 * m * (0.5 + 0.5 * streak)).astype(np.float32)
            h[sub] -= (0.05 * m * (0.6 + 0.4 * streak)).astype(np.float32)
        h += 0.004 * (F(300, 3, 21) - 0.5)
        # albedo: broad patches + fine regolith grain
        a = (1 + 0.45 * (F(6, 5, 30) - 0.5) + P["grain"] * (F(260, 2, 31) - 0.5) + 0.5 * P["grain"] * (F(900, 2, 32) - 0.5)) * albm
        base = np.array(P["base"], np.float32)
        # lighting bake
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * mpt)
        gz = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * mpt)
        inv = 1 / np.sqrt(gx * gx + 1 + gz * gz)
        L = sun_vector(P["sun_el"], P["sun_az"])
        lam = np.clip(-gx * inv * L[0] + inv * L[1] - gz * inv * L[2], 0, 1)
        shadow = self._shadow(h, L, mpt)
        lit = P["amb"] + P["exposure"] * lam * (1 - 0.68 * shadow)
        col = a[..., None] * base * lit[..., None]
        col = 1 - np.exp(-1.35 * col)
        self.mips = [col.astype(np.float32)]
        while self.mips[-1].shape[0] > 8:
            m_ = self.mips[-1]
            self.mips.append((m_[0::2, 0::2] + m_[1::2, 0::2] + m_[0::2, 1::2] + m_[1::2, 1::2]) / 4)
        self.sun = L
        self._cache = {}
        print(f"terrain '{kind}' baked in {time.time() - t0:.1f}s")

    @staticmethod
    def _shadow(h, L, mpt):
        dxz = np.array([L[0], L[2]], np.float32)
        dxz /= np.linalg.norm(dxz)
        tan_el = L[1] / math.hypot(L[0], L[2])
        occ = np.zeros(h.shape, bool)
        for k in sorted({int(1.12 ** i) for i in range(0, 50)} | set(range(1, 9))):
            hk = np.roll(h, (-int(round(dxz[1] * k)), -int(round(dxz[0] * k))), (0, 1))
            occ |= hk > h + k * mpt * tan_el
        s = occ.astype(np.float32)
        for _ in range(24):
            s = (s + np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)) / 5
        return s

    def _bilinear(self, lvl, U, V):
        T = self.mips[lvl]
        m, s = T.shape[0], 1.0 / (2 ** lvl)
        x, y = U * s - 0.5, V * s - 0.5
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = (x - x0)[..., None].astype(np.float32), (y - y0)[..., None].astype(np.float32)
        x1, y1 = (x0 + 1) % m, (y0 + 1) % m
        x0, y0 = x0 % m, y0 % m
        return T[y0, x0] * (1 - fx) * (1 - fy) + T[y0, x1] * fx * (1 - fy) + T[y1, x0] * (1 - fx) * fy + T[y1, x1] * fx * fy

    def render_ground(self, img, cam, cam_x=0.0, cam_z=0.0, quality=0.5):
        """Paint the ground below the horizon into img (perspective-correct, mip-mapped)."""
        key = (round(cam_x, 3), round(cam_z, 3), cam.f, cam.cam_h, cam.hy, img.width, img.height, quality)
        if key not in self._cache:
            self._cache = {key: self._render(img.width, img.height, cam, cam_x, cam_z, quality)}
        hy, strip = self._cache[key]
        img.paste(strip, (0, hy))

    def _render(self, W_, H_, cam, cam_x, cam_z, q):
        hy = int(math.ceil(cam.hy))
        Wq, Rq = max(8, int(W_ * q)), max(8, int((H_ - hy) * q))
        xs = (np.arange(Wq) + 0.5) / q
        ys = hy + (np.arange(Rq) + 0.5) / q
        z = cam.f * cam.cam_h / (ys - cam.hy)
        X = cam_x + (xs[None, :] - cam.cx) / cam.f * z[:, None]
        Zw = cam_z + np.broadcast_to(z[:, None], X.shape)
        U, V = X / self.size * self.n, Zw / self.size * self.n
        fx = (z / cam.f) / self.mpt / q
        fz = (z * z / (cam.f * cam.cam_h)) / self.mpt / q
        top = len(self.mips) - 1
        lvl = np.clip(np.log2(np.maximum(np.maximum(fx, 0.6 * fz), 1e-6)), 0, top - 0.001)
        out = np.zeros((Rq, Wq, 3), np.float32)
        base = np.floor(lvl).astype(int)
        for L in np.unique(base):
            rows = np.nonzero(base == L)[0]
            a = self._bilinear(L, U[rows], V[rows])
            b = self._bilinear(min(L + 1, top), U[rows], V[rows])
            f = (lvl[rows] - L)[:, None, None]
            out[rows] = a * (1 - f) + b * f
        im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")
        if q != 1:
            im = im.resize((W_, H_ - hy), Image.BILINEAR)
        return hy, im


def _hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def cast_shadow(img, cam, sun, x, z, radius, height, alpha=150):
    """Shadow of an object (footprint circle of `radius` m, `height` m tall) projected on the ground in perspective."""
    horiz = math.hypot(sun[0], sun[2])
    length = height / max(sun[1] / horiz, 0.05)
    sx_, sz_ = -sun[0] / horiz * length, -sun[2] / horiz * length
    pts = []
    for k in range(20):
        a = 2 * math.pi * k / 20
        for ox, oz in ((0.0, 0.0), (sx_, sz_)):
            zz = z + math.sin(a) * radius + oz
            if zz > 1.0:
                px, py, _ = cam.project(x + math.cos(a) * radius + ox, zz)
                pts.append((px, py))
    if len(pts) >= 3:
        ImageDraw.Draw(img, "RGBA").polygon(_hull(pts), fill=(5, 5, 9, alpha))


def make_rock(seed, light, tone=0.38, N=220):
    """A lit 3D-looking boulder sprite (faceted, grainy, dark underside). light is toward-sun in sprite space:
    x right, y up, z toward the viewer."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    u, v = (xx - N / 2) / (N / 2), (yy - N * 0.60) / (N / 2)
    th = np.arctan2(v * 1.35, u)
    harm = sum(a * np.sin(k * th + rng.uniform(0, 6.28)) for k, a in ((2, 0.10), (3, 0.08), (5, 0.05), (7, 0.03)))
    rb = 0.80 * (1 + harm)
    r = np.sqrt(u * u + (v * 1.35) ** 2)
    z = np.sqrt(np.clip(1 - (r / rb) ** 2, 0, 1)) ** 0.8
    one = np.ones_like(u)
    n1 = fbm(u * 2.2 + seed, v * 2.2, one * (seed + 3.0), 4, seed)
    n2 = 1 - np.abs(2 * fbm(u * 4.5 + 9, v * 4.5, one * (seed + 7.0), 3, seed + 5) - 1)
    z = z * (0.78 + 0.5 * (n1 - 0.5)) + 0.14 * n2 * (z > 0.05)
    d = 2.0 / N
    zx, zy = np.gradient(z, d, axis=1) * 0.55, np.gradient(z, d, axis=0) * 0.55
    nrm = np.sqrt(zx * zx + zy * zy + 1)
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip((-zx * L[0] + zy * L[1] + L[2]) / nrm, 0, 1)
    grain = 0.82 + 0.36 * fbm(u * 26, v * 26, one * (seed + 11.0), 2, seed + 9)
    ao = 1 - 0.5 * np.clip((v - 0.10) / 0.45, 0, 1) ** 1.6
    shade = (0.10 + 1.15 * lam) * grain * ao
    col = np.clip(tone * shade, 0, 1)[..., None] * np.array([1.0, 0.985, 0.96], np.float32)
    col = 1 - np.exp(-1.35 * col)
    alpha = np.clip((1 - r / rb) * N * 0.32, 0, 1)
    arr = np.dstack([np.clip(col, 0, 1) * 255, alpha * 255]).astype(np.uint8)
    im = Image.fromarray(arr, "RGBA")
    return im.crop(im.getbbox())


def place_rock(img, cam, sprite, x, z, width_m):
    sx, sy, s = cam.project(x, z)
    w = max(2, int(width_m * s))
    h = max(2, int(w * sprite.height / sprite.width))
    if sx + w < 0 or sx - w > img.width or sy - h > img.height:
        return
    spr = sprite.resize((w, h), Image.LANCZOS if w < sprite.width else Image.BILINEAR)
    img.paste(spr, (int(sx - w / 2), int(sy - h * 0.92)), spr)


def perspective_scene(planet_earth=None, t=0.0, terrain=None, seed=5):
    """A lunar surface in true perspective: lit regolith, craters, boulders, footprints, lander, flag, astronaut."""
    W_, H_ = sa.W, sa.H
    cam = GroundCamera(W_, H_)
    terrain = terrain or Terrain("moon", blast=(5.5, 52.0, 9.0))
    sun = terrain.sun
    light_sprite = (sun[0], sun[1], -sun[2])
    img = sa.vgrad((1, 2, 8), (6, 7, 14), h=H_, w=W_)
    sa.draw_stars(img, 1.0, t, ymax=int(cam.hy) - 30)
    if planet_earth is not None:
        sa.paste_center(img, planet_earth.render(150, rot=30 + t * 4, tilt=15, light=(0.55, -0.35, 0.75)), 780, 480)
    rng = random.Random(seed)
    # distant ridges, lit from the left
    ridge = [(-20, cam.hy + 3)]
    for x in range(-20, W_ + 60, 24):
        ridge.append((x, cam.hy - 30 - 24 * math.sin(x * 0.011 + 1) - 14 * math.sin(x * 0.037) - rng.uniform(0, 5)))
    ridge.append((W_ + 40, cam.hy + 3))
    gm = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(gm).polygon(ridge, fill=255)
    img.paste(sa.vgrad((92, 92, 98), (34, 34, 40), h=H_, w=W_), (0, int(cam.hy) - 70 + 70 - 70), gm) if False else \
        img.paste(Image.new("RGB", (W_, H_), (58, 58, 64)), (0, 0), gm)
    d = ImageDraw.Draw(img, "RGBA")
    for a, b in zip(ridge, ridge[1:]):
        if b[1] > a[1]:
            d.line([a, b], fill=(120, 120, 128, 150), width=3)          # sunlit left slopes
    terrain.render_ground(img, cam)
    d = ImageDraw.Draw(img, "RGBA")

    sprites = [make_rock(s, light_sprite, terrain.P["rock"] * (0.75 + 0.5 * random.Random(s).random())) for s in range(1, 15)]
    rocks = []
    for _ in range(90):
        z = 6 + rng.random() ** 1.7 * 170
        rocks.append((z, rng.uniform(-0.68, 0.68) * z, rng.choice(sprites), rng.uniform(0.15, 0.7) * (1 + z / 60)))
    for _ in range(18):
        z = 3.5 + rng.random() * 8
        rocks.append((z, rng.uniform(-0.6, 0.6) * z, rng.choice(sprites), rng.uniform(0.10, 0.4)))
    rocks.sort(key=lambda r: -r[0])
    for z, x, spr, wm in rocks:
        cast_shadow(img, cam, sun, x, z, wm * 0.42, wm * 0.5, 120)
        place_rock(img, cam, spr, x, z, wm)

    for k in range(10):                                                  # boot prints, lit by the same sun
        fx, fz = -1.9 - 0.72 * k, 7.0 + (0.17 if k % 2 else -0.17)
        sx, sy, s = cam.project(fx, fz)
        w, h = 0.30 * s, max(2.0, 0.12 * s * cam.cam_h / fz * 2.2)
        d.ellipse([sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2], fill=(34, 34, 40, 150))
        d.arc([sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2], 300, 80, fill=(210, 210, 216, 170), width=2)
        d.arc([sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2], 110, 250, fill=(8, 8, 12, 190), width=3)

    def place_h(z, x, height_m, sprite_px):
        sx, sy, s = cam.project(x, z)
        return sx, sy, (s * height_m) / sprite_px

    sx, sy, sc = place_h(52, 5.5, 12.0, 520)
    cast_shadow(img, cam, sun, 5.5, 52, 2.6, 11.0, 150)
    sa.paste_center(img, sa.rocket_layer(sa.ROCKET_LEGS, 0, 1.4, random.Random(0), 0, sc), sx, sy - 250 * sc)
    fx_, fy_, fsc = place_h(15, 3.2, 2.0, 340)
    cast_shadow(img, cam, sun, 3.2, 15, 0.05, 2.0, 130)
    sa.draw_flag(img, fx_, fy_, fsc, 1.0)
    ax_, ay_, asc = place_h(7.0, -0.6, 1.8, 390)
    cast_shadow(img, cam, sun, -0.6, 7.0, 0.3, 1.8, 150)
    sa.draw_astronaut(img, ax_, ay_, asc, "walk", 0.18, 1, t)
    return img



# ================================================================ ion tug
def tug_layer(deploy, chem, ion, rng, angle, scale):
    """The rocket with solar-panel wings and either a chemical flame (chem = flame length) or a blue ion beam
    (ion = 0..1). Rotates about the rocket centre, like sa.rocket_layer."""
    rig = Image.new("RGBA", (360, 1500), (0, 0, 0, 0))
    fx = Image.new("RGBA", (360, 1500), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fx, "RGBA")
    if chem > 2:
        sa.draw_flame(fd, chem, 1.9, rng)
    if ion > 0.02:
        L = 340 * ion * (0.96 + 0.08 * rng.random())
        for w, col in ((30, (110, 140, 255, 60)), (16, (150, 195, 255, 140)), (7, (235, 245, 255, 255))):
            fd.polygon([(180 - w * 0.5, 985), (180 + w * 0.5, 985), (180 + w * 0.1, 985 + L), (180 - w * 0.1, 985 + L)], fill=col)
        fd.ellipse([164, 976, 196, 1002], fill=(175, 215, 255, 255))
    rig.alpha_composite(fx.filter(ImageFilter.GaussianBlur(3)))
    rig.alpha_composite(sa.ROCKET, (0, 470))
    if deploy > 0.01:
        d = ImageDraw.Draw(rig, "RGBA")
        Lp = 190 * deploy
        for sgn in (-1, 1):
            x0, x1 = 180 + sgn * 65, 180 + sgn * (65 + Lp)
            xa, xb = sorted((x0, x1))
            d.line([(x0, 790), (x1, 790)], fill=(150, 156, 170, 255), width=4)
            d.rectangle([xa, 735, xb, 845], fill=(24, 52, 120, 255), outline=(170, 180, 200, 255), width=2)
            for k in range(1, max(2, int(Lp / 32))):
                x = xa + (xb - xa) * k / max(2, int(Lp / 32))
                d.line([(x, 735), (x, 845)], fill=(90, 130, 210, 255), width=1)
            d.line([(xa, 790), (xb, 790)], fill=(90, 130, 210, 255), width=1)
    if scale != 1:
        rig = rig.resize((max(1, int(360 * scale)), max(1, int(1500 * scale))), Image.BILINEAR)
    if abs(angle) > 0.05:
        rig = rig.rotate(-angle, resample=Image.BICUBIC, expand=True)
    return rig


# ===================================================================== demos
def demo_planets(path):
    import time
    t0 = time.time()
    planets = {k: Planet(k) for k in ("earth", "moon", "mars", "jupiter")}
    print(f"textures built in {time.time() - t0:.1f}s")
    sheet = sa.vgrad((2, 3, 10), (8, 10, 22), h=1500, w=1080)
    d = ImageDraw.Draw(sheet, "RGBA")
    f = sa.get_font(30)
    t1 = time.time()
    cells = [("earth", 0, 23, (-0.62, -0.30, 0.72)), ("moon", 40, 0, (-0.62, -0.30, 0.72)),
             ("mars", 70, 25, (-0.62, -0.30, 0.72)), ("jupiter", 10, 3, (-0.62, -0.30, 0.72))]
    for i, (k, rot, tilt, light) in enumerate(cells):
        x, y = 270 + (i % 2) * 540, 270 + (i // 2) * 540
        sa.paste_center(sheet, planets[k].render(215, rot, tilt, light), x, y)
        d.text((x - 250, y - 262), k, fill=(255, 255, 255, 200), font=f)
    for i, (rot, light, lab) in enumerate(((0, (-0.62, -0.30, 0.72), "rotation 0"), (130, (-0.62, -0.30, 0.72), "rotation 130"),
                                           (60, (0.95, 0.05, -0.2), "crescent"))):
        x, y = 180 + i * 360, 1270
        sa.paste_center(sheet, planets["earth"].render(150, rot, 23, light), x, y)
        d.text((x - 120, y - 190), lab, fill=(255, 255, 255, 200), font=sa.get_font(26))
    sheet.save(path)
    print(f"rendered {path} in {time.time() - t1:.1f}s")
    return planets


def demo_perspective(path, planets=None):
    earth = (planets or {}).get("earth") or Planet("earth")
    perspective_scene(earth, 1.0).save(path)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sciart_world demos")
    ap.add_argument("--demo", default="out/world", help="output folder for planets.png and perspective.png")
    out = Path(ap.parse_args().demo)
    out.mkdir(parents=True, exist_ok=True)
    pl = demo_planets(out / "planets.png")
    demo_perspective(out / "perspective.png", pl)
