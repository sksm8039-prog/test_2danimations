#!/usr/bin/env python3
"""
sciart_earth.py - the Earth, ray-traced from the launch pad to orbit.

EarthTerrain : tileable living ground (grass, forest, fields, bare soil, rock) with sunlight and cast shadows,
               plus an optional concrete launch pad.
EarthView    : ray-traced Earth seen from any camera position: four terrain tiles + a global texture blend by pixel
               footprint, oceans with sun glint, a cloud shell, haze and a limb-glowing atmosphere.
Site frame (same as the Moon): x east, y up at the launch site, z north, Earth centre at the origin.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

import sciart as sa
import sciart_world as sw
from sciart_world import Terrain, _norm01, fbm_p, sun_vector

R_E = 6.371e6
H_ATM, TAU_Z = 8000.0, 0.12


def _ss(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class EarthTerrain(Terrain):
    def __init__(self, size_m=160.0, tex=2048, seed=3, relief=8.0, pad=False, sun_el=38.0, sun_az=35.0):
        import time
        t0 = time.time()
        n = tex
        self.n, self.size, self.mpt = n, size_m, size_m / n
        mpt = self.mpt
        ux = (np.arange(n, dtype=np.float32) / n)[None, :]
        vy = (np.arange(n, dtype=np.float32) / n)[:, None]

        def F(p0, octs, sd):
            return _norm01(fbm_p(ux, vy, p0, octs, seed + sd)).astype(np.float32)

        h = relief * (F(3, 5, 1) - 0.5) * 2 + 0.25 * relief * (F(14, 4, 2) - 0.5) + 0.05 * relief * (F(70, 3, 3) - 0.5)
        xs = (np.arange(n, dtype=np.float32) - n / 2) * mpt
        dist = np.hypot(xs[None, :], xs[:, None])
        if pad:
            h = h * _ss((dist - 40.0) / 45.0).astype(np.float32)
        m, fields = F(4, 4, 9), F(18, 3, 11)
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * mpt)
        gz = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * mpt)
        slope = np.sqrt(gx * gx + gz * gz)
        e = (h - h.min()) / max(float(h.max() - h.min()), 1e-6)
        C = lambda *c: np.array(c, np.float32)
        dry = _ss((0.55 - m) / 0.2)[..., None]
        col = C(0.30, 0.46, 0.17) * (1 - dry) + C(0.55, 0.50, 0.27) * dry
        forest = (_ss((m - 0.55) / 0.15) * (1 - _ss((e - 0.55) / 0.2)))[..., None]
        col = col * (1 - forest) + C(0.11, 0.27, 0.11) * forest
        col = col * (1 + ((np.floor(fields * 6) / 6 - 0.5) * 0.22))[..., None]
        bare = (_ss((F(40, 3, 13) - 0.66) / 0.10) * 0.6)[..., None]
        col = col * (1 - bare) + C(0.40, 0.31, 0.22) * bare
        rock = np.clip(_ss((slope - 0.35) / 0.3) + _ss((e - 0.78) / 0.15), 0, 1)[..., None]
        col = col * (1 - rock) + C(0.46, 0.44, 0.41) * rock
        col = col * (1 + 0.20 * (F(400, 2, 20) - 0.5))[..., None]
        if pad:
            conc = _ss((30.0 - dist) / 4.0)[..., None]
            scorch = (_ss((46.0 - dist) / 14.0) * (1 - conc[..., 0]))[..., None]
            col = col * (1 - 0.55 * scorch) + C(0.18, 0.16, 0.14) * 0.55 * scorch
            col = col * (1 - conc) + C(0.56, 0.56, 0.57) * conc
            trench = (np.abs(xs[None, :] - 12.0) < 4.0) & (np.abs(xs[:, None]) < 24.0)
            col[trench] = C(0.12, 0.12, 0.13)
        inv = 1 / np.sqrt(gx * gx + 1 + gz * gz)
        L = sun_vector(sun_el, sun_az)
        lam = np.clip(-gx * inv * L[0] + inv * L[1] - gz * inv * L[2], 0, 1)
        shadow = self._shadow(h, L, mpt)
        lit = 0.30 + 1.35 * lam * (1 - 0.55 * shadow)
        col = np.clip(col * lit[..., None], 0, 1) ** (1 / 1.12)
        self.mips = [col.astype(np.float32)]
        while self.mips[-1].shape[0] > 8:
            a = self.mips[-1]
            self.mips.append((a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4)
        self.sun = L
        print(f"earth tile {size_m:,.0f} m baked in {time.time() - t0:.1f}s")


class EarthView:
    LODS = [(3200e3, 1024, dict(relief=6000.0)), (200e3, 2048, dict(relief=900.0)),
            (8e3, 2048, dict(relief=60.0)), (160.0, 2048, dict(relief=3.0, pad=True))]

    def __init__(self, seed=3):
        import time
        t0 = time.time()
        self.planet = sw.Planet("earth", tex=(1280, 640))
        self.tiles = [EarthTerrain(size, tex, seed + 5 * k, **over) for k, (size, tex, over) in enumerate(self.LODS)]
        self.sun = np.array(sun_vector(38.0, 35.0), np.float64)
        ux = (np.arange(512, dtype=np.float32) / 512)[None, :]
        vy = (np.arange(512, dtype=np.float32) / 512)[:, None]
        self.noise = _norm01(fbm_p(ux, vy, 12, 5, 91)).astype(np.float32)
        self._pick_site()
        print(f"earth view ready in {time.time() - t0:.1f}s")

    def _pick_site(self):
        """Choose where on the global map the launch site sits: land with ocean to the east."""
        oc = self.planet.tex["ocean"]
        th, tw = oc.shape
        best = (-9, 0, 0)
        for i in range(0, th, 4):
            lat = math.pi / 2 - (i + 0.5) / th * math.pi
            if abs(lat) > 0.55:
                continue
            for j in range(0, tw, 4):
                if oc[i, j] > 0.1:
                    continue
                east = oc[i, [(j + k) % tw for k in range(12, 30, 3)]].mean()
                west = oc[i, [(j - k) % tw for k in range(6, 24, 6)]].mean()
                sc = east - 0.6 * west - 0.3 * abs(lat)
                if sc > best[0]:
                    best = (sc, i, j)
        _, i, j = best
        self.lat_s = math.pi / 2 - (i + 0.5) / th * math.pi
        self.lon_s = (j + 0.5) / tw * 2 * math.pi - math.pi
        a = np.array([math.cos(self.lat_s) * math.sin(self.lon_s), math.cos(self.lat_s) * math.cos(self.lon_s), math.sin(self.lat_s)])
        e = np.array([math.cos(self.lon_s), -math.sin(self.lon_s), 0.0])
        nth = np.array([-math.sin(self.lat_s) * math.sin(self.lon_s), -math.sin(self.lat_s) * math.cos(self.lon_s), math.cos(self.lat_s)])
        self.Mg = np.stack([e, a, nth], axis=1)           # site basis (east, up, north) -> texture basis

    def _noise(self, E, N, scale):
        return self.noise[(N / scale % 1.0 * 512).astype(int) % 512, (E / scale % 1.0 * 512).astype(int) % 512]

    def render(self, cam, fwd, up_hint, f, W_=sa.W, H_=sa.H, cy=900.0, q=0.5, t=0.0):
        R = R_E
        Wq, Hq = int(W_ * q), int(H_ * q)
        xs, ys = (np.arange(Wq) + 0.5) / q, (np.arange(Hq) + 0.5) / q
        C = np.array(cam, np.float64)
        fwd = np.array(fwd, np.float64)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(np.array(up_hint, np.float64), fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        d = fwd[None, None, :] + right[None, None, :] * ((xs - W_ / 2) / f)[None, :, None] \
            + up[None, None, :] * (-(ys - cy) / f)[:, None, None]
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        D = d.reshape(-1, 3)
        Cn = float(np.linalg.norm(C))
        h_c = Cn - R
        b = D @ C
        disc = b * b - (Cn * Cn - R * R)
        hit = (disc > 0) & (b < 0)
        tt = np.where(hit, -b - np.sqrt(np.maximum(disc, 0)), 0.0)
        S = self.sun
        K = len(D)
        rgb = np.zeros((K, 3), np.float32)
        alpha = np.zeros(K, np.float32)
        m = self.planet

        def tex_coords(nvec):
            nt = nvec @ self.Mg.T
            return np.arcsin(np.clip(nt[:, 2], -1, 1)), np.arctan2(nt[:, 0], nt[:, 1])

        def smp(arr, lat_g, lon_g):
            u = (lon_g + np.pi) / (2 * np.pi) * m.tw - 0.5
            v = np.clip((np.pi / 2 - lat_g) / np.pi * m.th - 0.5, 0, m.th - 1.001)
            return m._sample(arr, u, v)

        ih = np.nonzero(hit)[0]
        if len(ih):
            P = C[None, :] + tt[ih, None] * D[ih]
            n = P / R
            g = (tt[ih] / f) / np.sqrt(np.clip(-(D[ih] * n).sum(1), 0.04, 1.0))
            lat, lon = np.arcsin(np.clip(n[:, 2], -1, 1)), np.arctan2(n[:, 0], n[:, 1])
            E, N = R * lon, R * lat
            lam = np.clip(n @ S, 0, 1)
            lg, lo = tex_coords(n)
            alb = smp(self.planet.tex["albedo"], lg, lo)
            ov = smp(self.planet.tex["ocean"], lg, lo)
            glob = alb * (0.10 + 1.05 * lam ** 0.9)[:, None]
            ocean_w = _ss((ov + 0.20 * (self._noise(E, N, 7000.0) - 0.5) + 0.07 * (self._noise(E, N, 700.0) - 0.5) - 0.45) / 0.10)
            col = np.zeros((len(ih), 3), np.float32)
            wrem = np.ones(len(ih), np.float32)
            for k in (3, 2, 1, 0):
                tile = self.tiles[k]
                uk = _ss((tile.size / np.maximum(g, 1e-6) - 700.0) / 700.0) * (1 - _ss((np.abs(lat) - 0.9) / 0.35))
                if k == 3:
                    uk = uk * (1 - _ss((np.hypot(E, N) - 55.0) / 25.0))
                wk = (uk * wrem).astype(np.float32)
                sel = np.nonzero(wk > 0.003)[0]
                if len(sel):
                    col[sel] += wk[sel, None] * tile.sample(E[sel], N[sel], g[sel])
                wrem = wrem * (1 - uk).astype(np.float32)
            tw_ = 1 - wrem
            land_w = (tw_ * (1 - ocean_w)).astype(np.float32)
            surf = col / np.maximum(tw_, 1e-6)[:, None] * land_w[:, None] + glob * (1 - land_w)[:, None]
            Hv = S[None, :] - D[ih]
            Hv /= np.linalg.norm(Hv, axis=1, keepdims=True)
            spec = np.clip((n * Hv).sum(1), 0, 1) ** 220 * ocean_w * 0.9
            surf = surf + spec[:, None] * np.array([1.0, 0.96, 0.88], np.float32)
            rgb[ih], alpha[ih] = surf, 1.0
            cos_inc = np.clip(-(D[ih] * n).sum(1), 0.03, 1.0)
            tau = TAU_Z * (1 - math.exp(-max(h_c, 0) / H_ATM)) / cos_inc + 1.4e-5 * math.exp(-max(h_c, 0) / H_ATM) * tt[ih]
            haze = (1 - np.exp(-tau))[:, None]
            rgb[ih] = rgb[ih] * (1 - haze) + np.array([0.55, 0.72, 1.0], np.float32) * (0.35 + 0.65 * lam)[:, None] * haze
        im_ = np.nonzero(~hit)[0]
        if len(im_):                                                    # sky and limb glow
            Dm = D[im_]
            Ch = C / Cn
            cosz = Dm @ Ch
            bb = Dm @ C
            hmin = np.sqrt(np.maximum(Cn * Cn - bb * bb, 0)) - R
            tau = np.where(cosz > 0, TAU_Z * math.exp(-max(h_c, 0) / H_ATM) / np.maximum(cosz, 0.04),
                           TAU_Z * math.sqrt(2 * math.pi * R / H_ATM) * np.exp(-np.maximum(hmin, 0) / H_ATM))
            kfac = 1.0 + 12.0 * math.exp(-max(h_c, 0) / 12000.0)
            a = np.clip((1 - np.exp(-tau)) * kfac, 0, 1)
            hz = np.clip(tau / 1.5, 0, 1)[:, None] * 0.6
            rgb[im_] = np.array([0.30, 0.52, 1.0], np.float32) * (1 - hz) + np.array([0.80, 0.88, 1.0], np.float32) * hz
            alpha[im_] = a
        # cloud shell at 9 km (seen from above it)
        Rs = R + 9000.0
        cs = Cn * Cn - Rs * Rs
        if cs > 0:
            ds = b * b - cs
            ok = (ds > 0) & (b < 0)
            ic = np.nonzero(ok)[0]
            if len(ic):
                ts = -b[ic] - np.sqrt(ds[ic])
                Ps = C[None, :] + ts[:, None] * D[ic]
                ns = Ps / Rs
                lg, lo = tex_coords(ns)
                cv = smp(self.planet.tex["clouds"], lg, lo + 0.0004 * t)
                Es, Ns = R * np.arctan2(ns[:, 0], ns[:, 1]), R * np.arcsin(np.clip(ns[:, 2], -1, 1))
                cv = cv * (0.55 + 0.9 * self._noise(Es, Ns, 60000.0)) * (0.8 + 0.4 * self._noise(Es, Ns, 9000.0))
                ca = np.clip((cv - 0.30) * 2.4, 0, 1) * 0.94
                lam_s = np.clip(ns @ S, 0, 1)
                ccol = np.clip(0.16 + 0.95 * lam_s, 0, 1)[:, None] * np.array([1.0, 1.0, 1.0], np.float32)
                ccol = ccol * np.array([0.97, 0.98, 1.0], np.float32)
                rgb[ic] = rgb[ic] * (1 - ca)[:, None] + ccol * ca[:, None]
                alpha[ic] = np.maximum(alpha[ic], ca)
        out = np.dstack([np.clip(rgb, 0, 1) * 255, alpha * 255]).astype(np.uint8).reshape(Hq, Wq, 4)
        im = Image.fromarray(out, "RGBA")
        return im.resize((W_, H_), Image.BILINEAR) if q != 1 else im
