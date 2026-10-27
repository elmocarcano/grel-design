"""Exact-geometry renderer for TuttoTondo (sphere-traced SDF, numpy).

Geometry (cm), y up, table centred on the origin:
  plate A  : 70 x 40, 2 thick, in the XY plane (z in [-1, 1])
  plate B  : same, rotated 90 deg (in the ZY plane, x in [-1, 1])
  each plate has a radius-30 half-circle cut from the centre of its top edge
  slots    : A is slotted from the floor up to 5, B from 5 up to the arc (10)
  top      : disc, radius 30, 2 thick, top surface flush at y = 40

The script renders the lighting once into buffers (per-pixel part/face class
plus light terms) so any colour combination can then be composited instantly.
"""
import numpy as np

R_EDGE = 0.16  # small edge radius, gives the soft highlight on lacquered edges


def box2(px, py, hx, hy):
    qx, qy = np.abs(px) - hx, np.abs(py) - hy
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def extrude(d2, w, h):
    qx, qy = d2, np.abs(w) - h
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def plate2d(u, y, kind):
    d = box2(u, y - 20, 35, 20)
    d = np.maximum(d, -(np.hypot(u, y - 40) - 30))          # half-circle cut
    if kind == 'A':
        d = np.maximum(d, -box2(u, y - 2.0, 1.0, 3.0))       # slot y 0..5
    else:
        d = np.maximum(d, -box2(u, y - 7.6, 1.0, 2.6))       # slot y 5..10
    return d


def sdf_parts(x, y, z):
    r = R_EDGE
    a = extrude(plate2d(x, y, 'A') + r, z, 1 - r) - r
    b = extrude(plate2d(z, y, 'B') + r, x, 1 - r) - r
    dd = extrude(np.hypot(x, z) - 30 + r, y - 39, 1 - r) - r
    return a, b, dd


def sdf(x, y, z):
    a, b, d = sdf_parts(x, y, z)
    return np.minimum(np.minimum(a, b), d)


def normal(x, y, z, e=0.01):
    nx = sdf(x + e, y, z) - sdf(x - e, y, z)
    ny = sdf(x, y + e, z) - sdf(x, y - e, z)
    nz = sdf(x, y, z + e) - sdf(x, y, z - e)
    n = np.stack([nx, ny, nz], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def march(o, d, tmax=900, steps=260, eps=0.004):
    n = o.shape[0]
    t = np.zeros(n)
    hit = np.zeros(n, bool)
    act = np.arange(n)
    for _ in range(steps):
        if act.size == 0:
            break
        p = o[act] + d[act] * t[act, None]
        s = sdf(p[:, 0], p[:, 1], p[:, 2])
        done = s < eps
        hit[act[done]] = True
        t[act] += np.where(done, 0, s * 0.9)
        far = t[act] > tmax
        act = act[~done & ~far]
    return t, hit


def soft_shadow(p, L, k=14.0, tmax=220, steps=90):
    n = p.shape[0]
    res = np.ones(n)
    t = np.full(n, 0.06)
    act = np.arange(n)
    for _ in range(steps):
        if act.size == 0:
            break
        Ld = L[act] if L.ndim == 2 else L[None, :]
        q = p[act] + Ld * t[act, None]
        h = sdf(q[:, 0], q[:, 1], q[:, 2])
        res[act] = np.minimum(res[act], k * h / t[act])
        t[act] += np.clip(h, 0.03, 4.0)
        act = act[(h > 1e-4) & (t[act] < tmax)]
    return np.clip(res, 0, 1)


def ao(p, n, samples=6, step=1.6):
    occ = np.zeros(p.shape[0])
    w = 1.0
    for i in range(1, samples + 1):
        h = step * i
        q = p + n * h
        d = sdf(q[:, 0], q[:, 1], q[:, 2])
        occ += w * np.clip(h - d, 0, None) / h
        w *= 0.6
    return np.clip(1 - 0.55 * occ, 0, 1)


def camera_rays(W, H, az, el, dist, fov, target):
    az, el = np.radians(az), np.radians(el)
    eye = np.array([dist * np.cos(el) * np.sin(az), target[1] + dist * np.sin(el), dist * np.cos(el) * np.cos(az)])
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 1 / np.tan(np.radians(fov) / 2)
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = 1 - (np.arange(H) + 0.5) / H * 2
    X, Y = np.meshgrid(xs * W / H, ys)
    d = fwd[None, None, :] * f + right[None, None, :] * X[..., None] + up[None, None, :] * Y[..., None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return eye, d.reshape(-1, 3)


def render_buffers(W, H, az=34, el=20, dist=330, fov=22, target=(0, 17, 0),
                   light_pos=(210, 300, 120), fill_dir=(-0.8, 0.3, 0.5)):
    eye, d = camera_rays(W, H, az, el, dist, fov, target)
    o = np.broadcast_to(eye, d.shape).copy()
    t, hit = march(o, d)
    N = d.shape[0]

    cls = np.zeros(N, np.int8)  # 0 bg, 1 body, 2 accent, 9 ground
    diff = np.zeros(N); amb = np.zeros(N); spec = np.zeros(N); gshadow = np.ones(N)

    idx = np.where(hit)[0]
    p = o[idx] + d[idx] * t[idx, None]
    n = normal(p[:, 0], p[:, 1], p[:, 2])
    a, b, dd = sdf_parts(p[:, 0], p[:, 1], p[:, 2])
    part = np.argmin(np.stack([a, b, dd], -1), -1)
    c = np.ones(idx.size, np.int8)
    # plate B end faces are the accent detail
    c[(part == 1) & (np.abs(p[:, 2]) > 34.8) & (np.abs(n[:, 2]) > 0.5)] = 2
    # the top: upper face and its edge carry the accent, underside the body
    c[(part == 2) & (n[:, 1] > -0.5)] = 2
    cls[idx] = c

    lp = np.asarray(light_pos, float)
    Lv = lp[None, :] - p
    dist_l = np.linalg.norm(Lv, axis=-1, keepdims=True)
    Lp = Lv / dist_l
    att = (np.linalg.norm(lp) / dist_l[:, 0]) ** 2          # 1 at the table centre
    pe = p + n * 0.02
    sh = soft_shadow(pe, Lp, k=10.0)
    occ = ao(p, n, samples=8, step=1.3)
    ndl = np.clip(np.sum(n * Lp, -1), 0, 1)
    F = np.asarray(fill_dir, float); F /= np.linalg.norm(F)
    fill = np.clip(n @ F, 0, 1) * 0.22
    diff[idx] = ndl * sh * att + fill * occ
    hemi = 0.5 + 0.5 * n[:, 1]
    amb[idx] = (0.50 * hemi + 0.22) * occ
    hvec = Lp - d[idx]; hvec /= np.linalg.norm(hvec, axis=-1, keepdims=True)
    spec[idx] = (np.clip(np.sum(n * hvec, -1), 0, 1) ** 40) * sh * 0.06 + \
                0.035 * (1 - np.clip(-np.sum(n * d[idx], -1), 0, 1)) ** 4 * occ

    # ground plane: shadow only
    g = np.where(~hit & (d[:, 1] < 0))[0]
    tg = -o[g, 1] / d[g, 1]
    pg = o[g] + d[g] * tg[:, None]
    near = np.hypot(pg[:, 0], pg[:, 2]) < 160
    gi = g[near]; pg = pg[near]
    Lg = np.asarray(light_pos, float)[None, :] - pg; Lg /= np.linalg.norm(Lg, axis=-1, keepdims=True)
    s = soft_shadow(pg + np.array([0, 0.02, 0]), Lg, k=6.0)
    up = np.tile([0.0, 1.0, 0.0], (gi.size, 1))
    occ_g = ao(pg, up, samples=6, step=2.5)
    gshadow[gi] = (0.45 + 0.55 * s) * (0.35 + 0.65 * occ_g)
    cls[gi] = 9
    bounce = np.zeros(N); bounce[idx] = np.clip(0.5 - 0.5 * n[:, 1], 0, 1) * occ
    return dict(W=W, H=H, cls=cls, diff=diff, amb=amb, spec=spec, gshadow=gshadow, bounce=bounce,
                hidx=idx, pos=p.astype(np.float32), nrm=n.astype(np.float32), part=part.astype(np.int8))


def srgb_to_lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _hash_noise(x, y, z, scale):
    """Cheap smooth value noise from summed sines (deterministic)."""
    v = (np.sin(x * 0.91 * scale + y * 1.7 * scale) + np.sin(y * 1.31 * scale - z * 0.77 * scale + 1.3)
         + np.sin(z * 1.13 * scale + x * 0.53 * scale + 2.1))
    return v / 3


def oak_albedo(p, n, part):
    """Procedural rift-sawn oak: straight grain along each part's length."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    across = np.where(part == 0, y + 0.6 * z, np.where(part == 1, y + 0.6 * x, z + 0.3 * y))
    along = np.where(part == 0, x, np.where(part == 1, z, x))
    drift = 0.9 * np.sin(along * 0.045 + across * 0.11) + 0.5 * np.sin(along * 0.013 - across * 0.07 + 1.7)
    g = across + drift
    # irregular fine streaks: several incommensurate frequencies, fading in and out along the grain
    streak = np.zeros_like(g)
    for f, ph, w in ((1.37, 0.3, .30), (2.71, 1.9, .25), (4.13, 4.2, .2), (6.07, 2.6, .15), (9.31, 5.1, .10)):
        streak += w * np.sin(g * f * 6.283 / 2.5 + ph + 0.6 * np.sin(along * 0.02 * f + ph))
    streak = np.clip(streak * 1.6, -1, 1)
    fade = 0.6 + 0.4 * np.sin(along * 0.07 + g * 0.9)
    figure = 0.5 + 0.5 * np.sin(g * 0.45 + 0.8 * np.sin(along * 0.03))
    t = np.clip(0.5 + 0.22 * streak * fade + 0.22 * (figure - 0.5), 0, 1)
    base = srgb_to_lin([214, 172, 122])
    dark = srgb_to_lin([176, 128, 82])
    col = base[None, :] * (1 - t[:, None]) + dark[None, :] * t[:, None]
    endg = ((part == 0) & (np.abs(n[:, 0]) > 0.6)) | ((part == 1) & (np.abs(n[:, 2]) > 0.6)) | \
           ((part == 2) & (np.abs(n[:, 1]) < 0.4))
    col[endg] *= 0.85
    return col


_OAK = None


def _oak_tex():
    """Oak veneer sampled from a reference photo (lighting removed), as linear albedo."""
    global _OAK
    if _OAK is None:
        import os
        from PIL import Image
        a = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oak-plate.png')).convert('RGB')).astype(float)
        lin = srgb_to_lin(a)
        target = srgb_to_lin([200, 150, 100])
        _OAK = lin * (target / lin.reshape(-1, 3).mean(0))[None, None, :]
    return _OAK


def _sample_mirror(tex, s, t, cm_per_px=0.11, cm_per_px_t=0.20):
    """Bilinear lookup with mirrored repeat; s across the grain, t along it (cm)."""
    H, W = tex.shape[:2]
    def mirror(u, n):
        u = np.mod(u, 2 * (n - 1)); return np.where(u > n - 1, 2 * (n - 1) - u, u)
    x = mirror(s / cm_per_px, W); y = mirror(t / cm_per_px_t, H)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy) +
            tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def oak_albedo_photo(p, n, part):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    tex = _oak_tex()
    # plates: grain runs up the height; top: grain runs along x. Offsets keep parts from matching.
    s = np.where(part == 0, x + 13.0, np.where(part == 1, z + 41.0, z + 7.0))
    t = np.where(part == 2, x + 30.0, y + np.where(part == 0, 0.5, 1.0))
    col = _sample_mirror(tex, s, t)
    endg = ((part == 0) & (np.abs(n[:, 0]) > 0.6)) | ((part == 1) & (np.abs(n[:, 2]) > 0.6)) | \
           ((part == 2) & (np.abs(n[:, 1]) < 0.4))
    col[endg] *= 0.86
    return col


def tone(x):
    """Gentle highlight shoulder so pale tops keep their tone."""
    k = 0.72
    return np.where(x < k, x, k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k))))


def composite(buf, body_hex, accent_hex, bg=(241, 232, 222), key=0.80, ambk=0.52, oak=False):
    W, H = buf['W'], buf['H']
    cls = buf['cls']
    bgl = srgb_to_lin(bg)
    # studio sweep: a touch brighter behind the table, softly darker at the edges
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx / W - 0.5) * 1.1, (yy / H - 0.42))
    sweep = (1.06 - 0.22 * r ** 2).reshape(-1)
    out = bgl[None, :] * sweep[:, None]
    g = cls == 9
    out[g] *= buf['gshadow'][g, None]
    hidx, pos, nrm, part = buf['hidx'], buf['pos'], buf['nrm'], buf['part']
    sub = cls[hidx]
    for k, hx in ((1, body_hex), (2, accent_hex)):
        m = sub == k
        ii = hidx[m]
        if oak:
            alb = oak_albedo_photo(pos[m].astype(float), nrm[m].astype(float), part[m])
        else:
            alb = np.tile(srgb_to_lin([int(hx[i:i + 2], 16) for i in (1, 3, 5)]), (ii.size, 1))
            # very slight unevenness of a sprayed lacquer
            alb *= (1 + 0.012 * _hash_noise(pos[m, 0], pos[m, 1], pos[m, 2], 0.35))[:, None]
        light = key * buf['diff'][ii] + ambk * buf['amb'][ii]
        warm = 0.30 * buf['bounce'][ii][:, None] * bgl[None, :]      # floor bounce
        out[ii] = alb * (light[:, None] + warm) + buf['spec'][ii, None] * (1.4 if oak else 1.0)
    out = tone(out)
    rng = np.random.default_rng(7)
    srgb = lin_to_srgb(out) + rng.normal(0, 0.005, out.shape)
    img = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(H, W, 3)
    return img


def composite_rgba(buf, body_hex, accent_hex, oak=False, shadow_strength=0.85, key=0.80, ambk=0.52, shadow=True):
    """Cut-out version: transparent background, the floor shadow kept as soft alpha."""
    W, H = buf['W'], buf['H']
    cls = buf['cls']
    bgl = srgb_to_lin((241, 232, 222))  # used only for the warm floor bounce
    rgb = np.zeros((cls.size, 3)); alpha = np.zeros(cls.size)
    g = cls == 9
    alpha[g] = np.clip((1 - buf['gshadow'][g]) * shadow_strength, 0, 1) if shadow else 0
    rgb[g] = 0
    hidx, pos, nrm, part = buf['hidx'], buf['pos'], buf['nrm'], buf['part']
    sub = cls[hidx]
    for k, hx in ((1, body_hex), (2, accent_hex)):
        m = sub == k
        ii = hidx[m]
        if oak:
            alb = oak_albedo_photo(pos[m].astype(float), nrm[m].astype(float), part[m])
        else:
            alb = np.tile(srgb_to_lin([int(hx[i:i + 2], 16) for i in (1, 3, 5)]), (ii.size, 1))
            alb *= (1 + 0.012 * _hash_noise(pos[m, 0], pos[m, 1], pos[m, 2], 0.35))[:, None]
        light = key * buf['diff'][ii] + ambk * buf['amb'][ii]
        warm = 0.30 * buf['bounce'][ii][:, None] * bgl[None, :]
        rgb[ii] = alb * (light[:, None] + warm) + buf['spec'][ii, None] * (1.4 if oak else 1.0)
        alpha[ii] = 1
    rgb = lin_to_srgb(tone(rgb))
    out = np.dstack([rgb.reshape(H, W, 3), alpha.reshape(H, W)])
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def shadow_rgba(buf, strength=0.8):
    """Floor shadow only (black + alpha), to be layered with mix-blend-mode: multiply."""
    W, H = buf['W'], buf['H']
    a = np.where(buf['cls'] == 9, np.clip((1 - buf['gshadow']) * strength, 0, 1), 0)
    out = np.zeros((H, W, 4)); out[..., 3] = a.reshape(H, W)
    return (out * 255 + 0.5).astype(np.uint8)
