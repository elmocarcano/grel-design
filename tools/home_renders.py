"""Images for the homepage: hero object, the four verbs, the closing angle, the colour picker set.
Run from the repo root: python3 tools/home_renders.py [part]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import tt_render as R, tt_scene as S
OUT = 'img/home/'
L = dict(light_pos=(90, 300, 240), fill_dir=(0.5, 0.3, -0.8))
CREAM, BORD = '#E8DFD0', '#7E2428'
FIELD = (21, 20, 19)
C = {'cream': '#E8DFD0', 'butter': '#E3D5AE', 'rose': '#DDB3AA', 'terracotta': '#C98A74',
     'sage': '#A9B79F', 'lake': '#A6BACB', 'bordeaux': '#7E2428', 'graphite': '#3B3A38'}

def cut(rgba, pad=24):
    ys, xs = np.where(rgba[..., 3] > 6)
    return Image.fromarray(rgba[max(ys.min()-pad,0):ys.max()+pad, max(xs.min()-pad,0):xs.max()+pad], 'RGBA')

def save_rgba(im, name, w):
    im = im.convert('RGBa'); h = round(im.height * w / im.width)
    im.resize((w, h), Image.LANCZOS).convert('RGBA').save(OUT + name, quality=88, method=6)

def save_rgb(arr, name, w, q=86):
    im = Image.fromarray(arr); h = round(im.height * w / im.width)
    im.resize((w, h), Image.LANCZOS).save(OUT + name, quality=q, optimize=True, progressive=True)

part = sys.argv[1] if len(sys.argv) > 1 else 'all'
if part in ('hero', 'all'):
    S.use_pose('final')
    b = R.render_buffers(2400, 1800, az=36, el=9, dist=200, fov=24, target=(0, 19, 0), **L)
    save_rgba(cut(R.composite_rgba(b, CREAM, BORD, shadow=False)), 'object.webp', 2000)
    b = R.render_buffers(2200, 1650, az=128, el=32, dist=215, fov=24, target=(0, 16, 0), **L)
    save_rgba(cut(R.composite_rgba(b, CREAM, BORD, shadow=False)), 'object-back.webp', 1800)
if part in ('verbs', 'all'):
    for pose, cam, name in [('cut', dict(az=18, el=58, dist=260, target=(0, 0, 0)), 'cut'),
                            ('slot', dict(az=28, el=14, dist=300, target=(22, 24, 0)), 'slot'),
                            ('drop', dict(az=40, el=12, dist=280, target=(0, 33, 0)), 'drop'),
                            ('final', dict(az=62, el=38, dist=230, target=(0, 18, 0)), 'turn')]:
        S.use_pose(pose)
        b = R.render_buffers(2400, 1350, fov=24, **cam, **L)
        save_rgb(R.composite(b, CREAM, BORD, bg=FIELD), f'verb-{name}.jpg', 2400)
if part in ('picker', 'all'):
    S.use_pose('final')
    b = R.render_buffers(1400, 1050, az=45, el=21, dist=225, fov=24, target=(0, 16.5, 0), **L)
    ref = R.composite_rgba(b, '#888888', '#888888', shadow=False)
    ys, xs = np.where(ref[..., 3] > 6); y0, y1, x0, x1 = ys.min()-8, ys.max()+8, xs.min()-8, xs.max()+8
    for bk, bh in C.items():
        for ak, ah in C.items():
            im = Image.fromarray(R.composite_rgba(b, bh, ah, shadow=False)[y0:y1, x0:x1], 'RGBA')
            save_rgba(im, f'pick-{bk}-{ak}.webp', 900)

