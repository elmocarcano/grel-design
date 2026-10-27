"""Cut-out hero renders (transparent WebP). Run from repo root: python3 tools/hero.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import tt_render as R
buf = R.render_buffers(2200, 1650, az=45, el=15, dist=205, fov=24, target=(0, 19, 0),
                       light_pos=(90, 300, 240), fill_dir=(0.5, 0.3, -0.8))
C = {'cream': '#E8DFD0', 'butter': '#E3D5AE', 'rose': '#DDB3AA', 'terracotta': '#C98A74',
     'sage': '#A9B79F', 'lake': '#A6BACB', 'bordeaux': '#7E2428', 'graphite': '#3B3A38'}
SETS = [('cream', 'bordeaux'), ('sage', 'cream'), ('terracotta', 'lake'), ('rose', 'cream'),
        ('bordeaux', 'cream'), ('oak', 'oak')]
box = None
imgs = {}
for b, a in SETS:
    rgba = R.composite_rgba(buf, C.get(b, '#000000'), C.get(a, '#000000'), oak=(b == 'oak'))  # with shadow, for the crop box
    imgs[(b, a)] = rgba
    ys, xs = np.where(rgba[..., 3] > 6)
    bb = (xs.min(), ys.min(), xs.max(), ys.max())
    box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3]))
pad = 20
half = max(1100 - box[0], box[2] - 1100) + pad   # keep the table centred horizontally
x0, x1 = 1100 - half, 1100 + half
y0, y1 = box[1] - pad, box[3] + pad
print('crop', x0, y0, x1, y1)
def out(arr, name):
    im = Image.fromarray(arr[y0:y1, x0:x1], 'RGBA').convert('RGBa')
    w = 1400; h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS).convert('RGBA')
    im.save(name, quality=86, method=6)
    print(name, im.size, os.path.getsize(name) // 1024, 'KB')
out(R.shadow_rgba(buf), 'img/hero-shadow.webp')
for (b, a) in imgs:
    out(R.composite_rgba(buf, C.get(b, '#000000'), C.get(a, '#000000'), oak=(b == 'oak'), shadow=False), f'img/hero-{b}-{a}.webp')
