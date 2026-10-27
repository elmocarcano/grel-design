"""Render every TuttoTondo image used on the site from the exact geometry.
Run from the repo root:  python3 tools/final.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import tt_render as R
D = os.environ.get('OUT', 'img/')
t = time.time()
buf = R.render_buffers(1800, 1686, az=45, el=21, dist=225, fov=24, target=(0, 16.5, 0),
                       light_pos=(90, 300, 240), fill_dir=(0.5, 0.3, -0.8))
print('lighting', round(time.time() - t), 's')
def save(body, acc, name, oak=False):
    img = R.composite(buf, body, acc, oak=oak)
    Image.fromarray(img).resize((900, 843), Image.LANCZOS).save(D + name, quality=87, optimize=True, progressive=True)
save('#7E2428', '#E8DFD0', 'tt-burgundy-cream.jpg')
save('#000000', '#000000', 'tt-oak.jpg', oak=True)
C = {'cream': '#E8DFD0', 'butter': '#E3D5AE', 'rose': '#DDB3AA', 'terracotta': '#C98A74',
     'sage': '#A9B79F', 'lake': '#A6BACB', 'bordeaux': '#7E2428', 'graphite': '#3B3A38'}
for b in C:
    for a in C:
        save(C[b], C[a], f'c-{b}-{a}.jpg')
print('done', round(time.time() - t), 's')
