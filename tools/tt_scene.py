"""Posed TuttoTondo scenes (flat sheet, slotting, dropping the top) on top of tt_render.

Each pose maps a world point to every part's own coordinates; the part shapes are the exact
ones from tt_render (plates 70 x 40 x 2 with a radius-30 cut, top Ø 60 x 2)."""
import numpy as np
import tt_render as R

r = R.R_EDGE


def _plate(u, yl, w, kind):
    return R.extrude(R.plate2d(u, yl, kind) + r, w, 1 - r) - r


def _disc(rad, yl):
    return R.extrude(rad - 30 + r, yl, 1 - r) - r


POSES = {
    # parts lying in the sheet, pulled 2 cm apart
    'cut': dict(A=lambda x, y, z: (x, z + 42, y - 1), B=lambda x, y, z: (x, 42 - z, y - 1),
                D=lambda x, y, z: (np.hypot(x, z), y - 1)),
    # plate B standing, plate A coming down onto it, top waiting on the floor
    'slot': dict(A=lambda x, y, z: (x, y - 24, z), B=lambda x, y, z: (z, y, x),
                 D=lambda x, y, z: (np.hypot(x - 78, z + 14), y - 1)),
    # the cross assembled, the top coming down
    'drop': dict(A=lambda x, y, z: (x, y, z), B=lambda x, y, z: (z, y, x),
                 D=lambda x, y, z: (np.hypot(x, z), y - 39 - 26)),
    'final': dict(A=lambda x, y, z: (x, y, z), B=lambda x, y, z: (z, y, x),
                  D=lambda x, y, z: (np.hypot(x, z), y - 39)),
}


def use_pose(name):
    P = POSES[name]

    def parts(x, y, z):
        a = _plate(*P['A'](x, y, z), 'A')
        b = _plate(*P['B'](x, y, z), 'B')
        d = _disc(*P['D'](x, y, z))
        return a, b, d

    def classify(p, n, part):
        c = np.ones(p.shape[0], np.int8)
        ub = P['B'](p[:, 0], p[:, 1], p[:, 2])[0]
        c[(part == 1) & (np.abs(ub) > 34.75)] = 2                 # plate B's end faces
        yl = P['D'](p[:, 0], p[:, 1], p[:, 2])[1]
        c[(part == 2) & (yl > -0.85)] = 2                         # top face and edge
        return c

    R.sdf_parts = parts
    R.sdf = lambda x, y, z: np.minimum(np.minimum(*parts(x, y, z)[:2]), parts(x, y, z)[2])
    R.classify_parts = classify
