"""Seitenansicht (von Osten/Elbe) und Schraegansicht eines Ausschnitts der Welt."""
import sys
import numpy as np
import amulet
from PIL import Image, ImageDraw

_src = open("tools/render.py").read().split("level = amulet.load_level")[0]
_ns = {}
exec(_src.replace("world_dir, out_png = sys.argv[1], sys.argv[2]", ""), _ns)
color, AIR = _ns["color"], _ns["AIR"]

world_dir, out_prefix = sys.argv[1], sys.argv[2]
X0, X1, Z0, Z1 = (int(v) for v in sys.argv[3:7])
Y0, Y1 = -64, 40
DIM = "minecraft:overworld"

level = amulet.load_level(world_dir)
NX, NZ, NY = X1 - X0, Z1 - Z0, Y1 - Y0
vol = np.zeros((NX, NY, NZ, 3), np.uint8)
solid = np.zeros((NX, NY, NZ), bool)

for cx in range(X0 // 16, (X1 - 1) // 16 + 1):
    for cz in range(Z0 // 16, (Z1 - 1) // 16 + 1):
        try:
            ch = level.get_chunk(cx, cz, DIM)
        except Exception:
            continue
        names = [b.base_name for b in ch.block_palette]
        air = np.array([n in AIR or "grass" == n or n in ("short_grass", "tall_grass") for n in names])
        cols = np.array([color(n) for n in names], np.uint8)
        for cy in ch.blocks.sub_chunks:
            if not (Y0 <= cy * 16 < Y1):
                continue
            arr = np.asarray(ch.blocks.get_sub_chunk(cy))
            for lx in range(16):
                x = cx * 16 + lx
                if not (X0 <= x < X1):
                    continue
                for lz in range(16):
                    z = cz * 16 + lz
                    if not (Z0 <= z < Z1):
                        continue
                    ids = arr[lx, :, lz]
                    s = ~air[ids]
                    ys = cy * 16 + np.arange(16) - Y0
                    solid[x - X0, ys, z - Z0] = s
                    vol[x - X0, ys, z - Z0] = cols[ids]

print("Volumen gelesen", solid.sum(), flush=True)

# 1) Seitenansicht von Osten: Blick nach Westen, links = Norden? -> links Sueden, rechts Norden
S = 4
side = np.full((NY, NZ, 3), (205, 222, 235), np.uint8)
for zi in range(NZ):
    for yi in range(NY):
        col = solid[:, yi, zi]
        idx = np.nonzero(col)[0]
        if len(idx):
            xi = idx[-1]                       # oestlichster Block
            depth = (NX - 1 - xi) / NX
            c = vol[xi, yi, zi].astype(np.float32) * (1.0 - 0.45 * depth)
            side[NY - 1 - yi, NZ - 1 - zi] = np.clip(c, 0, 255)
rows = np.nonzero((side != (205, 222, 235)).any(axis=(1, 2)))[0]
side = side[max(rows.min() - 6, 0): rows.max() + 1]
Image.fromarray(side).resize((side.shape[1] * S, side.shape[0] * S), Image.NEAREST).save(out_prefix + "_seite.png")

# 2) Schraegansicht von Suedosten (isometrisch, Maler-Algorithmus)
T = 6  # halbe Kachelbreite in Pixel
W = (NX + NZ) * T + 40
H = (NX + NZ) * T // 2 + NY * T + 40
img = Image.new("RGB", (W, H), (205, 222, 235))
d = ImageDraw.Draw(img)


def pt(x, y, z):
    # Blick von Suedosten: x nach rechts-hinten, z nach links-vorne
    sx = (x - z) * T + NZ * T + 20
    sy = (x + z) * T // 2 - y * T + NY * T + 20
    return sx, sy


def shade(c, f):
    return tuple(int(min(255, v * f)) for v in c)


order = []
for x in range(NX):
    for z in range(NZ):
        order.append((x + z, x, z))
order.sort()
for _, x, z in order:
    col_s = solid[x, :, z]
    for y in np.nonzero(col_s)[0]:
        top_open = y + 1 >= NY or not solid[x, y + 1, z]
        e_open = x + 1 >= NX or not solid[x + 1, y, z]
        s_open = z + 1 >= NZ or not solid[x, y, z + 1]
        if not (top_open or e_open or s_open):
            continue
        c = tuple(int(v) for v in vol[x, y, z])
        if top_open:
            d.polygon([pt(x, y + 1, z), pt(x + 1, y + 1, z), pt(x + 1, y + 1, z + 1), pt(x, y + 1, z + 1)], fill=shade(c, 1.08))
        if e_open:
            d.polygon([pt(x + 1, y + 1, z), pt(x + 1, y + 1, z + 1), pt(x + 1, y, z + 1), pt(x + 1, y, z)], fill=shade(c, 0.78))
        if s_open:
            d.polygon([pt(x, y + 1, z + 1), pt(x + 1, y + 1, z + 1), pt(x + 1, y, z + 1), pt(x, y, z + 1)], fill=shade(c, 0.9))
img.save(out_prefix + "_schraeg.png")
print("fertig")
