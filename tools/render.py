"""Draufsicht einer Bedrock-Welt als PNG (1 Pixel = 1 Block)."""
import sys
import numpy as np
import amulet
from PIL import Image

world_dir, out_png = sys.argv[1], sys.argv[2]
DIM = "minecraft:overworld"
AIR = {"air", "cave_air", "void_air", "structure_void", "light_block", "barrier"}

RULES = [
    ("water", (52, 98, 186)), ("ice", (150, 190, 240)),
    ("leaves", (48, 104, 40)), ("azalea", (60, 120, 50)),
    ("grass_block", (98, 150, 62)), ("short_grass", (110, 160, 70)), ("tall_grass", (110, 160, 70)),
    ("fern", (90, 140, 60)), ("moss", (90, 130, 50)), ("flower", (190, 170, 80)),
    ("dirt", (130, 95, 65)), ("farmland", (115, 80, 50)), ("podzol", (110, 80, 45)), ("mud", (90, 75, 60)),
    ("sand", (220, 205, 150)), ("gravel", (140, 135, 130)),
    ("brick", (150, 70, 55)), ("granite", (160, 105, 85)),
    ("log", (100, 75, 45)), ("wood", (100, 75, 45)), ("planks", (165, 130, 85)),
    ("stairs", (120, 110, 105)), ("slab", (125, 115, 110)),
    ("glass", (175, 205, 215)),
    ("white_", (225, 225, 220)), ("light_gray", (160, 160, 155)), ("gray_", (90, 90, 95)),
    ("black", (35, 35, 40)), ("red_", (160, 50, 45)), ("orange", (210, 120, 40)),
    ("yellow", (220, 190, 60)), ("lime", (120, 180, 60)), ("green", (75, 110, 45)),
    ("cyan", (40, 130, 140)), ("light_blue", (100, 160, 210)), ("blue", (50, 70, 160)),
    ("purple", (120, 60, 160)), ("magenta", (170, 70, 160)), ("pink", (220, 140, 160)),
    ("brown", (110, 75, 50)), ("terracotta", (160, 95, 70)),
    ("smooth_stone", (165, 165, 165)), ("stone", (128, 128, 128)), ("cobble", (115, 115, 115)),
    ("andesite", (135, 135, 135)), ("diorite", (190, 190, 190)), ("deepslate", (75, 75, 80)),
    ("quartz", (230, 225, 215)), ("concrete", (180, 180, 180)), ("wool", (210, 210, 210)),
    ("rail", (130, 110, 90)), ("path", (150, 125, 80)), ("snow", (245, 245, 250)),
]


def color(name):
    for key, c in RULES:
        if key in name:
            return c
    return (150, 140, 130)


level = amulet.load_level(world_dir)
coords = list(level.all_chunk_coords(DIM))
xs = [c[0] for c in coords]
zs = [c[1] for c in coords]
cx0, cz0 = min(xs), min(zs)
W = (max(xs) - cx0 + 1) * 16
H = (max(zs) - cz0 + 1) * 16
img = np.zeros((H, W, 3), np.uint8)
height = np.full((H, W), -9999, np.int32)
print(f"{len(coords)} Chunks, Bild {W}x{H}, Ursprung Block {cx0*16},{cz0*16}", flush=True)

for n, (cx, cz) in enumerate(coords):
    try:
        ch = level.get_chunk(cx, cz, DIM)
    except Exception:
        continue
    pal = ch.block_palette
    names = [b.base_name for b in pal]
    is_air = np.array([nm in AIR for nm in names] + [True])
    cols = np.array([color(nm) for nm in names] + [(0, 0, 0)], np.uint8)
    top_id = np.full((16, 16), -1, np.int64)
    top_y = np.full((16, 16), -9999, np.int64)
    for cy in sorted(ch.blocks.sub_chunks, reverse=True):
        arr = np.asarray(ch.blocks.get_sub_chunk(cy))  # x, y, z
        for y in range(15, -1, -1):
            layer = arr[:, y, :]
            solid = ~is_air[np.clip(layer, 0, len(names))]
            new = solid & (top_id < 0)
            top_id[new] = layer[new]
            top_y[new] = cy * 16 + y
        if (top_id >= 0).all():
            break
    x0, z0 = (cx - cx0) * 16, (cz - cz0) * 16
    tid = np.where(top_id >= 0, top_id, len(names))
    img[z0:z0 + 16, x0:x0 + 16] = cols[tid].transpose(1, 0, 2)
    height[z0:z0 + 16, x0:x0 + 16] = top_y.T
    if n % 500 == 0:
        print(n, flush=True)

# Schattierung: Gebaeude/Hoehen werfen nach Suedosten Schatten
h = height.astype(np.float32)
nb = np.roll(np.roll(h, 1, 0), 1, 1)
shade = np.clip((h - nb) * 0.06, -0.35, 0.35) + 1.0
tall = np.clip((h - np.median(h)) * 0.012, -0.15, 0.25)
out = np.clip(img.astype(np.float32) * (shade + tall)[..., None], 0, 255).astype(np.uint8)
Image.fromarray(out).save(out_png)
np.save(out_png.replace(".png", "_height.npy"), height.astype(np.int16))
print("fertig", out_png)
