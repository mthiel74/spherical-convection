"""
render_v7.py — render the 800-frame extended v6 run to a ~40 s MP4.

Reuses the exact v6 rendering pipeline (visualize_v6.render_frame) and the v6
ffmpeg encoder (render_movie_v6.write_mp4).  Reads frames_v6_long.npz, writes
output_v7.mp4, and copies it to the iCloud "Claude" drop zone.  No GIF (800
frames of 720x720 would be ~100 MB).
"""

import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_v6 import FPS, FRAME_SKIP, DT, N_SPINUP
from visualize_v6 import render_frame, fig_to_rgb
from render_movie_v6 import write_mp4

SRC_NPZ = "frames_v6_long.npz"
OUT_MP4 = "output_v7.mp4"
ICLOUD_MP4 = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
    "2026-07-08_spherical-convection_v7.mp4"
)

def main():
    data = np.load(SRC_NPZ)
    frames = list(data["coeffs"])
    total = len(frames)
    print(f"Loaded {total} frames (T{int(data['lmax'])}) "
          f"-> {total / FPS:.1f} s @ {FPS} fps")

    t0 = N_SPINUP * DT
    dt_f = FRAME_SKIP * DT
    imgs = []
    for i, coeffs in enumerate(frames):
        t_val = t0 + i * dt_f
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  rendering frame {i+1}/{total}  t={t_val:.1f}", flush=True)
        fig = render_frame(coeffs, i, total, t_val)
        imgs.append(fig_to_rgb(fig))
        plt.close(fig)

    print(f"Encoding {OUT_MP4} …", flush=True)
    if write_mp4(imgs, OUT_MP4, FPS):
        try:
            os.makedirs(os.path.dirname(ICLOUD_MP4), exist_ok=True)
            shutil.copy2(OUT_MP4, ICLOUD_MP4)
            print(f"Copied MP4 -> {ICLOUD_MP4}")
        except OSError as e:
            print(f"  could not copy to iCloud: {e}")

if __name__ == "__main__":
    main()
