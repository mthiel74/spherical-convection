"""
End-to-end pipeline: simulate → render → save GIF (+ optional MP4).
"""

import sys
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (OUTPUT_GIF, OUTPUT_MP4, FPS, N_FRAMES, FRAME_SKIP, DT,
                    N_SPINUP)
from simulate import run_simulation
from visualize import render_frame, fig_to_rgb


def make_animation(frames, output_gif=OUTPUT_GIF, output_mp4=None):
    """Render all frames and write GIF (and optionally MP4)."""
    imgs = []
    t0   = N_SPINUP * DT          # simulation time at first recorded frame
    dt_f = FRAME_SKIP * DT        # time between frames

    total = len(frames)
    for i, field in enumerate(frames):
        t_val = t0 + i * dt_f
        print(f"  rendering frame {i+1}/{total}  t={t_val:.1f}", flush=True)
        fig = render_frame(field, i, total, t_val)
        rgb = fig_to_rgb(fig)
        imgs.append(rgb)
        plt.close(fig)

    print(f"Saving {output_gif} …", flush=True)
    imageio.mimsave(output_gif, imgs, fps=FPS, loop=0)
    print(f"Saved {output_gif}  ({len(imgs)} frames @ {FPS} fps)")

    if output_mp4:
        print(f"Saving {output_mp4} …", flush=True)
        imageio.mimsave(output_mp4, imgs, fps=FPS,
                        macro_block_size=None,
                        quality=7)
        print(f"Saved {output_mp4}")


def main():
    save_mp4 = '--mp4' in sys.argv

    print("=== Spherical Convection Simulation ===")
    frames = run_simulation()

    print(f"\n=== Rendering {len(frames)} frames ===")
    make_animation(frames,
                   output_gif=OUTPUT_GIF,
                   output_mp4=OUTPUT_MP4 if save_mp4 else None)

    print("\nDone.")


if __name__ == "__main__":
    main()
