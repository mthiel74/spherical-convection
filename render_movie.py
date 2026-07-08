"""
End-to-end pipeline: simulate → render → save GIF + MP4.

The MP4 is encoded directly with the system `ffmpeg` (raw RGB frames piped to
libx264, yuv420p) and a copy is pushed to the iCloud "Claude" drop zone.
"""

import os
import sys
import shutil
import subprocess

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (OUTPUT_GIF, OUTPUT_MP4, ICLOUD_MP4, FPS, FRAME_SKIP, DT,
                    N_SPINUP, OMEGA)
from simulate import run_simulation
from visualize import render_frame, fig_to_rgb


# ── MP4 via a raw-RGB pipe into ffmpeg ─────────────────────────────────────
def write_mp4(imgs, path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("  ffmpeg not found — skipping MP4")
        return False

    h, w, _ = imgs[0].shape
    # libx264 + yuv420p needs even dimensions
    w2, h2 = w - (w % 2), h - (h % 2)

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vf", f"crop={w2}:{h2}:0:0",
        # Apple-compatible baseline H.264.  Note: 720x720 = 2025 macroblocks
        # exceeds H.264 level 3.0 (max 1620 MB/frame), so x264 auto-selects the
        # true minimum level (3.1) — 3.0 is not attainable at this resolution.
        "-c:v", "libx264", "-profile:v", "baseline",
        "-pix_fmt", "yuv420p", "-crf", "18",
        "-preset", "slow", "-movflags", "+faststart",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for im in imgs:
        proc.stdin.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        print(f"  ffmpeg exited with code {rc}")
        return False
    print(f"Saved {path}  ({len(imgs)} frames @ {fps} fps)")
    return True


def make_animation(frames, output_gif=OUTPUT_GIF, output_mp4=None,
                   icloud_mp4=None):
    imgs = []
    # Time is displayed in rotation periods:  t_rot = t_nondim · Ω / (2π).
    # (One rotation period = 2π/Ω in these non-dimensional units.)
    rot = OMEGA / (2.0 * np.pi)
    t0 = N_SPINUP * DT * rot
    dt_f = FRAME_SKIP * DT * rot
    total = len(frames)
    for i, field in enumerate(frames):
        t_val = t0 + i * dt_f
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  rendering frame {i+1}/{total}  t={t_val:.1f}", flush=True)
        fig = render_frame(field, i, total, t_val)
        imgs.append(fig_to_rgb(fig))
        plt.close(fig)

    print(f"Saving {output_gif} …", flush=True)
    imageio.mimsave(output_gif, imgs, fps=FPS, loop=0)
    print(f"Saved {output_gif}  ({len(imgs)} frames @ {FPS} fps)")

    if output_mp4:
        print(f"Saving {output_mp4} …", flush=True)
        if write_mp4(imgs, output_mp4, FPS) and icloud_mp4:
            try:
                os.makedirs(os.path.dirname(icloud_mp4), exist_ok=True)
                shutil.copy2(output_mp4, icloud_mp4)
                print(f"Copied MP4 → {icloud_mp4}")
            except OSError as e:
                print(f"  could not copy to iCloud: {e}")


def main():
    save_mp4 = '--mp4' in sys.argv

    print("=== Spherical Convection Simulation ===")
    frames = run_simulation()

    print(f"\n=== Rendering {len(frames)} frames ===")
    make_animation(frames,
                   output_gif=OUTPUT_GIF,
                   output_mp4=OUTPUT_MP4 if save_mp4 else None,
                   icloud_mp4=ICLOUD_MP4 if save_mp4 else None)
    print("\nDone.")


if __name__ == "__main__":
    main()
