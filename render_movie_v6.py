"""
render_movie_v6.py — end-to-end v6 pipeline: simulate → render → GIF + MP4.

    python3 render_movie_v6.py            # simulate + render GIF
    python3 render_movie_v6.py --mp4      # also encode MP4 + copy to iCloud
    python3 render_movie_v6.py --from-npz # skip the sim, render frames_v6.npz

The MP4 is encoded with the system ffmpeg (raw RGB piped to libx264, baseline
profile, yuv420p, CRF 18, +faststart) and a copy is pushed to the iCloud
"Claude" drop zone.

Note: time is reported in NON-DIMENSIONAL time units (t = step·DT), NOT
"rotations".  v5 mislabeled the axis as rotations while using a Coriolis
coefficient √(4π) too large; v6 fixes the coefficient and reports honest units.
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

from config_v6 import (OUTPUT_GIF, OUTPUT_MP4, ICLOUD_MP4, FPS, FRAME_SKIP, DT,
                       N_SPINUP, FRAMES_NPZ, LMAX)
from simulate_v6 import run_simulation
from visualize_v6 import render_frame, fig_to_rgb


def write_mp4(imgs, path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("  ffmpeg not found — skipping MP4")
        return False
    h, w, _ = imgs[0].shape
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vf", f"crop={w2}:{h2}:0:0",
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
    t0 = N_SPINUP * DT                     # honest non-dimensional time
    dt_f = FRAME_SKIP * DT
    total = len(frames)
    for i, coeffs in enumerate(frames):
        t_val = t0 + i * dt_f
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  rendering frame {i+1}/{total}  t={t_val:.1f}", flush=True)
        fig = render_frame(coeffs, i, total, t_val)
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
    from_npz = '--from-npz' in sys.argv

    print("=== v6: 2-D barotropic vorticity on a rotating sphere ===")
    if from_npz:
        print(f"Loading {FRAMES_NPZ} …")
        data = np.load(FRAMES_NPZ)
        frames = list(data['coeffs'])
        print(f"  {len(frames)} frames, T{int(data['lmax'])}")
    else:
        frames, final = run_simulation()
        np.savez_compressed(FRAMES_NPZ, coeffs=np.array(frames), lmax=LMAX)
        print(f"Saved {FRAMES_NPZ}")

    print(f"\n=== Rendering {len(frames)} frames ===")
    make_animation(frames,
                   output_gif=OUTPUT_GIF,
                   output_mp4=OUTPUT_MP4 if save_mp4 else None,
                   icloud_mp4=ICLOUD_MP4 if save_mp4 else None)
    print("\nDone.")


if __name__ == "__main__":
    main()
