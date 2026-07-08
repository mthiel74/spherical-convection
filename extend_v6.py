"""
extend_v6.py — extend the v6 run to 800 frames (40 s @ 20 fps) WITHOUT re-spinning.

The 200 frames in frames_v6.npz are already the saturated, post-spinup state, so
we reload the LAST frame's spectral coefficients, continue the identical dynamics
(same DT / FRAME_SKIP, fresh white-in-time forcing stream) for 600 more frames,
and concatenate.  Result: 800 contiguous frames written to frames_v6_long.npz.
"""

import numpy as np

from config_v6 import FRAME_SKIP, LMAX
from simulate_v6 import SpectralVorticity, diagnostics, _fmt

SRC_NPZ = "frames_v6.npz"
OUT_NPZ = "frames_v6_long.npz"
N_EXTRA = 600            # 200 existing + 600 = 800 frames = 40 s @ 20 fps

def main():
    data = np.load(SRC_NPZ)
    existing = data["coeffs"]                    # (200, 2, L+1, L+1)
    print(f"Loaded {existing.shape[0]} existing frames (T{int(data['lmax'])})")

    model = SpectralVorticity()
    model.omega_lm = existing[-1].copy()         # resume from saturated state
    rng = np.random.default_rng(20260708)        # fresh forcing stream

    print(f"Continuing for {N_EXTRA} frames (every {FRAME_SKIP} steps) …",
          flush=True)
    extra = []
    for i in range(N_EXTRA):
        for _ in range(FRAME_SKIP):
            model.step(rng)
        extra.append(model.coeffs())
        if (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{N_EXTRA}   {_fmt(diagnostics(model.omega_lm))}",
                  flush=True)

    all_frames = np.concatenate([existing, np.array(extra)], axis=0)
    np.savez_compressed(OUT_NPZ, coeffs=all_frames, lmax=LMAX)
    print(f"\nSaved {OUT_NPZ}  ({all_frames.shape[0]} frames, T{LMAX})")

if __name__ == "__main__":
    main()
