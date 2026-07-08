"""
verify_v6.py — spectral diagnostics on the saturated v6 state.

Loads frames_v6.npz (the recorded spectral coefficients), averages the enstrophy
and kinetic-energy spectra over all frames, and reports whether the forward
(enstrophy → filaments) and inverse (energy → large scales) cascades developed —
i.e. that the spectrum is BROAD rather than trapped in the forcing band as in v5.

    python3 verify_v6.py            # print numbers + save spectrum_v6.png

Writes spectrum_v6.png locally and (if reachable) to the iCloud Claude drop zone.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_v6 import (FRAMES_NPZ, FORCE_LMIN, FORCE_LMAX, LINEAR_DRAG, OMEGA,
                       NU_HYPER, LMAX)


def main():
    data = np.load(FRAMES_NPZ)
    coeffs = data['coeffs']                    # (N, 2, L+1, L+1)
    N, _, Lp1, _ = coeffs.shape
    L = Lp1 - 1
    ll = np.arange(Lp1)

    # frame-averaged per-degree enstrophy Z_l = Σ_m (c0²+c1²)  and energy Z_l/λ
    c2 = coeffs[:, 0] ** 2 + coeffs[:, 1] ** 2         # (N, L+1, L+1)
    ens_lm = c2.mean(axis=0)                            # (L+1, L+1)
    Z_l = ens_lm.sum(axis=1)                            # (L+1,)
    lam = ll * (ll + 1); lam[0] = 1
    E_l = Z_l / lam                                    # KE per degree
    E_l[0] = 0.0

    totZ = Z_l.sum()
    below = Z_l[:FORCE_LMIN].sum() / totZ
    inband = Z_l[FORCE_LMIN:FORCE_LMAX + 1].sum() / totZ
    above = Z_l[FORCE_LMAX + 1:].sum() / totZ

    e_lm = ens_lm / lam[:, None]; e_lm[0] = 0.0
    zonal = e_lm[:, 0].sum() / e_lm.sum()
    omega_rms = np.sqrt(c2.sum(axis=(1, 2)).mean())

    # Rhines degree l_R = sqrt(beta / 2U): beta ~ 2Omega (equator, nondim R=1);
    # U from total KE  (½∫u² = 2π Σ E_l)  →  U = sqrt(2 * KE / (4π)).
    KE = 2 * np.pi * E_l.sum()
    U = np.sqrt(2 * KE / (4 * np.pi))
    beta = 2 * OMEGA
    l_R = np.sqrt(beta / (2 * U)) if U > 0 else np.nan

    print(f"frames={N}  T{L}")
    print(f"ω_rms = {omega_rms:.3f}   effective Re ≈ ω_rms/μ = {omega_rms/LINEAR_DRAG:.1f}")
    print(f"enstrophy fractions  below/in/above forcing band "
          f"(l={FORCE_LMIN}-{FORCE_LMAX}): "
          f"{below*100:.1f}% / {inband*100:.1f}% / {above*100:.1f}%")
    print(f"zonal (m=0) energy fraction = {zonal*100:.1f}%")
    print(f"rms velocity U ≈ {U:.3f}   Rhines degree l_R ≈ {l_R:.1f}  "
          f"(forcing at l={FORCE_LMIN}-{FORCE_LMAX} → scale separation "
          f"{'OK' if l_R < FORCE_LMIN else 'MARGINAL'})")
    print("\nInterpretation: a BROAD spectrum (in-band well below 100%, with "
          "power both below → inverse cascade/large scales and above → forward "
          "enstrophy cascade/filaments) is the developed-cascade signature v5 "
          "lacked (v5: 97% trapped in-band).")

    # ── spectrum figure ──────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    l = ll[1:]

    ax = axs[0]
    ax.loglog(l, Z_l[1:], color='C3', lw=1.6)
    ax.axvspan(FORCE_LMIN, FORCE_LMAX, color='0.85', label='forcing band')
    ax.axvline(l_R, color='C0', ls='--', lw=1.0, label=f'Rhines l_R≈{l_R:.0f}')
    # enstrophy-cascade reference slope Z(l) ~ l^{-1}  (E(k)~k^{-3})
    ref = Z_l[FORCE_LMAX] * (l / FORCE_LMAX) ** (-1.0)
    m = (l >= FORCE_LMAX) & (l <= LMAX - 10)
    ax.loglog(l[m], ref[m], color='0.4', ls=':', lw=1.2, label=r'$l^{-1}$ (fwd)')
    ax.set_xlabel('degree l'); ax.set_ylabel(r'enstrophy $Z_l$')
    ax.set_title('Enstrophy spectrum'); ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')

    ax = axs[1]
    ax.loglog(l, E_l[1:], color='C2', lw=1.6)
    ax.axvspan(FORCE_LMIN, FORCE_LMAX, color='0.85')
    ax.axvline(l_R, color='C0', ls='--', lw=1.0)
    ax.set_xlabel('degree l'); ax.set_ylabel(r'kinetic energy $E_l$')
    ax.set_title('Energy spectrum'); ax.grid(alpha=.3, which='both')

    fig.suptitle(f"v6 saturated spectra (T{L}, μ={LINEAR_DRAG}, forcing l="
                 f"{FORCE_LMIN}–{FORCE_LMAX})   ω_rms={omega_rms:.2f}, "
                 f"Re≈{omega_rms/LINEAR_DRAG:.0f}, zonalE={zonal*100:.1f}%",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("spectrum_v6.png", dpi=120)
    print("\nSaved spectrum_v6.png")

    icloud = os.path.expanduser(
        "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
        "2026-07-08_spherical-convection_v6_spectrum.png")
    try:
        os.makedirs(os.path.dirname(icloud), exist_ok=True)
        fig.savefig(icloud, dpi=120)
        print(f"Copied spectrum → {icloud}")
    except OSError as e:
        print(f"  could not copy to iCloud: {e}")


if __name__ == "__main__":
    main()
