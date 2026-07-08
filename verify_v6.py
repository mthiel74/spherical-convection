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
                       NU_HYPER, LMAX, FORCE_AMP)


def stationarity(coeffs, frac=0.10):
    """
    Is the recorded window statistically stationary?  Compare energy E,
    enstrophy Z and the zonal-energy fraction over the FIRST `frac` and LAST
    `frac` of the frames and report the drift.  A stationary run drifts ≈0; the
    v6 audit found the recorded window was a transient (E +3.7 %, zonal −26 %
    over 200 frames, §4.1), which this check would have caught.  Warns if any
    quantity drifts by more than 2 %.
    """
    N = coeffs.shape[0]
    k = max(1, int(round(frac * N)))
    ll = np.arange(coeffs.shape[2])
    lam = (ll * (ll + 1)).astype(float); lam[0] = 1.0

    def stats(block):
        c2 = block[:, 0] ** 2 + block[:, 1] ** 2       # (n, L+1, L+1)
        Z_lm = c2.mean(axis=0)
        Z = Z_lm.sum()
        E_lm = Z_lm / lam[:, None]; E_lm[0] = 0.0
        E = E_lm.sum()
        zonal = E_lm[:, 0].sum() / E
        return E, Z, zonal

    E0, Z0, z0 = stats(coeffs[:k])
    E1, Z1, z1 = stats(coeffs[-k:])
    drift = lambda a, b: (b - a) / a * 100.0 if a != 0 else np.nan
    dE, dZ, dz = drift(E0, E1), drift(Z0, Z1), drift(z0, z1)

    print(f"\nStationarity (first {k} vs last {k} of {N} frames):")
    print(f"  energy    E: {E0:.4e} → {E1:.4e}   drift {dE:+.1f}%")
    print(f"  enstrophy Z: {Z0:.4e} → {Z1:.4e}   drift {dZ:+.1f}%")
    print(f"  zonal frac : {z0*100:.2f}% → {z1*100:.2f}%   drift {dz:+.1f}%")
    worst = max(abs(dE), abs(dZ), abs(dz))
    if worst > 2.0:
        print(f"  ⚠️  NON-STATIONARY: max |drift| = {worst:.1f}% > 2% — the "
              f"recorded window is a transient, not a steady state. Spin up "
              f"longer (config N_SPINUP) and re-record.")
    else:
        print(f"  ✓ stationary (max |drift| = {worst:.1f}% ≤ 2%)")
    return dict(dE=dE, dZ=dZ, dzonal=dz)


def energy_budget(Z_l):
    """
    Energy & enstrophy budget: injection (from the forcing parameters) vs
    dissipation (from the frame-averaged spectrum × the linear dissipation
    eigenvalues).  In a stationary state injection ≈ dissipation.  The v6 audit
    found energy in/out = 1.10 (+10 % imbalance, still charging, §4.3).

    Injection (white-in-time forcing, amp²=FORCE_AMP²/λ per unit time, 2l+1 real
    coefficients per degree l in the band):
        Ż_inj = Σ_band (2l+1)·FORCE_AMP²/λ ,   Ė_inj = Σ_band (2l+1)·FORCE_AMP²/λ²
    Dissipation (exact linear factor exp[−(μ+νλ⁴)dt] ⇒ loss rate 2(μ+νλ⁴)):
        Ż_diss = 2 Σ_l (μ+νλ⁴)·Z_l ,          Ė_diss = 2 Σ_l (μ+νλ⁴)·E_l ,  E_l=Z_l/λ

    Warns if |injection/dissipation − 1| > 5 % for either invariant.
    """
    ll = np.arange(len(Z_l))
    lam = (ll * (ll + 1)).astype(float); lam[0] = 1.0
    E_l = Z_l / lam; E_l[0] = 0.0

    band = np.arange(FORCE_LMIN, FORCE_LMAX + 1)
    lamb = band * (band + 1.0)
    Z_inj = (FORCE_AMP ** 2 * (2 * band + 1) / lamb).sum()
    E_inj = (FORCE_AMP ** 2 * (2 * band + 1) / lamb ** 2).sum()

    Z_drag = (2.0 * LINEAR_DRAG * Z_l).sum()
    Z_hyp = (2.0 * NU_HYPER * lam ** 4 * Z_l).sum()
    E_drag = (2.0 * LINEAR_DRAG * E_l).sum()
    E_hyp = (2.0 * NU_HYPER * lam ** 4 * E_l).sum()
    Z_out, E_out = Z_drag + Z_hyp, E_drag + E_hyp

    print("\nEnergy / enstrophy budget (injection vs dissipation):")
    print(f"  energy    inj = {E_inj:.3e}   diss = {E_out:.3e} "
          f"(drag {E_drag:.3e} + hyper {E_hyp:.3e})")
    print(f"            in/out = {E_inj/E_out:.3f}   "
          f"hyper takes {E_hyp/E_out*100:.0f}% of energy dissipation")
    print(f"  enstrophy inj = {Z_inj:.3e}   diss = {Z_out:.3e} "
          f"(drag {Z_drag:.3e} + hyper {Z_hyp:.3e})")
    print(f"            in/out = {Z_inj/Z_out:.3f}")
    for name, r in (("energy", E_inj / E_out), ("enstrophy", Z_inj / Z_out)):
        if abs(r - 1.0) > 0.05:
            print(f"  ⚠️  {name} budget does not close: in/out = {r:.3f} "
                  f"(|ratio−1| = {abs(r-1)*100:.0f}% > 5%) — not in steady state.")
        else:
            print(f"  ✓ {name} budget closes (in/out = {r:.3f})")
    return dict(E_inj=E_inj, E_out=E_out, Z_inj=Z_inj, Z_out=Z_out)


def inviscid_conservation(coeffs0=None, n_steps=200, dt=None):
    """
    OPTIONAL solver regression guard (not run by default — call explicitly).

    Run the advection substep ONLY — no forcing, no dissipation — from a
    saturated field and check that energy and enstrophy are conserved.  The
    exact barotropic Jacobian conserves both; the v6 audit measured ~1e-6 drift
    per 200 steps (§2.4).  A large drift would signal an aliasing / Jacobian
    regression.

    coeffs0: a (2,L+1,L+1) starting field.  Defaults to the last recorded frame.
    """
    from simulate_v6 import SpectralVorticity
    from config_v6 import DT
    dt = DT if dt is None else dt

    if coeffs0 is None:
        coeffs0 = np.load(FRAMES_NPZ)['coeffs'][-1]

    model = SpectralVorticity()
    model.omega_lm = np.asarray(coeffs0, float).copy()
    model.omega_lm[:, 0, 0] = 0.0
    ll = np.arange(model.lmax + 1)
    lam = (ll * (ll + 1)).astype(float); lam[0] = 1.0

    def EZ(c):
        c2 = c[0] ** 2 + c[1] ** 2
        E_lm = c2 / lam[:, None]; E_lm[0] = 0.0
        return E_lm.sum(), c2.sum()

    E0, Z0 = EZ(model.omega_lm)
    for _ in range(n_steps):
        k1 = model._tendency(model.omega_lm)
        k2 = model._tendency(model.omega_lm + dt * k1)
        model.omega_lm = model.omega_lm + 0.5 * dt * (k1 + k2)
        model.omega_lm[:, 0, 0] = 0.0
    E1, Z1 = EZ(model.omega_lm)

    dE = (E1 - E0) / E0 if E0 else np.nan
    dZ = (Z1 - Z0) / Z0 if Z0 else np.nan
    print(f"\nInviscid conservation ({n_steps} advection-only steps):")
    print(f"  energy    {E0:.6e} → {E1:.6e}   rel drift {dE:+.2e}")
    print(f"  enstrophy {Z0:.6e} → {Z1:.6e}   rel drift {dZ:+.2e}")
    if max(abs(dE), abs(dZ)) < 1e-3:
        print("  ✓ solver conserves E and Z (Jacobian is effectively non-aliasing)")
    else:
        print("  ⚠️  conservation degraded — possible Jacobian / aliasing regression")
    return dict(dE=dE, dZ=dZ)


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
    print(f"ω_rms = {omega_rms:.3f}   drag ratio ω_rms/μ = {omega_rms/LINEAR_DRAG:.1f} "
          f"(NOT a Reynolds number — a nondim inverse-drag parameter; "
          f"velocity-based Re at the forcing scale is ~140)")
    print(f"enstrophy fractions  below/in/above forcing band "
          f"(l={FORCE_LMIN}-{FORCE_LMAX}): "
          f"{below*100:.1f}% / {inband*100:.1f}% / {above*100:.1f}%")
    print(f"zonal (m=0) energy fraction = {zonal*100:.1f}%")
    sep = FORCE_LMIN / l_R if l_R > 0 else np.nan     # forcing/Rhines ratio
    print(f"rms velocity U ≈ {U:.3f}   Rhines degree l_R ≈ {l_R:.1f}  "
          f"(forcing at l={FORCE_LMIN}-{FORCE_LMAX} → separation ratio "
          f"{sep:.1f}× → {'good' if sep >= 5 else 'MARGINAL — no jets expected'})")
    print("\nInterpretation: a BROAD spectrum (in-band well below 100%, with "
          "power both below → inverse cascade/large scales and above → forward "
          "enstrophy cascade/filaments) is the developed-cascade signature v5 "
          "lacked (v5: 97% trapped in-band).")

    # ── stationarity + budget (would have caught the v6 transient) ────────
    stationarity(coeffs)
    energy_budget(Z_l)

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
                 f"ω_rms/μ≈{omega_rms/LINEAR_DRAG:.0f}, zonalE={zonal*100:.1f}%",
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
