"""
verify_v7.py — spectral diagnostics on the saturated v7 state.

Loads frames_v7.npz (the recorded spectral coefficients) and reports, on the
frame-averaged spectrum:

  • enstrophy / energy spectra and the below/in/above-band fractions,
  • stationarity of the recorded window (energy, enstrophy, zonal fraction),
  • the energy & enstrophy budgets (injection vs dissipation, incl. SVV),
  • the Rhines degree and forcing→Rhines separation ratio,
  • [NEW, improvement #8] the spectral ENERGY and ENSTROPHY FLUXES Π_E(l), Π_Z(l)
    from the nonlinear Jacobian transfer — the rigorous, direct evidence of a
    dual cascade (a broad spectrum alone is only consistent with one).

Adapts every v6 diagnostic (verify_v6.py) to the v7 config/solver, and adds the
flux measurement + a flux panel next to the spectra.

    python3 verify_v7.py            # print numbers + save spectrum_v7.png

References: Boffetta & Ecke (2012) Annu. Rev. Fluid Mech. 44, 427; Frisch,
Turbulence (Cambridge, 1995).  Writes spectrum_v7.png locally and (if reachable)
to the iCloud Claude drop zone.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyshtools as pysh

from config_v7 import (FRAMES_NPZ, FORCE_LMIN, FORCE_LMAX, LINEAR_DRAG, OMEGA,
                       NU_HYPER, LMAX, FORCE_BAND_SUM,
                       SVV_ENABLED, SVV_EPS0, SVV_LCUT)
from simulate_v7 import effective_force_amp, _svv_rate


# ── operators for the nonlinear transfer (resolution-agnostic) ───────────────
# Standalone copies of the solver's building blocks so spectral_flux() works on
# any (2,L+1,L+1) field (any L) without instantiating the LMAX-fixed model.

def _inv_laplacian(lmax):
    """ψ_lm = ∇⁻²ω_lm = −ω_lm/λ, λ=l(l+1); the (2,L+1,L+1) multiplier (0 at l=0)."""
    inv = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(1, lmax + 1):
        inv[:, l, :l + 1] = -1.0 / (l * (l + 1))
    return inv


def _coriolis_lm(lmax, omega):
    """Planetary vorticity f = 2Ω sinφ = (2Ω/√3)·Y₁⁰ (4π-normalised); only (1,0)."""
    f = np.zeros((2, lmax + 1, lmax + 1))
    f[0, 1, 0] = 2.0 * omega / np.sqrt(3.0)
    return f


def _jacobian_lm(a_lm, b_lm, lmax):
    """
    J(A,B) = (∇A)_φ (∇B)_θ − (∇A)_θ (∇B)_φ, evaluated with pyshtools' physical
    (metric-consistent) horizontal gradients, then transformed back to spectral
    space — identical to the solver's Jacobian (verified correct in the audit).
    """
    ca = pysh.SHCoeffs.from_array(a_lm, normalization='4pi', csphase=1)
    cb = pysh.SHCoeffs.from_array(b_lm, normalization='4pi', csphase=1)
    ga = ca.gradient(radius=1.0)
    gb = cb.gradient(radius=1.0)
    jac = ga.phi.data * gb.theta.data - ga.theta.data * gb.phi.data
    jac_grid = pysh.SHGrid.from_array(jac, grid='DH')
    return jac_grid.expand(normalization='4pi', csphase=1,
                           lmax_calc=lmax).coeffs


def _transfer_one(omega_lm, omega_rot):
    """
    Per-degree nonlinear transfer for ONE vorticity field (improvement #8).

    With ψ = ∇⁻²ω, absolute vorticity q = ω + f, and the barotropic Jacobian
    J ≡ J(ψ, q), the nonlinear tendency is N = −J.  The nonlinear transfer INTO
    degree l (the rate the Jacobian feeds energy / enstrophy to degree l) is,
    summing over the order m at fixed degree l (real 4π harmonics ⇒ the "Re{·}"
    is just the sum of the two real cos/sin components):

        T_E(l) = Σ_m  ψ_lm · J_lm          (= Re{ψ*_lm J_lm},  the task's form;
                                             also = Σ_m (1/λ) ω_lm N_lm — energy)
        T_Z(l) = Σ_m  ω_lm · N_lm  = −Σ_m ω_lm · J_lm      (enstrophy)

    T_E uses ψ*·J directly.  T_Z is the enstrophy analogue built from the
    TENDENCY N=−J (equivalently −ω*·J); this sign is what makes the forward
    enstrophy flux Π_Z(l) POSITIVE above the forcing band (see spectral_flux).
    Both are conservative: Σ_l T_E = Σ_l T_Z = 0 up to dealiasing error, because
    the barotropic Jacobian conserves energy and enstrophy exactly.

    Returns (T_E, T_Z), each a length-(L+1) array indexed by degree l.
    """
    lmax = omega_lm.shape[1] - 1
    inv = _inv_laplacian(lmax)
    psi_lm = inv * omega_lm                       # ψ = −ω/λ
    q_lm = omega_lm + omega_rot                   # absolute vorticity ω + f
    J = _jacobian_lm(psi_lm, q_lm, lmax)          # J(ψ, ω+f)

    # sum the two real components (cos, sin) then over m at each degree l
    T_E = (psi_lm[0] * J[0] + psi_lm[1] * J[1]).sum(axis=1)          # Σ_m ψ·J
    T_Z = -(omega_lm[0] * J[0] + omega_lm[1] * J[1]).sum(axis=1)     # Σ_m ω·N
    return T_E, T_Z


def spectral_flux(coeffs):
    """
    Spectral energy & enstrophy FLUX from the nonlinear Jacobian transfer
    (improvement #8; scientific_improvements.md §8).

    Parameters
    ----------
    coeffs : (2,L+1,L+1) single field, or (N,2,L+1,L+1) stack of frames.
             A stack is transfer-averaged over frames (the flux is only
             meaningful as a statistical, steady-state quantity).

    Returns dict with, per degree l = 0…L,
        T_E, T_Z : nonlinear transfer INTO degree l (energy, enstrophy),
        Pi_E, Pi_Z : the cumulative spectral FLUXES

            Π_E(l) = −Σ_{l'=1}^{l} T_E(l'),   Π_Z(l) = −Σ_{l'=1}^{l} T_Z(l').

    Π_E(l) is the net rate energy is transferred from degrees ≤ l to degrees > l
    (Π_E = −cumsum T_E, since Σ_l T_E = 0 ⇒ Π_E(l) = Σ_{l'>l} T_E(l')).  The
    DUAL-CASCADE SIGNATURE is:
        • Π_E(l) < 0 for l BELOW the forcing band  → inverse ENERGY cascade
          (energy flows up-scale, toward small l), and
        • Π_Z(l) > 0 for l ABOVE the forcing band  → forward ENSTROPHY cascade
          (enstrophy flows down-scale, toward large l).
    A broad spectrum is merely consistent with a dual cascade; these flux signs
    are the direct evidence (Boffetta & Ecke 2012).
    """
    coeffs = np.asarray(coeffs, float)
    single = coeffs.ndim == 3
    stack = coeffs[None] if single else coeffs
    lmax = stack.shape[2] - 1
    f_lm = _coriolis_lm(lmax, OMEGA)

    T_E = np.zeros(lmax + 1)
    T_Z = np.zeros(lmax + 1)
    for fr in stack:
        te, tz = _transfer_one(fr, f_lm)
        T_E += te
        T_Z += tz
    T_E /= len(stack)
    T_Z /= len(stack)

    # Π(l) = −Σ_{l'=1}^{l} T(l');  T[0]=0 (ψ,ω have no l=0 part) so cumsum is exact
    Pi_E = -np.cumsum(T_E)
    Pi_Z = -np.cumsum(T_Z)
    return dict(l=np.arange(lmax + 1), T_E=T_E, T_Z=T_Z, Pi_E=Pi_E, Pi_Z=Pi_Z)


# ── stationarity (adapted verbatim from verify_v6) ───────────────────────────

def stationarity(coeffs, frac=0.10):
    """
    Is the recorded window statistically stationary?  Compare energy E,
    enstrophy Z and the zonal-energy fraction over the FIRST `frac` and LAST
    `frac` of the frames and report the drift.  A stationary run drifts ≈0.
    Warns if any quantity drifts by more than 2 %.  (v7 spins up with an
    auto-stationarity guard — improvement #2 — so this should now pass.)
    """
    N = coeffs.shape[0]
    k = max(1, int(round(frac * N)))
    ll = np.arange(coeffs.shape[2])
    lam = (ll * (ll + 1)).astype(float); lam[0] = 1.0

    def stats(block):
        c2 = block[:, 0] ** 2 + block[:, 1] ** 2
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
              f"recorded window is a transient. Spin up longer and re-record.")
    else:
        print(f"  ✓ stationary (max |drift| = {worst:.1f}% ≤ 2%)")
    return dict(dE=dE, dZ=dZ, dzonal=dz)


# ── energy / enstrophy budget (adapted from verify_v6; +SVV, +effective amp) ──

def energy_budget(Z_l):
    """
    Energy & enstrophy budget: injection (from the forcing) vs dissipation (from
    the frame-averaged spectrum × the linear dissipation rates).  In a stationary
    state injection ≈ dissipation.

    Injection uses the EFFECTIVE forcing amplitude (improvement #5a: literal, or
    derived from ε).  amp² = amp_eff², with (2l+1) real coefficients per degree:
        Ż_inj = Σ_band (2l+1)·amp²/λ ,   Ė_inj = Σ_band (2l+1)·amp²/λ² = amp²·S_band
    (Ė_inj is exactly ε.)  Dissipation rates per mode are 2×(drag + hyper + SVV):
        drag  2μ ,   hyper 2νλ⁴ ,   SVV 2·ε_SVV(l)·λ  (improvement #7, if enabled).

    Warns if |injection/dissipation − 1| > 5 % for either invariant.
    """
    lmax = len(Z_l) - 1
    ll = np.arange(len(Z_l))
    lam = (ll * (ll + 1)).astype(float); lam[0] = 1.0
    E_l = Z_l / lam; E_l[0] = 0.0

    amp = effective_force_amp()
    band = np.arange(FORCE_LMIN, FORCE_LMAX + 1)
    lamb = band * (band + 1.0)
    Z_inj = (amp ** 2 * (2 * band + 1) / lamb).sum()
    E_inj = (amp ** 2 * (2 * band + 1) / lamb ** 2).sum()   # = amp²·S_band = ε

    Z_drag = (2.0 * LINEAR_DRAG * Z_l).sum()
    Z_hyp = (2.0 * NU_HYPER * lam ** 4 * Z_l).sum()
    E_drag = (2.0 * LINEAR_DRAG * E_l).sum()
    E_hyp = (2.0 * NU_HYPER * lam ** 4 * E_l).sum()

    Z_svv = E_svv = 0.0
    if SVV_ENABLED:
        # per-degree SVV rate ε_SVV(l)·λ (m-independent → take the m=0 column)
        svv_l = _svv_rate(lmax, SVV_EPS0, SVV_LCUT)[0, :, 0]
        Z_svv = (2.0 * svv_l * Z_l).sum()
        E_svv = (2.0 * svv_l * E_l).sum()

    Z_out = Z_drag + Z_hyp + Z_svv
    E_out = E_drag + E_hyp + E_svv

    print("\nEnergy / enstrophy budget (injection vs dissipation):")
    print(f"  ε (energy injection rate) = {E_inj:.3e}   "
          f"(forcing amp_eff = {amp:.3f}, S_band = {FORCE_BAND_SUM:.3e})")
    svv_note = f" + SVV {E_svv:.3e}" if SVV_ENABLED else ""
    print(f"  energy    inj = {E_inj:.3e}   diss = {E_out:.3e} "
          f"(drag {E_drag:.3e} + hyper {E_hyp:.3e}{svv_note})")
    print(f"            in/out = {E_inj/E_out:.3f}   "
          f"hyper takes {E_hyp/E_out*100:.0f}% of energy dissipation")
    svv_note = f" + SVV {Z_svv:.3e}" if SVV_ENABLED else ""
    print(f"  enstrophy inj = {Z_inj:.3e}   diss = {Z_out:.3e} "
          f"(drag {Z_drag:.3e} + hyper {Z_hyp:.3e}{svv_note})")
    print(f"            in/out = {Z_inj/Z_out:.3f}")
    for name, r in (("energy", E_inj / E_out), ("enstrophy", Z_inj / Z_out)):
        if abs(r - 1.0) > 0.05:
            print(f"  ⚠️  {name} budget does not close: in/out = {r:.3f} "
                  f"(|ratio−1| = {abs(r-1)*100:.0f}% > 5%) — not in steady state.")
        else:
            print(f"  ✓ {name} budget closes (in/out = {r:.3f})")
    return dict(E_inj=E_inj, E_out=E_out, Z_inj=Z_inj, Z_out=Z_out)


# ── inviscid conservation guard (adapted from verify_v6 → v7 solver) ─────────

def inviscid_conservation(coeffs0=None, n_steps=200, dt=None):
    """
    OPTIONAL solver regression guard (not run by default — call explicitly).

    Run the advection substep ONLY — no forcing, no dissipation — from a
    saturated field and check that energy and enstrophy are conserved.  The exact
    barotropic Jacobian conserves both; a large drift would signal an aliasing /
    Jacobian regression.  Uses the v7 solver's own _tendency (Heun substep).

    coeffs0: a (2,L+1,L+1) starting field.  Defaults to the last recorded frame.
    """
    from simulate_v7 import SpectralVorticity
    from config_v7 import DT
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


# ── main ─────────────────────────────────────────────────────────────────────

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
    # U from total KE (½∫u² = 2π Σ E_l)  →  U = sqrt(2 * KE / (4π)).
    KE = 2 * np.pi * E_l.sum()
    U = np.sqrt(2 * KE / (4 * np.pi))
    beta = 2 * OMEGA
    l_R = np.sqrt(beta / (2 * U)) if U > 0 else np.nan

    print(f"frames={N}  T{L}")
    print(f"ω_rms = {omega_rms:.3f}   drag ratio ω_rms/μ = {omega_rms/LINEAR_DRAG:.1f} "
          f"(NOT a Reynolds number — a nondim inverse-drag parameter)")
    print(f"enstrophy fractions  below/in/above forcing band "
          f"(l={FORCE_LMIN}-{FORCE_LMAX}): "
          f"{below*100:.1f}% / {inband*100:.1f}% / {above*100:.1f}%")
    print(f"zonal (m=0) energy fraction = {zonal*100:.1f}%")
    sep = FORCE_LMIN / l_R if l_R > 0 else np.nan     # forcing/Rhines ratio
    print(f"rms velocity U ≈ {U:.3f}   Rhines degree l_R ≈ {l_R:.1f}  "
          f"(forcing at l={FORCE_LMIN}-{FORCE_LMAX} → separation ratio "
          f"{sep:.1f}× → {'good' if sep >= 5 else 'MARGINAL — no jets expected'})")

    # ── stationarity + budget ─────────────────────────────────────────────
    stationarity(coeffs)
    energy_budget(Z_l)

    # ── spectral flux (improvement #8) ────────────────────────────────────
    flux = spectral_flux(coeffs)
    Pi_E, Pi_Z = flux['Pi_E'], flux['Pi_Z']
    # conservation check: the total transfer should sum to ≈0 (Π at l=L)
    print("\nSpectral flux (nonlinear Jacobian transfer, improvement #8):")
    print(f"  Σ_l T_E = {flux['T_E'].sum():+.2e}   Σ_l T_Z = {flux['T_Z'].sum():+.2e}"
          f"   (both → 0: the Jacobian conserves E and Z)")
    # mid-band representative degrees below / above the forcing
    lo = max(1, FORCE_LMIN // 2)                        # a degree below the band
    hi = min(L - 1, (FORCE_LMAX + L) // 2)              # a degree above the band
    print(f"  Π_E(l={lo}) = {Pi_E[lo]:+.2e}  "
          f"({'✓ <0 → inverse energy cascade' if Pi_E[lo] < 0 else '⚠ not <0'})")
    print(f"  Π_Z(l={hi}) = {Pi_Z[hi]:+.2e}  "
          f"({'✓ >0 → forward enstrophy cascade' if Pi_Z[hi] > 0 else '⚠ not >0'})")
    inv_ok = Pi_E[lo] < 0
    fwd_ok = Pi_Z[hi] > 0
    if inv_ok and fwd_ok:
        print("  ✓ DUAL CASCADE confirmed: Π_E<0 below the band AND Π_Z>0 above it.")
    else:
        print("  ⚠️  dual-cascade signature incomplete (flux signs above) — the "
              "spectrum may be broad without a developed cascade.")

    # ── spectrum + flux figure ────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    l = ll[1:]

    ax = axs[0]
    ax.loglog(l, Z_l[1:], color='C3', lw=1.6)
    ax.axvspan(FORCE_LMIN, FORCE_LMAX, color='0.85', label='forcing band')
    ax.axvline(l_R, color='C0', ls='--', lw=1.0, label=f'Rhines l_R≈{l_R:.0f}')
    ref = Z_l[FORCE_LMAX] * (l / FORCE_LMAX) ** (-1.0)
    m = (l >= FORCE_LMAX) & (l <= LMAX - 10)
    ax.loglog(l[m], ref[m], color='0.4', ls=':', lw=1.2, label=r'$l^{-1}$ (fwd)')
    ax.set_xlabel('degree l'); ax.set_ylabel(r'enstrophy $Z_l$')
    ax.set_title('Enstrophy spectrum'); ax.legend(fontsize=8)
    ax.grid(alpha=.3, which='both')

    ax = axs[1]
    ax.loglog(l, E_l[1:], color='C2', lw=1.6)
    ax.axvspan(FORCE_LMIN, FORCE_LMAX, color='0.85')
    ax.axvline(l_R, color='C0', ls='--', lw=1.0)
    ax.set_xlabel('degree l'); ax.set_ylabel(r'kinetic energy $E_l$')
    ax.set_title('Energy spectrum'); ax.grid(alpha=.3, which='both')

    # flux panel (improvement #8): Π_E and Π_Z on a signed (symlog) axis
    ax = axs[2]
    ax.axhline(0.0, color='0.6', lw=0.8)
    ax.axvspan(FORCE_LMIN, FORCE_LMAX, color='0.85', label='forcing band')
    ax.plot(ll[1:], Pi_E[1:], color='C0', lw=1.6, label=r'$\Pi_E$ (energy)')
    ax.plot(ll[1:], Pi_Z[1:], color='C3', lw=1.6, label=r'$\Pi_Z$ (enstrophy)')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=max(1e-12,
                  0.01 * max(np.abs(Pi_E).max(), np.abs(Pi_Z).max(), 1e-12)))
    ax.set_xlabel('degree l'); ax.set_ylabel(r'spectral flux $\Pi(l)$')
    ax.set_title('Spectral fluxes (dual cascade)')
    ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')

    fig.suptitle(f"v7 saturated spectra & fluxes (T{L}, μ={LINEAR_DRAG}, forcing "
                 f"l={FORCE_LMIN}–{FORCE_LMAX})   ω_rms={omega_rms:.2f}, "
                 f"zonalE={zonal*100:.1f}%", fontsize=10)
    fig.tight_layout()
    fig.savefig("spectrum_v7.png", dpi=120)
    print("\nSaved spectrum_v7.png")

    icloud = os.path.expanduser(
        "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
        "2026-07-08_spherical-convection_v7_spectrum.png")
    try:
        os.makedirs(os.path.dirname(icloud), exist_ok=True)
        fig.savefig(icloud, dpi=120)
        print(f"Copied spectrum → {icloud}")
    except OSError as e:
        print(f"  could not copy to iCloud: {e}")


if __name__ == "__main__":
    main()
