"""
simulate_v6.py — forced–dissipative barotropic vorticity on a rotating sphere.

    ∂ω/∂t + J(ψ, ω+f) = −ν(−∇²)⁴ω − μω + F

    ω = ∇²ψ        relative vorticity (the scalar ω_z we plot)
    f = 2Ω sinφ    planetary vorticity / Coriolis  (only the l=1,m=0 mode)
    J              spherical Jacobian (advection of absolute vorticity q=ω+f)
    ν(−∇²)⁴        ∇⁸ hyperviscosity  (small-scale / filament cutoff)
    μ              uniform linear (Rayleigh) drag  (arrests the inverse cascade)
    F              stochastic forcing in a narrow high-l band

This is the SAME equation and the SAME (verified-correct) numerics as v5 — RK2
(Heun) for advection + exact integrating factor for dissipation (Lie splitting,
globally O(dt)), and a metric-consistent spectral Jacobian via physical gradient
components.  The nonlinear substep is 2nd-order Heun and the linear factor is
exact, but the Lie–Trotter split between them makes the GLOBAL scheme first-order
(step-halving error ratio = 2.00; v6_critical_audit.md §2.1).  This is fine for
strongly-dissipated steady-state statistics — and with white-in-time forcing the
strong order is ≤ 1 regardless — but the scheme is NOT globally 2nd-order.  Two
things differ from v5:

  1. The Coriolis coefficient is FIXED.  pyshtools with normalization='4pi'
     uses the real harmonic  Y₁⁰ = √3·cosθ = √3·sinφ  (∫Y²dΩ = 4π).  To get
     f = 2Ω sinφ the coefficient must be  2Ω/√3, NOT the v5 value
     2Ω·√(4π/3), which was too large by exactly √(4π) ≈ 3.545 (physics_audit.md).

  2. The parameters (config_v6.py) are chosen so the flow actually develops the
     turbulence it claims: forcing at l≈60–80 (well above the Rhines scale),
     weak drag μ=0.03, and enough resolution/spinup for both an inverse cascade
     (→ zonal jets, large vortices) and a forward enstrophy cascade
     (→ filaments) to run.  Diagnostics below verify this.

It is NOT a convection model — there is no buoyancy, stratification, energy
equation or vertical velocity.  See README_v6.md.
"""

import numpy as np
import pyshtools as pysh

from config_v6 import (OMEGA, LMAX, NU_HYPER, LINEAR_DRAG, FORCE_LMIN,
                       FORCE_LMAX, FORCE_AMP, DT, N_SPINUP, N_FRAMES,
                       FRAME_SKIP, FRAMES_NPZ)

# ── helpers ────────────────────────────────────────────────────────────────

def _laplacian_eigenvalues(lmax):
    """−l(l+1) for each (l,m) pair, matching pyshtools SHCoeffs layout."""
    ev = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        ev[:, l, :l + 1] = -l * (l + 1)
    return ev


def _dissipation_filter(lmax, nu, drag, dt):
    """
    Exact integrating factor for the linear part over one step:

        exp( -(μ + ν λ⁴) dt ),   λ = l(l+1).

    ν λ⁴ is scale-selective ∇⁸ hyperdiffusion (bites only near truncation); μ is
    uniform linear drag.  Returns a (2, L+1, L+1) multiplicative array.
    """
    ev = _laplacian_eigenvalues(lmax)   # negative
    lam4 = ev ** 4                       # positive
    return np.exp(-(drag + nu * lam4) * dt)


class SpectralVorticity:
    """Vorticity in spectral space (real 4π-normalised SH) + time stepping."""

    def __init__(self):
        self.lmax = LMAX
        self._ev = _laplacian_eigenvalues(self.lmax)          # (2,L+1,L+1)
        self._visc = _dissipation_filter(self.lmax, NU_HYPER, LINEAR_DRAG, DT)

        # Inverse Laplacian eigenvalues (ψ = ∇⁻²ω); l=0 mode is 0.
        self._inv_ev = np.zeros_like(self._ev)
        for l in range(1, self.lmax + 1):
            self._inv_ev[:, l, :l + 1] = -1.0 / (l * (l + 1))

        # Planetary vorticity  f = 2Ω sinφ = 2Ω cosθ  →  only (l,m)=(1,0).
        # pyshtools 4π-normalised real harmonic:  Y₁⁰ = √3 cosθ.
        # So f = 2Ω cosθ = (2Ω/√3)·Y₁⁰  →  coefficient = 2Ω/√3.
        # (v5 used 2Ω·√(4π/3), the ORTHONORMAL value — a factor √(4π) too big.)
        self._f_lm = np.zeros((2, self.lmax + 1, self.lmax + 1))
        self._f_lm[0, 1, 0] = 2.0 * OMEGA / np.sqrt(3.0)

        # Small random initial vorticity in the forcing band.
        rng = np.random.default_rng(42)
        omega_lm = np.zeros((2, self.lmax + 1, self.lmax + 1))
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            for m in range(l + 1):
                amp = FORCE_AMP * 0.1 / (l + 1)
                omega_lm[0, l, m] = rng.standard_normal() * amp
                if m > 0:
                    omega_lm[1, l, m] = rng.standard_normal() * amp
        self.omega_lm = omega_lm

    # ── spectral ↔ grid conversions ─────────────────────────────────────

    def _to_grid(self, clm_array):
        coeffs = pysh.SHCoeffs.from_array(clm_array, normalization='4pi',
                                          csphase=1)
        return coeffs.expand(grid='DH2')

    def _to_lm(self, grid):
        coeffs = grid.expand(normalization='4pi', csphase=1,
                             lmax_calc=self.lmax)
        return coeffs.coeffs

    # ── spherical Jacobian via physical gradient components ─────────────

    def _jacobian_lm(self, a_lm, b_lm):
        """
        J(A,B) = (∇A)_φ (∇B)_θ − (∇A)_θ (∇B)_φ  evaluated on the grid with the
        physical (metric-consistent) horizontal gradient components, then
        transformed back to spectral space.  (Verified correct in the audit.)
        """
        ca = pysh.SHCoeffs.from_array(a_lm, normalization='4pi', csphase=1)
        cb = pysh.SHCoeffs.from_array(b_lm, normalization='4pi', csphase=1)
        ga = ca.gradient(radius=1.0)
        gb = cb.gradient(radius=1.0)
        jac = ga.phi.data * gb.theta.data - ga.theta.data * gb.phi.data
        jac_grid = pysh.SHGrid.from_array(jac, grid='DH')
        return self._to_lm(jac_grid)

    # ── stochastic forcing (white in time, narrow band in l) ────────────

    def _stochastic_forcing(self, rng):
        f_lm = np.zeros_like(self.omega_lm)
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            amp = FORCE_AMP / np.sqrt(l * (l + 1)) * np.sqrt(DT)
            f_lm[0, l, :l + 1] = rng.standard_normal(l + 1) * amp
            if l >= 1:
                f_lm[1, l, 1:l + 1] = rng.standard_normal(l) * amp
        return f_lm

    # ── time step (Heun advection + exact linear factor, Lie split O(dt)) ─

    def _tendency(self, omega_lm):
        psi_lm = self._inv_ev * omega_lm
        abs_vor_lm = omega_lm + self._f_lm
        return -self._jacobian_lm(psi_lm, abs_vor_lm)

    def step(self, rng):
        k1 = self._tendency(self.omega_lm)
        k2 = self._tendency(self.omega_lm + DT * k1)
        rhs = self.omega_lm + 0.5 * DT * (k1 + k2)
        rhs += self._stochastic_forcing(rng)
        self.omega_lm = self._visc * rhs
        self.omega_lm[:, 0, 0] = 0.0          # keep mean vorticity zero

    # ── output ──────────────────────────────────────────────────────────

    def vorticity_grid(self):
        return self._to_grid(self.omega_lm).data

    def coeffs(self):
        return self.omega_lm.copy()


# ── spectral diagnostics ────────────────────────────────────────────────────

def diagnostics(omega_lm):
    """
    Report whether the cascade has developed.  With 4π normalisation and
    Parseval ∫ω²dΩ = 4π Σ c², the 4π factor cancels in every ratio here.

      • enstrophy Z_l = Σ_m c_lm²  per degree l  → fractions below / in / above
        the forcing band (a BROAD spectrum, not one trapped in the band, is the
        signature of a developed cascade).
      • kinetic energy E_lm = c_lm² / [l(l+1)]  → zonal (m=0) energy fraction
        (large ⇒ jets).
      • ω_rms = √(Σ c_lm²)  and the inverse-drag ratio ω_rms/μ.  NOTE: this is
        NOT a Reynolds number — it is a nondimensional drag parameter (=100).
        The velocity-based Re at the forcing scale is ~140 (v6_critical_audit
        §18); do not read ω_rms/μ as Re.
    """
    c2 = omega_lm[0] ** 2 + omega_lm[1] ** 2          # (L+1, L+1) over (l,m)
    ens_l = c2.sum(axis=1)                            # per-degree enstrophy
    tot = ens_l.sum() + 1e-300
    below = ens_l[:FORCE_LMIN].sum() / tot
    inband = ens_l[FORCE_LMIN:FORCE_LMAX + 1].sum() / tot
    above = ens_l[FORCE_LMAX + 1:].sum() / tot

    ll = np.arange(omega_lm.shape[1])
    denom = ll * (ll + 1)
    denom[0] = 1
    e_lm = c2 / denom[:, None]
    e_lm[0, :] = 0.0
    tot_e = e_lm.sum() + 1e-300
    zonal = e_lm[:, 0].sum() / tot_e                  # m=0 column

    omega_rms = np.sqrt(c2.sum())
    return dict(below=below, inband=inband, above=above, zonal=zonal,
                rms=omega_rms, drag_ratio=omega_rms / LINEAR_DRAG)


def _fmt(d):
    return (f"ens[below/in/above band]={d['below']*100:4.1f}/"
            f"{d['inband']*100:4.1f}/{d['above']*100:4.1f}%  "
            f"zonalE={d['zonal']*100:4.1f}%  "
            f"ω_rms={d['rms']:5.2f}  ω_rms/μ={d['drag_ratio']:5.1f}")


# ── run simulation ───────────────────────────────────────────────────────────

def run_simulation(n_spinup=N_SPINUP, n_frames=N_FRAMES, frame_skip=FRAME_SKIP,
                   verbose=True):
    """
    Spin up to a statistically steady state, then record n_frames snapshots of
    the spectral coefficients.  Returns (coeff_frames, final_diag) where
    coeff_frames is a list of (2, L+1, L+1) arrays.
    """
    rng = np.random.default_rng(0)
    model = SpectralVorticity()

    if verbose:
        print(f"Spinning up for {n_spinup} steps …", flush=True)
    for i in range(n_spinup):
        model.step(rng)
        if verbose and (i + 1) % 1000 == 0:
            print(f"  spinup {i+1:5d}/{n_spinup}   {_fmt(diagnostics(model.omega_lm))}",
                  flush=True)

    frames = []
    if verbose:
        print(f"Recording {n_frames} frames (every {frame_skip} steps) …",
              flush=True)
    for i in range(n_frames):
        for _ in range(frame_skip):
            model.step(rng)
        frames.append(model.coeffs())
        if verbose and (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{n_frames}   {_fmt(diagnostics(model.omega_lm))}",
                  flush=True)

    final = diagnostics(model.omega_lm)
    if verbose:
        print("\nSaturated-state diagnostics:")
        print("  " + _fmt(final), flush=True)
    return frames, final


if __name__ == "__main__":
    frames, final = run_simulation()
    arr = np.array(frames)                       # (N_FRAMES, 2, L+1, L+1)
    np.savez_compressed(FRAMES_NPZ, coeffs=arr, lmax=LMAX)
    print(f"Saved {FRAMES_NPZ}  ({arr.shape[0]} frames, T{LMAX})")
