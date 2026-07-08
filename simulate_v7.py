"""
simulate_v7.py — forced–dissipative barotropic vorticity on a rotating sphere.

    ∂ω/∂t + J(ψ, ω+f) = −ν(−∇²)⁴ω − μω + F

    ω = ∇²ψ        relative vorticity (the scalar ω_z we plot)
    f = 2Ω sinφ    planetary vorticity / Coriolis  (only the l=1,m=0 mode)
    J              spherical Jacobian (advection of absolute vorticity q=ω+f)
    ν(−∇²)⁴        ∇⁸ hyperviscosity  (small-scale / filament cutoff)
    μ              uniform linear (Rayleigh) drag  (arrests the inverse cascade)
    F              stochastic forcing in a narrow high-l band

This is the SAME equation and the SAME (verified-correct) spatial discretisation
as v6 — RK2 (Heun) for advection + an exact integrating factor for the linear
dissipation — with the FIXED Coriolis coefficient 2Ω/√3 (see ._f_lm).  The
parameters (config_v7) implement scientific improvement #1: a wide forcing–Rhines
separation (l_f≈100–120, l_R≈10, ratio ≈11; weak drag μ=0.01; T170) aimed at the
zonostrophic regime (target R_β ≳ 2) so that zonal jets can actually form.  See
config_v7.py for the full parameter derivation and honesty caveats, and
scientific_improvements.md §1.

Improvements implemented on top of the v6 solver:
  • #2  auto-stationarity detection during spin-up  (spinup_to_stationary)
  • #3  STRANG (2nd-order) operator splitting for the time step
        S(dt) = L(dt/2)·N(dt)·L(dt/2)  — replaces the old Lie–Trotter O(dt) split

It is NOT a convection model — there is no buoyancy, stratification, energy
equation or vertical velocity.  See README_v6.md.
"""

import numpy as np
import pyshtools as pysh

from config_v7 import (OMEGA, LMAX, NU_HYPER, LINEAR_DRAG, FORCE_LMIN,
                       FORCE_LMAX, FORCE_AMP, DT, N_SPINUP, N_FRAMES,
                       FRAME_SKIP, FRAMES_NPZ,
                       STATIONARITY_INTERVAL, STATIONARITY_WINDOW,
                       STATIONARITY_TOL, N_SPINUP_MIN)

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
        # Full-step linear integrating factor exp(−(μ+νλ⁴)·dt) …
        self._visc = _dissipation_filter(self.lmax, NU_HYPER, LINEAR_DRAG, DT)
        # … and the HALF-step factor exp(−(μ+νλ⁴)·dt/2) = √(visc) needed by the
        # symmetric Strang split L(dt/2)·N(dt)·L(dt/2)  (improvement #3).
        self._sqrt_visc = _dissipation_filter(self.lmax, NU_HYPER, LINEAR_DRAG,
                                              DT / 2.0)

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

    # ── time step (Strang split: ½-diss · Heun advection · ½-diss + forcing) ─

    def _tendency(self, omega_lm):
        """Nonlinear tendency N(ω) = −J(ψ, ω+f),  ψ = ∇⁻²ω."""
        psi_lm = self._inv_ev * omega_lm
        abs_vor_lm = omega_lm + self._f_lm
        return -self._jacobian_lm(psi_lm, abs_vor_lm)

    def step(self, rng):
        """
        Advance one step with STRANG (2nd-order) operator splitting
        (improvement #3).

        Write the vorticity equation as  ∂ω/∂t = L ω + N(ω), where
            L ω  = −(μ + ν λ⁴) ω        (linear dissipation + drag; λ = l(l+1))
            N(ω) = −J(ψ, ω+f)           (nonlinear advection of absolute vort.)
        Let L(τ) = exp(Lτ) be the EXACT linear flow (diagonal in spectral space;
        the half-step factor is self._sqrt_visc = exp(L·dt/2)) and N(dt) the
        full-step Heun (RK2) advance of the nonlinear part.  The SYMMETRIC
        composition (Strang 1968, SIAM J. Numer. Anal. 5, 506)

            S(dt) = L(dt/2) · N(dt) · L(dt/2)

        is 2nd-order accurate: local truncation error O(dt³), global error
        O(dt²).  By the Baker–Campbell–Hausdorff expansion the leading splitting
        error of the ASYMMETRIC Lie–Trotter product L(dt)·N(dt) is
        ½dt²[L,N] + O(dt³) — first order globally; symmetrising cancels that
        commutator term, leaving −(dt³/24)([L,[L,N]] + 2[N,[L,N]]) + …, i.e.
        second order.  This restores the 2nd order that Heun and the integrating
        factor already have individually but the previous Lie split destroyed
        (observed step-halving error ratio ≈ 2.0 → target ≈ 4.0).

        The stochastic forcing is white-in-time (a √dt Wiener increment, not a
        smooth tendency), so it is added as a separate increment AFTER the
        deterministic symmetric split — preserving the √dt scaling, exactly as
        before.
        """
        # ½-step dissipation:  ω ← exp(L·dt/2) ω   [ = ω · √(visc_filter) ]
        w = self._sqrt_visc * self.omega_lm
        # full-step Heun (RK2) advection of the nonlinear Jacobian:  N(dt)
        k1 = self._tendency(w)
        k2 = self._tendency(w + DT * k1)
        w = w + 0.5 * DT * (k1 + k2)
        # ½-step dissipation:  ω ← exp(L·dt/2) ω
        w = self._sqrt_visc * w
        # add stochastic forcing after the split (white-in-time √dt increment)
        w = w + self._stochastic_forcing(rng)
        self.omega_lm = w
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
        NOT a Reynolds number — it is a nondimensional drag parameter.  Do not
        read ω_rms/μ as Re.
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
    enstrophy = c2.sum()                              # Z = Σ_lm c_lm²
    energy = e_lm.sum()                               # E = Σ_lm c_lm²/[l(l+1)]
    return dict(below=below, inband=inband, above=above, zonal=zonal,
                rms=omega_rms, drag_ratio=omega_rms / LINEAR_DRAG,
                energy=energy, enstrophy=enstrophy)


def _fmt(d):
    return (f"ens[below/in/above band]={d['below']*100:4.1f}/"
            f"{d['inband']*100:4.1f}/{d['above']*100:4.1f}%  "
            f"zonalE={d['zonal']*100:4.1f}%  "
            f"ω_rms={d['rms']:5.2f}  ω_rms/μ={d['drag_ratio']:5.1f}")


# ── auto-stationarity detection (improvement #2) ─────────────────────────────

def _window_is_stationary(samples, tol):
    """
    True when EVERY monitored quantity has plateaued across the sliding window.

    `samples` is a sequence of (energy, enstrophy, zonal) tuples spanning the
    last STATIONARITY_WINDOW steps.  For each of the three columns q we form the
    fractional peak-to-peak drift

            drift(q) = (max q − min q) / |mean q|

    and require drift(q) < tol for all three.  Using the window spread rather
    than a first-vs-last difference rejects a run that merely happens to return
    to its starting value while still oscillating, and one that drifts slowly but
    monotonically — both are non-stationary.  Energy and enstrophy (the two
    quadratic invariants) settling together with the zonal fraction is the
    signature of a statistically steady 2-D turbulent flow (improvement #2).
    """
    arr = np.asarray(samples, dtype=float)            # (n_samples, 3)
    spread = arr.max(axis=0) - arr.min(axis=0)
    mean = np.abs(arr.mean(axis=0)) + 1e-300
    return bool(np.all(spread / mean < tol))


def spinup_to_stationary(model, rng, n_min=N_SPINUP_MIN, n_max=N_SPINUP,
                         interval=STATIONARITY_INTERVAL,
                         window=STATIONARITY_WINDOW, tol=STATIONARITY_TOL,
                         verbose=True):
    """
    Advance `model` until the flow is statistically stationary, then return the
    number of steps taken.

    Instead of a fixed spin-up length, monitor the energy E, enstrophy Z and
    zonal energy fraction every `interval` steps and keep the samples covering
    the last `window` steps.  As soon as all three have drifted by less than
    `tol` (2%) across that sliding window — AND at least `n_min` steps have been
    taken (min guard) — declare the flow statistically steady and stop.  `n_max`
    (= N_SPINUP) is the hard upper guard: if stationarity is never detected we
    stop there regardless, so the routine always terminates.

    The window holds  window // interval  samples (e.g. 5000/1000 = 5).  The
    stationarity test only fires once the window is full, i.e. after at least
    `window` steps of history exist.
    """
    from collections import deque
    maxlen = max(1, window // interval)
    samples = deque(maxlen=maxlen)                    # recent (E, Z, zonal)

    if verbose:
        print(f"Spinning up (auto-stationary: |drift|<{tol:.0%} of E,Z,zonal "
              f"over {window} steps; guards {n_min}…{n_max}) …", flush=True)

    steps = 0
    for i in range(n_max):
        model.step(rng)
        steps = i + 1
        if steps % interval == 0:
            d = diagnostics(model.omega_lm)
            samples.append((d['energy'], d['enstrophy'], d['zonal']))
            full = len(samples) == maxlen
            stationary = full and _window_is_stationary(samples, tol)
            if verbose:
                flag = " STATIONARY" if (stationary and steps >= n_min) else ""
                print(f"  spinup {steps:6d}/{n_max}   {_fmt(d)}{flag}", flush=True)
            if stationary and steps >= n_min:
                if verbose:
                    print(f"  → statistically steady after {steps} steps "
                          f"(< {n_max} max); begin recording.", flush=True)
                return steps

    if verbose:
        print(f"  → reached max spin-up {n_max} steps without meeting the "
              f"stationarity tolerance; begin recording anyway.", flush=True)
    return steps


# ── run simulation ───────────────────────────────────────────────────────────

def run_simulation(n_spinup=N_SPINUP, n_frames=N_FRAMES, frame_skip=FRAME_SKIP,
                   verbose=True):
    """
    Spin up to a statistically steady state (auto-detected — improvement #2),
    then record n_frames snapshots of the spectral coefficients.  Returns
    (coeff_frames, final_diag) where coeff_frames is a list of (2, L+1, L+1)
    arrays.  `n_spinup` is passed through as the maximum spin-up guard.
    """
    rng = np.random.default_rng(0)
    model = SpectralVorticity()

    spinup_to_stationary(model, rng, n_max=n_spinup, verbose=verbose)

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
