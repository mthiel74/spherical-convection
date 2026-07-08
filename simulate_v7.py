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
  • #3  STRANG (2nd-order) operator splitting for the time step  (default)
        S(dt) = L(dt/2)·N(dt)·L(dt/2)  — replaces the old Lie–Trotter O(dt) split
  • #4  optional ETDRK4 (4th-order exponential integrator) — selectable via
        config.TIME_SCHEME='etdrk4'; treats the diagonal linear operator exactly

It is NOT a convection model — there is no buoyancy, stratification, energy
equation or vertical velocity.  See README_v6.md.
"""

import numpy as np
import pyshtools as pysh

from config_v7 import (OMEGA, LMAX, NU_HYPER, LINEAR_DRAG, FORCE_LMIN,
                       FORCE_LMAX, FORCE_AMP, DT, N_SPINUP, N_FRAMES,
                       FRAME_SKIP, FRAMES_NPZ,
                       STATIONARITY_INTERVAL, STATIONARITY_WINDOW,
                       STATIONARITY_TOL, N_SPINUP_MIN,
                       TIME_SCHEME, ETDRK4_M,
                       FORCE_BAND_SUM, EPSILON_TARGET, FORCE_FROM_EPSILON,
                       FORCE_TYPE, FORCE_CORR_TIME,
                       SVV_ENABLED, SVV_EPS0, SVV_LCUT,
                       DIFF_ROT_ENABLED, DIFF_ROT_DELTA_OMEGA, DIFF_ROT_TAU)

# ── helpers ────────────────────────────────────────────────────────────────

def _laplacian_eigenvalues(lmax):
    """−l(l+1) for each (l,m) pair, matching pyshtools SHCoeffs layout."""
    ev = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        ev[:, l, :l + 1] = -l * (l + 1)
    return ev


def _svv_rate(lmax, eps0, lcut):
    """
    Spectral-vanishing-viscosity decay rate  ε_SVV(l)·λ  (improvement #7).

    SVV (Tadmor 1989) is a Laplacian-type sink −ε_SVV(l)·λ ω with the smooth
    cutoff kernel (config §7, with L = lmax)

        ε_SVV(l) = ε₀·exp[ −((L−l)/(l−l_cut))² ]   (l > l_cut),   0  (l ≤ l_cut).

    Returns the DECAY RATE array ε_SVV(l)·λ, λ = l(l+1), in the (2,L+1,L+1)
    SHCoeffs layout (≥ 0; identically 0 for l ≤ l_cut), ready to add to the
    linear operator.  Adding it as a rate keeps the linear part diagonal, so the
    exact integrating factor / ETDRK4 machinery is unchanged.
    """
    l = np.arange(lmax + 1).astype(float)
    lam = l * (l + 1.0)                                  # λ per degree
    kernel = np.zeros(lmax + 1)
    hi = l > lcut
    kernel[hi] = np.exp(-((lmax - l[hi]) / (l[hi] - lcut)) ** 2)
    rate_l = eps0 * kernel * lam                         # ε_SVV(l)·λ per degree
    rate = np.zeros((2, lmax + 1, lmax + 1))
    for ll in range(lmax + 1):
        rate[:, ll, :ll + 1] = rate_l[ll]
    return rate


def _dissipation_filter(lmax, nu, drag, dt, svv_rate=None):
    """
    Exact integrating factor for the linear part over one step:

        exp( -(μ + ν λ⁴ + ε_SVV(l)·λ) dt ),   λ = l(l+1).

    ν λ⁴ is scale-selective ∇⁸ hyperdiffusion (bites only near truncation); μ is
    uniform linear drag; ε_SVV(l)·λ is the optional spectral vanishing viscosity
    (improvement #7, 0 unless SVV_ENABLED).  Returns a (2,L+1,L+1) array.
    """
    ev = _laplacian_eigenvalues(lmax)   # negative
    lam4 = ev ** 4                       # positive
    svv = 0.0 if svv_rate is None else svv_rate
    return np.exp(-(drag + nu * lam4 + svv) * dt)


def _linear_operator(lmax, nu, drag, svv_rate=None):
    """
    The diagonal linear operator of the split  ∂ω/∂t = L ω + N(ω):

        L_l = −(μ + ν λ⁴ + ε_SVV(l)·λ),   λ = l(l+1).

    Includes the optional spectral vanishing viscosity ε_SVV(l)·λ (improvement
    #7, 0 unless SVV_ENABLED).  Returns a (2,L+1,L+1) array matching the SHCoeffs
    layout, ≤ 0 everywhere (pure decay).  exp(L·dt) is exactly the full-step
    integrating factor `_dissipation_filter`.
    """
    ev = _laplacian_eigenvalues(lmax)
    lam4 = ev ** 4
    svv = 0.0 if svv_rate is None else svv_rate
    return -(drag + nu * lam4 + svv)


def _etdrk4_coeffs(L, dt, M=32):
    """
    ETDRK4 exponential-integrator coefficients (Cox & Matthews 2002; Kassam &
    Trefethen 2005) for  ∂ω/∂t = L ω + N(ω)  with L DIAGONAL (array `L`).

    ── The scheme.  With h = dt and the matrix exponential of the linear part,
    ETDRK4 advances one step as (Kassam & Trefethen 2005, eqs. 2.5–2.9):

        Nu = N(uₙ)
        a  = E2·uₙ + Q·Nu
        Na = N(a)
        b  = E2·uₙ + Q·Na
        Nb = N(b)
        c  = E2·a  + Q·(2 Nb − Nu)
        Nc = N(c)
        uₙ₊₁ = E·uₙ + Nu·f1 + 2(Na+Nb)·f2 + Nc·f3

    where, with the φ-functions φ₁,φ₂,φ₃,

        E  = e^{hL},   E2 = e^{hL/2},   Q  = (h/2) φ₁(hL/2),
        f1 = h [ −4 − hL + e^{hL}(4 − 3hL + (hL)²) ] / (hL)³,
        f2 = h [  2 + hL + e^{hL}(−2 + hL)          ] / (hL)³,
        f3 = h [ −4 − 3hL − (hL)² + e^{hL}(4 − hL)  ] / (hL)³,
        Q  = h [ e^{hL/2} − 1 ] / (hL).

    ── Why the contour integral.  Evaluated naively these quotients suffer
    CATASTROPHIC CANCELLATION as z = hL → 0 (here |z| ≲ 5e-3 for every mode, so
    the naive form loses ~all significant digits).  φₖ(z) are ENTIRE functions,
    so by Cauchy's integral formula their value equals the mean over any circle
    enclosing z.  Kassam & Trefethen therefore replace each pointwise evaluation
    by the average of the integrand over M points equally spaced on a unit
    circle centred at z:

        φₖ(z) ≈ (1/M) Σ_{j} g_k( z + e^{iθ_j} ),   θ_j = 2π(j+½)/M,

    which is cancellation-free (the integrand is O(1) on the circle) and, because
    the imaginary parts cancel in conjugate pairs, its real part is exact.  We
    take the real part at the end since L is real.

    Returns (E, E2, Q, f1, f2, f3), each a real array with L's shape.
    """
    h = dt
    z = (h * L)[..., None]                                   # (...,1) centres
    theta = 2.0 * np.pi * (np.arange(1, M + 1) - 0.5) / M    # (M,)
    r = np.exp(1j * theta)                                   # unit-circle points
    zc = z + r                                               # (..., M) contour

    E  = np.exp(h * L)
    E2 = np.exp(h * L / 2.0)
    # Q = h·φ₁(z/2) = h·mean[(e^{z/2}−1)/z]
    Q  = h * np.mean((np.exp(zc / 2.0) - 1.0) / zc, axis=-1).real
    zc3 = zc ** 3
    f1 = h * np.mean((-4.0 - zc + np.exp(zc) * (4.0 - 3.0 * zc + zc ** 2)) / zc3,
                     axis=-1).real
    f2 = h * np.mean((2.0 + zc + np.exp(zc) * (-2.0 + zc)) / zc3, axis=-1).real
    f3 = h * np.mean((-4.0 - 3.0 * zc - zc ** 2 + np.exp(zc) * (4.0 - zc)) / zc3,
                     axis=-1).real
    return E, E2, Q, f1, f2, f3


def _diff_rot_target(lmax, delta_omega):
    """
    Target zonal-mean vorticity ω̄_target,l0 for the differential-rotation
    relaxation (improvement #9; scientific_improvements.md §9).

    ── The profile.  A solar-like angular velocity, fastest at the equator,
            Ω(φ) = Ω₀ − ΔΩ sin²φ ,   φ = LATITUDE (φ=0 equator, ±90° poles).
    Ω₀ is already carried by the Coriolis term f = 2Ω₀ sinφ, so only the
    DIFFERENTIAL part δΩ(φ) = Ω(φ) − Ω₀ = −ΔΩ sin²φ enters the relative flow.

    ── From Ω(φ) to a target vorticity.  δΩ is a zonal wind in the co-rotating
    frame, ū(φ) = δΩ(φ)·R cosφ (R cosφ = distance from the spin axis; R=1).  A
    zonal wind derives from a streamfunction by ū = −∂ψ̄/∂y = −(1/R)∂ψ̄/∂φ, so
            ∂ψ̄/∂φ = −R ū = R² ΔΩ sin²φ cosφ
            ⇒ ψ̄_target(φ) = (R² ΔΩ / 3) sin³φ     (integration constant → 0),
    and ω̄_target = ∇²ψ̄_target.  We build ψ̄_target(φ) on the DH2 grid (constant
    in longitude ⇒ a pure m=0 field), expand it, keep the m=0 column, then apply
    ∇² spectrally (eigenvalue −l(l+1)).  No hand-differentiation, so the target
    is exact to the truncation.

    Returns a (2,L+1,L+1) array that is zero except in the m=0 (c[0,l,0]) slots.
    """
    zero = pysh.SHCoeffs.from_zeros(lmax=lmax, normalization='4pi')
    grid = zero.expand(grid='DH2')
    lat = np.deg2rad(grid.lats())                       # latitude of each row
    psi_lat = (delta_omega / 3.0) * np.sin(lat) ** 3    # ψ̄_target(φ), R=1
    grid.data[:] = psi_lat[:, None]                     # constant in longitude
    psi_lm = grid.expand(normalization='4pi', csphase=1, lmax_calc=lmax).coeffs
    psi_target = np.zeros((2, lmax + 1, lmax + 1))
    psi_target[0, :, 0] = psi_lm[0, :, 0]               # keep m=0 only
    omega_target = _laplacian_eigenvalues(lmax) * psi_target   # ∇²ψ̄ = −l(l+1)ψ̄
    return omega_target


def check_dealiasing(lmax, verbose=True):
    """
    Verify the transform grid dealiases the quadratic Jacobian (improvement #6;
    scientific_improvements.md §6).

    ── The Orszag 2/3 (dealiasing) rule.  The only nonlinearity here is the
    QUADRATIC Jacobian J(ψ, ω+f): a product of two fields each band-limited to
    degree L.  A product of two degree-≤L fields contains content up to degree
    2L.  On a physical grid that resolves only up to some maximum degree, any
    content above that maximum is ALIASED — folded back onto the retained
    degrees, contaminating them.  To keep degrees 0…L alias-free after a
    quadratic product, the grid must faithfully represent degrees up to 3L/2:
    then the aliased content (from degrees 3L/2 … 2L) folds only onto degrees
    ≥ L/2 … but is discarded because the result is re-truncated at L before it
    can corrupt the resolved band.  Equivalently one keeps only the lowest 2/3 of
    the grid's resolvable wavenumbers — Orszag's "2/3 rule" (Orszag 1971, JAS 28,
    1074; Boyd, Chebyshev & Fourier Spectral Methods, 2001).

    ── Concretely.  Require the grid to have at least

            N ≥ ⌈ 3(L+1)/2 ⌉   points in BOTH latitude and longitude.

    pyshtools' DH2 (Driscoll–Healy, sampling=2) grid has ≈ 2(L+1) latitudes and
    ≈ 4(L+1) longitudes, both comfortably above 3(L+1)/2 — so quadratic products
    are dealiased at this truncation.  This routine confirms it and warns loudly
    if a future resolution/grid change ever violates the rule.

    Returns dict(nlat, nlon, required, ok).
    """
    required = int(np.ceil(1.5 * (lmax + 1)))       # ⌈ 3(L+1)/2 ⌉
    # Actual grid the solver uses: expand a zero field to read the DH2 dims.
    g = pysh.SHCoeffs.from_zeros(lmax=lmax, normalization='4pi').expand(grid='DH2')
    nlat, nlon = g.data.shape
    ok = (nlat >= required) and (nlon >= required)
    if verbose:
        status = "✓" if ok else "⚠️  VIOLATED"
        print(f"Dealiasing (Orszag 2/3) check: T{lmax}  DH2 grid {nlat}×{nlon} "
              f"(lat×lon), need ≥ {required} each  →  {status}", flush=True)
        if not ok:
            print("  ⚠️  the transform grid is too coarse to dealias the "
                  "quadratic Jacobian at this truncation — the resolved band "
                  "will be aliased. Increase the grid density or lower LMAX.",
                  flush=True)
    return dict(nlat=nlat, nlon=nlon, required=required, ok=ok)


def effective_force_amp():
    """
    The forcing amplitude actually used (improvement #5a).

    If config.FORCE_FROM_EPSILON is set, the amplitude is DERIVED from the target
    energy-injection rate by inverting  ε = FORCE_AMP² · S_band  (config eq. 5a):

            FORCE_AMP = √( EPSILON_TARGET / S_band ).

    Otherwise the literal config.FORCE_AMP is used (exact v6/legacy behaviour).
    Either way the resulting injection rate is  ε = amp² · S_band.
    """
    if FORCE_FROM_EPSILON:
        return float(np.sqrt(EPSILON_TARGET / FORCE_BAND_SUM))
    return float(FORCE_AMP)


class SpectralVorticity:
    """Vorticity in spectral space (real 4π-normalised SH) + time stepping."""

    def __init__(self):
        self.lmax = LMAX
        # Confirm the DH2 grid dealiases the quadratic Jacobian (improvement #6).
        self.dealiasing = check_dealiasing(self.lmax, verbose=True)
        self._ev = _laplacian_eigenvalues(self.lmax)          # (2,L+1,L+1)
        # Optional spectral vanishing viscosity rate ε_SVV(l)·λ (improvement #7);
        # 0 unless SVV_ENABLED.  Supplements the ∇⁸ hyperviscosity in the linear
        # operator (both act only near the truncation).
        self._svv = (_svv_rate(self.lmax, SVV_EPS0, SVV_LCUT)
                     if SVV_ENABLED else None)
        # Full-step linear integrating factor exp(−(μ+νλ⁴+ε_SVV·λ)·dt) …
        self._visc = _dissipation_filter(self.lmax, NU_HYPER, LINEAR_DRAG, DT,
                                         svv_rate=self._svv)
        # … and the HALF-step factor = √(visc) needed by the symmetric Strang
        # split L(dt/2)·N(dt)·L(dt/2)  (improvement #3).
        self._sqrt_visc = _dissipation_filter(self.lmax, NU_HYPER, LINEAR_DRAG,
                                              DT / 2.0, svv_rate=self._svv)

        # Time-integration scheme (improvement #4).  ETDRK4 needs the diagonal
        # linear operator L and its exponential φ-function coefficients; Strang
        # needs only the filters above.  Precompute ETDRK4 coefficients once.
        self.time_scheme = TIME_SCHEME
        if self.time_scheme not in ('strang', 'etdrk4'):
            raise ValueError(f"TIME_SCHEME must be 'strang' or 'etdrk4', "
                             f"got {self.time_scheme!r}")
        self._L = _linear_operator(self.lmax, NU_HYPER, LINEAR_DRAG,
                                   svv_rate=self._svv)
        if self.time_scheme == 'etdrk4':
            (self._E, self._E2, self._Q,
             self._f1, self._f2, self._f3) = _etdrk4_coeffs(self._L, DT, ETDRK4_M)

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

        # ── Differential-rotation relaxation (improvement #9) ──────────────
        # Precompute the target zonal-mean vorticity ω̄_target,l0 and the inverse
        # relaxation time; both unused unless DIFF_ROT_ENABLED.  Applied to the
        # m=0 modes only inside _tendency.
        self.diff_rot_enabled = DIFF_ROT_ENABLED
        if self.diff_rot_enabled:
            self._omega_target = _diff_rot_target(self.lmax, DIFF_ROT_DELTA_OMEGA)
            self._diff_rot_inv_tau = 1.0 / DIFF_ROT_TAU

        # ── Forcing set-up (improvement #5) ───────────────────────────────
        # Effective amplitude (literal, or derived from ε if FORCE_FROM_EPSILON).
        self._force_amp = effective_force_amp()
        # Realised energy-injection rate ε = amp²·S_band (config eq. 5a).
        self.epsilon = self._force_amp ** 2 * FORCE_BAND_SUM
        self.force_type = FORCE_TYPE
        if self.force_type not in ('white', 'ou'):
            raise ValueError(f"FORCE_TYPE must be 'white' or 'ou', "
                             f"got {self.force_type!r}")
        # Per-coefficient white-noise amplitude amp_l = amp/√[l(l+1)] on the band
        # (0 outside), broadcast to the (2,L+1,L+1) SHCoeffs layout.  The sin
        # component (index 1) has no m=0 entry, so it is masked below.
        self._amp_l = np.zeros((2, self.lmax + 1, self.lmax + 1))
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            a = self._force_amp / np.sqrt(l * (l + 1))
            self._amp_l[0, l, :l + 1] = a       # cos block: m = 0 … l
            self._amp_l[1, l, 1:l + 1] = a       # sin block: m = 1 … l (no m=0)

        # Ornstein–Uhlenbeck state (improvement #5b), only if FORCE_TYPE='ou'.
        # Exact OU update over dt:  f ← a·f + b·ξ,  a = e^{−dt/τc}, drawing the
        # increment std b so the process stays at its stationary variance
        # Var(f_lm) = amp_l²/(2 τc) (config §5b).  Initialise f at that variance
        # so there is no forcing spin-up transient.
        if self.force_type == 'ou':
            tau = FORCE_CORR_TIME
            self._ou_a = np.exp(-DT / tau)                       # decay factor
            self._ou_std = self._amp_l / np.sqrt(2.0 * tau)      # stationary std
            # increment std b with Var(b·ξ) = Var·(1−a²)  (exact OU discretisation)
            self._ou_b = self._ou_std * np.sqrt(1.0 - self._ou_a ** 2)
            ou_rng = np.random.default_rng(7)
            self._ou_f = self._ou_std * ou_rng.standard_normal(self._ou_std.shape)

        # Small random initial vorticity in the forcing band.
        rng = np.random.default_rng(42)
        omega_lm = np.zeros((2, self.lmax + 1, self.lmax + 1))
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            for m in range(l + 1):
                amp = self._force_amp * 0.1 / (l + 1)
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

    # ── stochastic forcing (improvement #5: white or OU, narrow band in l) ─

    def _forcing(self, rng):
        """
        Vorticity increment δω_lm contributed by the stochastic forcing over one
        step dt.  Dispatches on FORCE_TYPE (improvement #5b): a fresh √dt Wiener
        increment ('white') or the tendency f_lm·dt of the persistent
        Ornstein–Uhlenbeck field ('ou').
        """
        if self.force_type == 'ou':
            return self._ou_forcing(rng)
        return self._stochastic_forcing(rng)

    def _stochastic_forcing(self, rng):
        """
        White-in-time forcing: an independent Gaussian increment per band
        coefficient with std amp_l·√dt, amp_l = amp/√[l(l+1)] (config §5a).  This
        is a Wiener increment (∝ √dt), delta-correlated in time.  Kept as an
        explicit per-degree loop so the RNG draw order — hence the realised
        stream — is identical to v6 when the amplitude is unchanged.
        """
        f_lm = np.zeros_like(self.omega_lm)
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            amp = self._force_amp / np.sqrt(l * (l + 1)) * np.sqrt(DT)
            f_lm[0, l, :l + 1] = rng.standard_normal(l + 1) * amp
            if l >= 1:
                f_lm[1, l, 1:l + 1] = rng.standard_normal(l) * amp
        return f_lm

    def _ou_forcing(self, rng):
        """
        Ornstein–Uhlenbeck (coloured) forcing, correlation time τ_c (config §5b).

        Advance the persistent forcing field one step with the EXACT OU
        discretisation (no time-discretisation bias in the stationary statistics)

            f_lm ← a·f_lm + b·ξ ,   a = e^{−dt/τc},  b = std·√(1−a²),  ξ ~ N(0,1),

        where std² = amp_l²/(2 τc) is the OU stationary variance (chosen so the
        τc→0 limit reproduces the 'white' branch and the same ε; config §5b).
        The forcing enters the vorticity as a SMOOTH tendency over the step, so
        the increment returned is f_lm·dt (contrast the white branch's √dt Wiener
        increment).  Off-band and illegal (sin, m=0) entries stay identically
        zero because b and the initial field vanish there.
        """
        xi = rng.standard_normal(self._ou_f.shape)
        self._ou_f = self._ou_a * self._ou_f + self._ou_b * xi
        return self._ou_f * DT

    # ── time step (Strang split: ½-diss · Heun advection · ½-diss + forcing) ─

    def _tendency(self, omega_lm):
        """
        Nonlinear tendency N(ω) = −J(ψ, ω+f),  ψ = ∇⁻²ω, plus (if enabled) the
        differential-rotation Newtonian relaxation of the m=0 mean flow
        (improvement #9):

            N(ω) = −J(ψ, ω+f) − (1/τ_relax)·(ω̄ − ω̄_target)·[m=0 only].

        The relaxation is part of the nonlinear tendency, so it is integrated by
        whichever scheme (Strang / ETDRK4) evaluates N — the base case
        (DIFF_ROT_ENABLED=False) is untouched.
        """
        psi_lm = self._inv_ev * omega_lm
        abs_vor_lm = omega_lm + self._f_lm
        tend = -self._jacobian_lm(psi_lm, abs_vor_lm)
        if self.diff_rot_enabled:
            # relax only the zonal-mean (m=0) vorticity toward ω̄_target
            tend[0, :, 0] += -self._diff_rot_inv_tau * (
                omega_lm[0, :, 0] - self._omega_target[0, :, 0])
        return tend

    def step(self, rng):
        """Advance one step with the configured scheme (config.TIME_SCHEME)."""
        if self.time_scheme == 'etdrk4':
            self._step_etdrk4(rng)
        else:
            self._step_strang(rng)

    def _step_strang(self, rng):
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
        # add stochastic forcing after the split (white √dt Wiener increment, or
        # OU tendency — improvement #5)
        w = w + self._forcing(rng)
        self.omega_lm = w
        self.omega_lm[:, 0, 0] = 0.0          # keep mean vorticity zero

    def _step_etdrk4(self, rng):
        """
        Advance one step with ETDRK4 — exponential time-differencing Runge–Kutta,
        4th order (Cox & Matthews 2002; Kassam & Trefethen 2005; improvement #4).

        For  ∂ω/∂t = L ω + N(ω)  with L = −(μ + ν λ⁴) diagonal and
        N(ω) = −J(ψ, ω+f), the linear part is advanced EXACTLY by e^{Lh} while
        the nonlinear part is integrated at 4th order.  Using the precomputed
        φ-function coefficients (see _etdrk4_coeffs) E, E2, Q, f1, f2, f3:

            Nu = N(uₙ)
            a  = E2·uₙ + Q·Nu ;             Na = N(a)
            b  = E2·uₙ + Q·Na ;             Nb = N(b)
            c  = E2·a  + Q·(2 Nb − Nu) ;    Nc = N(c)
            uₙ₊₁ = E·uₙ + Nu·f1 + 2(Na+Nb)·f2 + Nc·f3

        This is the exact-linear analogue of classical RK4: when L→0 the φ-
        functions reduce to E=E2=1, Q=f2=h/6·…, and the update collapses to RK4.
        The stiff hyperviscous/drag decay is handled to machine precision, so no
        stability restriction comes from L.

        As in the Strang step, the white-in-time stochastic forcing is a √dt
        Wiener increment (not a smooth tendency), so it is added AFTER the
        deterministic ETDRK4 update rather than inside the N evaluations, which
        would mis-scale it.
        """
        u = self.omega_lm
        Nu = self._tendency(u)
        a = self._E2 * u + self._Q * Nu
        Na = self._tendency(a)
        b = self._E2 * u + self._Q * Na
        Nb = self._tendency(b)
        c = self._E2 * a + self._Q * (2.0 * Nb - Nu)
        Nc = self._tendency(c)
        u = (self._E * u + Nu * self._f1
             + 2.0 * (Na + Nb) * self._f2 + Nc * self._f3)
        # add stochastic forcing after the deterministic ETDRK4 update
        # (white √dt Wiener increment, or OU tendency — improvement #5)
        u = u + self._forcing(rng)
        self.omega_lm = u
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
