"""
two_layer_qg.py — two-layer quasi-geostrophic flow on a rotating sphere.

Scientific improvement #12 (scientific_improvements.md §12): the minimal model in
which jets arise SELF-CONSISTENTLY from an internal energy source — baroclinic
instability of an imposed vertical shear — rather than from ad-hoc stochastic
forcing.  Standalone solver; it reuses only the verified spectral primitives of
the barotropic code (pyshtools Jacobian, Laplacian eigenvalues, ETDRK4
coefficients) and the same Strang / ETDRK4 time stepping.

═══════════════════════════════════════════════════════════════════════════════
DERIVATION — the two-layer QG equations
═══════════════════════════════════════════════════════════════════════════════
Two immiscible layers of resting-state depth H₁ (upper) and H₂ (lower), a rigid
lid on top and a flat bottom, coupled across a density interface (reduced gravity
g′ = g Δρ/ρ).  Each layer has a streamfunction ψ_i (geostrophic: u_i = k̂×∇ψ_i)
and a QG potential vorticity

    q₁ = ∇²ψ₁ + f + F₁(ψ₂ − ψ₁),        F₁ = f₀²/(g′H₁)     (upper)
    q₂ = ∇²ψ₂ + f + F₂(ψ₁ − ψ₂),        F₂ = f₀²/(g′H₂)     (lower)

f = 2Ω sinφ is the planetary vorticity; the stretching terms F_i(ψ_j−ψ_i) are the
vortex-stretching by interface displacement η = (f₀/g′)(ψ₁−ψ₂).  Each layer
materially conserves its PV up to forcing/dissipation:

    ∂q₁/∂t + J(ψ₁, q₁) = 𝓕 + 𝓓₁          (forcing + dissipation)
    ∂q₂/∂t + J(ψ₂, q₂) =      𝓓₂          (dissipation; Ekman/PV drag on bottom)

J(A,B) is the spherical Jacobian (advection).  The coupling k_d² ≡ F₁+F₂ defines
the internal deformation wavenumber; l_d ≈ √(F₁+F₂) is the deformation degree.

───────────────────────────────────────────────────────────────────────────────
IMPOSED SHEAR AS A STEADY BACKGROUND  (the energy source)
───────────────────────────────────────────────────────────────────────────────
Split each field into a fixed background + evolving perturbation,
ψ_i^tot = Ψ_i + ψ_i,  q_i^tot = Q_i + q_i.  The background is SOLID-BODY rotation
of layer i at rate U_i:  Ψ_i = −U_i sinφ  (a pure Y₁₀), giving zonal wind
ū_i = U_i cosφ.  Because Ψ_i ∝ sinφ,

    Q_i = ∇²Ψ_i + f + F_i(Ψ_j − Ψ_i) = [2U_i + 2Ω + F_i(U_i − U_j)]·sinφ  ∝ Y₁₀,

so Q_i is ALSO ∝ sinφ and J(Ψ_i, Q_i) = J(sinφ, sinφ) = 0: the background is an
EXACT steady solution of the full nonlinear system for any U₁, U₂.  The vertical
shear U₁−U₂ tilts the interface Ψ₁−Ψ₂ ∝ sinφ and stores available potential
energy; perturbations tap it → BAROCLINIC INSTABILITY (Phillips 1954).  The
Charney–Stern necessary condition is that the meridional PV gradient dQ_i/dφ
change sign between layers, which the F_i(U_i−U_j) term arranges once the shear
exceeds a β-dependent critical value.

The perturbation PV evolves (subtracting the steady background, using
J(Ψ_i,Q_i)=0) exactly as the total advection of the total fields:

    ∂q_i/∂t = −J(ψ_i^tot, q_i^tot) + 𝓕_i + 𝓓_i(q_i),

i.e. −J(Ψ_i,q_i) [advection of eddy PV by the mean flow] − J(ψ_i,Q_i) [advection
of the mean PV gradient by eddies — the instability term] − J(ψ_i,q_i) [eddy–eddy]
The background is never dissipated (𝓓 acts on the perturbation q_i only), so the
mean shear is maintained externally, as intended.

───────────────────────────────────────────────────────────────────────────────
PV INVERSION IN SPECTRAL SPACE (the layer coupling)
───────────────────────────────────────────────────────────────────────────────
With ∇² → −λ, λ = l(l+1), the perturbation inversion q_i = ∇²ψ_i + F_i(ψ_j−ψ_i)
is, per (l,m), the 2×2 linear system  q = M ψ,

    M = [ −(λ+F₁)   F₁      ;   F₂     −(λ+F₂) ],
    det M = λ(λ + F₁ + F₂),

so ψ = M⁻¹ q with

    ψ₁ = [ −(λ+F₂) q₁ − F₁ q₂ ] / det
    ψ₂ = [ −F₂ q₁ − (λ+F₁) q₂ ] / det .

det = 0 only at l = 0 (undetermined mean streamfunction gauge); we set ψ = 0
there and hold the mean PV at zero, exactly as the barotropic solver does.

───────────────────────────────────────────────────────────────────────────────
ENERGETICS (barotropic / baroclinic diagnostics)
───────────────────────────────────────────────────────────────────────────────
Total energy  E = Σ_i ½ H_i ∫|∇ψ_i|² dΩ  +  ½ (f₀²/g′) ∫(ψ₁−ψ₂)² dΩ, the last term
the available potential energy (APE), with f₀²/g′ = F₁H₁ = F₂H₂.  Decomposing into
the mass-weighted barotropic mode ψ_τ = (H₁ψ₁+H₂ψ₂)/H_tot and the baroclinic mode
ψ_c = ψ₁−ψ₂:
    • barotropic KE   E_bt = ½ H_tot Σ_lm λ ψ_{τ,lm}²
    • baroclinic KE   E_bc = (total KE) − E_bt
    • APE             = ½ (f₀²/g′) Σ_lm ψ_{c,lm}²
Baroclinic instability converts mean APE → eddy (E_bc + APE) → E_bt (jets): the
signature is E_bt growing at the expense of the shear.

References: Phillips (1954) Tellus 6, 273; Panetta (1993) JAS 50, 2073; Salmon,
Lectures on Geophysical Fluid Dynamics (Oxford 1998), ch. 6; Vallis (2017), ch. 9.
"""

import numpy as np
import pyshtools as pysh

# Reuse the VERIFIED spectral primitives from the barotropic solver.
from simulate_v7 import _laplacian_eigenvalues, _etdrk4_coeffs


# ═════════════════════════════════════════════════════════════════════════════
# Parameters (standalone; sensible supercritical-shear defaults)
# ═════════════════════════════════════════════════════════════════════════════
LMAX      = 64          # spectral truncation (coupling ⇒ keep modest)
OMEGA     = 2.0         # rotation rate; planetary vorticity f = 2Ω sinφ, β = 2Ω
H1        = 1.0         # upper-layer resting depth
H2        = 1.0         # lower-layer resting depth
F1        = 50.0        # = f₀²/(g′H₁); deformation coupling (l_d = √(F₁+F₂) ≈ 10)
F2        = 50.0        # = f₀²/(g′H₂)
U1        = 0.25        # upper-layer solid-body rate (imposed)
U2        = -0.25       # lower-layer solid-body rate  → shear U₁−U₂ = 0.5
NU_HYPER  = 2.0e-14     # ∇⁸ hyperviscosity coefficient (small-scale enstrophy sink)
DRAG      = 0.02        # linear PV drag on the BOTTOM layer (equilibrates the BCI)
DT        = 2.0e-3      # time step
TIME_SCHEME = 'strang'  # 'strang' (default) or 'etdrk4'
ETDRK4_M  = 32          # contour points for the ETDRK4 φ-functions
SEED      = 1234        # RNG seed for the initial small perturbation
INIT_AMP  = 1.0e-3      # amplitude of the initial random perturbation


# ═════════════════════════════════════════════════════════════════════════════
# Spectral primitives
# ═════════════════════════════════════════════════════════════════════════════

def _jacobian_lm(a_lm, b_lm, lmax):
    """
    Spherical Jacobian J(A,B) = (∇A)_φ(∇B)_θ − (∇A)_θ(∇B)_φ via pyshtools'
    metric-consistent physical gradients, transformed back to spectral space.
    Identical convention to the (verified-correct) barotropic solver.
    """
    ca = pysh.SHCoeffs.from_array(a_lm, normalization='4pi', csphase=1)
    cb = pysh.SHCoeffs.from_array(b_lm, normalization='4pi', csphase=1)
    ga = ca.gradient(radius=1.0)
    gb = cb.gradient(radius=1.0)
    jac = ga.phi.data * gb.theta.data - ga.theta.data * gb.phi.data
    jac_grid = pysh.SHGrid.from_array(jac, grid='DH')
    return jac_grid.expand(normalization='4pi', csphase=1, lmax_calc=lmax).coeffs


def _inversion_coeffs(lmax, F1, F2):
    """
    Per-(l,m) 2×2 PV-inversion multipliers A = M⁻¹ (see module docstring):
        ψ₁ = A11 q₁ + A12 q₂,   ψ₂ = A21 q₁ + A22 q₂,
    with det = λ(λ+F₁+F₂), A11=−(λ+F₂)/det, A12=−F₁/det, A21=−F₂/det,
    A22=−(λ+F₁)/det.  The l=0 slot is 0 (mean-streamfunction gauge).  Each Aij is
    a (2,L+1,L+1) array depending only on l, ready to multiply an SHCoeffs field.
    """
    A11 = np.zeros((2, lmax + 1, lmax + 1))
    A12 = np.zeros_like(A11); A21 = np.zeros_like(A11); A22 = np.zeros_like(A11)
    for l in range(1, lmax + 1):
        lam = l * (l + 1.0)
        det = lam * (lam + F1 + F2)
        A11[:, l, :l + 1] = -(lam + F2) / det
        A12[:, l, :l + 1] = -F1 / det
        A21[:, l, :l + 1] = -F2 / det
        A22[:, l, :l + 1] = -(lam + F1) / det
    return A11, A12, A21, A22


def _sinphi_field(lmax, amp):
    """Spectral coefficients of the field  amp·sinφ  (a pure Y₁₀, coeff amp/√3)."""
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[0, 1, 0] = amp / np.sqrt(3.0)          # 4π-norm: Y₁₀ = √3 sinφ
    return c


# ═════════════════════════════════════════════════════════════════════════════
# Solver
# ═════════════════════════════════════════════════════════════════════════════

class TwoLayerQG:
    """
    Two-layer QG on the sphere.  Prognostic state: the PERTURBATION PVs q₁, q₂
    (each a (2,L+1,L+1) 4π-normalised SHCoeffs array).  The steady background
    (solid-body shear + planetary vorticity) is held fixed and added inside the
    tendency.
    """

    def __init__(self, lmax=LMAX, omega=OMEGA, H1=H1, H2=H2, F1=F1, F2=F2,
                 U1=U1, U2=U2, nu=NU_HYPER, drag=DRAG, dt=DT,
                 time_scheme=TIME_SCHEME, seed=SEED, init_amp=INIT_AMP):
        self.lmax = lmax
        self.omega, self.H1, self.H2 = omega, H1, H2
        self.F1, self.F2, self.U1, self.U2 = F1, F2, U1, U2
        self.nu, self.drag, self.dt = nu, drag, dt
        self.Htot = H1 + H2
        self.f0sq_over_gp = F1 * H1        # = f₀²/g′ = F₂H₂ (APE coefficient)

        self._ev = _laplacian_eigenvalues(lmax)                     # −λ
        self._lam4 = self._ev ** 4                                  # λ⁴ ≥ 0
        self._A = _inversion_coeffs(lmax, F1, F2)                   # (A11,A12,A21,A22)

        # ── steady background (solid-body shear + planetary vorticity) ──────
        # Ψ_i = −U_i sinφ ;  Q_i = [2U_i + 2Ω + F_i(U_i−U_j)]·sinφ  (all ∝ Y₁₀)
        self._Psi1 = _sinphi_field(lmax, -U1)
        self._Psi2 = _sinphi_field(lmax, -U2)
        self._Q1 = _sinphi_field(lmax, 2 * U1 + 2 * omega + F1 * (U1 - U2))
        self._Q2 = _sinphi_field(lmax, 2 * U2 + 2 * omega + F2 * (U2 - U1))

        # ── linear (diagonal) dissipation operators, per layer ──────────────
        # Layer 1: ∇⁸ hyperviscosity only.  Layer 2: hyperviscosity + PV drag.
        # (PV drag −μ q₂ is used, not Ekman −μ∇²ψ₂, so L stays DIAGONAL in q and
        # the exact integrating factor / ETDRK4 machinery carries over unchanged.)
        self._L1 = -(self.nu * self._lam4)
        self._L2 = -(self.drag + self.nu * self._lam4)
        self._E2_1 = np.exp(self._L1 * dt / 2.0)     # half-step (Strang)
        self._E2_2 = np.exp(self._L2 * dt / 2.0)

        self.time_scheme = time_scheme
        if time_scheme not in ('strang', 'etdrk4'):
            raise ValueError(f"time_scheme must be 'strang' or 'etdrk4', got "
                             f"{time_scheme!r}")
        if time_scheme == 'etdrk4':
            (self._E_1, self._E2f_1, self._Q_1,
             self._f1_1, self._f2_1, self._f3_1) = _etdrk4_coeffs(self._L1, dt, ETDRK4_M)
            (self._E_2, self._E2f_2, self._Q_2,
             self._f1_2, self._f2_2, self._f3_2) = _etdrk4_coeffs(self._L2, dt, ETDRK4_M)

        # ── initial small random perturbation (seeds the instability) ───────
        rng = np.random.default_rng(seed)
        self.q1 = self._random_field(rng, init_amp)
        self.q2 = self._random_field(rng, init_amp)

    def _random_field(self, rng, amp):
        c = np.zeros((2, self.lmax + 1, self.lmax + 1))
        for l in range(1, self.lmax + 1):
            s = amp / (l + 1.0)                       # red-ish so it is smooth
            c[0, l, :l + 1] = rng.standard_normal(l + 1) * s
            c[1, l, 1:l + 1] = rng.standard_normal(l) * s
        c[:, 0, 0] = 0.0
        return c

    # ── PV inversion ────────────────────────────────────────────────────────
    def invert(self, q1, q2):
        """ψ₁, ψ₂ from perturbation q₁, q₂ (spectral 2×2 solve, see docstring)."""
        A11, A12, A21, A22 = self._A
        psi1 = A11 * q1 + A12 * q2
        psi2 = A21 * q1 + A22 * q2
        return psi1, psi2

    # ── nonlinear tendency ────────────────────────────────────────────────────
    def _tendency(self, q1, q2):
        """
        N_i = −J(ψ_i^tot, q_i^tot),  ψ_i^tot = ψ_i + Ψ_i,  q_i^tot = q_i + Q_i.
        Because J(Ψ_i, Q_i) = 0, this equals the perturbation tendency
        −J(Ψ_i,q_i) − J(ψ_i,Q_i) − J(ψ_i,q_i) (mean-advection + instability +
        eddy–eddy).  Linear dissipation is applied separately (it is the diagonal
        operator L, handled by the time integrator).
        """
        psi1, psi2 = self.invert(q1, q2)
        p1 = psi1 + self._Psi1
        p2 = psi2 + self._Psi2
        Q1 = q1 + self._Q1
        Q2 = q2 + self._Q2
        n1 = -_jacobian_lm(p1, Q1, self.lmax)
        n2 = -_jacobian_lm(p2, Q2, self.lmax)
        return n1, n2

    # ── time stepping ─────────────────────────────────────────────────────────
    def step(self):
        if self.time_scheme == 'etdrk4':
            self._step_etdrk4()
        else:
            self._step_strang()

    def _step_strang(self):
        """Strang split  L(dt/2)·N(dt)·L(dt/2)  with a coupled Heun (RK2) N."""
        dt = self.dt
        # ½-step dissipation (per-layer diagonal integrating factor)
        w1 = self._E2_1 * self.q1
        w2 = self._E2_2 * self.q2
        # full-step Heun on the coupled nonlinear tendency
        k1a, k1b = self._tendency(w1, w2)
        k2a, k2b = self._tendency(w1 + dt * k1a, w2 + dt * k1b)
        w1 = w1 + 0.5 * dt * (k1a + k2a)
        w2 = w2 + 0.5 * dt * (k1b + k2b)
        # ½-step dissipation
        self.q1 = self._E2_1 * w1
        self.q2 = self._E2_2 * w2
        self.q1[:, 0, 0] = 0.0
        self.q2[:, 0, 0] = 0.0

    def _step_etdrk4(self):
        """ETDRK4 per layer (diagonal L_i), coupled through the N evaluations."""
        u1, u2 = self.q1, self.q2
        Nu1, Nu2 = self._tendency(u1, u2)
        a1 = self._E2f_1 * u1 + self._Q_1 * Nu1
        a2 = self._E2f_2 * u2 + self._Q_2 * Nu2
        Na1, Na2 = self._tendency(a1, a2)
        b1 = self._E2f_1 * u1 + self._Q_1 * Na1
        b2 = self._E2f_2 * u2 + self._Q_2 * Na2
        Nb1, Nb2 = self._tendency(b1, b2)
        c1 = self._E2f_1 * a1 + self._Q_1 * (2.0 * Nb1 - Nu1)
        c2 = self._E2f_2 * a2 + self._Q_2 * (2.0 * Nb2 - Nu2)
        Nc1, Nc2 = self._tendency(c1, c2)
        self.q1 = (self._E_1 * u1 + Nu1 * self._f1_1
                   + 2.0 * (Na1 + Nb1) * self._f2_1 + Nc1 * self._f3_1)
        self.q2 = (self._E_2 * u2 + Nu2 * self._f1_2
                   + 2.0 * (Na2 + Nb2) * self._f2_2 + Nc2 * self._f3_2)
        self.q1[:, 0, 0] = 0.0
        self.q2[:, 0, 0] = 0.0

    # ── diagnostics ───────────────────────────────────────────────────────────
    def energies(self):
        """
        Barotropic / baroclinic energy diagnostics of the PERTURBATION (see the
        Energetics section of the module docstring).  Returns a dict with the
        barotropic KE, baroclinic KE, APE, their sum, and the total.
        (4π Parseval constants cancel in every ratio, so they are dropped.)
        """
        psi1, psi2 = self.invert(self.q1, self.q2)
        ll = np.arange(self.lmax + 1)
        lam = (ll * (ll + 1.0))                            # λ per degree

        def sum_lam(psi):
            c2 = psi[0] ** 2 + psi[1] ** 2                 # (L+1,L+1)
            return (c2.sum(axis=1) * lam).sum()            # Σ_lm λ ψ²

        def sum_sq(psi):
            return (psi[0] ** 2 + psi[1] ** 2).sum()

        KE1 = 0.5 * self.H1 * sum_lam(psi1)
        KE2 = 0.5 * self.H2 * sum_lam(psi2)
        KE = KE1 + KE2
        psi_bt = (self.H1 * psi1 + self.H2 * psi2) / self.Htot
        E_bt = 0.5 * self.Htot * sum_lam(psi_bt)
        E_bc = KE - E_bt
        APE = 0.5 * self.f0sq_over_gp * sum_sq(psi1 - psi2)
        return dict(KE1=KE1, KE2=KE2, KE=KE, barotropic=E_bt, baroclinic_KE=E_bc,
                    APE=APE, baroclinic=E_bc + APE, total=KE + APE)

    def vorticity_grids(self):
        """Upper/lower relative-vorticity grids ∇²ψ_i for visualisation."""
        psi1, psi2 = self.invert(self.q1, self.q2)
        z1 = pysh.SHCoeffs.from_array(self._ev * psi1, normalization='4pi').expand(grid='DH2').data
        z2 = pysh.SHCoeffs.from_array(self._ev * psi2, normalization='4pi').expand(grid='DH2').data
        return z1, z2


# ═════════════════════════════════════════════════════════════════════════════
# Verification: baroclinic instability is the energy source
# ═════════════════════════════════════════════════════════════════════════════

def verify(nsteps=4000, report_every=500):
    """
    Demonstrate that the imposed shear (not any external forcing) is the energy
    source, via baroclinic instability:

      • WITH shear (U₁≠U₂): the perturbation energy grows exponentially in the
        linear phase and the barotropic (jet) energy rises as APE is released.
      • WITHOUT shear (U₁=U₂=0): the same initial perturbation only DECAYS under
        dissipation (no internal energy source).

    Also runs an inviscid, no-shear conservation check: total energy KE+APE is
    conserved by the nonlinear advection + coupled inversion.
    """
    print("=" * 74)
    print("Two-layer QG — baroclinic-instability verification")
    print("=" * 74)

    # ── sheared run: expect growth ────────────────────────────────────────────
    m = TwoLayerQG()
    ld = np.sqrt(m.F1 + m.F2)
    print(f"\nT{m.lmax}  Ω={m.omega}  F₁=F₂={m.F1}  (deformation degree l_d≈{ld:.1f})"
          f"  shear U₁−U₂={m.U1 - m.U2:.2f}  scheme={m.time_scheme}")
    e0 = m.energies()['total']
    hist = [(0, e0, m.energies()['barotropic'])]
    for i in range(1, nsteps + 1):
        m.step()
        if i % report_every == 0:
            e = m.energies()
            hist.append((i, e['total'], e['barotropic']))
            print(f"  step {i:5d}  E_total={e['total']:.4e}  "
                  f"E_bt={e['barotropic']:.4e}  E_bc={e['baroclinic']:.4e}  "
                  f"APE={e['APE']:.4e}")
    grew = hist[-1][1] > 10.0 * hist[0][1]
    bt_grew = hist[-1][2] > 10.0 * (hist[0][2] + 1e-30)
    # linear growth rate from the exponential phase (first half of the record)
    tt = np.array([h[0] for h in hist]) * m.dt
    ee = np.array([h[1] for h in hist])
    half = len(ee) // 2 + 1
    sigma = np.polyfit(tt[:half], np.log(ee[:half]), 1)[0] / 2.0   # energy→amp
    print(f"  → energy grew ×{hist[-1][1]/hist[0][1]:.1f};  "
          f"linear growth rate σ ≈ {sigma:.3f} /time  "
          f"({'✓ UNSTABLE' if grew else '⚠ no growth'})")
    print(f"  → barotropic (jet) energy grew: {'✓' if bt_grew else '⚠ no'} "
          f"(APE released into eddies then the barotropic mode)")

    # ── no-shear run: expect decay ────────────────────────────────────────────
    m0 = TwoLayerQG(U1=0.0, U2=0.0)
    e_start = m0.energies()['total']
    for i in range(nsteps):
        m0.step()
    e_end = m0.energies()['total']
    decayed = e_end < e_start
    print(f"\nNo-shear control (U₁=U₂=0): E_total {e_start:.3e} → {e_end:.3e}  "
          f"({'✓ DECAYS — no internal energy source' if decayed else '⚠ grew?!'})")

    # ── inviscid, no-shear conservation of total energy ───────────────────────
    mc = TwoLayerQG(U1=0.0, U2=0.0, nu=0.0, drag=0.0)
    E0 = mc.energies()['total']
    for _ in range(300):
        mc.step()
    E1 = mc.energies()['total']
    rel = abs(E1 - E0) / E0
    print(f"\nInviscid no-shear conservation (300 steps): "
          f"E {E0:.6e} → {E1:.6e}  rel drift {rel:.2e}  "
          f"({'✓ conserved' if rel < 1e-2 else '⚠ drift'})")

    ok = grew and bt_grew and decayed and rel < 1e-2
    print("\n" + ("✓ ALL CHECKS PASSED — baroclinic instability confirmed"
                  if ok else "⚠ some checks failed (see above)"))
    return ok


if __name__ == "__main__":
    verify()
