"""
shallow_water.py — rotating shallow-water equations on the sphere (spectral,
vorticity–divergence formulation).

Scientific improvement #13 (scientific_improvements.md §13): advance from the
single non-divergent vorticity equation to the full shallow-water system, which
restores horizontal divergence, gravity waves and geostrophic adjustment — the
next rung above the barotropic model.  Standalone solver reusing only the
verified spectral primitives (pyshtools Jacobian/gradient, Laplacian
eigenvalues, ETDRK4 coefficients) and the same Strang / ETDRK4 time stepping.

═══════════════════════════════════════════════════════════════════════════════
DERIVATION — vorticity–divergence form
═══════════════════════════════════════════════════════════════════════════════
The rotating shallow-water equations for layer height h and horizontal velocity
u on a sphere of radius a=1 are, in vector-invariant form,

    ∂h/∂t + ∇·(h u) = 0                                            (mass)
    ∂u/∂t + (f+ζ) k̂×u + ∇(Φ + K) = 𝓓 ,                            (momentum)

with Φ = g h (geopotential), K = ½|u|² (kinetic energy per mass), ζ = k̂·∇×u the
relative vorticity, f = 2Ω sinφ the planetary vorticity, and 𝓓 an optional
hyperdiffusion.  The prognostic scalars are the relative VORTICITY ζ, the
DIVERGENCE δ = ∇·u, and the height h.  The velocity is recovered by the Helmholtz
decomposition through a streamfunction ψ and velocity potential χ,

    ψ = ∇⁻²ζ ,   χ = ∇⁻²δ ,   u = k̂×∇ψ + ∇χ ,

so u_θ = −(∂_φψ)/… etc. are obtained from the (verified) pyshtools gradient.

Taking k̂·∇× and ∇· of the momentum equation and using the identities
k̂·∇×(η∇a)=k̂·(∇η×∇a)≡J(η,a), k̂·∇×(η k̂×∇a)=∇η·∇a+η∇²a, ∇·(η∇a)=∇η·∇a+η∇²a and
∇·(η k̂×∇a)=−J(η,a), with η≡ζ+f the absolute vorticity, gives the closed system

    ∂ζ/∂t = −J(ψ, η) − ∇η·∇χ − η δ                         (vorticity)
    ∂δ/∂t = J(η, χ) + ∇η·∇ψ + η ζ − ∇²(Φ + K)              (divergence)
    ∂h/∂t = −J(ψ, h) − ∇h·∇χ − h δ                         (mass / height)

Here J(a,b) ≡ k̂·(∇a×∇b) = (k̂×∇a)·∇b is exactly the spherical advection
operator of the barotropic solver (advection of b by the flow with streamfunction
a), realised by the same pyshtools primitive `_jacobian_lm`.  ∇a·∇b is the grid
dot product of the two gradients; ∇²(Φ+K) is diagonal in spectral space
(eigenvalue −l(l+1)).  In the NON-DIVERGENT limit (δ=χ=0) the vorticity equation
collapses to ∂ζ/∂t = −J(ψ, ζ+f) — the barotropic model — which the code checks
against the reference solver.  The −∇²Φ term in the divergence equation and the
−hδ mass term are the two sides of the gravity-wave/geostrophic-adjustment
mechanism absent from the barotropic model.

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION — Williamson et al. (1992) test case 2
═══════════════════════════════════════════════════════════════════════════════
Test case 2 is global STEADY solid-body rotation (flow along the equator, α=0):

    u = u₀ cosφ ,  v = 0 ,   ζ = 2u₀ sinφ ,  δ = 0 ,
    g h = g h₀ − (a Ω u₀ + ½u₀²) sin²φ .

This is an exact time-independent solution: the pressure-gradient force balances
the Coriolis + metric terms (gradient-wind balance), so ∂ζ/∂t = ∂δ/∂t = ∂h/∂t = 0.
`verify()` initialises this state and checks (i) all three tendencies vanish to
truncation error, and (ii) the height-field error norms stay tiny under
integration — the standard TC2 diagnostic (Williamson et al. 1992, JCP 102, 211;
Galewsky, Scott & Polvani 2004, Tellus A 56, 429).

References: Bourke (1972) Mon. Wea. Rev. 100, 683; Hack & Jakob (1992) NCAR/TN-343;
Williamson et al. (1992) J. Comput. Phys. 102, 211.
"""

import numpy as np
import pyshtools as pysh

from simulate_v7 import _laplacian_eigenvalues, _etdrk4_coeffs


# ═════════════════════════════════════════════════════════════════════════════
# Parameters (nondimensional; unit sphere a=1).  Defaults set up Williamson TC2.
# ═════════════════════════════════════════════════════════════════════════════
LMAX      = 42          # spectral truncation
OMEGA     = 1.0         # rotation rate; f = 2Ω sinφ
GRAV      = 1.0         # gravitational acceleration g
H0        = 1.0         # reference mean height h₀
U0        = 0.2         # solid-body flow speed (TC2)
NU_HYPER  = 0.0         # ∇⁸ hyperviscosity (0 for the TC2 steady-state test)
DT        = 6.0e-3      # time step (must resolve gravity waves c=√(gh))
TIME_SCHEME = 'strang'  # 'strang' (default) or 'etdrk4'
ETDRK4_M  = 32


# ═════════════════════════════════════════════════════════════════════════════
# Spectral primitives (shared grid: pyshtools DH2, matching the gradient output)
# ═════════════════════════════════════════════════════════════════════════════

def _grad(clm):
    """Physical gradient components (∇f)_θ, (∇f)_φ on the DH2 grid (radius 1)."""
    g = pysh.SHCoeffs.from_array(clm, normalization='4pi', csphase=1).gradient(radius=1.0)
    return g.theta.data, g.phi.data


def _to_grid(clm):
    return pysh.SHCoeffs.from_array(clm, normalization='4pi',
                                    csphase=1).expand(grid='DH2').data


def _to_lm(grid_arr, lmax):
    return pysh.SHGrid.from_array(grid_arr, grid='DH').expand(
        normalization='4pi', csphase=1, lmax_calc=lmax).coeffs


def _jacobian_lm(a_lm, b_lm, lmax):
    """J(a,b) = k̂·(∇a×∇b): advection of b by the flow with streamfunction a.
    Identical convention/primitive to the verified barotropic solver."""
    at, ap = _grad(a_lm)
    bt, bp = _grad(b_lm)
    jac = ap * bt - at * bp
    return _to_lm(jac, lmax)


def _graddot_lm(a_lm, b_lm, lmax):
    """∇a·∇b on the grid, transformed to spectral coefficients."""
    at, ap = _grad(a_lm)
    bt, bp = _grad(b_lm)
    return _to_lm(at * bt + ap * bp, lmax)


def _sinphi_field(lmax, amp):
    """Spectral coefficients of amp·sinφ (a pure Y₁₀, coeff amp/√3)."""
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[0, 1, 0] = amp / np.sqrt(3.0)
    return c


def _dh2_lats(lmax):
    """Latitudes (degrees) of the DH2 grid the solver uses."""
    return pysh.SHCoeffs.from_zeros(lmax=lmax,
                                    normalization='4pi').expand(grid='DH2').lats()


# ═════════════════════════════════════════════════════════════════════════════
# Solver
# ═════════════════════════════════════════════════════════════════════════════

class ShallowWater:
    """
    Rotating shallow water on the sphere.  Prognostic spectral fields: relative
    vorticity ζ (zeta), divergence δ (delta), height h.  ζ and δ have zero mean
    (l=0); h carries a nonzero mean (its l=0 is conserved by mass continuity).
    """

    def __init__(self, lmax=LMAX, omega=OMEGA, grav=GRAV, nu=NU_HYPER, dt=DT,
                 time_scheme=TIME_SCHEME):
        self.lmax = lmax
        self.omega, self.grav, self.nu, self.dt = omega, grav, nu, dt
        self._ev = _laplacian_eigenvalues(lmax)                 # −λ
        self._lam4 = self._ev ** 4                              # λ⁴ ≥ 0
        # inverse Laplacian (ψ=∇⁻²ζ, χ=∇⁻²δ); l=0 → 0
        self._inv = np.zeros((2, lmax + 1, lmax + 1))
        for l in range(1, lmax + 1):
            self._inv[:, l, :l + 1] = -1.0 / (l * (l + 1))
        # planetary vorticity f = 2Ω sinφ (only Y₁₀)
        self._f = _sinphi_field(lmax, 2.0 * omega)

        # linear (diagonal) ∇⁸ hyperdiffusion, applied to ζ, δ, h
        self._L = -(self.nu * self._lam4)
        self._E2 = np.exp(self._L * dt / 2.0)                   # Strang half step
        self.time_scheme = time_scheme
        if time_scheme not in ('strang', 'etdrk4'):
            raise ValueError(f"time_scheme must be 'strang' or 'etdrk4', got "
                             f"{time_scheme!r}")
        if time_scheme == 'etdrk4':
            (self._E, self._E2f, self._Q,
             self._f1, self._f2, self._f3) = _etdrk4_coeffs(self._L, dt, ETDRK4_M)

        self.zeta = np.zeros((2, lmax + 1, lmax + 1))
        self.delta = np.zeros((2, lmax + 1, lmax + 1))
        self.h = np.zeros((2, lmax + 1, lmax + 1))

    # ── nonlinear tendency (the full RHS minus the diagonal hyperdiffusion) ────
    def _tendency(self, zeta, delta, h):
        """Return (dζ, dδ, dh) from the vorticity–divergence equations (docstring)."""
        lmax = self.lmax
        psi = self._inv * zeta
        chi = self._inv * delta
        eta = zeta + self._f                                    # absolute vorticity

        # gradients (each one grid synthesis of two components)
        pt, pp = _grad(psi)
        ct, cp = _grad(chi)
        et, ep = _grad(eta)
        ht, hp = _grad(h)

        # velocity u = k̂×∇ψ + ∇χ  →  (u_θ, u_φ) = (−ψ_φ+χ_θ, ψ_θ+χ_φ)
        u_th = -pp + ct
        u_ph = pt + cp
        K = 0.5 * (u_th ** 2 + u_ph ** 2)                       # kinetic energy

        # grid values of the fields entering the products
        eta_g = _to_grid(eta)
        zeta_g = _to_grid(zeta)
        delta_g = _to_grid(delta)
        h_g = _to_grid(h)
        Phi_g = self.grav * h_g

        # Jacobians  J(a,b)=k̂·(∇a×∇b)=ap*bt−at*bp  (same primitive as barotropic)
        J_psi_eta = _to_lm(pp * et - pt * ep, lmax)             # J(ψ, η)
        J_eta_chi = _to_lm(ep * ct - et * cp, lmax)             # J(η, χ)
        J_psi_h = _to_lm(pp * ht - pt * hp, lmax)               # J(ψ, h)
        # gradient dot products
        deta_dchi = _to_lm(et * ct + ep * cp, lmax)             # ∇η·∇χ
        deta_dpsi = _to_lm(et * pt + ep * pp, lmax)             # ∇η·∇ψ
        dh_dchi = _to_lm(ht * ct + hp * cp, lmax)               # ∇h·∇χ
        # field products (grid → spectral)
        eta_delta = _to_lm(eta_g * delta_g, lmax)
        eta_zeta = _to_lm(eta_g * zeta_g, lmax)
        h_delta = _to_lm(h_g * delta_g, lmax)
        # ∇²(Φ+K): expand (Φ+K), multiply by −λ (=self._ev)
        PhiK_lm = _to_lm(Phi_g + K, lmax)
        lap_PhiK = self._ev * PhiK_lm

        dzeta = -J_psi_eta - deta_dchi - eta_delta
        ddelta = J_eta_chi + deta_dpsi + eta_zeta - lap_PhiK
        dh = -J_psi_h - dh_dchi - h_delta
        # ζ, δ have no mean; h's mean is conserved by continuity (dh l=0 ≈ 0)
        dzeta[:, 0, 0] = 0.0
        ddelta[:, 0, 0] = 0.0
        return dzeta, ddelta, dh

    # ── time stepping ─────────────────────────────────────────────────────────
    def step(self):
        if self.time_scheme == 'etdrk4':
            self._step_etdrk4()
        else:
            self._step_strang()

    def _step_strang(self):
        dt = self.dt
        z = self._E2 * self.zeta
        d = self._E2 * self.delta
        hh = self._E2 * self.h
        k1z, k1d, k1h = self._tendency(z, d, hh)
        k2z, k2d, k2h = self._tendency(z + dt * k1z, d + dt * k1d, hh + dt * k1h)
        z = z + 0.5 * dt * (k1z + k2z)
        d = d + 0.5 * dt * (k1d + k2d)
        hh = hh + 0.5 * dt * (k1h + k2h)
        self.zeta = self._E2 * z
        self.delta = self._E2 * d
        self.h = self._E2 * hh
        self.zeta[:, 0, 0] = 0.0
        self.delta[:, 0, 0] = 0.0

    def _step_etdrk4(self):
        E, E2, Q, f1, f2, f3 = self._E, self._E2f, self._Q, self._f1, self._f2, self._f3
        z, d, hh = self.zeta, self.delta, self.h
        Nz, Nd, Nh = self._tendency(z, d, hh)
        az, ad, ah = E2 * z + Q * Nz, E2 * d + Q * Nd, E2 * hh + Q * Nh
        Naz, Nad, Nah = self._tendency(az, ad, ah)
        bz, bd, bh = E2 * z + Q * Naz, E2 * d + Q * Nad, E2 * hh + Q * Nah
        Nbz, Nbd, Nbh = self._tendency(bz, bd, bh)
        cz, cd, ch = E2 * az + Q * (2*Nbz - Nz), E2 * ad + Q * (2*Nbd - Nd), E2 * ah + Q * (2*Nbh - Nh)
        Ncz, Ncd, Nch = self._tendency(cz, cd, ch)
        self.zeta = E * z + Nz * f1 + 2*(Naz + Nbz) * f2 + Ncz * f3
        self.delta = E * d + Nd * f1 + 2*(Nad + Nbd) * f2 + Ncd * f3
        self.h = E * hh + Nh * f1 + 2*(Nah + Nbh) * f2 + Nch * f3
        self.zeta[:, 0, 0] = 0.0
        self.delta[:, 0, 0] = 0.0

    # ── diagnostics ───────────────────────────────────────────────────────────
    def energy(self):
        """Total shallow-water energy  E = ∫(½h|u|² + ½g h²) dΩ  (grid quadrature).
        Conserved by the inviscid, unforced flow — a global integral invariant."""
        psi = self._inv * self.zeta
        chi = self._inv * self.delta
        pt, pp = _grad(psi); ct, cp = _grad(chi)
        u_th = -pp + ct; u_ph = pt + cp
        h_g = _to_grid(self.h)
        ke = 0.5 * h_g * (u_th ** 2 + u_ph ** 2)
        pe = 0.5 * self.grav * h_g ** 2
        # area-weighted mean over the DH2 grid (∝ ∫·dΩ); weight by sinθ (colat)
        nlat, nlon = h_g.shape
        colat = np.linspace(0, np.pi, nlat, endpoint=False) + 0.5 * np.pi / nlat
        w = np.sin(colat)[:, None]
        return float(((ke + pe) * w).sum() / w.sum() / nlon * (4 * np.pi))

    def height_grid(self):
        return _to_grid(self.h)


# ═════════════════════════════════════════════════════════════════════════════
# Williamson TC2 initial state
# ═════════════════════════════════════════════════════════════════════════════

def williamson2(model, u0=U0, h0=H0):
    """
    Initialise `model` with Williamson et al. (1992) test case 2 (steady
    solid-body rotation, α=0):  ζ = 2u₀ sinφ, δ = 0,
    g h = g h₀ − (Ω u₀ + ½u₀²) sin²φ   (a = 1).  Returns the analytic height grid.
    """
    lmax = model.lmax
    model.zeta = _sinphi_field(lmax, 2.0 * u0)          # ζ = 2u₀ sinφ
    model.delta = np.zeros((2, lmax + 1, lmax + 1))
    C = model.omega * u0 + 0.5 * u0 ** 2                 # a=1
    h_grid = _to_grid(np.zeros((2, lmax + 1, lmax + 1)))
    nlat, nlon = h_grid.shape
    lat = np.deg2rad(_dh2_lats(lmax))
    sin2 = (np.sin(lat) ** 2)[:, None] * np.ones((1, nlon))
    h_analytic = h0 - (C / model.grav) * sin2
    model.h = _to_lm(h_analytic, lmax)
    return h_analytic


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify(nsteps=400):
    print("=" * 74)
    print("Shallow water — Williamson (1992) test case 2 verification")
    print("=" * 74)

    # ── 1. Non-divergent (barotropic) limit consistency ───────────────────────
    m = ShallowWater()
    rng = np.random.default_rng(0)
    z = np.zeros((2, m.lmax + 1, m.lmax + 1))
    for l in range(1, 12):
        z[0, l, :l + 1] = rng.standard_normal(l + 1) * 0.05 / (l + 1)
        z[1, l, 1:l + 1] = rng.standard_normal(l) * 0.05 / (l + 1)
    m.zeta, m.delta = z, np.zeros_like(z)
    m.h = _sinphi_field(m.lmax, 0.0); m.h[0, 0, 0] = m.grav  # h=const ⇒ Φ=const
    dz, dd, dh = m._tendency(m.zeta, m.delta, m.h)
    psi = m._inv * m.zeta
    ref = -_jacobian_lm(psi, m.zeta + m._f, m.lmax)          # barotropic tendency
    err = np.max(np.abs(dz - ref))
    print(f"\n1. Non-divergent limit: max|∂ζ/∂t_SW − (−J(ψ,ζ+f))| = {err:.2e} "
          f"({'✓' if err < 1e-12 else '⚠'} matches barotropic solver)")

    # ── 2. TC2 steady state: all tendencies vanish ────────────────────────────
    m2 = ShallowWater()
    h_an = williamson2(m2)
    dz, dd, dh = m2._tendency(m2.zeta, m2.delta, m2.h)
    # scale each tendency by a characteristic magnitude of its field's dynamics
    sz = np.max(np.abs(m2.zeta)) + 1e-30
    sh = np.max(np.abs(m2.h)) + 1e-30
    rz = np.max(np.abs(dz)) / sz
    rd = np.max(np.abs(dd)) / (m2.grav * sh)                 # δ forced by ∇²Φ ~ gh
    rh = np.max(np.abs(dh)) / sh
    print(f"\n2. TC2 steady state — relative tendencies (should be ~0):")
    print(f"   |∂ζ/∂t|/|ζ| = {rz:.2e}   |∂δ/∂t|/(g|h|) = {rd:.2e}   "
          f"|∂h/∂t|/|h| = {rh:.2e}")
    steady = max(rz, rd, rh) < 1e-6
    print(f"   → {'✓ steady state maintained' if steady else '⚠ not steady'}")

    # ── 3. TC2 time integration: height error norms stay small ────────────────
    m3 = ShallowWater()
    h_an = williamson2(m3)
    def herr():
        hg = m3.height_grid()
        diff = hg - h_an
        l2 = np.sqrt((diff ** 2).mean()) / np.sqrt((h_an ** 2).mean())
        linf = np.max(np.abs(diff)) / np.max(np.abs(h_an))
        return l2, linf
    E0 = m3.energy()
    for i in range(nsteps):
        m3.step()
    l2, linf = herr()
    E1 = m3.energy()
    print(f"\n3. TC2 integration ({nsteps} steps, scheme={m3.time_scheme}):")
    print(f"   height error   l2 = {l2:.2e}   l∞ = {linf:.2e}")
    print(f"   energy drift   {abs(E1-E0)/E0:.2e}   (E={E0:.4e})")
    integ_ok = l2 < 1e-4 and linf < 1e-4

    # ── 4. Inviscid energy conservation under a gravity-wave transient ────────
    m4 = ShallowWater()
    williamson2(m4)
    # add a small height bump → excites gravity waves; energy must be conserved
    bump = np.zeros((2, m4.lmax + 1, m4.lmax + 1)); bump[0, 4, 2] = 0.02
    m4.h = m4.h + bump
    Eg0 = m4.energy()
    for _ in range(nsteps):
        m4.step()
    Eg1 = m4.energy()
    edrift = abs(Eg1 - Eg0) / Eg0
    print(f"\n4. Gravity-wave transient energy conservation ({nsteps} steps): "
          f"drift {edrift:.2e}  ({'✓' if edrift < 5e-3 else '⚠'})")

    ok = (err < 1e-12) and steady and integ_ok and edrift < 5e-3
    print("\n" + ("✓ ALL CHECKS PASSED — TC2 steady state preserved, gravity "
                  "waves conserve energy" if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
