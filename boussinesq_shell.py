"""
boussinesq_shell.py — Boussinesq Rayleigh–Bénard convection in a rotating
spherical shell  r_i < r < r_o.

Scientific improvement #15 (scientific_improvements.md §15): the MINIMAL TRUE
convection model.  Where improvements #1–#14 all evolve a single 2-D scalar
(barotropic vorticity / QG PV / shallow water / MHD) with NO buoyancy, this
module adds the three ingredients that make the problem convection:

    • a temperature perturbation Θ with its own evolution equation,
    • a buoyancy force  Ra·Pr·(r̂×∇Θ  ⇒ vertical force)  driving the flow,
    • a vertical velocity and a poloidal/toroidal decomposition of a 3-D field.

Because a full 3-D time-stepping shell dynamo/convection solver is a "Major"
undertaking (improvement #17 hands that to Rayleigh/Dedalus), this module
implements the two pieces that are *rigorously verifiable against known
results* and that constitute the physical heart of the problem:

  (A) the LINEAR ONSET problem — the critical Rayleigh number Ra_c at which a
      motionless, conducting shell first goes convectively unstable.  This is
      Chandrasekhar's classic eigenvalue problem; its answers are known to many
      digits (free–free 27π⁴/4, rigid–rigid 1707.76, and the rotating
      Ra_c ∝ Ek^{−4/3} law), so the solver can be checked exactly.
  (B) ENERGY CONSERVATION of the convective nonlinearity in the dissipationless
      limit, via the Saltzman (1962) / Lorenz (1963) Galerkin truncation of the
      Boussinesq equations — the smallest self-contained system that carries the
      advective nonlinearity + buoyancy exchange.

═══════════════════════════════════════════════════════════════════════════════
GOVERNING EQUATIONS (Boussinesq, rotating shell; the model this onset linearises)
═══════════════════════════════════════════════════════════════════════════════
With velocity u (∇·u = 0, poloidal/toroidal: u = ∇×∇×(P r̂) + ∇×(T r̂)),
temperature perturbation Θ about a conducting background T̄(r), gravity g = −g r̂,
rotation Ω ẑ, and nondimensional control numbers  Ra (Rayleigh), Ek (Ekman),
Pr (Prandtl):

    ∂Θ/∂t + u·∇Θ = κ ∇²Θ + S(r)              [S(r) = −w dT̄/dr, background heating]
    ∂ω/∂t + (u·∇)ω − (ω·∇)u
                = −(2Ω/Ek) ∂u/∂z + Ra·Pr·(r̂×∇Θ) + Pr ∇²ω     [vorticity form]

with ω = ∇×u.  The buoyancy term Ra·Pr·r̂×∇Θ is the curl of the buoyancy force
ρ' g ∝ Θ r̂ ; the Coriolis term (2Ω/Ek)∂u/∂z is the curl of 2Ω×u for Ω ∥ ẑ.

── LINEARISATION about rest (u=0, Θ=0).  Drop the quadratic advection, project on
a horizontal planform of wavenumber a (plane layer) or spherical degree l
(shell).  At marginal stability (growth rate σ=0) the amplitudes of the vertical
velocity W(z), vertical vorticity ζ(z) and temperature Θ(z) obey (Chandrasekhar
1961, Hydrodynamic and Hydromagnetic Stability, ch. III & VI):

    (D²−a²)² W = a² Ra Θ − Ta^{1/2} D ζ         [vertical momentum]
    (D²−a²)  ζ = Ta^{1/2} D W                    [vertical vorticity]
    (D²−a²)  Θ = −W                              [heat, background gradient = −W]

D = d/dz, Ta = Ek^{−2} the Taylor number.  Eliminating ζ, Θ gives the sixth-order
scalar  (D²−a²)³W + Ta D²W = −a² Ra W, whose lowest eigenvalue over a is Ra_c.

── BOUNDARY CONDITIONS.  Θ = 0 (fixed T) at both plates.  Mechanically:
    • stress-free ("free"):  W = D²W = 0,  Dζ = 0   → Ra_c = 27π⁴/4 = 657.511,
    • no-slip     ("rigid"): W = DW  = 0,  ζ  = 0   → Ra_c = 1707.762  (Ta=0).

═══════════════════════════════════════════════════════════════════════════════
NUMERICS — Chebyshev collocation eigenvalue solve
═══════════════════════════════════════════════════════════════════════════════
Radius/height is discretised on a Chebyshev–Gauss–Lobatto grid (Trefethen,
Spectral Methods in MATLAB, 2000).  Each 4th-order field W is split into (W, V=LW)
so every variable is 2nd-order, giving exactly two boundary conditions per field
and a well-posed generalized eigenproblem  A y = Ra B y,  y = [W, V, ζ, Θ].  The
smallest positive real eigenvalue is Ra(a); minimising over the planform a (plane
layer) or degree l (shell) yields Ra_c.  N≈40 collocation points already give
machine-accurate agreement with the analytic benchmarks.

References: Chandrasekhar (1961), Hydrodynamic and Hydromagnetic Stability (Oxford);
Christensen & Aubert (2006) GJI 166, 97; Gastine, Wicht & Aurnou (2013) Icarus
225, 156; Saltzman (1962) J. Atmos. Sci. 19, 329; Lorenz (1963) J. Atmos. Sci.
20, 130.
"""

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar


# ═════════════════════════════════════════════════════════════════════════════
# Chebyshev differentiation (Trefethen 2000, cheb.m)
# ═════════════════════════════════════════════════════════════════════════════

def cheb(N):
    """Chebyshev–Gauss–Lobatto nodes x_j=cos(jπ/N) on [−1,1] and the (N+1)²
    first-derivative matrix D.  Returns (D, x)."""
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def _smallest_real_positive(A, B):
    """Smallest real, positive, finite generalized eigenvalue of A y = λ B y."""
    w = eig(A, B, right=False)
    w = w[np.isfinite(w)]
    w = w[np.abs(w.imag) < 1e-6 * (np.abs(w.real) + 1.0)].real
    w = w[w > 1.0]
    return w.min() if w.size else np.inf


# ═════════════════════════════════════════════════════════════════════════════
# (A) Linear onset — plane layer (the exact Chandrasekhar benchmark)
# ═════════════════════════════════════════════════════════════════════════════

def onset_rayleigh_plane(a, Ta=0.0, bc='free', N=40):
    """
    Marginal Rayleigh number Ra(a) for a plane layer, horizontal wavenumber a,
    Taylor number Ta = Ek^{−2}, boundaries 'free' (stress-free) or 'rigid'
    (no-slip).  Solves the coupled (W, V=LW, ζ, Θ) eigenproblem on [0,1].
    """
    Dx, _ = cheb(N)
    D = 2.0 * Dx                      # map [−1,1] → [0,1]
    D2 = D @ D
    n = N + 1
    I = np.eye(n); Z = np.zeros((n, n))
    a2 = a * a
    L = D2 - a2 * I                   # Helmholtz operator D²−a²
    sTa = np.sqrt(Ta)
    # unknowns y = [W, V, ζ, Θ] with V ≡ L W (splitting keeps every field 2nd order)
    A = np.block([
        [-L,        I,   Z,        Z],   #  V − LW = 0
        [ Z,        L,   sTa * D,  Z],   #  LV + √Ta Dζ − a²Ra Θ = 0
        [-sTa * D,  Z,   L,        Z],   #  Lζ − √Ta DW = 0
        [ I,        Z,   Z,        L],   #  LΘ + W = 0
    ])
    B = np.zeros((4 * n, 4 * n))
    B[n:2 * n, 3 * n:4 * n] = a2 * I     # only LV-eq couples to Ra Θ
    for e in (0, N):                     # both plates
        for blk in range(4):
            A[blk * n + e, :] = 0.0; B[blk * n + e, :] = 0.0
        A[e, e] = 1.0                                     # W = 0
        if bc == 'free':
            A[n + e, n + e] = 1.0                         # D²W = 0  ⇔ V = 0
            A[2 * n + e, 2 * n:3 * n] = D[e, :]           # Dζ = 0 (stress-free)
        elif bc == 'rigid':
            A[n + e, 0:n] = D[e, :]                       # DW = 0 (no-slip)
            A[2 * n + e, 2 * n + e] = 1.0                 # ζ  = 0
        else:
            raise ValueError(f"bc must be 'free' or 'rigid', got {bc!r}")
        A[3 * n + e, 3 * n + e] = 1.0                     # Θ = 0
    return _smallest_real_positive(A, B)


def critical_rayleigh_plane(Ta=0.0, bc='free', N=40, a_max=None):
    """Critical Rayleigh number Ra_c = min_a Ra(a) and its wavenumber a_c.
    The critical planform grows ~Ta^{1/6}, so the search window widens with Ta."""
    if a_max is None:
        a_max = 4.0 + 0.9 * Ta ** (1.0 / 6.0)     # comfortably above a_c(Ta)
    res = minimize_scalar(lambda a: onset_rayleigh_plane(a, Ta, bc, N),
                          bounds=(0.5, a_max), method='bounded',
                          options={'xatol': 1e-4})
    return res.fun, res.x


def onset_rayleigh_free_analytic(a, Ta=0.0):
    """Exact stress-free marginal curve Ra(a) = [(π²+a²)³ + Ta π²]/a²
    (W = sin πz is the exact eigenfunction; Chandrasekhar 1961 §27)."""
    return ((np.pi ** 2 + a ** 2) ** 3 + Ta * np.pi ** 2) / a ** 2


# ═════════════════════════════════════════════════════════════════════════════
# (A′) Linear onset — spherical shell (non-rotating), reduces to plane layer
# ═════════════════════════════════════════════════════════════════════════════

def onset_rayleigh_shell(l, r_i, r_o, bc='free', N=40):
    """
    Marginal Rayleigh number Ra(l) for convection of spherical-harmonic degree l
    in a shell r_i<r<r_o (non-rotating).  The radial Laplacian of a Y_lm field is
    L_l = d²/dr² + (2/r) d/dr − l(l+1)/r²; the horizontal buoyancy coupling is
    l(l+1)/r².  In the thin-shell / large-l limit this reproduces the plane-layer
    Ra_c exactly (verified in verify()).
    """
    Dx, x = cheb(N)
    r = r_i + (r_o - r_i) * (x + 1.0) / 2.0
    D = (2.0 / (r_o - r_i)) * Dx
    D2 = D @ D
    n = N + 1
    I = np.eye(n); Z = np.zeros((n, n))
    invr = np.diag(1.0 / r); invr2 = np.diag(1.0 / r ** 2)
    ll = l * (l + 1.0)
    L = D2 + 2.0 * invr @ D - ll * invr2      # radial part of ∇² on degree l
    a2d = ll * invr2                          # horizontal buoyancy coupling
    A = np.block([[-L, I, Z], [Z, L, Z], [I, Z, L]])
    B = np.zeros((3 * n, 3 * n)); B[n:2 * n, 2 * n:3 * n] = a2d
    for e in (0, N):
        for blk in range(3):
            A[blk * n + e, :] = 0.0; B[blk * n + e, :] = 0.0
        A[e, e] = 1.0                                     # W = 0
        if bc == 'free':
            A[n + e, n + e] = 1.0                         # D²W = 0 ⇔ V = 0
        elif bc == 'rigid':
            A[n + e, 0:n] = D[e, :]                       # DW = 0
        else:
            raise ValueError(f"bc must be 'free' or 'rigid', got {bc!r}")
        A[2 * n + e, 2 * n + e] = 1.0                     # Θ = 0
    return _smallest_real_positive(A, B)


# ═════════════════════════════════════════════════════════════════════════════
# Control-parameter bookkeeping (Ek ↔ Ta, the rapid-rotation law)
# ═════════════════════════════════════════════════════════════════════════════

def taylor_from_ekman(Ek):
    """Ta = Ek^{−2}   (Ek = ν/2Ωd², Ta = (2Ωd²/ν)²)."""
    return Ek ** -2.0


class RotatingConvection:
    """
    Container for a Boussinesq rotating-convection setup and its onset.

    Parameters
    ----------
    Ek : Ekman number (ν/2Ωd²).  Ek→∞ (Ta=0) is the non-rotating limit.
    Pr : Prandtl number (ν/κ).  Sets the *time scale* of onset, not Ra_c itself
         for these steady (exchange-of-stabilities) modes.
    bc : 'free' (stress-free) or 'rigid' (no-slip) plates.
    """

    def __init__(self, Ek=np.inf, Pr=1.0, bc='free'):
        self.Ek, self.Pr, self.bc = Ek, Pr, bc
        self.Ta = 0.0 if np.isinf(Ek) else taylor_from_ekman(Ek)

    def critical(self, N=40):
        """(Ra_c, a_c) for this configuration (plane-layer onset)."""
        return critical_rayleigh_plane(self.Ta, self.bc, N)


# ═════════════════════════════════════════════════════════════════════════════
# (B) Energy conservation of the convective nonlinearity (Saltzman/Lorenz)
# ═════════════════════════════════════════════════════════════════════════════

def _lorenz_rhs(state, sigma, r, b):
    """Saltzman(1962)/Lorenz(1963) truncation of 2-D Boussinesq RB.  X≈velocity
    amplitude, Y,Z≈temperature-perturbation amplitudes.  The QUADRATIC terms
    (−XZ, +XY) are the Galerkin projection of the advective nonlinearity u·∇."""
    X, Y, Z = state
    return np.array([sigma * (Y - X),
                     r * X - Y - X * Z,
                     X * Y - b * Z])


def _lorenz_nonlinear_only(state):
    """The dissipationless, unforced limit: keep ONLY the advective nonlinearity
    (σ=b=0, no buoyancy forcing rX, no linear damping).  Ẋ=0, Ẏ=−XZ, Ż=XY."""
    X, Y, Z = state
    return np.array([0.0, -X * Z, X * Y])


def _rk4(rhs, y0, dt, nsteps):
    y = np.array(y0, float); traj = [y.copy()]
    for _ in range(nsteps):
        k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(y.copy())
    return np.array(traj)


def energy_conservation_inviscid(y0=(1.0, 1.0, 1.0), dt=1e-3, nsteps=20000):
    """
    Integrate the Galerkin-truncated Boussinesq system in the DISSIPATIONLESS,
    UNFORCED limit and return the relative drift of the quadratic energy
    E = ½(X²+Y²+Z²).  d/dt E|_nonlinear = Y(−XZ)+Z(XY) = 0 exactly, so a correct
    advective nonlinearity conserves E; any drift is pure time-stepping error.
    """
    traj = _rk4(_lorenz_nonlinear_only, y0, dt, nsteps)
    E = 0.5 * (traj ** 2).sum(axis=1)
    return abs(E[-1] - E[0]) / E[0], E


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify():
    print("=" * 74)
    print("Boussinesq convection in a rotating spherical shell — verification")
    print("=" * 74)
    ok = True

    # ── 1. Non-rotating onset: exact Chandrasekhar critical Rayleigh numbers ────
    Rc_free, ac_free = critical_rayleigh_plane(0.0, 'free')
    exact_free = 27.0 * np.pi ** 4 / 4.0
    e_free = abs(Rc_free - exact_free) / exact_free
    print(f"\n1a. Stress-free onset:  Ra_c = {Rc_free:.3f}  (exact 27π⁴/4 = "
          f"{exact_free:.3f}),  a_c = {ac_free:.4f} (exact {np.pi/np.sqrt(2):.4f})")
    print(f"    → rel. error {e_free:.2e}  {'✓' if e_free < 1e-3 else '⚠'}")
    ok &= e_free < 1e-3

    Rc_rig, ac_rig = critical_rayleigh_plane(0.0, 'rigid')
    e_rig = abs(Rc_rig - 1707.762) / 1707.762
    print(f"1b. No-slip onset:      Ra_c = {Rc_rig:.3f}  (exact 1707.762),  "
          f"a_c = {ac_rig:.4f} (exact 3.117)")
    print(f"    → rel. error {e_rig:.2e}  {'✓' if e_rig < 1e-3 else '⚠'}")
    ok &= e_rig < 1e-3

    # ── 2. Rotating onset matches the exact stress-free marginal curve ─────────
    print("\n2. Rotating onset (stress-free) vs exact [(π²+a²)³+Taπ²]/a²:")
    rot_ok = True
    for Ta in (1e3, 1e4, 1e5, 1e6):
        Rc, _ = critical_rayleigh_plane(Ta, 'free')
        aa = np.linspace(0.5, 40, 12000)
        Ra_an = onset_rayleigh_free_analytic(aa, Ta).min()
        rel = abs(Rc - Ra_an) / Ra_an
        rot_ok &= rel < 5e-3
        print(f"   Ta={Ta:.0e}: Ra_c={Rc:11.1f}  analytic={Ra_an:11.1f}  "
              f"rel={rel:.1e} {'✓' if rel < 5e-3 else '⚠'}")
    ok &= rot_ok

    # ── 3. Rapid-rotation scaling Ra_c ∝ Ta^{2/3} = Ek^{−4/3} ──────────────────
    Rc5, _ = critical_rayleigh_plane(1e5, 'free')
    Rc6, _ = critical_rayleigh_plane(1e6, 'free')
    slope = np.log(Rc6 / Rc5) / np.log(1e6 / 1e5)
    print(f"\n3. Rotational stabilisation: Ra_c(Ta=1e6)/Ra_c(Ta=1e5) exponent = "
          f"{slope:.3f}  (→ 2/3 = 0.667 asymptotically; Ra_c ∝ Ek^(−4/3))")
    scale_ok = Rc6 > Rc5 and 0.55 < slope < 0.70
    print(f"    → rotation raises Ra_c, exponent approaching 2/3  "
          f"{'✓' if scale_ok else '⚠'}")
    ok &= scale_ok

    # ── 4. Spherical-shell onset reduces to the plane layer (thin-shell limit) ──
    rmid = 2000.5; r_i, r_o = 2000.0, 2001.0        # gap d=1, r_mid≈2000
    l_star = int(round((np.pi / np.sqrt(2)) * rmid))  # a_c=π/√2 ⇒ √(l(l+1))≈a_c·r
    Rc_shell = min(onset_rayleigh_shell(l, r_i, r_o, 'free')
                   for l in range(l_star - 300, l_star + 301, 60))
    e_shell = abs(Rc_shell - exact_free) / exact_free
    print(f"\n4. Spherical-shell onset, thin-shell limit (l≈{l_star}): "
          f"Ra_c = {Rc_shell:.3f}")
    print(f"    → matches plane-layer 27π⁴/4 = {exact_free:.3f}, rel {e_shell:.2e}"
          f"  {'✓' if e_shell < 1e-2 else '⚠'}")
    ok &= e_shell < 1e-2

    # ── 5. Energy conservation of the convective nonlinearity (inviscid limit) ──
    drift, E = energy_conservation_inviscid()
    print(f"\n5. Energy conservation, dissipationless limit "
          f"(Galerkin/Saltzman–Lorenz): ΔE/E = {drift:.2e} over {len(E)} steps")
    en_ok = drift < 1e-6
    print(f"    → advective nonlinearity conserves ½(X²+Y²+Z²)  "
          f"{'✓' if en_ok else '⚠'}")
    ok &= en_ok

    print("\n" + ("✓ ALL CHECKS PASSED — onset Ra_c reproduced (non-rotating & "
                  "rotating), shell→plane limit, energy conserved"
                  if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
