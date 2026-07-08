"""
mhd_barotropic.py — two-dimensional magnetohydrodynamics (barotropic MHD) on a
rotating sphere.

Scientific improvement #14 (scientific_improvements.md §14): add a magnetic field
and the Lorentz force to the barotropic model.  This is the physically correct
direction for the SOLAR framing — in the tachocline a toroidal field suppresses
the inverse cascade and can quench or reorganise jets (Tobias, Diamond & Hughes
2007; Gilman 2000).  Standalone solver reusing the verified spectral primitives
(pyshtools Jacobian, Laplacian eigenvalues, ETDRK4 coefficients) and the same
Strang / ETDRK4 time stepping.

═══════════════════════════════════════════════════════════════════════════════
DERIVATION — 2-D incompressible MHD (reduced / barotropic)
═══════════════════════════════════════════════════════════════════════════════
In 2-D the velocity and magnetic field derive from scalar potentials,

    u = k̂×∇ψ ,   ω = ∇²ψ  (vorticity) ;    B = k̂×∇A ,   j = ∇²A  (current),

where A is the magnetic flux/streamfunction and j the out-of-plane current
density (units ρ=μ₀=1).  The incompressible MHD equations reduce to two coupled
advection equations for ω and A (Biskamp, Magnetohydrodynamic Turbulence 2003):

    ∂ω/∂t + J(ψ, ω+f) = J(A, j) + ν(−∇²)ⁿ⁻ᵗ… + F      (vorticity + Lorentz)
    ∂A/∂t + J(ψ, A)    = η ∇² A                        (induction)

with f = 2Ω sinφ the planetary vorticity, ν the (hyper)viscosity, η the magnetic
diffusivity and F an optional forcing.  BOTH the advection J(ψ,·) and the Lorentz
tension J(A,j) are the SAME spherical Jacobian primitive as the barotropic solver
— the magnetic tension is J of the flux function with the current, exactly
analogous to the advection of vorticity.

── SIGN CONVENTION (important).  The task statement writes the current as
j = −∇²A and the Lorentz term as +J(A,j).  Those two choices together give
+J(A,−∇²A) = −J(A,∇²A), whose linearisation yields σ² = −(k·v_A)² < 0 — a
spurious INSTABILITY, not Alfvén waves.  The physically correct magnetic tension
(the one that reproduces Alfvén waves, verified below) is +J(A,∇²A).  We adopt
the self-consistent convention j = ∇²A with the Lorentz term +J(A,j) = +J(A,∇²A);
this is identical physics to the task's B = ∇A×k̂ (which flips the sign of j) with
the correspondingly corrected tension sign.  Either way σ² = +(k·v_A)² > 0.

═══════════════════════════════════════════════════════════════════════════════
CONSERVATION LAWS (ideal limit ν=η=F=0, f arbitrary)
═══════════════════════════════════════════════════════════════════════════════
2-D MHD conserves three quadratic invariants, which the code checks:
    • total energy      E = ½∫(|u|²+|B|²)dΩ = ½Σ_lm λ(ψ_lm²+A_lm²)
    • mean-square flux  ⟨A²⟩ = Σ_lm A_lm²      (the 2-D magnetic Casimir)
    • cross-helicity    H_c = ∫u·B dΩ = Σ_lm λ ψ_lm A_lm .

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION — Alfvén waves
═══════════════════════════════════════════════════════════════════════════════
Linearising the ideal equations (ν=η=f=0) about a background field B₀ = k̂×∇A₀,
a perturbation of wavenumber k obeys  ∂ₜₜξ = (k·v_A)² ∂-of-nothing … i.e. it
OSCILLATES at the Alfvén frequency  ω_A = |k·v_A|,  v_A = B₀ (ρ=1).  A true
uniform field cannot exist on S² (the only harmonic scalars are constants), so we
verify the unambiguous Alfvén SIGNATURES, robust to O(1) geometric prefactors:
    (1) the perturbation OSCILLATES (bounded) rather than growing — magnetic
        tension is restoring, not destabilising (correct Lorentz sign);
    (2) kinetic and magnetic energy EXCHANGE periodically at fixed total energy;
    (3) the oscillation frequency scales LINEARLY with the field strength,
        ω_A ∝ |B₀|  (double B₀ ⇒ double frequency, since v_A ∝ B₀);
    (4) the frequency scales with the perturbation wavenumber, ω_A ∝ k.
Together (1)–(4) are the defining dispersion ω_A = k·v_A of Alfvén waves.

References: Biskamp (2003), Magnetohydrodynamic Turbulence (CUP); Tobias, Diamond
& Hughes (2007) ApJ 667, L113; Gilman (2000) ApJ 544, L79.
"""

import numpy as np
import pyshtools as pysh

from simulate_v7 import _laplacian_eigenvalues, _etdrk4_coeffs


# ═════════════════════════════════════════════════════════════════════════════
# Parameters (nondimensional; unit sphere, ρ=μ₀=1)
# ═════════════════════════════════════════════════════════════════════════════
LMAX      = 64
OMEGA     = 0.0         # rotation (f = 2Ω sinφ); 0 for the pure-Alfvén tests
NU_HYPER  = 0.0         # ∇⁸ hyperviscosity on ω (0 = ideal)
ETA_MAG   = 0.0         # magnetic diffusivity η (0 = ideal, frozen-in flux)
DRAG      = 0.0         # linear drag on ω
DT        = 2.0e-3
TIME_SCHEME = 'strang'
ETDRK4_M  = 32


def _jacobian_lm(a_lm, b_lm, lmax):
    """J(a,b)=k̂·(∇a×∇b) — the verified barotropic spherical Jacobian primitive."""
    ca = pysh.SHCoeffs.from_array(a_lm, normalization='4pi', csphase=1)
    cb = pysh.SHCoeffs.from_array(b_lm, normalization='4pi', csphase=1)
    ga = ca.gradient(radius=1.0)
    gb = cb.gradient(radius=1.0)
    jac = ga.phi.data * gb.theta.data - ga.theta.data * gb.phi.data
    return pysh.SHGrid.from_array(jac, grid='DH').expand(
        normalization='4pi', csphase=1, lmax_calc=lmax).coeffs


# ═════════════════════════════════════════════════════════════════════════════
# Solver
# ═════════════════════════════════════════════════════════════════════════════

class MHDBarotropic:
    """
    2-D barotropic MHD on the sphere.  Prognostic spectral fields: vorticity ω
    (omega) and magnetic flux function A.  Diagnostics: ψ=∇⁻²ω, B=k̂×∇A, j=∇²A.
    """

    def __init__(self, lmax=LMAX, omega=OMEGA, nu=NU_HYPER, eta=ETA_MAG,
                 drag=DRAG, dt=DT, time_scheme=TIME_SCHEME):
        self.lmax = lmax
        self.omega_rot, self.nu, self.eta, self.drag, self.dt = omega, nu, eta, drag, dt
        self._ev = _laplacian_eigenvalues(lmax)                 # −λ  (= ∇²)
        self._lam4 = self._ev ** 4
        self._inv = np.zeros((2, lmax + 1, lmax + 1))           # ∇⁻² (l=0→0)
        for l in range(1, lmax + 1):
            self._inv[:, l, :l + 1] = -1.0 / (l * (l + 1))
        self._f = np.zeros((2, lmax + 1, lmax + 1))             # f = 2Ω sinφ
        self._f[0, 1, 0] = 2.0 * omega / np.sqrt(3.0)

        # diagonal linear operators: ω gets ∇⁸ hyperviscosity + drag; A gets the
        # magnetic diffusion η∇² (regular Laplacian, eigenvalue −ηλ).
        self._Lw = -(self.drag + self.nu * self._lam4)
        self._La = self.eta * self._ev                          # = −η λ
        self._E2w = np.exp(self._Lw * dt / 2.0)
        self._E2a = np.exp(self._La * dt / 2.0)

        self.time_scheme = time_scheme
        if time_scheme not in ('strang', 'etdrk4'):
            raise ValueError(f"time_scheme must be 'strang' or 'etdrk4', got "
                             f"{time_scheme!r}")
        if time_scheme == 'etdrk4':
            (self._Ew, self._E2fw, self._Qw, self._f1w, self._f2w, self._f3w) = \
                _etdrk4_coeffs(self._Lw, dt, ETDRK4_M)
            (self._Ea, self._E2fa, self._Qa, self._f1a, self._f2a, self._f3a) = \
                _etdrk4_coeffs(self._La, dt, ETDRK4_M)

        self.omega = np.zeros((2, lmax + 1, lmax + 1))          # vorticity ω
        self.A = np.zeros((2, lmax + 1, lmax + 1))              # flux function A

    # ── nonlinear tendency (advection + Lorentz + induction advection) ─────────
    def _tendency(self, omega, A):
        """
        N_ω = −J(ψ, ω+f) + J(A, j),   j = ∇²A   (advection + Lorentz tension)
        N_A = −J(ψ, A)                            (flux advection; η∇²A is linear L)
        """
        psi = self._inv * omega
        j = self._ev * A                                        # j = ∇²A = −λ A
        n_omega = (-_jacobian_lm(psi, omega + self._f, self.lmax)
                   + _jacobian_lm(A, j, self.lmax))
        n_A = -_jacobian_lm(psi, A, self.lmax)
        return n_omega, n_A

    # ── time stepping ─────────────────────────────────────────────────────────
    def step(self):
        if self.time_scheme == 'etdrk4':
            self._step_etdrk4()
        else:
            self._step_strang()

    def _step_strang(self):
        dt = self.dt
        w = self._E2w * self.omega
        a = self._E2a * self.A
        k1w, k1a = self._tendency(w, a)
        k2w, k2a = self._tendency(w + dt * k1w, a + dt * k1a)
        w = w + 0.5 * dt * (k1w + k2w)
        a = a + 0.5 * dt * (k1a + k2a)
        self.omega = self._E2w * w
        self.A = self._E2a * a
        self.omega[:, 0, 0] = 0.0
        self.A[:, 0, 0] = 0.0

    def _step_etdrk4(self):
        w, a = self.omega, self.A
        Nw, Na = self._tendency(w, a)
        aw = self._E2fw * w + self._Qw * Nw
        aa = self._E2fa * a + self._Qa * Na
        Naw, Naa = self._tendency(aw, aa)
        bw = self._E2fw * w + self._Qw * Naw
        ba = self._E2fa * a + self._Qa * Naa
        Nbw, Nba = self._tendency(bw, ba)
        cw = self._E2fw * aw + self._Qw * (2 * Nbw - Nw)
        ca = self._E2fa * aa + self._Qa * (2 * Nba - Na)
        Ncw, Nca = self._tendency(cw, ca)
        self.omega = self._Ew * w + Nw * self._f1w + 2 * (Naw + Nbw) * self._f2w + Ncw * self._f3w
        self.A = self._Ea * a + Na * self._f1a + 2 * (Naa + Nba) * self._f2a + Nca * self._f3a
        self.omega[:, 0, 0] = 0.0
        self.A[:, 0, 0] = 0.0

    # ── diagnostics ───────────────────────────────────────────────────────────
    def invariants(self):
        """Total energy, kinetic/magnetic split, mean-square flux, cross-helicity."""
        ll = np.arange(self.lmax + 1)
        lam = (ll * (ll + 1.0))
        psi = self._inv * self.omega
        cpsi = (psi[0] ** 2 + psi[1] ** 2).sum(axis=1)
        cA = (self.A[0] ** 2 + self.A[1] ** 2).sum(axis=1)
        KE = 0.5 * (cpsi * lam).sum()
        ME = 0.5 * (cA * lam).sum()
        A2 = (self.A[0] ** 2 + self.A[1] ** 2).sum()
        Hc = (lam * (psi[0] * self.A[0] + psi[1] * self.A[1]).sum(axis=1)).sum()
        return dict(KE=KE, ME=ME, E=KE + ME, A2=A2, Hc=Hc)


# ═════════════════════════════════════════════════════════════════════════════
# Alfvén-wave verification
# ═════════════════════════════════════════════════════════════════════════════

def _measure_alfven_freq(b0, kmode, nsteps=6000, dt=2.0e-3):
    """
    Measure the Alfvén oscillation frequency of a single perturbation mode.

    Set a background flux A₀ (strength b0, a large-scale l=1 field), seed a small
    vorticity perturbation ω' at degree `kmode`, integrate the IDEAL equations
    (ν=η=f=0), and track the SEEDED coefficient ω'_{k,k}(t).  Magnetic tension
    makes it oscillate at the mode's Alfvén frequency ω_A = |k·v_A|.  Because no
    UNIFORM field exists on S² (harmonic scalars are constant), a degree-k mode is
    a superposition of local plane waves whose k·B₀ projection varies over the
    sphere, so its power spectrum spans 0…k·v_A.  We return two summaries of the
    seeded coefficient's power spectrum:
      • CENTROID (power-weighted mean frequency): scales cleanly with |B₀| (v_A∝B₀
        rescales every local frequency uniformly);
      • high-frequency EDGE (highest frequency holding >2% of the peak power):
        the k∥B₀-aligned component ω_A = k·v_A,max, which scales with k.
    Returns (centroid, edge, KE(t)).
    """
    lmax = max(40, kmode + 8)
    m = MHDBarotropic(lmax=lmax, omega=0.0, nu=0.0, eta=0.0, drag=0.0, dt=dt)
    m.A[0, 1, 1] = b0                            # background l=1 flux, strength b0
    m.omega[0, kmode, kmode] = 1.0e-4            # seed a single degree-k mode
    sig, KE = [], []
    for _ in range(nsteps):
        m.step()
        sig.append(m.omega[0, kmode, kmode])     # track the seeded coefficient
        KE.append(m.invariants()['KE'])
    sig = np.array(sig); KE = np.array(KE)
    s = sig - sig.mean()                         # detrend
    power = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), d=dt)           # cycles per time unit
    centroid = (fr[1:] * power[1:]).sum() / power[1:].sum()
    edge = fr[1:][power[1:] > 0.02 * power[1:].max()].max()
    return centroid, edge, KE


def verify():
    print("=" * 74)
    print("2-D barotropic MHD — Alfvén-wave & invariant verification")
    print("=" * 74)

    # ── 1. Ideal invariants conserved (no rotation: E, ⟨A²⟩ AND H_c) ──────────
    m = MHDBarotropic(lmax=48, omega=0.0, nu=0.0, eta=0.0, drag=0.0)
    rng = np.random.default_rng(0)
    for l in range(1, 10):
        m.omega[0, l, :l + 1] = rng.standard_normal(l + 1) * 0.1 / (l + 1)
        m.omega[1, l, 1:l + 1] = rng.standard_normal(l) * 0.1 / (l + 1)
        m.A[0, l, :l + 1] = rng.standard_normal(l + 1) * 0.1 / (l + 1)
        m.A[1, l, 1:l + 1] = rng.standard_normal(l) * 0.1 / (l + 1)
    m.omega[:, 0, 0] = 0.0; m.A[:, 0, 0] = 0.0
    I0 = m.invariants()
    for _ in range(500):
        m.step()
    I1 = m.invariants()
    dE = abs(I1['E'] - I0['E']) / abs(I0['E'])
    dA2 = abs(I1['A2'] - I0['A2']) / abs(I0['A2'])
    dHc = abs(I1['Hc'] - I0['Hc']) / (abs(I0['Hc']) + 1e-30)
    print(f"\n1. Ideal invariants (500 steps, no rotation): "
          f"ΔE/E={dE:.2e}  Δ⟨A²⟩/⟨A²⟩={dA2:.2e}  ΔH_c/H_c={dHc:.2e}")
    inv_ok = dE < 5e-3 and dA2 < 5e-3 and dHc < 5e-2
    print(f"   → {'✓ energy, mean-square flux & cross-helicity conserved' if inv_ok else '⚠ drift'}")

    # ── 2. Alfvén oscillation: bounded + energy exchange ──────────────────────
    c1, e1, KE1 = _measure_alfven_freq(b0=1.0, kmode=8)
    bounded = np.max(KE1) < 1e3 * (np.mean(KE1) + 1e-30) and np.isfinite(c1)
    exchange = KE1.std() / (KE1.mean() + 1e-30)
    print(f"\n2. Alfvén oscillation (B₀=1.0, k=8): ω_A(centroid) ≈ {c1:.2f}, "
          f"KE oscillation amplitude/mean = {exchange:.2f}")
    print(f"   → {'✓ bounded oscillatory energy exchange (restoring tension)' if bounded and exchange > 0.01 else '⚠ not oscillatory'}")

    # ── 3. ω_A ∝ |B₀|  (v_A = B₀: double the field → double the frequency) ─────
    ca, _, _ = _measure_alfven_freq(b0=0.5, kmode=8)
    cb, _, _ = _measure_alfven_freq(b0=1.0, kmode=8)
    ratio_b = cb / ca
    print(f"\n3. ω_A ∝ |B₀| (centroid):  ω(B₀=0.5)={ca:.3f}, ω(B₀=1.0)={cb:.3f}  "
          f"ratio={ratio_b:.2f} (expect ≈2.0)")
    b_ok = 1.7 < ratio_b < 2.3

    # ── 4. ω_A ∝ k  (spectral edge = k·v_A,max, the k∥B₀-aligned component) ─────
    _, ek1, _ = _measure_alfven_freq(b0=1.0, kmode=5)
    _, ek2, _ = _measure_alfven_freq(b0=1.0, kmode=10)
    ratio_k = ek2 / ek1
    print(f"\n4. ω_A ∝ k (spectral edge):  ω(k=5)={ek1:.3f}, ω(k=10)={ek2:.3f}  "
          f"ratio={ratio_k:.2f} (expect ≈2.0 for k·v_A)")
    k_ok = 1.5 < ratio_k < 2.5

    ok = inv_ok and bounded and b_ok and k_ok
    print("\n" + ("✓ ALL CHECKS PASSED — Alfvén waves reproduced (ω_A=k·v_A), "
                  "invariants conserved" if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
