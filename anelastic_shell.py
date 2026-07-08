"""
anelastic_shell.py — anelastic convection in a density-stratified spherical
shell  r_i < r < r_o.

Scientific improvement #16 (scientific_improvements.md §16): replace the
Boussinesq approximation of improvement #15 with the ANELASTIC approximation.
Boussinesq treats the background density as uniform — valid only when the layer
is thin compared with a density scale height.  The solar convection zone spans
~14 density scale heights (ρ varies by ~10⁶ from base to photosphere), so
Boussinesq is invalid there.  The anelastic approximation (Gough 1969; Braginsky
& Roberts 1995) FILTERS sound waves (the fast, dynamically irrelevant acoustic
mode) while RETAINING the strongly varying background density ρ̄(r).  Its two
defining consequences, both verified here:

  (1) the continuity equation becomes the ANELASTIC MASS CONSTRAINT
          ∇·(ρ̄ u) = 0                                     (not ∇·u = 0),
      so the velocity itself is compressible: ∇·u = −u·∇ln ρ̄ ≠ 0.  This is
      what produces the up/down-flow ASYMMETRY of stratified convection —
      broad slow upflows, narrow fast downdrafts.
  (2) the background is a hydrostatic POLYTROPE T̄, ρ̄, p̄ over N_ρ density scale
      heights, whose super-adiabaticity sets the convective driving.

═══════════════════════════════════════════════════════════════════════════════
BACKGROUND POLYTROPE  (the Jones et al. 2011 anelastic benchmark reference state)
═══════════════════════════════════════════════════════════════════════════════
With aspect ratio η = r_i/r_o, polytropic index n, and gravity g ∝ 1/r² (all mass
interior — appropriate for a self-gravitating envelope), the hydrostatic
polytropic solution is (Jones, Boronski, Brun, Glatzmaier, Gastine, Miesch &
Wicht 2011, Icarus 216, 120, §2):

    ζ(r) = c₀ + c₁ d/r ,        d = r_o − r_i,
    c₀ = (2ζ_i − η − 1)/(1−η),  c₁ = (1+η)(1−ζ_i)/(1−η)²,
    ζ_i = (η+1)/(η e^{N_ρ/n} + 1),

and the background fields are powers of the dimensionless temperature ζ:

    T̄ = ζ ,   ρ̄ = ζⁿ ,   p̄ = ζⁿ⁺¹ ,          (ideal gas p̄ = ρ̄ T̄ ✓)

here rescaled so T̄ = ρ̄ = 1 at the outer boundary r_o.  Then:
  • N_ρ ≡ ln[ρ̄(r_i)/ρ̄(r_o)] is EXACTLY the requested number of density scale
    heights (verified below),
  • hydrostatic balance  dp̄/dr = −ρ̄ ḡ  holds with  ḡ(r) = (n+1)c₁ d /(ζ_o r²),
  • the polytrope is p̄ ∝ ρ̄^{(n+1)/n}; it is exactly ISENTROPIC (adiabatic) when
    n = n_ad = 1/(γ−1), and SUPER-adiabatic (convectively unstable, ds̄/dr < 0)
    when n < n_ad — the physical regime for convection.

═══════════════════════════════════════════════════════════════════════════════
ANELASTIC MASS CONSTRAINT via poloidal/toroidal mass flux
═══════════════════════════════════════════════════════════════════════════════
Writing the MASS FLUX ρ̄u (not u) with poloidal/toroidal scalars,
    ρ̄u = ∇×∇×(P r̂) + ∇×(T r̂),
makes ∇·(ρ̄u) = ∇·(∇×…) ≡ 0 identically.  For a single spherical-harmonic degree
l the poloidal mass flux has
    (ρ̄u)_r        = l(l+1) P(r) / r²,
    horizontal part derived from the potential  φ(r) = P′(r),
and the constraint ∇·(ρ̄u) = (1/r²) d/dr(r²(ρ̄u)_r) − l(l+1)φ/r² = 0 holds to
machine precision (verified).  The velocity u = (ρ̄u)/ρ̄ then has ∇·u ≠ 0,
with the exact relation ∇·u = −u_r d ln ρ̄/dr — genuine anelastic compressibility.

References: Gough (1969) J. Atmos. Sci. 26, 448; Braginsky & Roberts (1995)
GAFD 79, 1; Jones et al. (2011) Icarus 216, 120; Lantz & Fan (1999) ApJS 121, 247.
"""

import numpy as np

from boussinesq_shell import cheb          # verified Chebyshev primitive (impr. #15)


# ═════════════════════════════════════════════════════════════════════════════
# Anelastic polytropic reference state
# ═════════════════════════════════════════════════════════════════════════════

class AnelasticReferenceState:
    """
    Hydrostatic polytropic background for anelastic convection in a shell, on a
    Chebyshev radial grid.  Fields (T̄, ρ̄, p̄, ḡ, s̄) are exposed as arrays over
    self.r, normalised so T̄ = ρ̄ = 1 at the outer boundary.

    Parameters
    ----------
    n_rho : number of density scale heights  N_ρ = ln[ρ̄(r_i)/ρ̄(r_o)].
    poly_n : polytropic index n  (ρ̄ ∝ ζⁿ).  n < 1/(γ−1) ⇒ super-adiabatic.
    eta : aspect ratio r_i/r_o.
    gamma : ratio of specific heats (5/3 for a monatomic ideal gas / plasma).
    N : number of Chebyshev radial modes.
    """

    def __init__(self, n_rho=3.0, poly_n=2.0, eta=0.35, gamma=5.0 / 3.0, N=48):
        self.n_rho, self.n, self.eta, self.gamma = n_rho, poly_n, eta, gamma
        # gap fixed to d = 1 (only the ratio η matters for the profile shape)
        self.r_i = eta / (1.0 - eta)
        self.r_o = 1.0 / (1.0 - eta)
        d = self.r_o - self.r_i

        Dx, x = cheb(N)
        self.r = self.r_i + d * (x + 1.0) / 2.0          # x=+1 → r_o (index 0)
        self.D = (2.0 / d) * Dx                          # d/dr on this grid

        zeta_i = (eta + 1.0) / (eta * np.exp(n_rho / poly_n) + 1.0)
        c0 = (2.0 * zeta_i - eta - 1.0) / (1.0 - eta)
        c1 = (1.0 + eta) * (1.0 - zeta_i) / (1.0 - eta) ** 2
        zeta = c0 + c1 * d / self.r
        zeta_o = zeta[np.argmax(self.r)]                 # value at r_o
        zeta = zeta / zeta_o                             # normalise T̄(r_o)=1
        self.c0, self.c1, self.zeta_o = c0, c1, zeta_o

        self.T_bar = zeta                                # T̄ = ζ
        self.rho_bar = zeta ** poly_n                    # ρ̄ = ζⁿ
        self.p_bar = zeta ** (poly_n + 1.0)              # p̄ = ζⁿ⁺¹ = ρ̄ T̄
        self.g = (poly_n + 1.0) * c1 * d / (zeta_o * self.r ** 2)   # ḡ ∝ 1/r²
        cv = 1.0 / (gamma - 1.0)
        self.entropy = cv * np.log(self.p_bar / self.rho_bar ** gamma)   # s̄

    # ── diagnostics ──────────────────────────────────────────────────────────
    def density_scale_heights(self):
        """N_ρ = ln[ρ̄(r_i)/ρ̄(r_o)] measured from the built profile."""
        return np.log(self.rho_bar[np.argmin(self.r)] /
                      self.rho_bar[np.argmax(self.r)])

    def hydrostatic_residual(self):
        """max relative residual of  dp̄/dr + ρ̄ ḡ = 0."""
        res = self.D @ self.p_bar + self.rho_bar * self.g
        return np.max(np.abs(res)) / np.max(np.abs(self.rho_bar * self.g))

    def polytrope_residual(self):
        """max relative residual of  p̄ = ρ̄^{(n+1)/n}  (polytropic relation)."""
        target = self.rho_bar ** ((self.n + 1.0) / self.n)
        return np.max(np.abs(self.p_bar - target)) / np.max(np.abs(self.p_bar))

    def entropy_gradient(self):
        """ds̄/dr (array).  <0 super-adiabatic (unstable), 0 adiabatic, >0 stable."""
        return self.D @ self.entropy

    def is_superadiabatic(self):
        return self.n < 1.0 / (self.gamma - 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Anelastic mass constraint  ∇·(ρ̄u) = 0  via a poloidal mass flux
# ═════════════════════════════════════════════════════════════════════════════

def poloidal_mass_flux(ref, l, potential=None):
    """
    Build a single-degree-l poloidal MASS FLUX  M = ρ̄u = ∇×∇×(P r̂)  on ref.r.

    Returns a dict with the radial mass flux M_r, the horizontal potential φ = P′,
    and the derived velocity components u_r = M_r/ρ̄, φ_u = φ/ρ̄.  By construction
    ∇·M = 0 (anelastic constraint); ∇·u ≠ 0 (compressible).
    """
    r, D = ref.r, ref.D
    if potential is None:                          # smooth P vanishing at walls
        potential = np.sin(np.pi * (r - ref.r_i) / (ref.r_o - ref.r_i))
    P = potential
    Pp = D @ P
    M_r = l * (l + 1.0) * P / r ** 2               # (ρ̄u)_r
    phi = Pp                                        # horizontal potential φ = P′
    return dict(l=l, P=P, M_r=M_r, phi=phi,
                u_r=M_r / ref.rho_bar, phi_u=phi / ref.rho_bar)


def anelastic_divergence(ref, flux):
    """∇·(ρ̄u) for the poloidal mass-flux mode (should be ~0)."""
    r, D, l = ref.r, ref.D, flux['l']
    return (1.0 / r ** 2) * (D @ (r ** 2 * flux['M_r'])) \
        - l * (l + 1.0) / r ** 2 * flux['phi']


def velocity_divergence(ref, flux):
    """∇·u for the derived velocity (≠0: anelastic compressibility)."""
    r, D, l = ref.r, ref.D, flux['l']
    return (1.0 / r ** 2) * (D @ (r ** 2 * flux['u_r'])) \
        - l * (l + 1.0) / r ** 2 * flux['phi_u']


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify():
    print("=" * 74)
    print("Anelastic convection in a stratified spherical shell — verification")
    print("=" * 74)
    ok = True
    ref = AnelasticReferenceState(n_rho=3.0, poly_n=2.0, eta=0.35)

    # ── 1. Reference state: N_ρ, hydrostatic, polytrope, ideal gas ─────────────
    Nrec = ref.density_scale_heights()
    e_N = abs(Nrec - ref.n_rho) / ref.n_rho
    print(f"\n1a. Density scale heights: N_ρ(built) = {Nrec:.6f}  (input "
          f"{ref.n_rho})  rel {e_N:.1e}  {'✓' if e_N < 1e-6 else '⚠'}")
    ok &= e_N < 1e-6
    print(f"    ρ̄(r_o)={ref.rho_bar[np.argmax(ref.r)]:.3f}, "
          f"ρ̄(r_i)={ref.rho_bar[np.argmin(ref.r)]:.3f}  → ρ̄ varies ×"
          f"{ref.rho_bar[np.argmin(ref.r)]/ref.rho_bar[np.argmax(ref.r)]:.1f}")

    hres = ref.hydrostatic_residual()
    print(f"1b. Hydrostatic balance dp̄/dr = −ρ̄ḡ:  max rel residual {hres:.1e}  "
          f"{'✓' if hres < 1e-8 else '⚠'}")
    ok &= hres < 1e-8

    pres = ref.polytrope_residual()
    print(f"1c. Polytropic p̄ = ρ̄^((n+1)/n):        max rel residual {pres:.1e}  "
          f"{'✓' if pres < 1e-10 else '⚠'}")
    ok &= pres < 1e-10

    ig = np.max(np.abs(ref.p_bar - ref.rho_bar * ref.T_bar)) / np.max(ref.p_bar)
    print(f"1d. Ideal gas p̄ = ρ̄T̄:                  max rel residual {ig:.1e}  "
          f"{'✓' if ig < 1e-12 else '⚠'}")
    ok &= ig < 1e-12

    # ── 2. Anelastic mass constraint ∇·(ρ̄u)=0 while ∇·u≠0 ─────────────────────
    flux = poloidal_mass_flux(ref, l=3)
    divM = anelastic_divergence(ref, flux)[2:-2]        # trim endpoint stencils
    divu = velocity_divergence(ref, flux)[2:-2]
    scale = np.max(np.abs(flux['M_r']))
    dm = np.max(np.abs(divM)) / scale
    print(f"\n2a. Anelastic constraint ∇·(ρ̄u) = 0:   max |∇·(ρ̄u)|/|ρ̄u| = {dm:.1e}"
          f"  {'✓' if dm < 1e-9 else '⚠'}")
    ok &= dm < 1e-9
    print(f"2b. Velocity is compressible ∇·u ≠ 0:  max |∇·u| = "
          f"{np.max(np.abs(divu)):.3f}  {'✓' if np.max(np.abs(divu)) > 1e-2 else '⚠'}")
    ok &= np.max(np.abs(divu)) > 1e-2

    # exact anelastic relation ∇·u = −u_r d ln ρ̄/dr
    dlnrho = (ref.D @ ref.rho_bar) / ref.rho_bar
    expected = (-flux['u_r'] * dlnrho)[3:-3]
    rel = np.max(np.abs(velocity_divergence(ref, flux)[3:-3] - expected)) \
        / np.max(np.abs(expected))
    print(f"2c. ∇·u = −u_r dlnρ̄/dr (compressibility): max rel residual {rel:.1e}"
          f"  {'✓' if rel < 1e-8 else '⚠'}")
    ok &= rel < 1e-8

    # ── 3. Thermodynamic / energy-budget consistency of the stratification ────
    print("\n3. Entropy stratification ds̄/dr = c_v[(n+1)/n − γ] dlnρ̄/dr "
          f"(γ={ref.gamma:.3f}, n_ad=1/(γ−1)={1/(ref.gamma-1):.2f}):")
    cv = 1.0 / (ref.gamma - 1.0)
    for pn in (1.0, 1.5, 2.0):
        rr = AnelasticReferenceState(n_rho=3.0, poly_n=pn, eta=0.35)
        ds = rr.entropy_gradient()
        analytic = cv * ((pn + 1) / pn - rr.gamma) * (rr.D @ np.log(rr.rho_bar))
        resid = np.max(np.abs(ds[3:-3] - analytic[3:-3]))          # absolute
        regime = ("super-adiabatic (convecting)" if rr.is_superadiabatic()
                  else "adiabatic (neutral)" if abs(pn - 1 / (rr.gamma - 1)) < 1e-9
                  else "sub-adiabatic (stable)")
        good = resid < 1e-10
        ok &= good
        print(f"   n={pn}: mean ds̄/dr = {ds.mean():+.4f}  [{regime}],  "
              f"|ds̄/dr − analytic| = {resid:.1e}  {'✓' if good else '⚠'}")
    # the n = n_ad polytrope must be exactly isentropic
    r_ad = AnelasticReferenceState(n_rho=3.0, poly_n=1.0 / (5.0 / 3.0 - 1.0),
                                   eta=0.35)
    ds_ad = np.max(np.abs(r_ad.entropy_gradient()))
    print(f"   n=n_ad=1.5: max|ds̄/dr| = {ds_ad:.1e} → exactly isentropic  "
          f"{'✓' if ds_ad < 1e-10 else '⚠'}")
    ok &= ds_ad < 1e-10

    print("\n" + ("✓ ALL CHECKS PASSED — polytropic reference (N_ρ exact, "
                  "hydrostatic, ideal gas), anelastic ∇·(ρ̄u)=0 with "
                  "compressible u, entropy budget consistent"
                  if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
