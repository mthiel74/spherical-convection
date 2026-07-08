"""
honest_interior.py — physically honest radial reconstruction of the cutaway
interior.

Scientific improvement #20 (scientific_improvements.md §20).  The v6 renderer
(visualize_v6.py) continues the surface field inward with a MIXING-LENGTH-scaled
factor (r/R)^(l/L_REF) using L_REF = 10, and paints the inner core on a
disconnected colour scale ~32× amplified.  Both OVERSTATE what a 2-D barotropic
field can say about the deep interior:

  • the true harmonic (potential) continuation of a degree-l surface pattern that
    is regular at the origin is (r/R)^l — the l/L_REF exponent with L_REF=10
    inflates the e-folding depth by ~10×;
  • for the forcing-scale modes (l ≈ 60–80) the honest (r/R)^l factor is ~4e-11
    at the base r = 0.71 R, i.e. the interior is essentially EMPTY.  That is the
    correct answer: a 2-D barotropic model carries NO radial information, so any
    visible interior structure is decoration, not physics.

This module provides the HONEST alternatives named in the spec:
  (a) surface_field()      — show ONLY the 2-D surface field, no interior at all;
  (b) interior_coeffs() / meridional_slice() — render the TRUE (r/R)^l evanescent
      continuation (l_ref = 1), which is surface-confined for forcing-scale modes.

WHAT (r/R)^l IS.  It is the interior harmonic solution of Laplace's equation
∇²Φ = 0 that is regular at r = 0:  Φ(r,θ,φ) = Σ_lm (r/R)^l ω_lm Y_lm(θ,φ).  It is
the mathematical potential continuation of the SURFACE pattern — NOT solved
interior dynamics.  It is evanescent: high-l (small-scale) structure decays
within a thin skin (depth ~ R/l), while only the largest scales reach the base.
For the true barotropic vorticity field this continuation carries no dynamical
meaning at all; it is shown only to make honest what the earlier renderer faked.

Reference: project audits physics_audit.md and v6_critical_audit.md §4.4, §5.1;
for a genuine 3-D field see scientific_improvements.md items 15–17.
"""

import numpy as np
import pyshtools as pysh

from config_v6 import R_INNER, R_OUTER, LMAX, CORE_LMAX, L_REF


# ═════════════════════════════════════════════════════════════════════════════
# Radial continuation factors
# ═════════════════════════════════════════════════════════════════════════════

def radial_factor(radii, lmax=LMAX, l_ref=1.0, r_outer=R_OUTER):
    """
    Radial continuation factor (r/R_outer)^(l/l_ref) for every degree l.

    l_ref = 1.0  → the TRUE (r/R)^l harmonic continuation (regular at the origin).
    l_ref = 10   → the v6 mixing-length rescaling (overstates penetration ~10×).

    Parameters
    ----------
    radii : (Nr,) array of radii (same units as r_outer)
    lmax  : maximum degree
    l_ref : reference degree in the exponent l/l_ref
    r_outer : surface radius

    Returns
    -------
    fac : (Nr, lmax+1) array;  fac[i, l] = (radii[i]/r_outer)^(l/l_ref).
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    l = np.arange(lmax + 1)
    return (radii[:, None] / r_outer) ** (l[None, :] / l_ref)


def interior_coeffs(surface_coeffs, radius, l_ref=1.0, r_outer=R_OUTER):
    """
    Spectral coefficients of the field continued inward to a single `radius`.

    Each degree-l coefficient is multiplied by (radius/R_outer)^(l/l_ref).  With
    the default l_ref = 1 this is the exact harmonic (potential) continuation.
    """
    lmax = surface_coeffs.shape[1] - 1
    fac = radial_factor(radius, lmax, l_ref, r_outer)[0]     # (lmax+1,) over degree l
    return surface_coeffs * fac[None, :, None]


# ═════════════════════════════════════════════════════════════════════════════
# Honest visualisations
# ═════════════════════════════════════════════════════════════════════════════

def surface_field(surface_coeffs, grid='DH2'):
    """
    Option (a): the honest 2-D surface field only — expand the coefficients on the
    sphere and return the SHGrid.  No interior is fabricated.
    """
    c = pysh.SHCoeffs.from_array(surface_coeffs, normalization='4pi', csphase=1)
    return c.expand(grid=grid)


def meridional_slice(surface_coeffs, longitude_deg=0.0, n_radial=60,
                     l_ref=1.0, r_outer=R_OUTER, r_inner=R_INNER):
    """
    Option (b): the TRUE (r/R)^l evanescent continuation on a meridional plane.

    Returns a pole-to-pole × radius slice of the continued field at the given
    longitude (and its antipode, longitude+180°, to close the meridian):

        field[i, k] = Σ_lm surface_coeffs_lm · (r_i/R)^(l/l_ref) · Y_lm(θ_k, φ),

    Parameters
    ----------
    longitude_deg : meridian longitude φ (deg); the returned slice spans θ∈[0,π]
                    on this meridian and θ∈[π,2π] wrapped is the φ+180° meridian.
    n_radial : number of radial nodes across the shell r∈[r_inner, r_outer].

    Returns
    -------
    radii : (n_radial,) radial nodes (ascending, r_inner … r_outer)
    colat : (Nθ,) colatitude nodes (rad, 0 … π) along the meridian
    field : (n_radial, Nθ) continued field on the meridional plane.
    """
    lmax = surface_coeffs.shape[1] - 1
    radii = np.linspace(r_inner, r_outer, n_radial)
    fac = radial_factor(radii, lmax, l_ref, r_outer)         # (Nr, lmax+1)

    # angular design: expand once per radius is cheap enough via pyshtools grid,
    # then pull the requested meridian column.  Use a DH grid and its longitudes.
    dummy = pysh.SHCoeffs.from_zeros(lmax, normalization='4pi')
    lats = dummy.expand(grid='DH').lats()                    # +90 … −90 (deg)
    lons = dummy.expand(grid='DH').lons()                    # 0 … 360 (deg)
    colat = np.deg2rad(90.0 - lats)                          # 0 … π
    jlon = int(np.argmin(np.abs(lons - (longitude_deg % 360.0))))

    field = np.empty((n_radial, colat.size))
    for i in range(n_radial):
        scaled = surface_coeffs * fac[i][None, :, None]      # continue to r_i
        g = pysh.SHCoeffs.from_array(scaled, normalization='4pi',
                                     csphase=1).expand(grid='DH')
        field[i] = g.data[:, jlon]                           # the meridian column
    return radii, colat, field


# ═════════════════════════════════════════════════════════════════════════════
# Honesty diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def amplitude_profile(surface_coeffs, radii, l_ref=1.0, r_outer=R_OUTER):
    """
    RMS amplitude of the continued field vs radius (Parseval / power spectrum).

    variance(r) = Σ_lm ω_lm² (r/R)^(2l/l_ref);  RMS(r) = √variance(r).  Normalised
    by the SURFACE RMS (r = R_outer, factor 1), so the returned profile is the
    fraction of surface amplitude reaching each radius.  This is the honesty
    diagnostic: with l_ref=1 the forcing-scale power collapses within a thin skin.
    """
    lmax = surface_coeffs.shape[1] - 1
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    power_l = (surface_coeffs ** 2).sum(axis=(0, 2))         # Σ_m ω_lm² per degree
    fac = radial_factor(radii, lmax, l_ref, r_outer)         # (Nr, lmax+1)
    rms = np.sqrt((fac ** 2 * power_l[None, :]).sum(axis=1))
    surf = np.sqrt(power_l.sum())                            # RMS at r = R_outer
    return rms / (surf if surf > 0 else 1.0)


def penetration_depth(l, r_outer=R_OUTER):
    """
    e-folding penetration depth of the TRUE (r/R)^l continuation for degree l:
    the depth below the surface at which the factor falls to 1/e, i.e.
    (r/R)^l = e^{-1}  →  r/R = e^{-1/l}  →  depth = R(1 − e^{-1/l}) ≈ R/l.
    Decreases with l — small scales are surface-trapped.
    """
    l = np.asarray(l, dtype=float)
    return r_outer * (1.0 - np.exp(-1.0 / l))


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def _forcing_scale_spectrum(lmax=LMAX, l_peak=70, width=8.0, seed=0):
    """A vorticity spectrum peaked at the forcing scale l_peak (Gaussian in l)."""
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal((2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        coeffs[:, l, l + 1:] = 0.0                           # zero unused m>l slots
        coeffs[:, l, :l + 1] *= np.exp(-0.5 * ((l - l_peak) / width) ** 2)
    coeffs[:, 0, 0] = 0.0
    return coeffs


def verify():
    print("=" * 74)
    print("Honest interior reconstruction — (r/R)^l verification")
    print("=" * 74)
    ok = True

    # ── 1. (r/R)^l gives the correct decay rate at several radii/degrees ─────────
    radii = np.array([0.9, 0.8, 0.71, 0.5])
    fac = radial_factor(radii, lmax=LMAX, l_ref=1.0)         # (Nr, L+1)
    max_err = 0.0
    for i, r in enumerate(radii):
        for l in (1, 5, 20, 70):
            expected = (r / R_OUTER) ** l
            max_err = max(max_err, abs(fac[i, l] - expected))
    print(f"\n1. (r/R)^l decay factor vs analytic, over r∈{list(radii)}, "
          f"l∈[1,5,20,70]:\n   max |factor − (r/R)^l| = {max_err:.2e}")
    decay_ok = max_err < 1e-12
    print(f"   → {'✓ radial factor is exactly (r/R)^l' if decay_ok else '⚠ mismatch'}")
    ok &= decay_ok

    # ── 2. low-l modes penetrate deeper than high-l modes ───────────────────────
    r_probe = 0.71                                           # base of the shell
    f_l = radial_factor(np.array([r_probe]), lmax=LMAX, l_ref=1.0)[0]
    monotone = np.all(np.diff(f_l[1:]) < 0)                  # strictly decreasing in l
    depths = penetration_depth(np.array([2, 10, 40, 70]))
    depth_monotone = np.all(np.diff(depths) < 0)             # depth shrinks with l
    print(f"\n2. Penetration vs degree (factor at r/R={r_probe}):")
    print(f"   (r/R)^l at l=2,10,40,70 : "
          f"{f_l[2]:.3e}, {f_l[10]:.3e}, {f_l[40]:.3e}, {f_l[70]:.3e}")
    print(f"   e-folding depth (R units) l=2,10,40,70 : "
          + ", ".join(f"{d:.3f}" for d in depths))
    pen_ok = monotone and depth_monotone and f_l[2] > f_l[10] > f_l[40] > f_l[70]
    print(f"   → {'✓ low-l penetrate deeper; high-l are surface-trapped' if pen_ok else '⚠ non-monotone'}")
    ok &= pen_ok

    # ── 3. L_REF=1 (true) gives far less interior amplitude than L_REF=10 ────────
    coeffs = _forcing_scale_spectrum(lmax=LMAX, l_peak=70)
    radii = np.linspace(R_INNER, R_OUTER, 40)
    rms_true = amplitude_profile(coeffs, radii, l_ref=1.0)   # normalised to surface
    rms_v6 = amplitude_profile(coeffs, radii, l_ref=L_REF)
    base_true = rms_true[0]                                  # amplitude at r=R_INNER
    base_v6 = rms_v6[0]
    ratio = base_v6 / max(base_true, 1e-300)
    print(f"\n3. Interior amplitude at the base r=R_INNER={R_INNER} "
          f"(forcing-scale spectrum, l≈70):")
    print(f"   true  (r/R)^l   , RMS/surface = {base_true:.3e}")
    print(f"   v6 (r/R)^(l/10) , RMS/surface = {base_v6:.3e}")
    print(f"   overstatement factor L_REF=10 vs 1 = {ratio:.3e}")
    amp_ok = base_true < 1e-6 and base_v6 > 1e-3 and ratio > 1e3
    print(f"   → {f'✓ true continuation leaves the interior essentially EMPTY; '
                  f'L_REF=10 inflates it ~{ratio:.0e}×' if amp_ok else '⚠ unexpected amplitudes'}")
    ok &= amp_ok

    print("\n" + ("✓ ALL CHECKS PASSED — honest (r/R)^l continuation is "
                  "surface-confined; the fabricated interior is exposed"
                  if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
