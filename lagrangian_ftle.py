"""
lagrangian_ftle.py — Lagrangian particle tracking and finite-time Lyapunov
exponents (FTLE) on the spectral barotropic velocity field.

Scientific improvement #19 (scientific_improvements.md §19): advect passive
tracers with the (verified-correct) velocity field and compute FTLE to extract
Lagrangian coherent structures (LCS).  A snapshot of vorticity is an EULERIAN
object; it cannot by itself say where fluid is stirred versus transported
coherently.  FTLE ridges are the material transport barriers — and on a rotating
sphere the zonal JETS are exactly such barriers: fluid does not cross a jet core,
so an FTLE ridge sits on each jet while the flanks (roll-up regions) mix.  This
turns the pretty movie into evidence about transport and coherent structure.

Reference: Haller (2015) Annu. Rev. Fluid Mech. 47, 137 (LCS review); Shadden,
Lekien & Marsden (2005) Physica D 212, 271 (FTLE ridges as LCS).

═══════════════════════════════════════════════════════════════════════════════
VELOCITY FROM SPECTRAL VORTICITY
═══════════════════════════════════════════════════════════════════════════════
The barotropic streamfunction is the inverse Laplacian of the vorticity,
ψ_lm = −ω_lm/[l(l+1)] (rigid lid) or −ω_lm/[l(l+1)+1/L_d²] (equivalent
barotropic), and the non-divergent velocity is u = k̂×∇ψ.  In (θ,φ) colatitude/
longitude components (unit sphere, k̂ = ê_r):

    u = ê_r × ∇ψ = ê_r × (ê_θ ∂ψ/∂θ + ê_φ (1/sinθ) ∂ψ/∂φ)
                 = ê_φ ∂ψ/∂θ − ê_θ (1/sinθ) ∂ψ/∂φ ,

so   u_θ = −(1/sinθ) ∂ψ/∂φ ,   u_φ = ∂ψ/∂θ .

pyshtools' SHCoeffs.gradient returns the horizontal gradient components
g.theta = ∂ψ/∂θ and g.phi = (1/sinθ) ∂ψ/∂φ on a Driscoll–Healy grid, so
    u_θ = −g.phi ,   u_φ = +g.theta .
The tracer ODE on the unit sphere (arc-length elements ds_θ = dθ, ds_φ = sinθ dφ)
is therefore
    dθ/dt = u_θ ,   dφ/dt = u_φ / sinθ .

═══════════════════════════════════════════════════════════════════════════════
FTLE
═══════════════════════════════════════════════════════════════════════════════
Seed a uniform (θ,φ) grid of tracers, advect for time T under the (steady, i.e.
snapshot) velocity field, and form the flow-map gradient (deformation gradient)
F = ∂x(T)/∂x(0) by finite differencing neighbouring tracers.  Carrying the sphere
metric (physical arc-length displacements dθ and sinθ dφ),

    F = [[ ∂Θ/∂θ₀ ,          (1/sinθ₀) ∂Θ/∂φ₀      ],
         [ sinΘ ∂Φ/∂θ₀ ,     (sinΘ/sinθ₀) ∂Φ/∂φ₀   ]] ,

where (Θ,Φ) = flow map of (θ₀,φ₀).  With σ_max the largest singular value of F
(equivalently √λ_max of the right Cauchy–Green tensor C = FᵀF), the FTLE is

    σ_FTLE(x) = (1/|T|) ln σ_max(x) .

Ridges of σ_FTLE are the finite-time transport barriers.
"""

import numpy as np
import pyshtools as pysh
from scipy.interpolate import RegularGridInterpolator

from simulate_v7 import _inverse_laplacian_ev

_POLE_EPS = 1.0e-3          # keep tracers this far (rad) from the poles


# ═════════════════════════════════════════════════════════════════════════════
# Velocity field from spectral vorticity
# ═════════════════════════════════════════════════════════════════════════════

def velocity_grids(omega_lm, deformation_radius=None):
    """
    Return the (steady) velocity field of a spectral vorticity snapshot on the
    Driscoll–Healy grid.

    Parameters
    ----------
    omega_lm : (2, L+1, L+1) array   spectral vorticity ω_lm (pyshtools 4π layout)
    deformation_radius : float or None   finite L_d (equivalent barotropic) or None

    Returns
    -------
    theta : (Nθ,)  colatitude nodes (rad), ascending 0…π (both poles included)
    phi   : (Nφ,)  longitude nodes (rad), ascending 0…2π (the pyshtools DH grid
            already carries the wrap column φ=2π ≡ φ=0, so it is periodic-ready)
    u_theta, u_phi : (Nθ, Nφ)  velocity components at the nodes
    """
    lmax = omega_lm.shape[1] - 1
    inv = _inverse_laplacian_ev(lmax, deformation_radius)
    psi_lm = inv * omega_lm
    cpsi = pysh.SHCoeffs.from_array(psi_lm, normalization='4pi', csphase=1)
    g = cpsi.gradient(radius=1.0)                 # g.theta = ∂ψ/∂θ, g.phi = (1/sinθ)∂ψ/∂φ

    lats = g.theta.lats()                          # +90 … −90 (deg), descending
    lons = g.theta.lons()                          # 0 … 360 (deg), ascending (incl. 360)
    theta = np.deg2rad(90.0 - lats)                # 0 … π ascending
    phi = np.deg2rad(lons)                         # 0 … 2π ascending

    u_theta = -g.phi.data                          # u_θ = −(1/sinθ)∂ψ/∂φ
    u_phi = g.theta.data                           # u_φ = ∂ψ/∂θ
    return theta, phi, u_theta, u_phi


def _velocity_interpolators(omega_lm, deformation_radius=None):
    """Build vectorised (θ,φ)→(u_θ,u_φ) interpolators for a vorticity snapshot."""
    theta, phi, u_theta, u_phi = velocity_grids(omega_lm, deformation_radius)
    kw = dict(method='linear', bounds_error=False, fill_value=None)
    it = RegularGridInterpolator((theta, phi), u_theta, **kw)
    ip = RegularGridInterpolator((theta, phi), u_phi, **kw)

    def rhs(th, ph):
        """Tracer ODE RHS: dθ/dt = u_θ, dφ/dt = u_φ/sinθ (steady field)."""
        thc = np.clip(th, _POLE_EPS, np.pi - _POLE_EPS)
        pts = np.stack([thc, np.mod(ph, 2.0 * np.pi)], axis=-1)
        ut = it(pts)
        up = ip(pts)
        return ut, up / np.sin(thc)

    return rhs


# ═════════════════════════════════════════════════════════════════════════════
# Particle advection (RK4 on the sphere)
# ═════════════════════════════════════════════════════════════════════════════

def advect(omega_lm, theta0, phi0, T, dt=None, deformation_radius=None,
           rhs=None, unwrap_phi=False):
    """
    Advect passive tracers under the steady velocity field of ω_lm for time T
    (T<0 integrates backward) using classical RK4.

    Parameters
    ----------
    theta0, phi0 : arrays of initial colatitude/longitude (rad)
    T   : total advection time (signed)
    dt  : step (defaults to |T|/200, sign taken from T)
    unwrap_phi : if True, do NOT wrap φ into [0,2π) in the OUTPUT (needed so the
                 flow-map gradient sees a smooth field); interpolation always uses
                 the wrapped angle internally.

    Returns
    -------
    theta, phi : final tracer positions (rad); same shape as inputs.
    """
    if rhs is None:
        rhs = _velocity_interpolators(omega_lm, deformation_radius)
    th = np.array(theta0, dtype=float).copy()
    ph = np.array(phi0, dtype=float).copy()
    if T == 0.0:
        return th, ph
    nsteps = max(1, int(np.ceil(abs(T) / (abs(dt) if dt else abs(T) / 200.0))))
    h = T / nsteps                                 # signed step
    for _ in range(nsteps):
        k1t, k1p = rhs(th, ph)
        k2t, k2p = rhs(th + 0.5 * h * k1t, ph + 0.5 * h * k1p)
        k3t, k3p = rhs(th + 0.5 * h * k2t, ph + 0.5 * h * k2p)
        k4t, k4p = rhs(th + h * k3t, ph + h * k3p)
        th = th + (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        ph = ph + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        th = np.clip(th, _POLE_EPS, np.pi - _POLE_EPS)   # reflect off poles
    if not unwrap_phi:
        ph = np.mod(ph, 2.0 * np.pi)
    return th, ph


# ═════════════════════════════════════════════════════════════════════════════
# FTLE field
# ═════════════════════════════════════════════════════════════════════════════

def compute_ftle(omega_lm, T, ngrid=48, dt=None, deformation_radius=None,
                 direction=1):
    """
    Finite-time Lyapunov exponent field of a vorticity snapshot.

    Seeds an ngrid×ngrid uniform (θ,φ) grid (poles excluded), advects for signed
    time `direction·|T|`, forms the metric flow-map gradient F by central
    differences, and returns σ_FTLE = (1/|T|) ln σ_max(F).

    direction : +1 forward FTLE (repelling LCS), −1 backward FTLE (attracting LCS).

    Returns
    -------
    theta_c, phi_c : (ngrid,) seed-grid colatitude/longitude nodes (rad)
    ftle : (ngrid, ngrid) FTLE field (interior points; edges set to NaN)
    """
    T = direction * abs(T)
    theta_c = np.linspace(_POLE_EPS + 0.05, np.pi - _POLE_EPS - 0.05, ngrid)
    phi_c = np.linspace(0.0, 2.0 * np.pi, ngrid, endpoint=False)
    TH0, PH0 = np.meshgrid(theta_c, phi_c, indexing='ij')

    rhs = _velocity_interpolators(omega_lm, deformation_radius)
    TH, PH = advect(omega_lm, TH0.ravel(), PH0.ravel(), T, dt=dt,
                    rhs=rhs, unwrap_phi=True)
    TH = TH.reshape(ngrid, ngrid)
    PH = PH.reshape(ngrid, ngrid)

    ftle = np.full((ngrid, ngrid), np.nan)
    for i in range(1, ngrid - 1):
        dth0 = theta_c[i + 1] - theta_c[i - 1]
        s0 = np.sin(theta_c[i])
        for j in range(1, ngrid - 1):
            dph0 = phi_c[j + 1] - phi_c[j - 1]
            sT = np.sin(TH[i, j])
            dTh_dth = (TH[i + 1, j] - TH[i - 1, j]) / dth0
            dTh_dph = (TH[i, j + 1] - TH[i, j - 1]) / dph0
            dPh_dth = (PH[i + 1, j] - PH[i - 1, j]) / dth0
            dPh_dph = (PH[i, j + 1] - PH[i, j - 1]) / dph0
            F = np.array([[dTh_dth,       dTh_dph / s0],
                          [sT * dPh_dth,  sT * dPh_dph / s0]])
            smax = np.linalg.svd(F, compute_uv=False)[0]
            ftle[i, j] = np.log(max(smax, 1e-300)) / abs(T)
    return theta_c, phi_c, ftle


# ═════════════════════════════════════════════════════════════════════════════
# Test velocity fields
# ═════════════════════════════════════════════════════════════════════════════

def _solid_body(lmax=32, c=1.0):
    """Pure (l=1,m=0) vorticity → solid-body rotation about the polar axis."""
    omega = np.zeros((2, lmax + 1, lmax + 1))
    omega[0, 1, 0] = c
    return omega


def _zonal_jet(lmax=32, modes=((5, 1.0), (6, 1.0)), amp=1.0):
    """
    A steady zonal (m=0) flow built from a few Legendre modes → latitudinal jets.
    Mixing an odd and an even degree breaks the equatorial symmetry so the shear
    has a UNIQUE global maximum (no mirror-image ambiguity in the ridge test).
    """
    omega = np.zeros((2, lmax + 1, lmax + 1))
    for ell, w in modes:
        omega[0, ell, 0] = amp * w
    return omega


def _shear_proxy(omega_lm, T, theta_query):
    """
    Analytic FTLE proxy for a STEADY ZONAL flow, at colatitudes `theta_query`.

    For u_φ=u_φ(θ), θ_T=θ₀ and Φ=φ₀+Ω(θ)T with Ω=u_φ/sinθ, the metric flow-map
    gradient is F=[[1,0],[a,1]] with a(θ)=sinθ·Ω'(θ)·T, whose largest singular
    value gives σ_FTLE(θ)=(1/T)ln σ_max — the value the numerical FTLE must match.
    """
    thg, phg, ut, up = velocity_grids(omega_lm)
    uphi_zon = up.mean(axis=1)                         # u_φ(θ), zonal (m=0 only)
    interior = (thg > 0.15) & (thg < np.pi - 0.15)     # drop poles (sinθ→0)
    thi = thg[interior]
    Omega = uphi_zon[interior] / np.sin(thi)
    dOmega = np.gradient(Omega, thi)
    a = np.sin(thi) * dOmega * T
    sigma_max = np.sqrt((2 + a**2 + np.abs(a) * np.sqrt(a**2 + 4)) / 2.0)
    ftle = np.log(sigma_max) / abs(T)
    return np.interp(theta_query, thi, ftle)


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def verify():
    print("=" * 74)
    print("Lagrangian particle tracking & FTLE — verification")
    print("=" * 74)
    ok = True

    # ── 1. Solid-body rotation: latitude preserved, tracers return after 1 period
    omega = _solid_body(lmax=32, c=1.0)
    rhs = _velocity_interpolators(omega)
    # rotation rate Ω_rot = u_φ/sinθ, uniform in latitude; measure it at the equator
    ut_eq, dphi_eq = rhs(np.array([np.pi / 2]), np.array([0.0]))
    Omega_rot = dphi_eq[0]
    period = 2.0 * np.pi / abs(Omega_rot)
    # seed a ring of tracers away from the poles
    th0 = np.deg2rad(np.array([40., 60., 90., 120., 140.]))
    ph0 = np.deg2rad(np.array([0., 72., 144., 216., 288.]))
    # solid body => u_θ = 0 everywhere: check latitude drift over a quarter period
    thq, phq = advect(omega, th0, ph0, 0.25 * period, dt=period / 2000.0, rhs=rhs)
    lat_drift = np.max(np.abs(thq - th0))
    # full period: tracers should return to their start
    thf, phf = advect(omega, th0, ph0, period, dt=period / 2000.0, rhs=rhs)
    dth = np.abs(thf - th0)
    dph = np.abs((phf - ph0 + np.pi) % (2 * np.pi) - np.pi)
    ret_err = np.max(np.hypot(dth, np.sin(th0) * dph))
    print(f"\n1. Solid-body rotation (Ω_rot={Omega_rot:.4f}, period={period:.3f}):")
    print(f"   latitude drift over ¼ period : {lat_drift:.2e} rad (expect ≈0, u_θ=0)")
    print(f"   return error after 1 period  : {ret_err:.2e} rad")
    sb_ok = lat_drift < 1e-3 and ret_err < 5e-3
    print(f"   → {'✓ tracers return to start, latitude preserved' if sb_ok else '⚠ drift too large'}")
    ok &= sb_ok

    # ── 2. FTLE of a zonal shear flow: ridges at maximum-shear latitudes ─────────
    omega = _zonal_jet(lmax=32, modes=((5, 1.0), (6, 1.0)), amp=3.0)
    T = 2.0
    theta_c, phi_c, ftle = compute_ftle(omega, T, ngrid=60, dt=T / 400.0)
    ftle_lat = np.full(theta_c.size, np.nan)          # zonal-mean FTLE(θ)
    ftle_lat[1:-1] = np.nanmean(ftle[1:-1], axis=1)   # interior rows only (edges all-NaN)
    valid = np.isfinite(ftle_lat)
    tc = theta_c[valid]
    fl = ftle_lat[valid]
    # analytic FTLE proxy σ=(1/T)ln σ_max(F), F=[[1,0],[sinθ·Ω'(θ)·T,1]]
    proxy = _shear_proxy(omega, T, tc)
    corr = np.corrcoef(fl, proxy)[0, 1]               # profile match across ALL latitudes
    lat_ftle_peak = tc[np.argmax(fl)]
    lat_shear_peak = tc[np.argmax(proxy)]
    zonal_spread = (np.nanstd(ftle[1:-1], axis=1)
                    / (np.nanmean(np.abs(ftle[1:-1]), axis=1) + 1e-12))
    print(f"\n2. Zonal shear flow FTLE (T={T}):")
    print(f"   FTLE ridge latitude θ  : {np.rad2deg(lat_ftle_peak):6.1f}°")
    print(f"   max-shear latitude θ   : {np.rad2deg(lat_shear_peak):6.1f}°")
    print(f"   FTLE(θ) vs analytic shear proxy correlation : {corr:.3f}")
    print(f"   FTLE is zonal (mean per-θ longitudinal spread): "
          f"{np.nanmean(zonal_spread):.2e}")
    ridge_ok = (corr > 0.9) and abs(lat_ftle_peak - lat_shear_peak) < np.deg2rad(8.0)
    print(f"   → {'✓ FTLE ridge sits at the maximum-shear latitude' if ridge_ok else '⚠ ridge misplaced'}")
    ok &= ridge_ok

    # ── 3. Direction independence for a (time-reversible) steady flow ────────────
    _, _, ftle_fwd = compute_ftle(omega, T, ngrid=60, dt=T / 400.0, direction=+1)
    _, _, ftle_bwd = compute_ftle(omega, T, ngrid=60, dt=T / 400.0, direction=-1)
    m = np.isfinite(ftle_fwd) & np.isfinite(ftle_bwd)
    rel = np.abs(ftle_fwd[m] - ftle_bwd[m]) / (np.abs(ftle_fwd[m]) + 1e-9)
    med_rel = np.median(rel)
    print(f"\n3. Direction independence (forward vs backward FTLE magnitude):")
    print(f"   median |σ₊−σ₋|/|σ₊| over the field : {med_rel:.2e}")
    dir_ok = med_rel < 0.05
    print(f"   → {'✓ forward and backward FTLE agree in magnitude' if dir_ok else '⚠ mismatch'}")
    ok &= dir_ok

    print("\n" + ("✓ ALL CHECKS PASSED — advection reversible & FTLE ridges at "
                  "transport barriers" if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
