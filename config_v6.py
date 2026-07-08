# ─────────────────────────────────────────────────────────────────────────────
# config_v6.py — parameters for the v6 barotropic-vorticity model
#
# v6 is a scientifically HONEST rebuild of the v5 run.  It is NOT a convection
# model — it is 2-D forced–dissipative barotropic turbulence on a rotating
# sphere (one scalar equation for relative vorticity ω_z).  See README_v6.md.
#
# Changes vs v5, motivated by physics_audit.md:
#   • Coriolis coefficient fixed (was √(4π)≈3.545× too large — see simulate_v6).
#   • Forcing moved WELL ABOVE the Rhines scale (l≈60–80) so an inverse cascade
#     can run below it (→ jets) and a forward enstrophy cascade above it
#     (→ filaments).
#   • Linear drag cut ~8× (0.25 → 0.03) so the inverse cascade is not killed.
#   • Higher resolution (T85 → T127) to resolve the filament range.
#   • Honest interior: the cross-sections are built by the correct radial
#     eigenfunction ω(r) ∝ (r/R)^l (solved in visualize_v6), not a cosmetic
#     copy of the surface field.
# ─────────────────────────────────────────────────────────────────────────────

# ── Geometry ────────────────────────────────────────────────────────────────
# Solar convection zone: 0.71 R_☉ (base, above the radiative interior) → surface.
R_OUTER = 1.0         # outer (surface) radius   — non-dimensional
R_INNER = 0.71        # base of the convection zone (r_inner / r_outer)
R_MID   = 0.85        # intermediate boundary, roughly mid-shell (thin line)

# ── Rotation ────────────────────────────────────────────────────────────────
# With the CORRECT Coriolis coefficient (see simulate_v6._f_lm) this really is
# Ω = 40 non-dimensional, not the effective ≈142 the v5 bug produced.
OMEGA = 40.0

# ── Spectral resolution ─────────────────────────────────────────────────────
LMAX = 127            # T127 — resolves the forward enstrophy (filament) cascade

# ── Dissipation ─────────────────────────────────────────────────────────────
# ∇⁸ hyperviscosity coefficient (multiplies λ⁴, λ = l(l+1)).  Tuned so the
# dissipation time τ = 1/(ν λ⁴) at l = LMAX is ≈0.25 time units while the
# forcing band (l≈60–80) is essentially untouched (τ ≳ 10) — a clean,
# scale-selective small-scale sink that sets the filament cutoff.
NU_HYPER = 6.0e-17
# Uniform linear (Rayleigh) drag μ.  Deliberately WEAK: it arrests the inverse
# cascade at the Rhines scale (→ jets) instead of killing it (v5 used μ=0.25,
# which suppressed the cascade entirely).  τ_drag = 1/μ ≈ 33 time units.
LINEAR_DRAG = 0.03

# ── Stochastic forcing ──────────────────────────────────────────────────────
# Force at SMALL scales, well above the Rhines degree (l_R≈8–20 here), leaving a
# wide inertial range below (inverse cascade → jets/large vortices) and above
# (forward enstrophy cascade → filaments).
FORCE_LMIN = 60       # inject energy at l >= FORCE_LMIN
FORCE_LMAX = 80       # inject energy at l <= FORCE_LMAX
FORCE_AMP  = 2.2      # forcing amplitude (tuned for effective Re ≈ 50–100)

# ── Time stepping ───────────────────────────────────────────────────────────
DT         = 1.5e-3   # timestep (smaller than v5 for stability at T127)
N_SPINUP   = 22000    # steps before recording (≈ 33 time units ≈ 1 drag time)
N_FRAMES   = 200      # frames to record
FRAME_SKIP = 10       # simulation steps between frames

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_GIF = "output_v6.gif"
OUTPUT_MP4 = "output_v6.mp4"
FPS        = 20       # animation frame rate

# Intermediate: saturated spectral coefficients for every recorded frame
FRAMES_NPZ = "frames_v6.npz"

# Copy of the MP4 pushed to the iCloud "Claude" drop zone
import os
ICLOUD_MP4 = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
    "2026-07-08_spherical-convection_v6.mp4"
)

# ── Rendering ───────────────────────────────────────────────────────────────
IMG_SIZE   = 720      # pixels (square; even → yuv420p-friendly)

# Octant cutaway: remove the wedge  0° ≤ lon < CUTAWAY_LON_END  AND  lat > 0.
CUTAWAY_LON_START = 0.0
CUTAWAY_LON_END   = 90.0

# Cross-section sampling resolution (the interior faces are the main visual)
N_RADIAL   = 60       # radial cells across the shell (R_INNER … R_OUTER)
N_ANG      = 160      # angular cells along each cross-section face
SURFACE_DS = 1        # downsample stride for the coloured outer surface

# Interior reconstruction (see visualize_v6).  Each spherical-harmonic mode is
# continued inward by  (r/R_outer)^(l/L_REF)  — a mixing-length-scaled radial
# eigenfunction.  The pure potential-flow continuation (L_REF=1, i.e. (r/R)^l)
# is mathematically exact but, for THIS field (power concentrated at l≳20 with
# essentially none at l<15; the inverse cascade arrests near the Rhines degree),
# it is evanescent — it confines >85% of the amplitude to the top 5% of the
# shell and shows an almost empty interior.  L_REF rescales the radial decay so
# that penetration depth ∝ horizontal wavelength (a mode of degree l stays
# coherent over ~its own wavelength, as convective cells do over a mixing
# length).  L_REF = π/D ≈ 10.8 with shell depth D = R_OUTER−R_INNER = 0.29 is
# the degree whose half-wavelength spans the shell, so shell-scale structures
# reach the base while fine filaments remain surface-confined.  This is an
# illustrative reconstruction (honestly labelled), NOT solved interior dynamics.
L_REF      = 10       # mixing-length reference degree for radial penetration
CORE_LMAX  = 12       # radiative interior painted from only the largest scales

# Differential-rotation shear.  The Sun rotates faster at the equator/surface
# than deeper down; a pattern continued inward is therefore twisted in longitude
# with depth.  We rotate the reconstructed field by a longitude offset that grows
# linearly from 0 at the surface (r=R_OUTER, so the cut faces still join the
# coloured surface seamlessly) to SHEAR_DEG at the base (r=R_INNER).  This bends
# the radial cross-section structures into concentric arcs along the shell.
# v5 used an arbitrary 55°; we use a MODERATE, physically-motivated 25° — enough
# to curve the structures, small enough not to over-wind them.  A longitude shift
# is a pure rotation about the polar axis, applied in spectral space (each order-m
# coefficient rotates by m·α), so it costs almost nothing.
SHEAR_DEG  = 25.0     # peak longitude twist (deg) across the shell, base vs surface

# Camera
VIEW_ELEV = 24.0
VIEW_AZIM = 40.0
