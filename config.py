# Physical and numerical parameters for rotating spherical shell convection

# ── Geometry ────────────────────────────────────────────────────────────────
# Solar convection zone: extends from 0.71 R_☉ (base, above the radiative
# interior) out to the surface R_☉.  The shell is relatively thin (depth ≈ 0.29).
R_OUTER = 1.0         # outer (surface) radius   — non-dimensional
R_INNER = 0.71        # base of the convection zone (r_inner / r_outer)
R_MID   = 0.85        # intermediate boundary, roughly mid-shell (thin line)

# ── Rotation ────────────────────────────────────────────────────────────────
# Increased for the thinner shell: a thin shell needs a faster rotation rate to
# keep the flow rotationally constrained into elongated banana cells.
OMEGA = 40.0          # rotation rate (non-dimensional); high = elongated banana cells

# ── Spectral resolution ─────────────────────────────────────────────────────
LMAX = 85             # maximum spherical harmonic degree (T85 → fine filaments)

# ── Dissipation ─────────────────────────────────────────────────────────────
# ∇^8 hyperviscosity (coefficient on λ^4, λ = l(l+1)).  Weak — it bites only
# near the truncation so the enstrophy cascade fills l ≈ 45–80 with filaments.
NU_HYPER = 3e-15
# Uniform linear (Rayleigh) drag — removes large-scale energy so the flow
# reaches a statistically steady state instead of a growing condensate.
LINEAR_DRAG = 0.25

# ── Stochastic forcing ──────────────────────────────────────────────────────
# Force a fairly high band.  For the thinner convection zone the characteristic
# convective cell scales with the (smaller) shell depth, so inject at higher l
# than the thick-shell case — smaller cells, appropriate to a thin shell.
FORCE_LMIN = 24       # inject energy at l >= FORCE_LMIN  (higher l → finer scales)
FORCE_LMAX = 52       # inject energy at l <= FORCE_LMAX
FORCE_AMP  = 1.3      # forcing amplitude per timestep

# ── Time stepping ───────────────────────────────────────────────────────────
DT         = 2e-3     # timestep (non-dimensional)
N_SPINUP   = 8000     # steps before recording (≈ 102 rotation periods elapsed)
N_FRAMES   = 200      # frames to record
FRAME_SKIP = 8        # simulation steps between frames

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_GIF = "output.gif"
OUTPUT_MP4 = "output_v5.mp4"
FPS        = 20       # animation frame rate

# Copy of the MP4 pushed to the iCloud "Claude" drop zone
import os
ICLOUD_MP4 = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
    "2026-07-08_spherical-convection_v5.mp4"
)

# ── Rendering ───────────────────────────────────────────────────────────────
IMG_SIZE   = 720      # pixels (square; even → yuv420p-friendly)

# Octant cutaway: remove the wedge  0° ≤ lon < CUTAWAY_LON_END  AND  lat > 0.
# This exposes an equatorial quarter-annulus (z = 0) plus two meridional faces.
CUTAWAY_LON_START = 0.0
CUTAWAY_LON_END   = 90.0

# Cross-section sampling resolution (the interior faces are the main visual)
N_RADIAL   = 60       # radial cells across the shell (R_INNER … R_OUTER)
N_ANG      = 160      # angular cells along each cross-section face
SURFACE_DS = 1        # downsample stride for the coloured outer surface

# Interior reconstruction: deeper shells are rotated in longitude (a
# differential-rotation shear) so the interior field follows curved arcs that
# run TANGENTIAL to the shell (concentric with the surface) instead of radial
# spokes.  The shear vanishes at the outer rim so the faces still join the
# surface colours seamlessly.
SHEAR_DEG  = -55.0    # total longitude shear across the shell (deg)

# Camera
VIEW_ELEV = 24.0
VIEW_AZIM = 40.0
