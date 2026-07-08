# Physical and numerical parameters for rotating spherical shell convection

# ── Geometry ────────────────────────────────────────────────────────────────
R_OUTER = 1.0         # outer (surface) radius   — non-dimensional
R_INNER = 0.35        # inner core radius (r_inner / r_outer)
R_MID   = 0.66        # one intermediate layer boundary (drawn as a thin line)

# ── Rotation ────────────────────────────────────────────────────────────────
OMEGA = 15.0          # rotation rate (non-dimensional); high = elongated banana cells

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
# Force a fairly high band: in this model the self-generated forward cascade is
# weak, so direct injection at l = 18–45 is what fills the fine-filament range.
FORCE_LMIN = 18       # inject energy at l >= FORCE_LMIN  (higher l → finer scales)
FORCE_LMAX = 45       # inject energy at l <= FORCE_LMAX
FORCE_AMP  = 1.3      # forcing amplitude per timestep

# ── Time stepping ───────────────────────────────────────────────────────────
DT         = 2e-3     # timestep (non-dimensional)
N_SPINUP   = 5000     # steps before recording (≈ 10 non-dim time units)
N_FRAMES   = 200      # frames to record
FRAME_SKIP = 8        # simulation steps between frames

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_GIF = "output.gif"
OUTPUT_MP4 = "output.mp4"
FPS        = 20       # animation frame rate

# Copy of the MP4 pushed to the iCloud "Claude" drop zone
import os
ICLOUD_MP4 = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
    "2026-07-08_spherical-convection_v4.mp4"
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

# Camera
VIEW_ELEV = 24.0
VIEW_AZIM = 40.0
