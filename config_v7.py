# ─────────────────────────────────────────────────────────────────────────────
# config_v7.py — parameters for the v7 barotropic-vorticity model
#
# v7 implements scientific improvement #1 (scientific_improvements.md §1):
# "Widen the forcing–Rhines scale separation so jets actually form."
#
# Same equation and numerics as v6 (simulate_v7 imports the v6 solver verbatim,
# only swapping this config).  It is still 2-D forced–dissipative barotropic
# turbulence on a rotating sphere — NOT convection.  What changes are the
# PARAMETERS, retuned so the inverse energy cascade has room to run from the
# forcing scale up to the Rhines scale and be arrested there by β into a handful
# of zonal jets.
#
# ── The defect v7 fixes ──────────────────────────────────────────────────────
# v6 forced at l_f≈60–80 with a Rhines degree l_R≈23 — a separation ratio of
# only l_f/l_R ≈ 2.6, far too small.  The inverse cascade had barely a factor of
# ~3 in scale to organise before hitting the planetary-vorticity gradient, so
# zonal energy stayed ~3.7% and no jets formed (v6_critical_audit §4.2).  Jet
# formation needs l_f/l_R ≳ 5–10 (Vallis & Maltrud 1993; Galperin & Read 2019).
#
# ── Strategy (the three levers of improvement #1) ────────────────────────────
#   (i)   force at higher l  (FORCE 100–120, needs T170 — improvement #6);
#   (ii)  cut friction further (μ: 0.03 → 0.01) so the cascade is not arrested
#         prematurely and runs the full inverse range;
#   (iii) tune Ω so l_R lands at ~10 (a handful of jets), giving l_f/l_R ≈ 11.
# Target the zonostrophy index R_β = k_β/k_R ≳ 2 (Galperin et al. 2006): v7's
# design estimate is R_β ≈ 2.6 — the zonostrophic (jet-bearing) regime.
#
# ── HONESTY on the numbers below ─────────────────────────────────────────────
# The scale estimates (l_R, ε, U, k_β, R_β) are ORDER-OF-MAGNITUDE predictions
# from an energy-balance argument calibrated against the v6 run (see the derivation
# in each block).  The velocity U, injection rate ε and hence l_R and R_β must be
# MEASURED from the saturated v7 flow before any jet claim is made — that is
# exactly improvement #2 (run to steady state, then time-average and diagnose).
# These parameters aim the run at the zonostrophic regime; they do not prove it.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Geometry ────────────────────────────────────────────────────────────────
# Solar convection zone: 0.71 R_☉ (base, above the radiative interior) → surface.
# (Geometry unchanged from v6 — it only affects the cosmetic cutaway render.)
R_OUTER = 1.0         # outer (surface) radius   — non-dimensional
R_INNER = 0.71        # base of the convection zone (r_inner / r_outer)
R_MID   = 0.85        # intermediate boundary, roughly mid-shell (thin line)

# ── Rotation ────────────────────────────────────────────────────────────────
# Planetary vorticity f = 2Ω sinφ; the β-effect that arrests the inverse cascade
# is β ≡ df/dy|_eq = 2Ω on the unit sphere (R=1).  Ω is chosen to place the
# Rhines degree at l_R ≈ 10 — i.e. a handful of jets.
#
#   MATH — Rhines scale.  The Rhines wavenumber is where the eddy turnover rate
#   matches the Rossby-wave frequency; in the convention of improvement #1,
#           l_R = sqrt( 2Ω / U )                              (β = 2Ω, R = 1)
#   with U the rms velocity of the energy-containing (large-scale) flow.  U is
#   fixed by the energy balance (see LINEAR_DRAG / FORCE_AMP blocks): the v7
#   estimate is U ≈ 0.13.  Solving for Ω at the target l_R = 10:
#           Ω = l_R² · U / 2 = 100 · 0.13 / 2 ≈ 6.5.
#   Check:  l_R = sqrt(2·6.5 / 0.132) = 9.9  →  l_f/l_R = 110/9.9 ≈ 11  ✓ (≳5–10)
#   (v6 used Ω=40, which — with its lower-l forcing and larger U — gave l_R≈23.)
OMEGA = 6.5

# ── Spectral resolution ─────────────────────────────────────────────────────
# T170 (improvement #6): a wider inertial range than v6's T127.  Two ranges must
# fit below the ∇⁸ cutoff: the INVERSE range l_R…l_f (≈10…110) and a FORWARD
# enstrophy range l_f…LMAX (≈120…170).
#
#   HONESTY — the forward range is thin.  LMAX/FORCE_LMAX = 170/120 ≈ 1.42, so
#   only ~0.5 of an octave of forward enstrophy cascade sits above the forcing
#   before hyperviscosity bites (τ_dissip(l=120) ≈ 4 tu; see NU_HYPER).  The
#   headline gain is the WIDE INVERSE range (l_f/l_R ≈ 11); the forward range is
#   sacrificed to keep the forcing high-l at affordable cost.  T255 (the upper
#   end of improvement #6) would open the forward range but costs ∝ L³ ≈ 3.4×
#   more per step again.  Orszag 2/3 dealiasing: pyshtools' DH2 grid oversamples
#   (2(L+1) lat × 2(L+1) lon vs the L needed), so quadratic products are
#   dealiased at this truncation (improvement #6's verification note).
LMAX = 170

# ── Dissipation ─────────────────────────────────────────────────────────────
# ∇⁸ hyperviscosity ν (multiplies λ⁴, λ = l(l+1)).  Exact integrating factor in
# the solver: the linear decay per step is exp(−ν λ⁴ dt).
#
#   MATH — set the small-scale cutoff time.  Dissipation time τ = 1/(ν λ⁴).
#   Require τ(l=LMAX) ≈ 0.25 tu (same clean cutoff as v6):
#           λ_max = 170·171 = 29070,   λ_max⁴ = 7.14e17
#           ν = 1/(0.25 · λ_max⁴) = 5.6e-18.
#   Scale-selectivity check (τ in time units):
#           τ(l=170) = 0.25   (cutoff — filaments die in a quarter time unit)
#           τ(l=120) = 4.0    (top of forcing band — mildly damped, see LMAX note)
#           τ(l=100) = 17     (bottom of forcing band — essentially untouched)
#           τ(l=10)  = 8.8e6  (Rhines/jet scale — utterly untouched, as required)
#   So ν drains enstrophy at the grid scale while leaving the inverse-cascade
#   range (l ≲ 100) inviscid, as a hyperdiffusion should.
NU_HYPER = 5.6e-18

# Uniform linear (Rayleigh) drag μ.  This is the PRIMARY large-scale energy sink
# and it sets both the saturated energy and the arrest of the inverse cascade.
#
#   WHY WEAKER THAN v6.  v6 used μ=0.03 (τ_drag = 1/μ = 33 tu).  Cutting it to
#   0.01 (τ_drag = 100 tu) lets the inverse cascade run ~3× longer before drag
#   halts it, pushing the frictional arrest scale to larger scale (smaller l) so
#   the cascade reaches the Rhines scale instead of being killed first.
#           Frictional arrest wavenumber  k_fr ~ (β³/ε)^{1/5} scaling aside,
#           the direct effect is energy level:  E_sat = ε/(2μ) doubles+ vs v6.
#   HONESTY: weaker drag ⇒ longer spin-up (see N_SPINUP) and a genuine risk the
#   flow is still transient unless run the full 5 τ_drag.
LINEAR_DRAG = 0.01

# ── Stochastic forcing ──────────────────────────────────────────────────────
# White-in-time, narrow band in l.  Injected as ω_lm += f_lm each step with
# per-coefficient amplitude  amp = FORCE_AMP/√(l(l+1)) · √dt  (solver unchanged).
#
#   WHERE — well above the Rhines scale.  FORCE 100–120 sits a factor ~11 in
#   scale above l_R≈10, giving the wide inverse range jets need.  It also sits
#   well above the transitional (zonostrophic) wavenumber k_β≈26 (below), so the
#   ordering is  l_R (≈10)  <  k_β (≈26)  <  l_f (≈110)  — the textbook
#   zonostrophic arrangement (Galperin et al. 2006; Zonal Jets 2019, ch. 2).
#
#   MATH — energy-injection rate ε and the resulting U.  Energy per mode is
#   E_lm = c_lm²/[l(l+1)].  Per step the forcing adds ⟨f_lm²⟩ = amp² =
#   FORCE_AMP²/[l(l+1)] · dt, so the energy injected per unit time is
#           ε = FORCE_AMP² · Σ_{l=100}^{120} (2l+1)/[l(l+1)]²   (2l+1 coeffs/deg)
#             ≈ FORCE_AMP² · 3.17e-5.
#   Calibrating the O(1) Parseval/normalisation constant against v6 (which gives
#   A≈1.13, i.e. this simple sum reproduces v6's l_R≈23 to ~10%):
#           ε ≈ 1.7e-4    (v7),    vs   ε ≈ 6.9e-4 (v6).
#   Energy balance with drag dominant, E_sat = ε/(2μ), U = √(2E_sat) = √(ε/μ):
#           U ≈ √(1.7e-4 / 0.01) ≈ 0.13.
#   FORCE_AMP kept at 2.2 for continuity with v6; ε is lower than v6 only because
#   the same amplitude at higher l injects less ENERGY (∝ 1/[l(l+1)]²).
FORCE_LMIN = 100      # inject energy at l >= FORCE_LMIN
FORCE_LMAX = 120      # inject energy at l <= FORCE_LMAX
FORCE_AMP  = 2.2      # forcing amplitude (unchanged from v6)

# ── Zonostrophy target (documentation) ───────────────────────────────────────
# Zonostrophy index (Galperin, Sukoriansky et al. 2006, Nonlin. Proc. Geophys.):
#           R_β = k_β / k_R,   k_β = (β³/ε)^{1/5},   k_R = l_R = √(2Ω/U).
# Regimes:  R_β ≳ 2.5 zonostrophic (sharp jets) · 1.5–2.5 transitional ·
#           <1.5 friction-dominated (no jets).  v7 DESIGN estimate:
#           β = 2Ω = 13,  ε ≈ 1.7e-4  →  k_β = (13³/1.7e-4)^{1/5} ≈ 26,
#           k_R ≈ 10  →  R_β ≈ 2.6.   TARGET: R_β ≳ 2  ✓ (borderline zonostrophic).
# This is the quantitative goal of improvement #1; it must be re-measured from
# the saturated run (improvement #2) — the estimate assumes drag-dominated
# balance and a calibrated normalisation constant, both good only to ~factor 2.
ZONOSTROPHY_TARGET = 2.0   # want measured R_β ≳ this

# ── Time stepping ───────────────────────────────────────────────────────────
# The linear (dissipation+drag) part is integrated EXACTLY by the integrating
# factor, so dt is limited only by the advective CFL of the Heun (RK2) nonlinear
# substep, not by the stiff hyperviscosity.
#
#   MATH — CFL.  Hold the same effective Courant number as the stable v6 run:
#           C = dt · U · L_max.
#   v6:  C = 1.5e-3 · 0.151 · 127 = 0.029  (well below the RK2 limit; the safety
#   margin covers the max — not rms — straining velocity).  For v7 at U≈0.13,
#   L=170:  dt = 0.029/(0.13·170) = 1.3e-3.  Round DOWN to 1.2e-3 (C = 0.027).
#   Rossby waves do NOT limit dt: fastest |ω_R| = 2Ω·m/[l(l+1)] ≤ Ω = 6.5, period
#   1/Ω ≈ 0.15 tu ≫ dt.
DT         = 1.2e-3   # timestep (smaller than v6's 1.5e-3 for T170 stability)

#   N_SPINUP — reach a genuine statistically steady state (improvement #2).
#   Weaker drag ⇒ longer memory: τ_drag = 1/μ = 100 tu.  Require ≥5 drag times:
#           N_SPINUP = 5·τ_drag/dt = 5·100/1.2e-3 = 4.17e5  →  420000 steps
#                    = 504 time units.
#   HONESTY — cost.  This is ~3.8× more steps than v6 and each T170 step is
#   ~(170/127)³ ≈ 2.4× costlier, so v7 spin-up is ~9× v6's wall-clock.  That is
#   the price of a wide inverse range at low friction; there is no cheaper route
#   to a saturated jet state.
N_SPINUP   = 420000   # steps before recording (= 504 tu ≈ 5 drag times)
N_FRAMES   = 200      # frames to record
FRAME_SKIP = 10       # simulation steps between frames

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_GIF = "output_v7.gif"
OUTPUT_MP4 = "output_v7.mp4"
FPS        = 20       # animation frame rate

# Intermediate: saturated spectral coefficients for every recorded frame
FRAMES_NPZ = "frames_v7.npz"

# Copy of the MP4 pushed to the iCloud "Claude" drop zone
ICLOUD_MP4 = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Claude/"
    "2026-07-08_spherical-convection_v7.mp4"
)

# ── Rendering ───────────────────────────────────────────────────────────────
# (Rendering parameters carried over from v6 unchanged — they concern only the
# cosmetic cutaway, which later improvements #9/#20 address.  See config_v6.py
# for the full honesty caveats on L_REF and SHEAR_DEG; they are reproduced here
# verbatim so simulate/render_v7 are self-contained.)
IMG_SIZE   = 720      # pixels (square; even → yuv420p-friendly)

# Octant cutaway: remove the wedge  0° ≤ lon < CUTAWAY_LON_END  AND  lat > 0.
CUTAWAY_LON_START = 0.0
CUTAWAY_LON_END   = 90.0

# Cross-section sampling resolution
N_RADIAL   = 60       # radial cells across the shell (R_INNER … R_OUTER)
N_ANG      = 160      # angular cells along each cross-section face
SURFACE_DS = 1        # downsample stride for the coloured outer surface

# Interior reconstruction: mixing-length-scaled radial continuation (r/R)^(l/L_REF).
# HONESTY: this OVERSTATES radial penetration ~L_REF× vs the true (r/R)^l
# eigenfunction; a 2-D barotropic model carries NO radial structure.  Purely
# illustrative (v6_critical_audit §4.4; improvement #20).
L_REF      = 10       # mixing-length reference degree for radial penetration
CORE_LMAX  = 12       # radiative interior painted from only the largest scales

# Longitude twist for the cutaway faces (VISUALIZATION only, not dynamics —
# improvement #9 would replace it with a real differential-rotation term).
SHEAR_DEG  = 25.0     # peak longitude twist (deg) across the shell — a viz choice

# Camera
VIEW_ELEV = 24.0
VIEW_AZIM = 40.0
