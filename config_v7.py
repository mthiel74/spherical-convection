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
FORCE_AMP  = 2.2      # forcing amplitude (unchanged from v6; see FORCE_FROM_EPSILON)

# ── Physically-grounded forcing (improvement #5; scientific_improvements.md §5) ─
#
# ── 5a.  Controlled energy-injection rate ε ──────────────────────────────────
# The white-in-time forcing adds an independent Gaussian increment δω_lm to every
# coefficient in the band each step, with per-coefficient standard deviation
#           std(δω_lm) = amp_l·√dt,   amp_l = FORCE_AMP / √[l(l+1)].
# Because the increment is independent of the current field, it adds ENSTROPHY at
# rate ⟨δω_lm²⟩/dt = amp_l² per coefficient and ENERGY at rate amp_l²/[l(l+1)]
# (energy per mode is E_lm = ω_lm²/[l(l+1)]).  Summing over the (2l+1) real
# coefficients per degree l in the band gives the total energy-injection rate
#
#           ε = Σ_{l=LMIN}^{LMAX} (2l+1) · amp_l² / [l(l+1)]
#             = FORCE_AMP² · Σ_{l=LMIN}^{LMAX} (2l+1) / [l(l+1)]²
#             ≡ FORCE_AMP² · S_band .                                    (5a)
#
# So ε is fixed by FORCE_AMP and the band alone.  S_band is a pure geometric
# constant of the forcing band (computed below); inverting (5a) lets us set the
# amplitude FROM a target injection rate rather than choosing it arbitrarily:
#           FORCE_AMP = √( EPSILON_TARGET / S_band ).
# ε matters physically because it sets the frictional-arrest / jet-spacing scale
# k_fr ~ (β³/ε)^{1/5} (Maltrud & Vallis 1991; improvement #1's zonostrophy k_β).
#
# S_band = Σ_band (2l+1)/[l(l+1)]²  — the per-amplitude² injection efficiency.
FORCE_BAND_SUM = sum((2 * l + 1) / (l * (l + 1)) ** 2
                     for l in range(FORCE_LMIN, FORCE_LMAX + 1))   # ≈ 3.17e-5

# Target energy-injection rate ε (per unit time).  The default equals the ε the
# legacy FORCE_AMP=2.2 injects (≈ 1.53e-4), so the default v7 run is unchanged.
# Set FORCE_FROM_EPSILON=True to make ε the control knob and DERIVE FORCE_AMP.
EPSILON_TARGET     = FORCE_AMP ** 2 * FORCE_BAND_SUM   # ≈ 1.53e-4
FORCE_FROM_EPSILON = False   # True ⇒ FORCE_AMP := √(EPSILON_TARGET / S_band)

# ── 5b.  Forcing temporal correlation: white vs Ornstein–Uhlenbeck ───────────
# FORCE_TYPE selects the temporal statistics of the forcing:
#   'white' — delta-correlated in time (the default; exact v6 behaviour).  Each
#             step draws a fresh independent increment δω_lm = amp_l·√dt·ξ.
#   'ou'    — a persistent forcing field f_lm with a FINITE correlation time τ_c,
#             evolving as the Ornstein–Uhlenbeck (OU) stochastic process
#                     df_lm = −(f_lm/τ_c) dt + σ_l dW ,
#             and added to the vorticity as a smooth tendency  δω_lm = f_lm·dt.
#
#   MATH — the OU process.  The OU SDE above is the UNIQUE stationary Gaussian
#   Markov process; its stationary autocovariance is the exponential
#           C(s) = ⟨f(t) f(t+s)⟩ = (σ_l² τ_c / 2) · exp(−|s| / τ_c),
#   with stationary variance C(0) = σ_l² τ_c / 2 and integral time τ_c.  We fix
#   σ_l by demanding that, as τ_c → 0, the OU tendency δω = f·dt converge to the
#   SAME white-noise forcing as the 'white' branch (so ε is unchanged and the two
#   branches are directly comparable).  A tendency driven by a stationary process
#   f with ∫₋∞^∞ C(s) ds = amp_l² is white-noise-equivalent (spectral density
#   amp_l² at zero frequency); for OU ∫ C ds = σ_l² τ_c², hence
#           σ_l² = amp_l² / τ_c²      ⇒   Var(f_lm) = C(0) = amp_l² / (2 τ_c).
#   The field is initialised at this stationary variance (no forcing transient).
#   Coloured forcing removes the unphysical white-noise artefact and changes the
#   injection statistics (Constantinou, Farrell & Ioannou 2014, JAS 71, 1818).
FORCE_TYPE      = 'white'   # 'white' (default, = v6) or 'ou'
FORCE_CORR_TIME = 1.0       # OU correlation time τ_c (time units); used iff 'ou'

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

# ── Spectral vanishing viscosity (improvement #7; scientific_improvements.md §7)
# SVV (Tadmor 1989, SIAM J. Numer. Anal. 26, 30) is a scale-selective sink that
# is IDENTICALLY ZERO over the resolved/inertial range and switches on smoothly
# only near the truncation.  Unlike a plain Laplacian it does not damp the large
# scales; unlike a sharp spectral cutoff it is smooth (no Gibbs); and being a
# genuine (modified) viscosity it provides the entropy dissipation that
# guarantees convergence to the correct weak solution — Tadmor's result.
#
#   THE SVV KERNEL.  SVV acts as a Laplacian viscosity −ε_SVV(l)·λ ω (λ=l(l+1))
#   with an l-dependent coefficient that ramps up above a cutoff degree l_cut:
#
#       ε_SVV(l) = ε₀ · exp[ −((L − l)/(l − l_cut))² ]   for l > l_cut,
#                = 0                                       for l ≤ l_cut,
#
#   with L = LMAX.  The kernel is 0 at l = l_cut, rises smoothly (it is ≈0 for a
#   band just above l_cut because (l−l_cut) is tiny there), and reaches ε₀ at the
#   truncation l = L.  The standard choice is l_cut ~ √L (Maday & Tadmor 1989):
#           l_cut = √170 ≈ 13.
#   SVV SUPPLEMENTS the ∇⁸ hyperviscosity here (both act only near the cutoff);
#   set NU_HYPER = 0 to use SVV as the sole small-scale sink (a pure replacement).
#
#   TUNING ε₀.  At the truncation the SVV decay rate is ε₀·λ_max = ε₀·L(L+1).
#   Matching the hyperviscous cutoff rate there (ν λ_max⁴ ≈ 4 ⇒ τ ≈ 0.25 tu):
#           ε₀ ≈ 4 / (170·171) ≈ 1.4e-4.
SVV_ENABLED = False      # supplement the ∇⁸ sink with SVV (default off = v6)
SVV_LCUT    = 13         # l_cut ≈ √LMAX — SVV is exactly 0 for l ≤ l_cut
SVV_EPS0    = 1.4e-4     # ε₀ — SVV strength at the truncation (τ≈0.25 tu at l=L)

# ═════════════════════════════════════════════════════════════════════════════
# Tier B — NEW PHYSICS (scientific_improvements.md §§9–11).  Each of the three
# terms below is an OPTIONAL extra term in the SAME barotropic solver, DISABLED
# by default so the base v7 run (§§1–8) is byte-for-byte unchanged.  Enable one
# (or several) by flipping its flag.  Improvements §§12–14 (two-layer QG, shallow
# water, MHD) are separate standalone modules, not flags here.
# ═════════════════════════════════════════════════════════════════════════════

# ── 9.  Differential-rotation mean flow (Newtonian relaxation) ────────────────
# scientific_improvements.md §9.  Relax the ZONAL-MEAN (m=0) vorticity toward a
# prescribed differential-rotation profile, à la the Held–Suarez (1994) core, so
# the mean shear is a genuine term in the evolution rather than a cosmetic twist.
# The relaxation seeds a barotropically-unstable mean flow that can shed eddies
# and organise jets self-consistently.
#
#   THE PROFILE.  A solar-like angular velocity, fastest at the equator:
#           Ω(φ) = Ω₀ − ΔΩ sin²φ ,     φ = LATITUDE  (φ=0 equator, ±90° poles).
#   (θ in the task = latitude here; written φ to avoid clashing with colatitude.
#   sin²φ = 0 at the equator ⇒ Ω = Ω₀ there — the fast-equator solar law.  Ω₀ is
#   already carried by the Coriolis term f = 2Ω₀ sinφ, so only the DIFFERENTIAL
#   part δΩ(φ) = Ω(φ) − Ω₀ = −ΔΩ sin²φ enters the relative flow.)
#
#   FROM Ω(φ) TO A TARGET VORTICITY.  The differential rotation is a zonal wind
#   in the co-rotating frame,  ū(φ) = δΩ(φ)·R cosφ  (R cosφ = distance from the
#   spin axis; R=1).  A zonal wind derives from a streamfunction by ū = −∂ψ̄/∂y =
#   −(1/R)∂ψ̄/∂φ, so
#           ∂ψ̄/∂φ = −R ū = R² ΔΩ sin²φ cosφ  ⇒  ψ̄_target(φ) = (R² ΔΩ/3) sin³φ,
#   and the target zonal-mean vorticity is ω̄_target = ∇²ψ̄_target, i.e. in
#   spectral space ω̄_target,l0 = −l(l+1)·ψ̄_target,l0 (a pure m=0 spectrum).  The
#   solver builds ψ̄_target(φ) on the grid, expands it, keeps m=0, and applies
#   ∇² spectrally — no hand-differentiation, so the construction is exact to
#   truncation.
#
#   THE TERM.  Added to the tendency of the m=0 modes ONLY (Held–Suarez form):
#           ∂ω̄/∂t |_relax = −(1/τ_relax)·(ω̄ − ω̄_target).
#   τ_relax is the relaxation time (short ⇒ stiff mean forcing; long ⇒ gentle).
DIFF_ROT_ENABLED     = False   # relax the m=0 mean flow toward the DR profile
DIFF_ROT_DELTA_OMEGA = 0.3     # ΔΩ — equator-to-pole angular-velocity contrast
DIFF_ROT_TAU         = 10.0    # τ_relax — Newtonian relaxation time (time units)

# ── 10.  Topographic β / bottom relief ───────────────────────────────────────
# scientific_improvements.md §10.  Add a FIXED topography h(θ,φ) to the potential
# vorticity, so the quantity ADVECTED by the flow becomes
#           q = ω + f + f₀ h / H         (h = relief, H = layer depth, f₀ = 2Ω₀),
# i.e. q = ω + f + η with the stationary topographic vorticity η(θ,φ) ≡ f₀ h/H.
# The tendency N = −J(ψ, q) then splits as
#           N = −J(ψ, ω+f) − J(ψ, η),
# so the ONLY change is the extra advection term −J(ψ, η) of a FIXED field η by
# the flow ψ = ∇⁻²ω.  η is not part of the streamfunction inversion (it is an
# external PV source, not flow vorticity), so ψ is unchanged.  Physically the
# stationary PV-gradient ∇η radiates topographic Rossby waves, exerts form drag,
# and can anchor standing eddies / lock jets, breaking the artificial zonal
# symmetry (Vallis & Maltrud 1993; Vallis 2017 §14).
#
#   REPRESENTATION.  η is a few low-order spherical harmonics (continental-scale
#   ridges).  TOPO_MODES maps (l,m) → amplitude of the REAL cosine coefficient
#   η_lm (4π-normalised c[0,l,m] slot).  f₀ h/H is folded into the amplitude, so
#   TOPO_MODES values are already in vorticity units.  Only m≤l, l≥1 are legal
#   (an l=0 topography is a constant PV offset with zero gradient — no dynamics).
TOPO_ENABLED = False
# Default relief: an l=2,m=0 axisymmetric belt + an l=3,m=2 zonally-varying ridge.
TOPO_MODES   = {(2, 0): 0.5, (3, 2): 0.3}

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

# ── Time-integration scheme (improvements #3 / #4) ───────────────────────────
# 'strang'  — Strang (2nd-order) operator split L(dt/2)·N(dt)·L(dt/2) (default,
#             improvement #3): cheap, 2nd order, exact linear part.
# 'etdrk4'  — ETDRK4 exponential time-differencing Runge–Kutta (improvement #4,
#             Cox & Matthews 2002; Kassam & Trefethen 2005): treats the diagonal
#             linear operator L = −(μ + ν λ⁴) EXACTLY (to machine precision) and
#             the nonlinear Jacobian at 4th order.  ~4 Jacobian evaluations/step
#             vs Strang's 2, so ~2× costlier per step but far more accurate on
#             the stiff linear term — worth it at higher resolution/longer dt.
# The white-in-time stochastic forcing is added as a separate √dt increment
# after the deterministic substep in BOTH schemes (it is a Wiener increment, not
# a smooth tendency, so it must not be folded into the RK evaluations of N).
TIME_SCHEME = 'strang'
# Kassam–Trefethen contour-integral averaging for the ETDRK4 φ-functions: number
# of equally-spaced points on the unit circle around each eigenvalue.  M=32 is
# the standard choice; it removes the catastrophic cancellation in (e^z−1)/z and
# the higher φ-functions when |z| = |L·dt| is small (our case: |L·dt| ≲ 5e-3).
ETDRK4_M = 32

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

# ── Auto-stationarity detection (improvement #2) ─────────────────────────────
# Rather than always spinning up for the full fixed N_SPINUP, monitor three
# integral invariants of the flow and stop as soon as the flow is statistically
# steady — but never before N_SPINUP_MIN nor after N_SPINUP (the guards).
#
#   WHAT IS MONITORED (all three must settle; see simulate_v7.diagnostics):
#     • energy       E = Σ_lm c_lm²/[l(l+1)]     (total kinetic energy, Parseval)
#     • enstrophy    Z = Σ_lm c_lm²              (total enstrophy)
#     • zonal frac.  E_zonal/E   (m=0 energy fraction — the jet indicator)
#   Energy and enstrophy are the two quadratic invariants of 2-D turbulence;
#   the zonal fraction is the quantity whose 200-frame −26% drift flagged the v6
#   window as a transient (v6_critical_audit §4.2).  Requiring all three to
#   plateau is a stricter, more honest stationarity test than any single one.
#
#   THE TEST.  Sample the three quantities every STATIONARITY_INTERVAL steps and
#   keep the samples spanning the last STATIONARITY_WINDOW steps (a sliding
#   window of W/interval samples).  The flow is declared statistically stationary
#   when, for EACH quantity q over the window,
#           (max q − min q) / |mean q|  <  STATIONARITY_TOL .
#   i.e. the peak-to-peak drift across a full window is below the tolerance.
#   Using the window spread (not a two-point difference) rejects both slow drift
#   and large sampling oscillations.
STATIONARITY_INTERVAL = 1000     # steps between stationarity samples
STATIONARITY_WINDOW   = 5000     # sliding-window length (steps); W/interval samples
STATIONARITY_TOL      = 0.02     # max fractional peak-to-peak drift over window (2%)
N_SPINUP_MIN          = 20000    # never declare stationary before this (min guard)

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
