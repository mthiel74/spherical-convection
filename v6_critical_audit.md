# v6 Critical Audit — forced–dissipative barotropic vorticity on a rotating sphere

**Date:** 2026-07-08
**Scope:** `config_v6.py`, `simulate_v6.py`, `visualize_v6.py`, `render_movie_v6.py`, `verify_v6.py`.
**Method:** every checkable claim was re-derived and tested numerically against the *actual*
saturated field (`frames_v6.npz`, 200 frames) with pyshtools 4.14.1. Scripts:
`scratchpad/audit_checks.py`, `audit_checks2.py` (reproduce every number below).
**Posture:** merciless. Credit is given only where a claim survived a numerical test.

---

## 0. Bottom line (the paragraph that matters)

**The core numerics are genuinely sound — sounder than the docstrings admit — and I found no new
hard numerical BUG.** The Coriolis fix is correct (f(pole)=80.000 with `2Ω/√3`), the spherical
Jacobian is metric-consistent, and — contradicting my own starting hypothesis — **the un-dealiased
quadratic Jacobian is effectively alias-free** (0.07 % vs a properly 2×-padded reference) and the
advection **conserves energy and enstrophy to ~1e-6 per step**. So the machinery is trustworthy.

**The problems are not in the solver; they are in what the run actually produced and what the code
claims about it.** Three things fail on inspection of the field itself:

1. **The recorded movie is a non-stationary transient, not a saturated steady state.** Over the
   200-frame window the energy is still *rising* (+3.7 %) and the zonal fraction is *falling*
   (−26 %). The spinup is exactly **one drag time** (33 tu); the energy-containing scales need
   several. An independent energy budget confirms it: injection exceeds dissipation by **+10 %** at
   the "saturated" state.
2. **The advertised zonal jets do not exist.** Zonal (m=0) energy is **3.7 %** and *declining* —
   essentially isotropic turbulence with a whisper of anisotropy. v5 was 1.3 %; v6 is barely better
   and moving the wrong way. The README/config claim "inverse cascade → jets/large vortices" is
   **not borne out by the field**, the same category of overclaim the previous audit flagged.
   Root causes: the Rhines/forcing scale separation is **marginal** (l_R≈23, forcing 60–80, ratio
   2.6 — not "well above"), and **42 % of injected energy is dissipated by hyperviscosity at small
   scales** instead of cascading upscale.
3. **The interior and core remain decoration, with quantified overstatements.** The mixing-length
   radial reconstruction overstates the penetration of forcing-scale structure by ~**10×** (a
   true `(r/R)^l` continuation puts l≈70 at 4×10⁻¹¹ of surface amplitude at the base; v6 shows 9 %).
   The inner core is coloured on a **hidden scale amplified 32×** relative to the one displayed
   colorbar — a saturated red on the core is ω≈0.20 but reads as 6.5 against the legend.

The genuine v6 improvement is real but narrower than claimed: the spectrum is now **broad**
(53 % below / 31 % in-band / 16 % above the forcing band, vs v5's 97 % trapped in-band), so a
forward enstrophy cascade (filaments) and *some* inverse transfer exist. But "developed turbulence
with jets, in steady state" is not what `frames_v6.npz` contains.

---

## 1. Findings, classified

| # | Area | Finding | Class |
|---|------|---------|-------|
| 1 | Numerics | Time integration is globally **1st-order** (Lie split of RK2 advection ⊗ exact linear factor), not the "2nd-order Heun" the docstring implies. Verified: step-halving error ratio = **2.00**. | CONCERN |
| 2 | Numerics | White-noise √dt forcing scaling is **correct** (variance/time independent of dt; budget closes). | ✅ PASS |
| 3 | Numerics | Per-mode forcing amplitude `∝1/√(l(l+1))` is **arbitrary** — neither flat-energy nor flat-enstrophy injection; varies 1.77× (enstrophy) / 3.1× (energy) across the band. | CONCERN |
| 4 | Numerics | Jacobian sign is a consistent global flip (= mirror flow / Ω→−Ω); statistics unaffected. | ✅ PASS |
| 5 | Numerics | Advection **conserves energy & enstrophy** to 2e-6 / 9e-6 per step (inviscid test). | ✅ PASS |
| 6 | Spectral | **Aliasing is negligible** — `gradient()` returns a 257×513 grid, oversampled enough that the un-dealiased Jacobian matches a 2×-padded reference to 0.07 %. My suspected aliasing bug does **not** materialize. | ✅ PASS |
| 7 | Spectral | `gradient()` returns **physical** (metric-consistent) components (grad Y₁⁰ matches −√3 sinθ, φ-comp ~1e-16). | ✅ PASS |
| 8 | Spectral | 4π-normalisation needs **no (2−δ_m0) factor**; `c2=c0²+c1²` equals pyshtools power exactly. Enstrophy/zonal diagnostics are correct. | ✅ PASS |
| 9 | Physics | Recorded window is **not statistically stationary** (energy +3.7 %, enstrophy +3.3 %, zonal −26 % across the 200 frames). Spinup = **1 drag time**, too short. | CONCERN (major) |
| 10 | Physics | **No zonal jets**: zonal energy 3.7 % and falling. README/config claim of jets from the inverse cascade is **not realised**. | CONCERN (major) / overclaim |
| 11 | Physics | Rhines/forcing separation is **marginal**: l_R≈23.4 vs forcing 60–80 (ratio 2.6), not "well above the Rhines scale" as commented. | CONCERN |
| 12 | Physics | **42 % of injected energy is removed by hyperviscosity** (not drag) — forcing sits too close to the dissipation range, leaking energy that should cascade upscale. | CONCERN |
| 13 | Physics | `L_REF=10` radial reconstruction **overstates penetration ~10×**; forcing-scale structure is honestly dead at the base (`0.71⁷⁰≈4e-11`) but shown at 9 % amplitude. Honestly labelled, but a large visual overstatement. | CONCERN / COSMETIC |
| 14 | Physics | `SHEAR_DEG=25°` conflates the solar **25–30 % rate** difference with a **25° arc** twist (unit confusion), and applies it as a depth-dependent **solid** rotation (uniform in latitude), which is not latitudinal differential rotation. | CONCERN (justification) / COSMETIC |
| 15 | Viz | Inner core is coloured on its **own 97th-percentile scale, 32× the displayed colorbar** — one legend, two hidden scales. Misleading. | COSMETIC (misleading) |
| 16 | Viz | Lambert core normal vector is **correct** (outward unit normal = position). | ✅ PASS |
| 17 | Viz | Octant cutaway logic (remove lon∈[0,90), lat>0; three faces line the cavity) is **geometrically consistent**. | ✅ PASS |
| 18 | Diag | `Re = ω_rms/μ` is **not a Reynolds number** — it is an inverse-drag parameter (=100). The actual velocity-based Re at the forcing scale is ~140; the flow speed is U≈0.073. | COSMETIC (mislabel) |
| 19 | Diag | Rhines uses equatorial (max) β — a **conservative** choice — but β varies with latitude; a single l_R is a simplification. | minor caveat |
| 20 | Missing | No **energy/enstrophy budget** check (input vs output). Adding it immediately reveals the +10 % non-steady imbalance. | MISSING |
| 21 | Missing | No **stationarity** check across the recorded frames — would have caught #9/#10. | MISSING |
| 22 | Missing | No **conservation test** of the solver (inviscid E/Z). It passes, but it isn't in the repo. | MISSING |

**Score of the previous audit's three "fixes":** Coriolis ✅ genuinely fixed. Cascade ⚠️ partially
— spectrum broadened, but jets still absent and the run under-equilibrated. Interior ⚠️ — replaced
one decoration (ξ⁴ decay) with another (10×-overstated eigenfunction), honestly labelled but still
not physics.

---

## 2. Numerical correctness

### 2.1 RK2 / integrating-factor splitting — globally first order (CONCERN)
The step is `k1,k2 = RK2 advection only`; `rhs = ω + ½dt(k1+k2)`; `rhs += F`; `ω ← e^{-(μ+νλ⁴)dt}·rhs`.
The nonlinear substep is 2nd-order Heun, and the linear factor is *exact*, but the **splitting
between them is Lie–Trotter, not Strang**, so the global scheme is **O(dt)**. Direct test
(deterministic part, one step vs two half-steps vs four quarter-steps):

```
||big−half|| = 2.95e-6   ||half−quart|| = 1.48e-6   ratio = 2.00   → first order
```

A ratio of 2 is the unambiguous signature of first-order global accuracy. The docstring's "RK2
(Heun), 2nd-order" is therefore **overstated**. This is not a wrong *result* — for steady-state
statistics with strong dissipation it is fine, and with **white-in-time forcing the strong order is
≤ 1 regardless** — but the accuracy claim should be corrected, or the splitting made Strang
(`L(dt/2) N(dt) L(dt/2)`) to actually earn 2nd order in the deterministic limit.

### 2.2 Forcing scaling — correct √dt, arbitrary per-mode shape (PASS / CONCERN)
`amp = FORCE_AMP/√(l(l+1))·√dt`. Variance injected per mode per unit time = `FORCE_AMP²/(l(l+1))`,
**independent of dt** — the correct white-noise scaling (confirmed by the budget in §3.2). But the
`1/√(l(l+1))` shape is neither constant enstrophy injection (that needs `amp=const`) nor constant
energy injection per mode (`amp∝√(l(l+1))`) — it is **arbitrary**. Across l=60–80 the enstrophy
injection per mode falls 1.77× and the energy injection 3.1×. Because the band is narrow this
barely matters, but there is no physical principle behind the choice and none is claimed.

### 2.3 Sign, dissipation filter, mean removal (PASS)
`J_code = −J_true` applied to `ω+f` together — a consistent mirror, statistics unchanged (as the
prior audit established). The dissipation filter `exp(−(μ+νλ⁴)dt)` is the exact linear solution; it
also multiplies the freshly-added forcing, but in-band `e^{-…}≈0.9999`, a negligible bias. `ω₀₀=0`
correctly enforces zero-mean vorticity.

### 2.4 Conservation (PASS — strong result)
Running advection only (no forcing/dissipation) from a saturated field, 200 steps:

```
Energy:    5.421125e-3 → 5.421137e-3   (+2.2e-6)
Enstrophy: 9.235799e+0 → 9.235885e+0   (+9.3e-6)
```

The exact barotropic Jacobian conserves both; the discrete scheme conserves them to ~1e-6, i.e. the
**Jacobian is effectively non-aliasing and the time step is accurate at these amplitudes**. This is
the single strongest piece of evidence that the solver is correct.

---

## 3. Spectral method

### 3.1 Aliasing — I was wrong; it's clean (PASS)
The quadratic Jacobian is computed on the grid and transformed back with `lmax_calc=LMAX`, with **no
explicit 2/3 dealiasing** — which *should* alias product content in (L, 2L] back onto retained
modes. It does not, because `SHCoeffs.gradient()` returns a **257×513** grid (≈2(L+1)×4(L+1)),
oversampled enough that the product is resolved. Test against a Jacobian computed on a 2L-padded
grid and truncated:

```
||J_code − J_dealiased|| / ||J_dealiased|| = 0.0007
per-degree power ratio at l=10…127: all 1.00
```

So the method is de-facto alias-free, and the conservation test (§2.4) corroborates it. **This
concern is retired.** (Note it relies on pyshtools' grid choice; if someone later swaps the grid or
truncates the gradient, aliasing could reappear — a comment to that effect would be prudent.)

### 3.2 Gradients, csphase, normalisation (PASS)
- `gradient()` gives **physical** components: for ψ=Y₁⁰, `grad.theta` matches `−√3 sinθ` to 5
  digits and `grad.phi`~1e-16. The 1/sinθ metric factor is inside `.phi`, so the Jacobian is
  metric-consistent with no double counting.
- `csphase=1` is used in **every** transform (`_to_grid`, `_to_lm`, `_jacobian_lm`, `spharm`,
  `coeffs_to_surface`, `_inner_core`) — consistent.
- **No (2−δ_m0) factor** is needed: `Σ(c0²+c1²)` equals pyshtools' `spectrum()` exactly (ratio
  1.000000, per-l ratios all 1.0). The enstrophy-fraction and zonal-energy diagnostics are correct
  as written.

---

## 4. Physics

### 4.1 The run is not in steady state (CONCERN, major)
`N_SPINUP·DT = 33.0 tu = 1/μ` — exactly **one drag time**. The energy-containing scales relax on
the drag time, so one τ_drag is ~1–2 e-foldings, not equilibration. The recorded 200-frame window
(t=33.0→36.0, only 0.09 τ_drag long) is still drifting:

```
enstrophy: 8.883e0 → 9.178e0   (+3.3 %)      slope +0.15/tu
energy:    5.279e-3 → 5.476e-3 (+3.7 %)      still rising
zonal%:    4.02 %   → 2.96 %    (−26 %)      falling, not building
```

The movie therefore shows a **transient**, not the saturated turbulence it advertises. Fix: spin up
several drag times (≥5/μ ≈ 165 tu) and *verify* stationarity before recording.

### 4.2 No jets (CONCERN, major / overclaim)
Zonal (m=0) energy fraction = **3.66 %** and declining. Jupiter-like jet regimes sit at 50–90 %.
This is isotropic turbulence with negligible zonal organisation — the README's "builds large-scale /
zonal structure via the inverse cascade … zonal jets from an arrested inverse cascade" is **not
realised in the field**. Two reinforcing causes:

- **Marginal scale separation.** Measured l_R = √(β/2U) = **23.4** (β=2Ω=80, U=0.073), forcing at
  60–80 → ratio only **2.6**. The config comment "well above the Rhines scale (l_R≈8–20)"
  understates l_R; there is an inverse range but it is short.
- **Energy leaks at small scales** (§4.3), starving the inverse cascade.

The spectrum *is* genuinely broader than v5 (53/31/16 % below/in/above band vs v5's 97 % in-band),
so filaments and *some* upscale transfer exist — but not jets.

### 4.3 Energy budget leaks to hyperviscosity (CONCERN)
Independent budget on the mean spectrum:

```
energy injection      = 6.07e-4 /tu
energy dissipation    = drag 3.23e-4  +  hyperviscosity 2.29e-4  = 5.51e-4
  → in/out = 1.10 (still charging)   → 42 % of energy removed at small scales
enstrophy injection   = 2.905  ;  dissipation drag 0.544 + hyper 2.313 = 2.857  (in/out 1.017)
```

Enstrophy is correctly hyperviscosity-dominated (forward cascade). But **energy** should be removed
almost entirely by large-scale drag in a clean inverse-cascade setup; here hyperviscosity takes
42 %. That is a direct consequence of forcing at l=60–80, close to the l≳68 dissipation range — a
big chunk of injected energy never gets upscale. Pushing the forcing lower (or the hyperviscous
cutoff higher) would widen the inverse range.

### 4.4 Radial reconstruction overstates penetration ~10× (CONCERN / COSMETIC)
`(r/R)^(l/L_REF)` with L_REF=10. e-folding depth = L_REF/l in r/R, i.e. **L_REF× deeper** than the
true regular solid harmonic `(r/R)^l`:

```
l   true amp @base(0.71)   v6 amp @base   e-fold depth/shell: true → v6
10     3.3e-2               0.710          0.34 → 3.4  (reaches base, & then some)
60     1.2e-9               0.128          0.06 → 0.57
70     3.9e-11              0.091          0.05 → 0.49   (forcing scale)
80     1.3e-12              0.065          0.04 → 0.43
```

Forcing-scale structure (l≈70) is honestly **evanescent** — 4×10⁻¹¹ of surface amplitude at the
base — but v6 paints it at 9 % and lets it e-fold over half the shell. The docstring's "penetration
∝ wavelength, as convective cells over a mixing length" is dimensionally self-consistent, but there
is **no barotropic-vorticity content in the radial direction at all** (the model is 2-D); the whole
reconstruction is invented decoration. Honestly labelled, but a viewer sees a filled interior that
the physics does not support.

### 4.5 Differential-rotation shear (CONCERN justification / COSMETIC effect)
`SHEAR_DEG=25°` with the comment "the Sun's equator-pole differential rotation is about 25–30 %."
That number is a **fractional rate** difference (ΔΩ/Ω, latitudinal); the code applies a **25° arc**
twist as a function of **radius**, uniform in latitude. Three problems: (a) unit conflation (% vs
degrees); (b) the bulk convection zone is nearly iso-rotating on radial/conical lines — the strong
*radial* shear is confined to the tachocline (~0.7R) and the near-surface layer, not a smooth 0→25°
ramp; (c) a latitude-uniform solid twist is not *differential* rotation at all. It is a pleasant
visual bend, not a representation of solar rotation. Cosmetic in effect (visualization only), but
the justification is wrong.

---

## 5. Visualization

### 5.1 Inner-core colour scale is disconnected and 32× hotter (COSMETIC, misleading)
`_inner_core` receives the surface `vmax` but **overwrites it** with `np.percentile(|core|,97)`.
Measured on frame 100:

```
surface vmax (the colorbar shown) = 6.497
inner-core vmax (own hidden scale) = 0.2005   → 32.4× amplification
```

So a saturated core red means ω≈0.20 but reads as 6.5 against the single displayed legend. The code
comment is honest about *why* (otherwise the core is white), but the rendered figure gives the
viewer no way to know the core is on a 32× scale. For a figure whose whole selling point is honesty,
this is the most misleading element. Fix: either show a second colorbar/annotation for the core, or
grey/desaturate it to signal "not on the same scale," or state the amplification in the caption.

### 5.2 Normals and cutaway (PASS)
The core's outward normal equals its unit position vector (correct Lambert term), shaded as a
headlight `0.55+0.45·max(n·cam,0)` — fine for illustration. The octant removal (`lon∈[0,90) ∧
lat>0`) and the three lining faces (equatorial quarter-annulus + two meridional walls at lon=0,90)
are geometrically consistent and join at the right edges.

---

## 6. Diagnostics & what's missing

- **Rhines formula** `√(β/2U)` is dimensionally correct on the unit sphere (k≈l/R, R=1) and uses
  the equatorial **max** β — the conservative choice for asserting scale separation. Caveat: β=2Ω
  cosφ varies with latitude, so l_R is latitude-dependent and jets (if any) would favour low
  latitudes; a single number is a simplification, not an error.
- **`Re=ω_rms/μ=100` is mislabelled** — it is a nondimensional inverse-drag, not a Reynolds number.
  The velocity-based Re at the forcing scale is ~140; the actual rms speed is U≈0.073 with eddy
  turnover 0.19 tu ≪ drag time 33 tu (locally very nonlinear, which is why the spectrum broadens).
- **MISSING: energy/enstrophy budget** (§4.3) — trivial to add to `verify_v6.py` and it directly
  exposes the non-steady state.
- **MISSING: stationarity test** (§4.1) — plotting Z(t), E(t), zonal(t) across frames would have
  caught the transient.
- **MISSING: inviscid conservation test** (§2.4) — it passes, but belongs in the repo as a
  regression guard for the solver.
- **Angular momentum:** under forcing+drag nothing is conserved, so there is no invariant to check
  at runtime; but the *inviscid* solver should conserve axial angular momentum, and that (like E, Z)
  is untested. Minor.

---

## 7. What v6 gets right (earned credit)

1. **Coriolis is correctly fixed:** `2Ω/√3` gives f(pole)=80.000 exactly (Y₁⁰ peak =√3 confirmed).
2. **The solver is trustworthy:** metric-consistent Jacobian, effectively alias-free, energy/
   enstrophy conserving to 1e-6, correct √dt forcing, consistent csphase and 4π power. No hard bug.
3. **The spectrum genuinely broadened** (53/31/16 % vs v5's 97 % in-band) — a real, measurable
   improvement; filaments and a forward enstrophy cascade now exist.
4. **The honesty framing is mostly good** — the code says "illustrative, not solved dynamics," and
   the docstrings are candid about the model not being convection.

---

## 8. Ranked fixes

1. **Spin up to steady state and prove it.** Raise `N_SPINUP` to ≥5/μ (~165 tu) and add a
   stationarity check (Z, E, zonal vs frame) + an energy/enstrophy budget to `verify_v6.py`. Do not
   record until injection≈dissipation and the trends are flat. *(This is the single most important
   fix — the current movie is a transient.)*
2. **Stop claiming jets, or earn them.** At l_R≈23 with forcing 60–80 and 42 % energy lost to
   hyperviscosity, zonal energy is 3.7 % and falling. Either (a) push forcing lower / widen the
   inverse range and cut small-scale energy loss until real jets appear (then re-measure zonal
   fraction), or (b) delete "jets/zonal structure/banana" language from README_v6/config.
3. **Fix the accuracy claim.** Either relabel the scheme "1st-order operator split (Heun advection +
   exact linear factor)" or make the split Strang to earn 2nd order in the deterministic limit.
4. **Make the core colour scale honest.** Add a second colorbar or caption note for the 32×
   amplified inner core, or desaturate it — one legend must not silently cover two scales.
5. **Reword the interior/shear physics.** State the ~10× penetration overstatement (L_REF) and drop
   the "25 % ≈ 25°" differential-rotation justification; both are decoration, and the caption should
   say so numerically.
6. **Rename `Re`.** It is `ω_rms/μ`, a drag parameter — call it that.

*All numbers verified in `scratchpad/audit_checks.py` and `audit_checks2.py`, pyshtools 4.14.1,
against `frames_v6.npz` (200 frames). No literature claim is relied on that isn't standard 2-D
turbulence / helioseismology (Rhines 1975; Vallis 2017; tachocline: Christensen-Dalsgaard et al.
1991).*
