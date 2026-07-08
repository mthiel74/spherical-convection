# Physics & Equations Audit — Spherical Convection Simulation vs. NASA Solar Convection

**Date:** 2026-07-08
**Scope:** rigorous, quantitative comparison of the repo's 2‑D barotropic‑vorticity
"solar convection" model (`config.py`, `simulate.py`, `visualize.py`, v5 →
`output_v5.mp4`) against real 3‑D solar convection simulations of the type NASA/NCAR publish.
**Audience:** physicist. No hand‑waving. Every checkable claim was verified numerically
(`scratch_verify.py`, pyshtools 4.14.1) or against cited literature.

---

## 0. Executive summary (the one paragraph that matters)

The **numerical machinery is almost entirely correct** — the spectral barotropic solver, the
Laplacian inversion, the hyperviscosity operator, the integrating‑factor time stepping and the
RK2 scheme all check out. **One genuine quantitative bug exists**: the Coriolis coefficient is
too large by a factor **√(4π) ≈ 3.545** (verified numerically), so the model's *effective*
rotation rate is ≈142, not the labelled 40. The decisive finding, however, is that the run
**does not reproduce the structures it claims to**: measured on the actual saturated field,
**97 % of the enstrophy is trapped in the forcing band, the zonal (jet) energy is 1.3 %, and the
vortices are isotropic round blobs** — there are **no banana cells, no zonal jets, and no
filament cascade**, despite the README asserting all three. The cross‑section interior and the
inner core in the render are **cosmetic** (a ξ⁴‑decayed copy of the surface field + an unrelated
static random field), not solved physics.

**Correction to my first pass (before the reference GIF was available): the NASA reference is
*not* a 3‑D ASH/Rayleigh convection render.** Having now inspected the actual GIF
(`nasa_original.gif`, 360×360, 268 frames), it is an **ω'_z vorticity field on a cutaway sphere
— the *same visualization genre and the same 2‑D‑style model family as ours***, just far better
executed. So the honest head‑to‑head is **our 2‑D field vs a sibling 2‑D field**, and the
reference wins decisively on turbulence realism: it shows a **fully‑developed enstrophy cascade
(thin filaments, broad spectrum, l≈40–75 plus low‑l blobs)** and a **wide, warm‑skewed dynamic
range**, exactly the things ours lacks. *Neither* image is real solar convection physics — that
is 3‑D anelastic MHD (§13–15), which both models omit entirely. Bottom line: our numerics are
sound (bar one √(4π) factor), but as a realization of even a 2‑D vorticity field the run is
under‑developed and its interior/core are painted on; the reference demonstrates what the *same
kind* of model looks like when done well.

---

## PART 1 — Frame‑by‑frame comparison against the real NASA GIF

Figure: `frame_comparison.png` (+ iCloud copy). Source: `nasa_original.gif` (360×360, 268 frames,
slowly evolving; displayed time drifts only t≈46542→46582 across the loop → a statistically steady
field). Measurements from `scratch_measure_nasa.py` (NASA, from pixels) and from the spectral
coefficients (ours, exact).

**What the reference actually is.** Identical plot furniture to ours — title `ω'_z`, `t = …`, a
cutaway sphere with a solid inner boundary, a dashed intermediate boundary and a lat/lon graticule.
It is a **2‑D‑style relative‑vorticity field on a sphere**, *not* a 3‑D convection render (those
show radial velocity/temperature with columnar radial structure, not a scalar ω'_z). Treat it as a
**better‑resolved sibling** of our own model.

| Feature | NASA reference (measured from pixels) | Ours v5 (exact, from coefficients) |
|---|---|---|
| Surface morphology | **thin, stretched filaments** (developed enstrophy cascade) | **isotropic round blobs**, single scale |
| Characteristic degree | filaments **l≈40–75** (6.3° wavelength, ±30%) **+ low‑l interior blobs** → broad spectrum | l_peak=**24**, ⟨l⟩=**34**, 96.8% of enstrophy in l=24–52 (narrow) |
| Colour / amplitude | **warm/hot palette, strongly +skewed**: median(R−B)=**+0.30**, 85% warm / 13% cool; wide dynamic range | **symmetric RdBu_r, zero‑mean**: median 0.00, 43% warm / 41% cool; narrow |
| Cross‑section interior | large‑scale, low‑l vorticity filling **to the centre** | **cosmetic ξ⁴ decay** → ~0 by mid‑shell (not solved) |
| Inner core | **no distinct calm core** — structure continues inward | **distinct shaded low‑l core** with drift (faked, unrelated field) |
| Zonal jets / bananas | not obviously banded; weak large‑scale organization | **none** (zonal energy 1.3%) |

**Reading of the four zoom panels.** (a) NASA surface = fine curved filaments; ours = round cells.
(b) NASA interior = coherent large‑scale orange/blue field reaching the centre; ours = pale near‑white
with a shaded fake core. The single most important morphological difference is the **filaments**: the
reference has a wide inertial range (high effective Reynolds number); ours has essentially none.

**Caveat on the NASA numbers.** l is estimated from a foreshortened, curved rendered surface via a
patch FFT, so ±30%; the colour skew could partly reflect a warm colormap rather than a truly
positive‑skewed field. The *qualitative* contrasts (filaments vs blobs, broad vs narrow, to‑centre vs
faked‑decay) are unambiguous and do not depend on the exact numbers.

---

## PART 2 — Equations audit

The prognostic equation (README / `simulate.py` docstring):

∂ω/∂t + J(ψ, ω+f) = −ν(−∇²)⁴ω − μω + F

This is the standard **forced–dissipative barotropic vorticity equation** on the rotating sphere
(material conservation of absolute vorticity q = ω+f, plus hyperdiffusion, linear drag, stochastic
forcing). The written form is textbook‑correct. Item‑by‑item verification of the code:

### ✅ 1. Barotropic vorticity equation — CORRECT (as written)
`_tendency()` returns `−J(ψ, ω+f)`; forcing is added; the linear operators (−μ and −ν(−∇²)⁴) are
applied as an exact integrating factor. Structurally faithful to the equation. (The top‑of‑file
docstring drops the −μω term, but the code *does* include it via the dissipation filter — a
comment omission, not a code error.)

### ❌ 2. Coriolis parameter f = 2Ω sinφ — REAL BUG (verified: factor √(4π) = 3.545 too large)
Position in spectral space is **right**: f = 2Ω sinφ = 2Ω cosθ ∝ Y₁⁰, i.e. only (l,m) = (1,0).
The **coefficient is wrong**. The code sets

```python
self._f_lm[0, 1, 0] = 2.0 * OMEGA * np.sqrt(4.0 * np.pi / 3.0)
```

with the comment "Y₁⁰(4π) = √(3/4π) cosθ". That √(3/4π) is the **orthonormal** convention
(∫Y²dΩ = 1). But the solver uses `normalization='4pi'`, whose real harmonic is
**Y₁⁰ = √3 cosθ** (∫Y²dΩ = 4π). Numerically verified:

| quantity | value |
|---|---|
| unit‑coefficient Y₁⁰ peak in pyshtools 4π | **1.7321 = √3** (confirms 4π convention) |
| code coefficient 2Ω·√(4π/3) | 163.73 |
| **correct** coefficient 2Ω/√3 | 46.19 |
| f(pole) produced by the code | 283.6 |
| f(pole) intended (= 2Ω) | 80.0 |
| **ratio produced / intended** | **3.5449 = √(4π)** |

**Consequence:** the planetary vorticity — and hence the β‑effect that is the *only* way rotation
acts in this model — is 3.545× too strong. The **effective rotation rate is Ω_eff ≈ 142**, not 40.
This is a bug, but a *rescaling* one: since Ω is a free non‑dimensional parameter, the run is
internally consistent at Ω_eff ≈ 142. It does not corrupt the solution; it means the label is wrong
and every "rotation period" in the movie is mis‑scaled by 3.545.

### ✅ 3. Jacobian J(ψ, ω+f) via gradients — CORRECT (method is actually good)
`_jacobian_lm` forms the horizontal gradients with `SHCoeffs.gradient()` (physical spherical
components, which *include* the 1/sinθ metric factor) and computes
`(∇A)_φ(∇B)_θ − (∇A)_θ(∇B)_φ` on the grid, then transforms back. Using **physical** gradient
components is the correct, metric‑consistent way to evaluate the spherical Jacobian — it avoids the
1/cosφ singularity that a naive ∂λ,∂φ finite‑difference Jacobian would introduce. This is a
genuinely sound choice.

**Sign:** numerically, J_code(x, y) = −z, i.e. the code computes **−(∇A×∇B)·r̂**, the opposite of
the "u = ẑ×∇ψ" convention. **This is not a bug.** A global sign flip of J is identical to adopting
u = −ẑ×∇ψ (mirror the flow), and it is applied to (ω+f) *together*, so the relative sign of
self‑advection J(ψ,ω) and the β‑term J(ψ,f) is preserved. Rossby waves still propagate westward
relative to the mean flow (because "west" flips with u). Statistically the turbulence is
unaffected. Verified consistent.

### ✅ 4. Streamfunction inversion ψ = ∇⁻²ω — CORRECT
On the unit sphere ∇²Y_lm = −l(l+1)Y_lm, so ψ = ∇⁻²ω uses eigenvalue **−1/[l(l+1)]** with the l=0
mode set to zero. Exactly what `self._inv_ev` holds. Correct.

### ✅ 5. Hyperviscosity (−∇²)⁴ = λ⁴, λ = l(l+1) — CORRECT
`ev = −l(l+1)` (the ∇² eigenvalue), `lam4 = ev**4 = [l(l+1)]⁴ = λ⁴`. Since (−∇²) has eigenvalue
+l(l+1), (−∇²)⁴ has eigenvalue λ⁴. This **is** ∇⁸ hyperdiffusion (∇⁸ = (∇²)⁴ = (−∇²)⁴, even power).
Correct — not ∇⁴, not ∇⁶.

### ✅ 6. Implicit integrating factor exp(−(μ + νλ⁴)dt) — CORRECT
`_dissipation_filter` returns `exp(−(drag + ν·λ⁴)·dt)`, applied multiplicatively after the explicit
advection+forcing step. This is the exact solution of the linear part ω̇ = −(μ + νλ⁴)ω over dt —
unconditionally stable, standard operator splitting. Correct.

### ✅ 7. RK2 (Heun) — CORRECT
`k1 = f(ω)`, `k2 = f(ω + dt·k1)`, `ω* = ω + ½dt(k1+k2)` is exactly Heun's method (explicit trapezoid,
2nd‑order). Forcing (∝√dt, correct white‑noise scaling) is added, then the dissipation factor. The
splitting of nonlinear (RK2) from linear (exact) parts is a sensible, common construction. Correct.

**Equations verdict:** 6 of 7 correct; the 7th (Coriolis) is right in structure but off by √(4π) in
amplitude. As a *piece of numerics*, this solver is sound.

---

## PART 2 — Parameters vs. solar values

### 8. ✅ R_INNER = 0.71 — essentially exact
Helioseismic base of the convection zone: **r_c/R☉ = 0.713 ± 0.003** (Christensen‑Dalsgaard, Gough
& Thompson 1991; Basu & Antia 1997 refine to 0.713 ± 0.001). 0.71 is correct to ~0.4 %. `R_MID = 0.85`
is a plausible mid‑shell decoration. Good.

### 9. ⚠️ Ω = 40 → Rossby number
In a 2‑D barotropic model there is no convective velocity to define Ro the usual way; the meaningful
measure is the **ratio of relative to planetary vorticity**, Ro ~ ω_rms/f. Measured:
ω_rms ≈ 2.1, f(pole) = 283.6 (as coded) → **Ro ~ 0.007**; with the intended coefficient
(f = 80) → Ro ~ 0.026. Either way "strongly rotationally constrained" *on paper*. The Sun's deep
convection zone is indeed rotationally constrained (**Ro < 1**, transition near Ro ≈ 0.16;
Featherstone & Hindman 2016), so the *regime label* is defensible. **But** — see §16–17 — the model
shows **none** of the anisotropy that rotational constraint is supposed to produce, because in 2‑D
rotation only acts through β and the drag suppresses the inverse cascade that would build jets. The
rotation is dynamically almost inert here.

### 10. ⚠️ Forcing band l = 24–52 — plausible scale, wrong regime for "giant cells"
Angular half‑wavelength ≈ 180°/l: l=24 → 7.5° → **91 Mm** at the surface; l=52 → 3.5° → **42 Mm**.
So the forcing injects at **~40–90 Mm**, i.e. between **supergranulation** (~30 Mm, l≈120) and the
low end of **deep giant cells** (~100–200 Mm, **l ≲ 10–30**; Miesch et al. 2008, Hathaway et al.
2013). The band overlaps the giant‑cell range at its low end but is centred too small. More
important, injecting energy *directly* at the "cell" scale is the opposite of how real convection
works (buoyancy drives a broadband instability; scale is *selected*, not imposed).

### 11. ⚠️ Hyperviscosity ν = 3×10⁻¹⁵ — bites only very near truncation; modest effective Re
Damping rate νλ⁴ and its timescale by degree (verified):

| l | λ = l(l+1) | νλ⁴ | τ_hyper |
|---|---|---|---|
| 24 | 600 | 4×10⁻⁴ | 2572 |
| 40 | 1640 | 0.022 | 46 |
| 52 | 2756 | 0.173 | 5.8 |
| 60 | 3660 | 0.54 | 1.9 |
| 70 | 4970 | 1.83 | 0.55 |
| 85 (=LMAX) | 7310 | 8.57 | 0.12 |

So hyperviscosity is negligible through the forcing band and only becomes fast (τ<1) at **l ≳ 68** —
a correctly scale‑selective cutoff. The **effective Reynolds number** at the forcing scale, taken as
(nonlinear eddy rate)/(dissipation rate) ≈ ω_rms/μ ≈ 2.1/0.25 ≈ **8** (hyperviscosity there is
2×10⁻², negligible). Re ~ 8 is *weakly turbulent* — enough for some straining but the linear drag,
not viscosity, is the dominant sink. This is why (§16) the spectrum barely transports energy out of
the forcing band.

### 12. ✅/⚠️ Linear drag μ = 0.25 → damping time 1/μ = 4 time units
= **25.5 labelled rotations** (Ω=40) or **90 rotations** at Ω_eff=142. A reasonable large‑scale sink
in absolute terms — but it is *strong enough to kill the inverse cascade* before it can organise jets
(the Rhines mechanism needs the cascade to run for many drag times at scales below the forcing). It
is the proximate cause of the missing zonal jets.

---

## PART 2 — What the NASA simulation actually is (items 13–15)

### 13. What the reference GIF is — and what real solar‑convection imagery is
**The supplied reference (`nasa_original.gif`) is itself a 2‑D‑style ω'_z vorticity field on a
cutaway sphere** (see Part 1) — the same model *genre* as ours, not a 3‑D convection code. I cannot
verify its provenance or the code that produced it; on the visual evidence (scalar relative vorticity
ω'_z, cutaway sphere, filamentary 2‑D turbulence, no radial columnar structure) it is **not** an ASH/
Rayleigh anelastic‑convection render. So the "2‑D toy vs 3‑D NASA physics" contrast I drew before the
GIF was available does **not** apply to this reference; both are 2‑D vorticity fields, and the
reference is simply the better‑developed one.

For completeness, the *genuine* NASA/NCAR solar‑convection simulations (the physically correct target,
which **neither** image is) come from two 3‑D global codes:
- **ASH (Anelastic Spherical Harmonic)** — Clune, Miesch, Brun, Toomre, Glatzmaier (NCAR). Spherical
  harmonics horizontally × Chebyshev radially; an LES of the rotating spherical shell. Source of the
  classic red/blue **radial‑velocity** giant‑cell sphere with banana cells.
- **Rayleigh** — N. Featherstone (CU Boulder / CIG). Same physics class, modern, massively parallel.

(A separate, widely‑shared NASA cutaway is **GSFC SVS 3496 "Solar Dynamo: Plasma Flows"** — differential
rotation + meridional circulation + tachocline — but that is a flux‑transport‑dynamo *schematic*, not a
convection LES, and not the supplied reference either.)

### 14. Equations the 3‑D code solves (vs. ours)
Full **3‑D anelastic MHD** on (r,θ,φ), fields expanded about a stratified, near‑adiabatic reference
state ρ̄(r), P̄(r), T̄(r), S̄(r):

- anelastic continuity: **∇·(ρ̄ u) = 0** (filters sound waves; ρ̄ varies with depth)
- momentum: ρ̄(∂ₜu + u·∇u + **2Ω×u**) = −∇P′ + **ρ̄ g S′/c_p** (buoyancy) + (1/4π)(∇×B)×B + ∇·𝒟
- entropy/energy: ρ̄T̄(∂ₜS′ + u·∇(S̄+S′)) = ∇·[κ_r ρ̄ c_p ∇(T̄+T′) + κ ρ̄ T̄ ∇(S̄+S′)] + Φ + Ohmic
- induction (MHD): ∂ₜB = ∇×(u×B) − ∇×(η∇×B), ∇·B = 0

That is **5–8 coupled 3‑D fields**. Ours is **one scalar equation for ω on a 2‑D surface** — three
velocity components collapse to a non‑divergent horizontal (u_θ,u_φ) with **u_r ≡ 0**, and there is
no ρ, no S, no B, no energy equation.

### 15. Key physics we miss (all absent *by construction*, not by approximation)
Radial stratification (~4–5 density scale heights) · **buoyancy driving** (the actual engine —
ρ̄ g S′/c_p) · thermal/entropy transport and the Rayleigh number (no convective *instability* at all)
· **radial/vertical velocity** (the entire up/down‑flow signal in the NASA images) · compressibility
· **Taylor–Proudman balance** (a genuinely 3‑D constraint ∂u/∂z→0 — meaningless on a surface) ·
meridional circulation · **Reynolds‑stress‑driven differential rotation** · magnetic fields / dynamo.
In short: our model shares only the *rotating‑sphere geometry* and *2‑D‑turbulence kinematics*. It
contains none of the driving or constraining physics of convection.

---

## PART 2 — Quantitative comparison (items 16–19)

Measured on our saturated field (3000 steps ≈ 1.5 drag times; `scratch_verify.py`):

### 16. Characteristic vortex scale
- **Ours:** enstrophy spectrum peaks at **l_peak = 24**, energy‑weighted **⟨l⟩ = 34** → cells of
  **~40–90 Mm**. *Critically*, **96.8 %** of the enstrophy sits in l=24–52, only 0.6 % at l<24 and
  2.6 % at l>52. **There is no inverse cascade and no forward filament cascade** — the field is the
  forcing band ringing. (Contradicts the README's "enstrophy cascade fills l≈45–80 with filaments"
  and "inverse cascade to banana cells".)
- **Sun / ASH:** deep **giant cells l ≲ 10–30** (~200 Mm, lifetimes ~1 month); supergranulation
  l≈120 (~30 Mm). Our l overlaps the giant‑cell range but is centred a bit high, and — the real point
  — it is a *forced* scale, not a *selected* one.

### 17. Zonal vs. meridional structure — WE FAIL THIS COMPLETELY
- **Ours:** zonal (m=0) energy fraction = **1.3 %**. The surface field is **isotropic round blobs at
  all latitudes** (see the flat lat–lon panel in `frame_comparison.png`): no east–west elongation, no
  latitude dependence, no jets. The Rhines degree l_R = √(β/2U) ≈ **25** (intended β) or **46**
  (as‑coded β) falls *inside the forcing band* — there is **no scale separation** for jet formation,
  and the strong drag suppresses what little inverse cascade there might be.
- **Sun / ASH:** strongly **anisotropic** — north–south‑elongated **banana cells** at low latitude
  (outside the tangent cylinder, from Taylor–Proudman), plus a robust **zonal differential‑rotation**
  band structure (fast equator). This anisotropy is *the* signature of rotating convection, and we
  reproduce **none** of it.

### 18. Depth of penetration — NOT PHYSICS
- **Ours:** the cross‑section interior is **not simulated**. `visualize.spherical_field()` takes the
  single 2‑D surface field, applies a **steep quartic radial decay ξ⁴** (amplitude 0.06 by mid‑shell,
  ≈0 below), a **cosmetic longitude shear** (SHEAR_DEG·(1−ξ)) and **ad‑hoc sinusoidal "bands"** to
  fake tangential arcs. So "structures fade out by mid‑shell" is a chosen decay law, not a computed
  penetration depth.
- **Sun / ASH:** giant cells span the **full convection zone** (r/R ≈ 0.71→1.0) as coherent
  columns; penetration/overshoot at the base is a *result*.

### 19. Inner core — decorative, coincidentally reasonable
- **Ours:** `_inner_core()` paints a **separate, static, low‑degree (l=2–5) random field** at reduced
  amplitude with a slow longitudinal drift. It is unrelated to the outer solution and to any physics.
- **Sun / NASA:** the region below 0.71 R☉ is the **stable radiative interior** — nearly solid‑body
  rotation, **no convection**. A calm, featureless core is *qualitatively* the right idea, so this one
  reads acceptably — but ours is random noise dressed as calmness, not a modelled stable layer.

---

## PART 3 — Honest verdict

### What the simulation gets RIGHT
1. **The numerics are sound.** Spectral barotropic solver, Laplacian inversion, ∇⁸ hyperviscosity,
   exact integrating‑factor dissipation, Heun RK2 — all correct and correctly implemented. Using
   physical gradient components for the spherical Jacobian is a good, metric‑consistent choice.
2. **Geometry.** Convection‑zone base at 0.71 R☉ (vs 0.713) is essentially exact; the thin‑shell
   aspect ratio is right.
3. **Regime labels** (rotationally constrained, forcing at ~supergranular–giant‑cell scale) are in
   the right ballpark, even if not realised dynamically.
4. **The render is genuinely beautiful** and the occlusion/graticule engineering is careful and
   correct. As *scientific illustration* it is effective.

### What it gets WRONG
1. **It is not a convection model.** No buoyancy, no stratification, no energy equation, no vertical
   velocity. It is 2‑D vorticity turbulence on a sphere. Calling ω′_z "convection" is a category
   error — there is no convective instability anywhere in the code.
2. **Coriolis bug:** f too large by √(4π) = 3.545 (effective Ω ≈ 142, not 40); all "rotation period"
   labels are mis‑scaled accordingly.
3. **The advertised structures do not exist in the output.** No banana cells, no zonal jets (1.3 %
   zonal energy), no filament cascade (97 % of enstrophy trapped in the forcing band). The surface is
   isotropic blobs. The README/config claims of inverse cascade → banana cells and forward cascade →
   filaments are **not borne out by the field they produce**.
4. **The interior and core are cosmetic.** The cutaway faces are a ξ⁴‑decayed, longitude‑sheared copy
   of the surface field with sinusoidal "bands"; the core is an unrelated static random field. Nothing
   in the interior is solved. The most visually "3‑D" part of the movie is the least physical.

### Is the reference "more correct"? — Two different questions.
**(a) vs the supplied reference GIF (a 2‑D‑style ω'_z field, same genre as ours).** It is not "more
physically correct" — it is the **same class of model, better executed**. It exhibits a fully‑developed
enstrophy cascade (filaments, broad spectrum l≈40→low‑l), a wide dynamic range, and coherent
large‑scale interior structure, where ours is a narrow‑band, weakly‑nonlinear blob field with a faked
interior. In other words the reference is the existence proof that this kind of 2‑D model *can* look
rich and turbulent; our run doesn't, because of the parameter choices (§11–12, §16–17): weak
nonlinearity (effective Re ~ 8), strong drag, and forcing sitting on top of the Rhines scale, which
together kill both the inverse cascade (→ no jets) and the forward cascade (→ no filaments). This is
**tunable** — closer to a fixable gap than a fundamental one.

**(b) vs real solar convection (3‑D ASH/Rayleigh).** *Neither* image is correct. Real convection solves
the **actual governing equations** — 3‑D anelastic MHD with buoyancy, stratification, energy transport
and rotation on (r,θ,φ). It **selects** convective scales from an instability, **generates** differential
rotation from Reynolds stresses, **produces** banana cells via Taylor–Proudman, and resolves radial
structure and (in dynamo runs) the magnetic cycle. A 2‑D surface vorticity equation — ours *or* the
reference — contains none of that driving/constraining physics; any resemblance is 2‑D‑turbulence
pattern formation, not convection. (And even the 3‑D codes face the unresolved **"convective
conundrum"**: helioseismology finds deep flows 20–100× weaker than the LES predict, Hanasoge et al.
2012 — so the real codes are not "truth" either.)

### Unavoidable limitations of 2‑D barotropic vs. full 3‑D
These cannot be fixed by tuning — they are structural:
- **No radial dimension** ⇒ no depth structure, no penetration, no overshoot, no tachocline, no
  vertical velocity (the whole NASA color signal).
- **No buoyancy / energy equation** ⇒ no convective instability, no Rayleigh number, no heat flux.
  The flow is *stirred*, not *driven*.
- **No Taylor–Proudman** ⇒ banana cells cannot arise from the correct mechanism (they'd have to be
  faked via β‑turbulence, which here doesn't even produce jets).
- **No self‑consistent differential rotation** ⇒ Reynolds‑stress angular‑momentum transport needs
  correlated 3‑D velocity components that a single scalar ω cannot carry.
- **No magnetism** ⇒ no dynamo, no cycle, no sunspots.

A 2‑D barotropic model *can*, when tuned (forcing well below the Rhines scale, weak drag), produce
**Jupiter‑like zonal jets** — that is its legitimate physical niche. This run doesn't even reach that,
because the forcing sits at the Rhines scale and the drag is too strong.

### Are the equations mathematically correct even if physically simplified?
**Yes, with one amplitude bug.** The barotropic vorticity equation is correctly formulated and (Coriolis
coefficient aside) correctly discretised. As a numerical integrator of ∂ω/∂t + J(ψ,ω+f) = −ν(−∇²)⁴ω −
μω + F, the code is trustworthy. The mathematics is fine; it is simply the *wrong equation* for solar
convection.

### Are the parameters in the right ballpark?
**Mostly yes, individually; no, in combination.** R_INNER (0.71), the rotational‑constraint regime,
and the forcing scale (~supergranular–giant‑cell) are each defensible. But the **combination** —
forcing at l≈24–52 with Rhines degree l_R≈25–46 *inside* that band, plus μ=0.25 drag — guarantees the
run cannot produce jets or an inverse cascade, so the emergent field matches neither Jupiter‑style 2‑D
turbulence nor solar convection. And the Coriolis coefficient is a factor √(4π) off.

---

## Bottom line
- **As numerics / scientific illustration:** solid, careful, and (bar one √(4π) factor) correct.
- **vs the supplied reference GIF:** same 2‑D model genre, but the reference is a *much* better
  realization — it has the enstrophy cascade, filaments, dynamic range and coherent interior that ours
  lacks. Our shortfall here is largely **tunable** (parameters, not physics): the run is weakly
  nonlinear, over‑damped, and forced at the Rhines scale, so it produces neither jets nor filaments and
  the interior/core are painted on.
- **vs real solar convection:** *neither* image is convection. Both are 2‑D surface vorticity fields
  with no buoyancy, stratification, energy equation or vertical velocity — the wrong governing model
  for the Sun, not merely a coarser one. Real convection needs a 3‑D anelastic code.

### Concrete, ranked fixes
1. **Fix the Coriolis coefficient:** `2*OMEGA/np.sqrt(3.0)` (removes the √(4π) error). *(1 line.)*
2. **Stop advertising structures that aren't there** — the current field has no jets, bananas or
   filaments; either change the parameters or the claims (README/config say all three; the field has
   none).
3. **To match the reference's look (developed 2‑D turbulence):** widen the inertial range — push
   forcing to higher l (e.g. l≈60–80, above the current band), cut the linear drag μ by ~5–10× so an
   inverse cascade can run, and/or raise the effective Reynolds number. That yields filaments (forward
   enstrophy cascade) and, with weak enough drag, Jupiter‑like zonal jets (inverse cascade arrested at
   the Rhines scale). This is the legitimate physics a barotropic model delivers.
4. **Make the interior honest:** the cross‑sections and core are currently decoration. Either label
   them as such, or — since a 2‑D model has no interior to show — drop the cutaway and present the
   surface field (or a flat map) as what it is.
5. **To actually model solar convection:** you need a 3‑D anelastic code (Rayleigh is open‑source).
   No amount of 2‑D tuning bridges that gap.

*Quantitative claims verified in `scratch_verify.py` (solver: Coriolis, Jacobian, spectrum) and
`scratch_measure_nasa.py` (reference GIF scale/colour), pyshtools 4.14.1; side‑by‑side in
`frame_comparison.png`; reference `nasa_original.gif` (360×360, 268 frames). Literature:
Christensen‑Dalsgaard et al. 1991; Miesch et al. 2008
(arXiv:0707.1460); Featherstone & Hindman 2016 (arXiv:1609.05153); Hanasoge et al. 2012 (PNAS);
Hathaway et al. 2013 (arXiv:1401.0551).*
