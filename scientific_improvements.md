# 20 Ways to Scientifically Improve the Spherical-Convection Project

**Scope and honesty up front.** The current solver integrates a single scalar
equation — forced–dissipative **barotropic (2-D) vorticity on a rotating
sphere**,

$$\partial_t\omega + J(\psi,\,\omega+f) = -\nu(-\nabla^2)^4\omega - \mu\,\omega + F,
\qquad \omega=\nabla^2\psi,\quad f=2\Omega\sin\varphi .$$

It is **not convection**: there is no buoyancy, no stratification, no thermal or
entropy equation, and no vertical velocity. The numerics are verified correct
(two audits), but the physics on display is 2-D turbulence, and the v6 movie is
a **transient** recorded before the flow reached a statistically steady state.
The "interior" of the cutaway is a cosmetic radial continuation, not solved
dynamics.

The improvements below are ranked by **scientific return per unit effort toward
a defensible result** — i.e. the first entries most cheaply close the gap
between what the code claims and what it does; the middle entries add genuinely
new physics; the last entries replace the model with real convection. Each entry
gives: *what it is*, *why it matters* (the new physics/behaviour captured), a
*difficulty* estimate, and at least one *key reference*.

Difficulty legend: **Easy** (hours, parameter/diagnostic level) · **Medium**
(days, new numerics inside the existing solver) · **Hard** (weeks, new equation
set) · **Major** (a new code / research programme).

---

## Tier A — Make the current barotropic model do the physics it already claims

### 1. Widen the forcing–Rhines scale separation so jets actually form
**What.** The headline scientific defect. Jets emerge when the inverse energy
cascade runs from the forcing scale up to the Rhines scale and is halted there by
the β-effect. The current run forces at $l_f\!\approx\!60$–$80$ with a Rhines
degree $l_R\!\approx\!23$ — a separation ratio of only **2.6**, far below what is
needed. Push $l_f/l_R$ to $\gtrsim 5$–$10$ by (i) forcing at higher $l$ (needs
item 6's resolution), (ii) cutting friction further so the cascade is not
arrested prematurely, and (iii) tuning $\Omega$ so $l_R$ lands at $\sim$5–12 (a
handful of jets). Target the **zonostrophy index** $R_\beta\gtrsim 2$.
**Why.** This is the difference between "broadened spectrum, no jets" (the honest
v6 result) and the banded zonal flow the project is trying to illustrate. It
converts the model from *almost* showing the target physics to *showing* it.
**Difficulty.** Medium (couples to resolution and run length).
**Reference.** Rhines (1975) *J. Fluid Mech.* **69**, 417; Vallis & Maltrud
(1993) *J. Phys. Oceanogr.* **23**, 1346; Galperin & Read (eds.), *Zonal Jets*
(Cambridge, 2019); Scott & Dritschel (2012) *J. Fluid Mech.* **711**, 576.

### 2. Integrate to a genuine statistically steady state, then time-average
**What.** The recorded window drifts (energy +3.7 %, zonal fraction −26 % over
200 frames): it is a spin-up transient, not equilibrium. Extend spin-up until the
`stationarity` drift is <2 % across the whole record, and report *time- and
ensemble-averaged* spectra and jet profiles, not single snapshots.
**Why.** Every statistical claim (spectral slope, zonal fraction, cascade
directions) is only meaningful in the saturated state. Averaging also suppresses
the sampling noise that masks a weak but real cascade.
**Difficulty.** Easy (longer run + averaging; the diagnostic already exists).
**Reference.** McWilliams, *Fundamentals of Geophysical Fluid Dynamics*
(Cambridge, 2006), ch. on turbulence statistics.

### 3. Replace Lie–Trotter splitting with Strang (2nd-order) or IMEX
**What.** The scheme is Heun (RK2) for advection composed with an *exact*
integrating factor for the linear part, but the operators are split
Lie–Trotter — so the global order is **first**, not second (step-halving error
ratio ≈ 2.0). Symmetrise it (½-step linear · full nonlinear · ½-step linear) for
Strang splitting, or move to an **IMEX** Runge–Kutta pair that treats the stiff
linear term implicitly and the Jacobian explicitly.
**Why.** Recovers 2nd-order accuracy at essentially no extra cost, tightening the
energy/enstrophy budget (currently ~10 % of the energy leaks anomalously to
hyperviscosity) and letting larger stable time steps.
**Difficulty.** Easy–Medium.
**Reference.** Strang (1968) *SIAM J. Numer. Anal.* **5**, 506; Ascher, Ruuth &
Spiteri (1997) *Appl. Numer. Math.* **25**, 151.

### 4. Exponential time differencing (ETDRK4) for the stiff linear term
**What.** Because the linear operator $\mu+\nu\lambda^4$ is diagonal in spectral
space, exponential integrators are ideal: ETDRK4 treats it *exactly* while giving
**4th-order** accuracy on the nonlinear Jacobian. A drop-in replacement for the
current split step.
**Why.** Removes the stiffness-imposed time-step limit from hyperviscosity and
makes the linear dissipation exact to machine precision — directly improving the
budget closure and allowing higher resolution (item 6) without a punishing Δt.
**Difficulty.** Medium.
**Reference.** Cox & Matthews (2002) *J. Comput. Phys.* **176**, 430; Kassam &
Trefethen (2005) *SIAM J. Sci. Comput.* **26**, 1214.

### 5. Physically-grounded forcing: fix the spectral shape and correlation time
**What.** The stochastic forcing is white-in-time (correct $\sqrt{\Delta t}$
scaling) but has an arbitrary per-mode amplitude and injects at a fixed rate that
is only loosely tied to a target energy-injection rate $\varepsilon$. Replace it
with (a) an isotropic ring forcing of controlled $\varepsilon$, and/or (b) a
finite correlation time (Ornstein–Uhlenbeck) so the forcing is not delta-correlated.
**Why.** $\varepsilon$ sets the frictional arrest scale $k_{fr}\sim(\beta^3/\varepsilon)^{1/5}$
and hence the jet spacing; controlling it makes the jet regime *predictable* and
comparable to the theory in item 1. Coloured forcing changes the injection
statistics and removes an unphysical white-noise artefact.
**Difficulty.** Easy–Medium.
**Reference.** Maltrud & Vallis (1991) *J. Fluid Mech.* **228**, 321;
Constantinou, Farrell & Ioannou (2014) *J. Atmos. Sci.* **71**, 1818.

### 6. Increase resolution (T170–T255) with a verified dealiasing rule
**What.** T127 gives a thin inertial range between forcing ($l\!\sim\!70$) and the
∇⁸ cutoff. Going to T170 or T255 opens a wider forward-enstrophy range and lets
the forcing move to higher $l$ (item 1) while keeping the Rhines scale resolved.
Confirm the transform still satisfies the 2/3 (Orszag) dealiasing rule at the new
truncation.
**Why.** A wider inertial range is what makes the $k^{-3}$ enstrophy range and the
inverse-cascade range genuinely *ranges* rather than a couple of decades of
octaves. Necessary companion to item 1.
**Difficulty.** Easy (parameter) but costs CPU ∝ $L^3$.
**Reference.** Orszag (1971) *J. Atmos. Sci.* **28**, 1074; Boyd, *Chebyshev and
Fourier Spectral Methods*, 2nd ed. (Dover, 2001).

### 7. Better small-scale sink: spectral vanishing viscosity or tuned hyper-order
**What.** ∇⁸ hyperviscosity currently absorbs ~10 % of the *energy* (it should
predominantly drain enstrophy). Replace or supplement it with a spectral
vanishing viscosity (SVV) or a smooth exponential spectral filter that is inert
over the inertial range and bites only near truncation, and check the
enstrophy-dominated draining.
**Why.** Cleaner scale-selectivity means the inverse cascade is not artificially
damped, the energy budget closes, and the inertial-range slope is not
contaminated by a bottleneck at the cutoff.
**Difficulty.** Medium.
**Reference.** Tadmor (1989) *SIAM J. Numer. Anal.* **26**, 30 (SVV); Jablonowski
& Williamson (2011), in *Numerical Techniques for Global Atmospheric Models*
(Springer).

### 8. Measure spectral **flux**, not just spectral slope
**What.** Compute the spectral energy flux $\Pi_E(l)$ and enstrophy flux
$\Pi_Z(l)$ from the nonlinear transfer term, and plot them. A broad spectrum is
*consistent with* a dual cascade but does not *prove* one; a negative $\Pi_E$
below the forcing band and positive $\Pi_Z$ above it is the direct evidence.
**Why.** This is the rigorous diagnostic that distinguishes a genuine inverse
energy cascade from a merely broadened spectrum — exactly the claim the project
needs to substantiate before asserting jets are "almost" forming.
**Difficulty.** Medium (needs the triad transfer term; the Jacobian is already
in hand).
**Reference.** Boffetta & Ecke (2012) *Annu. Rev. Fluid Mech.* **44**, 427;
Frisch, *Turbulence* (Cambridge, 1995).

---

## Tier B — Extend the physics (new terms / new equation sets)

### 9. Impose a real differential-rotation mean flow instead of a cosmetic twist
**What.** The cutaway's 25° longitude "shear" is a pure visualization rotation,
not dynamics. Instead relax the zonal-mean flow toward a prescribed
differential-rotation profile $\bar u(\varphi)$ (Newtonian relaxation, as in a
Held–Suarez core), so the mean shear is an actual term in the evolution.
**Why.** Gives a self-consistent mean flow that can be barotropically unstable and
seed eddies/jets from the shear itself, replacing an arbitrary graphic with
physics.
**Difficulty.** Medium.
**Reference.** Held & Suarez (1994) *Bull. Amer. Meteorol. Soc.* **75**, 1825;
Vallis, *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. (Cambridge, 2017).

### 10. Topographic β / bottom-relief vorticity term
**What.** Add a fixed "topography" $h(\theta,\varphi)$ to the potential vorticity,
$q=\omega+f+f_0 h/H$, so a stationary vorticity-gradient pattern interacts with
the flow. On the sphere this is a prescribed source in the PV.
**Why.** Topographic Rossby waves and form drag anchor jets and change the
inverse-cascade anisotropy — a well-studied route to standing eddies and locked
jets, and a cheap way to break the artificial zonal symmetry.
**Difficulty.** Medium.
**Reference.** Vallis & Maltrud (1993) *J. Phys. Oceanogr.* **23**, 1346; Vallis,
*Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. (2017), §14.

### 11. Equivalent-barotropic model: add a finite deformation radius
**What.** Promote $q=\nabla^2\psi+f$ to $q=\nabla^2\psi-\psi/L_d^2+f$ (one extra
diagonal term in spectral space). This introduces an intrinsic length scale
$L_d$, the Rossby radius.
**Why.** With no $L_d$, the barotropic model has no dynamical length between
forcing and planetary scales; adding it produces far more realistic jet widths
and halts the inverse cascade at $L_d$ as well as at the Rhines scale. Minimal
code change, large physical gain.
**Difficulty.** Easy–Medium.
**Reference.** Rhines (1975) *J. Fluid Mech.* **69**, 417; Vallis (2017), §9.

### 12. Two-layer quasi-geostrophic model (baroclinic instability)
**What.** Two coupled QG PV equations with a mean vertical shear. The energy
source becomes **baroclinic instability** of the imposed shear, not artificial
stochastic forcing.
**Why.** This is the minimal model in which jets arise *self-consistently* from an
internal energy source (available potential energy → eddies → jets), rather than
being hand-forced. It is the standard theoretical laboratory for planetary jets
and a genuine physics upgrade over any single-layer forcing scheme.
**Difficulty.** Hard.
**Reference.** Phillips (1954) *Tellus* **6**, 273; Panetta (1993) *J. Atmos.
Sci.* **50**, 2073; Salmon, *Lectures on Geophysical Fluid Dynamics* (Oxford,
1998).

### 13. Shallow-water equations on the sphere (add divergence & gravity waves)
**What.** Advance from one vorticity equation to the rotating shallow-water
system (mass + two momentum components). This restores horizontal divergence,
gravity waves, and geostrophic adjustment.
**Why.** The barotropic model is the non-divergent limit; shallow water is the
next rung and captures wave–mean-flow interaction, gravity-wave radiation, and a
free surface — all absent now. Well-supplied with standard test cases for
verification.
**Difficulty.** Hard.
**Reference.** Williamson et al. (1992) *J. Comput. Phys.* **102**, 211 (test
suite); Galewsky, Scott & Polvani (2004) *Tellus A* **56**, 429.

### 14. Magnetohydrodynamic extension (β-plane / shallow-water MHD)
**What.** Add a magnetic field and the Lorentz force — e.g. 2-D MHD or
shallow-water MHD on the sphere — introducing Alfvén waves and magnetic tension.
**Why.** This is the physically correct direction for the *solar* framing: in the
tachocline a toroidal field suppresses the inverse cascade and can quench or
reorganise jets. It connects the toy model to real solar-interior dynamics.
**Difficulty.** Hard.
**Reference.** Tobias, Diamond & Hughes (2007) *Astrophys. J.* **667**, L113;
Gilman (2000) *Astrophys. J.* **544**, L79 (shallow-water MHD).

---

## Tier C — Toward actual convection (major projects)

### 15. Boussinesq Rayleigh–Bénard convection in a rotating spherical shell
**What.** The minimal *true* convection model: 3-D Navier–Stokes under the
Boussinesq approximation, coupled to a temperature equation with a
buoyancy force, in a shell $r_i<r<r_o$ heated from below/within. Control
parameters: Rayleigh, Ekman, Prandtl numbers.
**Why.** This is the smallest step that makes the project *convection* rather than
2-D turbulence: it has buoyancy, an energy equation, vertical velocity, and
self-organised convective cells and (via rotation) real differential rotation
and jets — the phenomena the visualization currently only mimics.
**Difficulty.** Major.
**Reference.** Christensen & Aubert (2006) *Geophys. J. Int.* **166**, 97;
Gastine, Wicht & Aurnou (2013) *Icarus* **225**, 156.

### 16. Anelastic convection in a stratified shell (correct for the Sun)
**What.** Replace Boussinesq with the **anelastic** approximation: filter sound
waves but retain a strongly varying background density over many pressure scale
heights, as in the solar convection zone ($\rho$ varies by $\sim10^6$).
**Why.** Boussinesq is invalid for the Sun's deep, highly stratified envelope;
anelastic is the standard, physically correct reduced model and captures
up/down-flow asymmetry, plumes, and the density-stratified energy transport that
set the real convective spectrum.
**Difficulty.** Major.
**Reference.** Gough (1969) *J. Atmos. Sci.* **26**, 448; Braginsky & Roberts
(1995) *Geophys. Astrophys. Fluid Dyn.* **79**, 1; Jones et al. (2011) *Icarus*
**216**, 120 (anelastic benchmark).

### 17. Adopt an established 3-D spherical solver (Rayleigh / Dedalus / MagIC)
**What.** Rather than hand-building 3-D convection, use a validated
community code: **Rayleigh** (pseudo-spectral anelastic dynamo/convection),
**Dedalus** (general spectral PDE framework, spherical bases), or **MagIC**.
**Why.** These codes are benchmarked, parallelised, and remove years of
solver-development and verification risk. Dedalus in particular can express the
full anelastic or Boussinesq shell system in a few dozen lines. This is the
pragmatic path from "toy" to "research-grade."
**Difficulty.** Major (learning curve + HPC), but far less than writing one.
**Reference.** Featherstone & Hindman (2016) *Astrophys. J.* **818**, 32
(Rayleigh); Burns et al. (2020) *Phys. Rev. Research* **2**, 023068 (Dedalus).

### 18. Couple to a realistic solar thermal/entropy stratification
**What.** Drive the convection (in item 15/16) with a background state taken from
a standard solar model (temperature, density, entropy, gravity vs. radius from
Model S / MESA), rather than a constant or ad-hoc profile.
**Why.** The convective length scales, velocities, and super-adiabaticity are set
by the real stratification; using Model S makes the simulation quantitatively
comparable to helioseismology and to the actual solar convection zone.
**Difficulty.** Major (only meaningful on top of items 16–17).
**Reference.** Christensen-Dalsgaard et al. (1996) *Science* **272**, 1286
(Model S); Miesch (2005) *Living Rev. Solar Phys.* **2**, 1.

---

## Tier D — Visualization & analysis (evidence, not decoration)

### 19. Lagrangian particle tracking / FTLE for transport structure
**What.** Advect passive tracers with the (verified-correct) velocity field and
compute finite-time Lyapunov exponents to extract Lagrangian coherent structures.
**Why.** Jets are transport *barriers*; FTLE ridges reveal mixing regions and the
barrier at each jet core directly, providing a dynamical diagnostic that a
snapshot of vorticity cannot. It turns the pretty movie into evidence about
mixing and coherent structures.
**Difficulty.** Medium.
**Reference.** Haller (2015) *Annu. Rev. Fluid Mech.* **47**, 137.

### 20. Honest interior: solve a radial structure, or stop rendering a fake one
**What.** The cutaway interior is a mixing-length-scaled continuation
$(r/R)^{l/L_{\rm ref}}$ that overstates penetration ~10× versus the true
$(r/R)^l$ potential continuation, and the inner core uses a disconnected colour
scale ~32× hotter. Either (a) drop the fabricated interior and show only the 2-D
surface field honestly, (b) render the *true* $(r/R)^l$ evanescent continuation,
or (c) obtain a real 3-D field from items 15–17 and visualise *that*.
**Why.** A 2-D barotropic model carries **no** radial information; painting a
plausible interior is scientifically misleading regardless of how it is labelled.
Aligning the visualization with the model's actual content (or upgrading the
model) is a correctness fix, not a cosmetic one.
**Difficulty.** Easy (options a/b) to Major (option c).
**Reference.** Project audits `physics_audit.md` and `v6_critical_audit.md` §4.4,
§5.1; for a true 3-D field see items 15–17.

---

## Summary ranking rationale

- **1–8 (Tier A)** are the highest-value moves *within the current framework*:
  items 1, 2, 6 and 8 together decide whether the model demonstrably produces
  jets and a dual cascade — the project's own stated goal — while 3, 4, 5, 7
  fix numerical honesty (order of accuracy, budget closure, forcing physics).
- **9–14 (Tier B)** add new physics in increasing order of ambition, from a one-
  term deformation-radius change (11) to self-consistent baroclinic jets (12),
  gravity waves (13), and the solar-relevant MHD route (14).
- **15–18 (Tier C)** are the only entries that make the project *actually
  convection*; they are major but are the honest destination if "solar
  convection" is to be more than a label.
- **19–20 (Tier D)** convert the output from illustration into evidence and
  remove the one remaining scientifically misleading element (the fabricated
  interior).

Two cross-cutting truths: (i) nothing here is blocked by the solver — the
numerics are correct; the gaps are *physical scope* and *run discipline*. (ii)
The single cheapest change with the largest scientific payoff is **items 1 + 2 +
8 together**: run long enough, with wider scale separation, and *measure the
flux* — that alone would let the project make a defensible claim about jets.
