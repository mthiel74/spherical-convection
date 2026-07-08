# v6 — 2-D barotropic vorticity on a rotating sphere (an honest rebuild)

**This is not a convection simulation.** It is a 2-D forced–dissipative
*barotropic vorticity* model on a rotating sphere — one scalar equation for the
vertical relative vorticity ω_z. It shares only the *rotating-sphere geometry*
and *2-D-turbulence kinematics* with solar convection; it contains none of the
driving or constraining physics of convection. v6 exists to be **scientifically
honest** about exactly that, after an audit (`physics_audit.md`) found v5 both
mislabeled and cosmetically faked.

v6 leaves all v5 files untouched. New files: `config_v6.py`, `simulate_v6.py`,
`visualize_v6.py`, `render_movie_v6.py`, this README. Outputs: `output_v6.mp4`,
`output_v6.gif`.

---

## What it solves

The forced–dissipative barotropic vorticity equation:

    ∂ω/∂t + J(ψ, ω + f) = −ν(−∇²)⁴ ω − μ ω + F

- `ω = ∇²ψ` — relative vorticity (the plotted scalar ω_z); `ψ` streamfunction.
- `J(ψ, ω+f)` — advection of **absolute** vorticity `q = ω + f` by the
  non-divergent horizontal flow `u = ẑ × ∇ψ`. Materially conserves `q` in the
  inviscid, unforced limit.
- `f = 2Ω sinφ` — planetary vorticity (Coriolis); the β-effect is the *only*
  way rotation enters a 2-D model.
- `−ν(−∇²)⁴ω` — ∇⁸ hyperviscosity; a scale-selective small-scale sink that sets
  the filament cutoff.
- `−μω` — uniform linear (Rayleigh) drag; a large-scale energy sink that
  arrests the inverse cascade.
- `F` — stochastic (white-in-time) forcing in a narrow band of degrees.

Numerics (all verified correct in the audit): real 4π-normalised spherical
harmonics (pyshtools); RK2/Heun for advection+forcing; an exact integrating
factor for the linear dissipation; and a metric-consistent spectral Jacobian
built from *physical* horizontal gradient components (avoids the 1/cosφ
singularity of a naive ∂λ,∂φ Jacobian).

## What changed from v5, and why

| | v5 | v6 | why |
|---|---|---|---|
| **Coriolis coefficient** | `2Ω·√(4π/3)` | `2Ω/√3` | v5 used the *orthonormal* value with the *4π* harmonics — too large by exactly √(4π) ≈ 3.545, so the effective rotation was ≈142, not 40. |
| **Forcing band** | l = 24–52 (on the Rhines scale) | l = 60–80 (well above it) | leaves room for an inverse cascade *below* (→ jets/large vortices) and a forward enstrophy cascade *above* (→ filaments). |
| **Linear drag μ** | 0.25 | 0.03 | v5's drag killed the inverse cascade (effective Re ≈ 8, no jets). Weak drag lets it run and arrest at the Rhines scale. |
| **Resolution** | T85 | T127 | resolves the filament (forward enstrophy) range. |
| **Spinup** | 8 000 steps | 22 000 steps | the cascade needs ~1 drag time to develop. |
| **Interior** | ξ⁴-decayed, longitude-sheared copy of the surface + sinusoidal "bands" (cosmetic) | **radial eigenfunction** `(r/R)^l` (physics, see below) | v5's interior was painted on. |
| **Inner core** | unrelated static random l=2–5 field | large-scale (l ≤ 6) part of the *actual* field, continued to r = R_inner | v5's core was noise dressed as calmness. |
| **Time axis** | "rotations" (mis-scaled by 3.545) | non-dimensional time units | honest labelling. |

## The radial reconstruction — why it's physically motivated, not fake

We simulate only a 2-D surface field ω(θ,φ). To draw a cutaway we must say
something about ω *inside* the shell. v6 uses the **correct radial
eigenfunction** rather than a decorative decay:

    ω(r, θ, φ) = Σ_{l,m}  ω_lm · (r/R_outer)^l · Y_lm(θ, φ).

`r^l Y_lm` is the *regular solid harmonic* — the solution of Laplace's equation
∇²(r^l Y_lm) = 0 that stays finite toward the origin. It is the natural radial
structure for a scalar harmonic field (e.g. the streamfunction / potential flow)
in a spherical shell, so continuing each mode inward by `(r/R)^l` is a
mathematically defined reconstruction, not an ad-hoc choice. Its consequences
are exactly what a well-resolved reference shows:

- **low-l (large-scale) structures reach deep** into the shell; **high-l
  filaments stay surface-confined** — because `(r/R)^l` decays faster the larger
  l is. The depth ordering is *derived*, not imposed;
- at `r = R_outer` the factor is 1, so the cut faces join the coloured surface
  **seamlessly** — no longitude shear needed;
- below `R_inner = 0.71 R_☉` (the base of the solar convection zone, from
  helioseismology `r_c/R_☉ = 0.713 ± 0.001`) lies the **stable radiative
  interior — no convection**. We paint the inner core only with the large-scale
  (l ≤ 6) part of the same field, continued to `r = R_inner`; it reads as a calm
  tint, honestly the most a 2-D surface field can say about the deep interior.

**Honest caveat.** This reconstruction is *kinematic geometry*, not a solved 3-D
flow. It shows how the surface modes would continue as harmonic fields; it does
**not** compute interior dynamics, radial velocity, or penetration depth. It is
labelled as a reconstruction, not a simulation.

## What v6 gets right

- **The numerics are sound** — correct barotropic solver, Laplacian inversion,
  ∇⁸ hyperviscosity, exact integrating-factor dissipation, RK2, and a
  metric-consistent spherical Jacobian.
- **The Coriolis coefficient is now correct** (Ω = 40 means Ω = 40).
- **The turbulence is actually developed** — with forcing above the Rhines
  scale and weak drag, the run produces a *broad* enstrophy spectrum (filaments
  via the forward enstrophy cascade) and builds large-scale / zonal structure
  via the inverse cascade, instead of v5's narrow-band blob field. The printed
  diagnostics (enstrophy fractions below/in/above the forcing band, zonal-energy
  fraction, ω_rms, and the drag ratio ω_rms/μ — a nondimensional inverse-drag
  parameter, *not* a Reynolds number) quantify this at run time.
- **The interior is honest** — the exact radial eigenfunction, labelled as a
  reconstruction; a calm radiative core tied to the real large-scale field.

## What it cannot capture (structural, not tunable)

A 2-D barotropic model omits, *by construction*:

- **No radial dimension** ⇒ no depth structure, penetration, overshoot,
  tachocline, or vertical velocity (the entire up/down-flow signal in real
  convection imagery).
- **No buoyancy / energy equation** ⇒ no convective *instability*, no Rayleigh
  number, no heat transport. The flow is **stirred, not driven** — convective
  scales are *imposed* by the forcing band, not *selected* by an instability.
- **No stratification / compressibility** (the Sun spans ~4–5 density scale
  heights).
- **No Taylor–Proudman constraint** (a genuinely 3-D balance) ⇒ real banana
  cells cannot arise from the correct mechanism.
- **No self-consistent differential rotation** (needs correlated 3-D velocity
  components a single scalar ω cannot carry).
- **No magnetism** ⇒ no dynamo, no cycle, no sunspots.

The legitimate physical niche of this model is **Jupiter-like 2-D geostrophic
turbulence** (zonal jets from an arrested inverse cascade), *not* solar
convection.

## To actually model solar convection

You need a **3-D anelastic (MHD) code** solving continuity, momentum (with
buoyancy `ρ̄ g S′/c_p` and the Coriolis force), an entropy/energy equation, and —
for a dynamo — induction, on `(r, θ, φ)`:

- **Rayleigh** — <https://github.com/geodynamics/Rayleigh> (open source,
  N. Featherstone / CIG).
- **ASH** (Anelastic Spherical Harmonic) — Clune, Miesch, Brun, Toomre,
  Glatzmaier (NCAR); the source of the classic radial-velocity giant-cell /
  banana-cell renders.

Even those 3-D LES face the unresolved **"convective conundrum"**
(helioseismology finds deep flows 20–100× weaker than the models predict;
Hanasoge et al. 2012), so they are not "truth" either — but they solve the
*actual governing equations*, which this model does not.

## Running it

```bash
python3 render_movie_v6.py --mp4      # simulate (T127) + render + MP4 + iCloud copy
python3 simulate_v6.py                # just the sim → frames_v6.npz + diagnostics
python3 render_movie_v6.py --from-npz # re-render from a saved frames_v6.npz
```

*References: Christensen-Dalsgaard, Gough & Thompson 1991; Basu & Antia 1997;
Miesch et al. 2008 (arXiv:0707.1460); Featherstone & Hindman 2016
(arXiv:1609.05153); Hanasoge et al. 2012 (PNAS); Rhines 1975. Full audit in
`physics_audit.md`.*
