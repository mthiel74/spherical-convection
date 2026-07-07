# Spherical Convection

Simulates rotating convection in a spherical shell and renders an animated
visualisation of the z-component of vorticity (ω′_z).

## Physics

Solves the **barotropic vorticity equation** on a rotating sphere using
spectral methods (real spherical harmonics via `pyshtools`):

```
∂ω/∂t + J(ψ, ω + f) = −ν (−∇²)^4 ω + F
```

where
- ω = ∇²ψ is the relative vorticity
- f = 2Ω sin φ is the planetary vorticity (Coriolis)
- J is the Jacobian (advection)
- ν is the hyperviscosity coefficient
- F is stochastic forcing injecting energy at convective scales (l ≈ 8–20)

Rapid rotation (high Ω) enforces the **Taylor-Proudman constraint**, producing
elongated "banana cell" structures aligned with the rotation axis near the
equator, and more isotropic turbulence near the poles.

## Visualisation

A 3-D sphere with a **quarter-wedge cutaway** reveals:
- Outer spherical surface coloured by ω′_z
- Equatorial cross-section (z = 0 plane)
- Meridional cross-section (two exposed faces of the wedge)

Interior fields are approximated from the surface field using the
Taylor-Proudman columnar-flow assumption.

Red = cyclonic (positive ω′_z), blue = anticyclonic (negative), white = zero.

## Quick start

```bash
pip install -r requirements.txt
python render_movie.py          # produces output.gif
python render_movie.py --mp4    # also produces output.mp4
```

## File layout

| File | Purpose |
|------|---------|
| `config.py` | Physical and numerical parameters |
| `simulate.py` | Spectral vorticity solver |
| `visualize.py` | 3-D sphere rendering with cutaway |
| `render_movie.py` | Pipeline: simulate → render → save |
| `requirements.txt` | Python dependencies |

## Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OMEGA` | 10.0 | Non-dimensional rotation rate |
| `LMAX` | 42 | Spectral truncation (T42) |
| `NU_HYPER` | 1e-12 | Hyperviscosity coefficient |
| `FORCE_LMIN/MAX` | 8/20 | Forcing band (spherical harmonic degree) |
| `N_SPINUP` | 4000 | Spin-up steps before recording |
| `N_FRAMES` | 250 | Frames in the animation |
| `DT` | 5e-4 | Non-dimensional timestep |
