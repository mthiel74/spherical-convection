"""
Barotropic vorticity equation on a rotating sphere (spectral method).

∂ω/∂t + J(ψ, ω+f) = −ν (−∇²)^4 ω + F

ω = ∇²ψ  (vorticity = Laplacian of streamfunction)
f = 2Ω sinφ  (planetary vorticity / Coriolis)
J = Jacobian operator
ν = hyperviscosity coefficient
F = stochastic forcing at convective scales
"""

import numpy as np
import pyshtools as pysh

from config import (OMEGA, LMAX, NU_HYPER, FORCE_LMIN, FORCE_LMAX,
                    FORCE_AMP, DT, N_SPINUP, N_FRAMES, FRAME_SKIP)

# ── helpers ────────────────────────────────────────────────────────────────

def _laplacian_eigenvalues(lmax):
    """−l(l+1) for each (l,m) pair, matching pyshtools SHCoeffs layout."""
    ev = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        ev[:, l, :l+1] = -l * (l + 1)
    return ev


def _hyperviscosity_filter(lmax, nu, dt):
    """
    Implicit integrating factor for hyperviscosity: exp(ν λ^4 dt)
    where λ = l(l+1).  Returns multiplicative coefficient array shaped (2,L+1,L+1).
    """
    ev = _laplacian_eigenvalues(lmax)   # negative
    lam4 = ev**4                         # positive (negative^4 = positive)
    return np.exp(-nu * lam4 * dt)


class SpectralVorticity:
    """
    Holds vorticity in spectral space and steps it forward in time.
    Uses real spherical harmonics via pyshtools (normalization='4pi').
    """

    def __init__(self):
        self.lmax = LMAX
        self._ev   = _laplacian_eigenvalues(self.lmax)      # shape (2,L+1,L+1)
        self._visc = _hyperviscosity_filter(self.lmax, NU_HYPER, DT)

        # Inverse Laplacian eigenvalues (ψ = ∇⁻² ω); l=0 mode is 0
        self._inv_ev = np.zeros_like(self._ev)
        for l in range(1, self.lmax + 1):
            self._inv_ev[:, l, :l+1] = -1.0 / (l * (l + 1))

        # Planetary vorticity f = 2Ω sinφ  (only l=1,m=0 in 4π-normalised SH)
        # Y_1^0 (4π-normalised) = sqrt(3/4π) cosθ = sqrt(3/4π) sinφ
        # So f = 2Ω sinφ → coefficient = 2Ω / sqrt(3/4π) = 2Ω * sqrt(4π/3)
        self._f_lm = np.zeros((2, self.lmax + 1, self.lmax + 1))
        self._f_lm[0, 1, 0] = 2.0 * OMEGA * np.sqrt(4.0 * np.pi / 3.0)

        # Initialise vorticity with small random noise
        rng = np.random.default_rng(42)
        omega_lm = np.zeros((2, self.lmax + 1, self.lmax + 1))
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            for m in range(l + 1):
                amp = FORCE_AMP * 0.1 / (l + 1)
                omega_lm[0, l, m] = rng.standard_normal() * amp
                if m > 0:
                    omega_lm[1, l, m] = rng.standard_normal() * amp
        self.omega_lm = omega_lm

    # ── spectral ↔ grid conversions ─────────────────────────────────────

    def _to_grid(self, clm_array):
        """(2,L+1,L+1) numpy array → SHGrid (Driscoll-Healy)."""
        coeffs = pysh.SHCoeffs.from_array(clm_array, normalization='4pi', csphase=1)
        return coeffs.expand(grid='DH2')

    def _to_lm(self, grid):
        """SHGrid → (2,L+1,L+1) numpy array."""
        coeffs = grid.expand(normalization='4pi', csphase=1, lmax_calc=self.lmax)
        return coeffs.coeffs

    # ── Jacobian J(A,B) = ∂A/∂φ ∂B/∂λ − ∂A/∂λ ∂B/∂φ in spectral space ─

    def _jacobian_lm(self, a_lm, b_lm):
        """
        Compute J(A,B) spectrally using the gradient approach.
        ∂A/∂λ, ∂A/∂φ computed via horizontal gradient in pyshtools.
        """
        ca = pysh.SHCoeffs.from_array(a_lm, normalization='4pi', csphase=1)
        cb = pysh.SHCoeffs.from_array(b_lm, normalization='4pi', csphase=1)

        # Gradient on unit sphere
        ga = ca.gradient(radius=1.0)
        gb = cb.gradient(radius=1.0)

        # ga, gb are SHGravTensor or SHGradient objects;
        # .theta is ∂/∂θ component (colatitude), .phi is ∂/∂λ component
        # J(A,B) = (∂A/∂λ)(∂B/∂θ) − (∂A/∂θ)(∂B/∂λ)   [on unit sphere]
        # All in grid space
        dA_lam = ga.phi.data   # ∂A/∂λ
        dA_th  = ga.theta.data # ∂A/∂θ
        dB_lam = gb.phi.data
        dB_th  = gb.theta.data

        jac_grid_data = dA_lam * dB_th - dA_th * dB_lam

        # Back to spectral
        jac_grid = pysh.SHGrid.from_array(jac_grid_data, grid='DH')
        return self._to_lm(jac_grid)

    # ── forcing ─────────────────────────────────────────────────────────

    def _stochastic_forcing(self, rng):
        """
        Add random forcing at convective scales (FORCE_LMIN ≤ l ≤ FORCE_LMAX).
        Energy is normalised so each band contributes roughly equally.
        """
        f_lm = np.zeros_like(self.omega_lm)
        for l in range(FORCE_LMIN, FORCE_LMAX + 1):
            for m in range(l + 1):
                amp = FORCE_AMP / np.sqrt(l * (l + 1))
                f_lm[0, l, m] = rng.standard_normal() * amp * np.sqrt(DT)
                if m > 0:
                    f_lm[1, l, m] = rng.standard_normal() * amp * np.sqrt(DT)
        return f_lm

    # ── time step (RK2 / explicit) ───────────────────────────────────────

    def _tendency(self, omega_lm):
        """Compute dω/dt (spectral), excluding hyperviscosity."""
        # streamfunction
        psi_lm = self._inv_ev * omega_lm

        # absolute vorticity in spectral space
        abs_vor_lm = omega_lm + self._f_lm

        # Jacobian of (ψ, ω+f) in spectral space
        jac_lm = self._jacobian_lm(psi_lm, abs_vor_lm)

        return -jac_lm   # dω/dt = -J(ψ, ω+f)

    def step(self, rng):
        """Advance one timestep with RK2 + implicit hyperviscosity."""
        k1 = self._tendency(self.omega_lm)
        k2 = self._tendency(self.omega_lm + DT * k1)
        rhs = self.omega_lm + 0.5 * DT * (k1 + k2)

        # Add stochastic forcing
        rhs += self._stochastic_forcing(rng)

        # Implicit hyperviscosity damping
        self.omega_lm = self._visc * rhs

        # Zero out l=0 (mean vorticity stays 0)
        self.omega_lm[:, 0, 0] = 0.0

    # ── output ───────────────────────────────────────────────────────────

    def vorticity_grid(self):
        """Return vorticity on the DH2 lat-lon grid as a 2-D numpy array."""
        grid = self._to_grid(self.omega_lm)
        return grid.data

    def streamfunction_grid(self):
        """Return ψ on the DH2 grid."""
        psi_lm = self._inv_ev * self.omega_lm
        return self._to_grid(psi_lm).data


# ── run simulation ─────────────────────────────────────────────────────────

def run_simulation():
    """
    Spin up, then collect N_FRAMES snapshots of the vorticity field.
    Returns list of 2-D numpy arrays (lat × lon).
    """
    rng = np.random.default_rng(0)
    model = SpectralVorticity()

    print(f"Spinning up for {N_SPINUP} steps …", flush=True)
    for i in range(N_SPINUP):
        model.step(rng)
        if (i + 1) % 500 == 0:
            print(f"  spinup step {i+1}/{N_SPINUP}", flush=True)

    frames = []
    print(f"Recording {N_FRAMES} frames (every {FRAME_SKIP} steps) …", flush=True)
    for i in range(N_FRAMES):
        for _ in range(FRAME_SKIP):
            model.step(rng)
        frames.append(model.vorticity_grid().copy())
        if (i + 1) % 50 == 0:
            rms = np.sqrt(np.mean(frames[-1]**2))
            print(f"  frame {i+1}/{N_FRAMES}  rms_ω={rms:.4f}", flush=True)

    return frames


if __name__ == "__main__":
    frames = run_simulation()
    np.save("frames.npy", np.array(frames))
    print("Saved frames.npy")
