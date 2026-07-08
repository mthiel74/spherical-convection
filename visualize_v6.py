"""
visualize_v6.py — 3-D spherical-shell renderer with an octant cutaway, using an
HONEST interior reconstruction.

The v5 renderer faked the interior: the cross-section faces were a ξ⁴-decayed,
longitude-sheared copy of the surface field with sinusoidal "bands", and the
inner core was an unrelated static random field (physics_audit.md §18–19).

v6 replaces both with a radial eigenfunction reconstruction.  A harmonic mode
of degree l continues inward through a spherical shell as (r/R)^l (the regular
solid harmonic, ∇²(r^l Y_lm)=0, finite at the origin).  We use the
mixing-length-scaled form

        ω(r, θ, φ) = Σ_{l,m}  ω_lm · (r/R_outer)^(l/L_REF) · Y_lm(θ, φ).

Why the L_REF rescaling (see config_v6): the pure form (L_REF=1) is exact but
evanescent — for THIS field, whose power sits at l≳20 with almost none at l<15
(the inverse cascade arrests near the Rhines degree), (r/R)^l pushes >85% of the
amplitude into the top 5% of the shell and shows an almost empty interior.
L_REF ≈ π/D ≈ 10.8 (shell depth D=0.29) is the degree whose half-wavelength
spans the shell; rescaling l→l/L_REF makes penetration depth ∝ horizontal
wavelength (as convective cells stay coherent over ~a mixing length).  It is an
illustrative reconstruction, honestly labelled — NOT solved interior dynamics.

Consequences, all physical in character rather than cosmetic:

  • large-scale (low-l) structures reach DEEP into the shell; fine filaments
    (high-l) stay surface-confined — the depth ordering follows from the
    eigenfunction, not from an ad-hoc decay law;
  • at r = R_outer the factor is 1, so the cut faces join the coloured surface
    seamlessly (no shear needed);
  • the region below R_inner = 0.71 R_☉ is the STABLE radiative interior (no
    convection).  We paint the inner core with only the large-scale (l ≤ CORE_LMAX)
    part of the same field, continued to r = R_inner — a calm, honest tint, not
    fabricated convection.

Everything that must occlude everything else (outer surface, inner core, the
three cut faces) is emitted into a single Poly3DCollection so matplotlib
depth-sorts all polygons together.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
import pyshtools as pysh

from config_v6 import (IMG_SIZE, LMAX, R_INNER, R_OUTER, R_MID,
                       CUTAWAY_LON_START, CUTAWAY_LON_END,
                       N_RADIAL, N_ANG, SURFACE_DS, CORE_LMAX, L_REF,
                       VIEW_ELEV, VIEW_AZIM)

# ── colormap ────────────────────────────────────────────────────────────────
CMAP = plt.cm.RdBu_r                     # red = +ω_z, blue = −ω_z


def _to_rgba(vals, vmax):
    """Field values → RGBA via the diverging colormap, clamped to ±vmax."""
    normed = (np.clip(vals / (vmax + 1e-12), -1.0, 1.0) + 1.0) / 2.0
    return CMAP(normed)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed spherical-harmonic design matrices for the interior faces.
#
# Each cut face is a (radius × angle) mesh.  The ANGULAR points are fixed, so we
# evaluate Y_lm there ONCE, then per frame apply the radial factor (r/R)^l and a
# single matrix product.  This makes the eigenfunction reconstruction cheap.
# ─────────────────────────────────────────────────────────────────────────────

_R_FACE = np.linspace(R_INNER, R_OUTER, N_RADIAL + 1)      # shared radial nodes
# mixing-length-scaled radial eigenfunction factor  (r/R)^(l/L_REF)   (Nr,L+1)
_RADFAC = (_R_FACE[:, None] / R_OUTER) ** (np.arange(LMAX + 1)[None, :] / L_REF)

_YCACHE = {}     # geometry-only design matrices, built lazily


def _ymat(theta_deg, phi_deg):
    """
    Design matrix Y for a set of angular points, shape (P, ncoef) with the
    coefficient axis flattened to match a (2, L+1, L+1) array.  theta is
    colatitude, phi longitude, both in degrees.
    """
    P = len(theta_deg)
    Y = np.empty((P, 2 * (LMAX + 1) * (LMAX + 1)))
    for p in range(P):
        ylm = pysh.expand.spharm(LMAX, theta_deg[p], phi_deg[p],
                                 normalization='4pi', csphase=1, degrees=True)
        Y[p] = ylm.reshape(-1)
    return Y


def _face_field(coeffs, ymat):
    """
    Reconstruct ω on a (radius × angle) face by the radial eigenfunction:

        field[p, i] = Σ_lm coeffs_lm · (r_i/R)^(l/L_REF) · Y_lm(angle_p).

    Returns an (P, Nr) array aligned with a meshgrid(r, angle) of shape
    (P=len(angle), Nr=len(r)).
    """
    L = LMAX
    # scaled coefficients per radius: (Nr, 2, L+1, L+1)
    scaled = coeffs[None] * _RADFAC[:, None, :, None]     # broadcast (r/R)^l over m
    scaled = scaled.reshape(_RADFAC.shape[0], -1)         # (Nr, ncoef)
    field = ymat @ scaled.T                               # (P, Nr)
    return field


# ── outer-surface sampler (bilinear, longitude-periodic) — from the grid field ─
def latlon_grid():
    dummy = pysh.SHCoeffs.from_zeros(LMAX, normalization='4pi')
    g = dummy.expand(grid='DH2')
    return g.lats(), g.lons()


def coeffs_to_surface(coeffs):
    """Expand spectral coefficients to the DH2 surface vorticity grid."""
    c = pysh.SHCoeffs.from_array(coeffs, normalization='4pi', csphase=1)
    return c.expand(grid='DH2').data


def make_sampler(surface_field, lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    order = np.argsort(lat)
    lat_s = lat[order]
    fld = surface_field[order, :]
    if lon[-1] >= 360.0 - 1e-6:
        lon = lon[:-1]
        fld = fld[:, :-1]
    lon_ext = np.concatenate([lon, [360.0]])
    fld_ext = np.concatenate([fld, fld[:, :1]], axis=1)
    interp = RegularGridInterpolator((lat_s, lon_ext), fld_ext,
                                     bounds_error=False, fill_value=None)

    def sample(lat_q, lon_q):
        la = np.clip(np.asarray(lat_q, float), -90.0, 90.0)
        lo = np.mod(np.asarray(lon_q, float), 360.0)
        return interp(np.stack([la, lo], axis=-1))

    return sample


# ── vectorised mesh → quads ────────────────────────────────────────────────
def _mesh_quads(X, Y, Z):
    P = np.stack([X, Y, Z], axis=-1)
    v0 = P[:-1, :-1]; v1 = P[1:, :-1]
    v2 = P[1:, 1:];   v3 = P[:-1, 1:]
    return np.stack([v0, v1, v2, v3], axis=2).reshape(-1, 4, 3)


def _node_to_face(C):
    return 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[1:, 1:] + C[:-1, 1:])


# ── individual surfaces (each returns verts (K,4,3), rgba (K,4)) ───────────
def _outer_surface(sample, vmax):
    lat = np.linspace(90.0, -90.0, 2 * (LMAX + 1))[::SURFACE_DS]
    lon = np.linspace(0.0, 360.0, 4 * (LMAX + 1) + 1)[::SURFACE_DS]
    LO, LA = np.meshgrid(lon, lat)
    phi = np.radians(LA); lam = np.radians(LO)
    X = R_OUTER * np.cos(phi) * np.cos(lam)
    Y = R_OUTER * np.cos(phi) * np.sin(lam)
    Z = R_OUTER * np.sin(phi)

    field = sample(LA, LO)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)

    loc = _node_to_face(LO); lac = _node_to_face(LA)
    removed = (loc >= CUTAWAY_LON_START) & (loc < CUTAWAY_LON_END) & (lac > 0.0)
    keep = ~removed.reshape(-1)
    return verts[keep], rgba[keep]


def _equatorial_face(coeffs, vmax):
    """z = 0 quarter-annulus under the removed wedge (lat = 0, lon 0…90)."""
    lam_deg = np.linspace(CUTAWAY_LON_START, CUTAWAY_LON_END, N_ANG + 1)
    key = ('eq', tuple(np.round(lam_deg, 6)))
    if key not in _YCACHE:
        _YCACHE[key] = _ymat(np.full_like(lam_deg, 90.0), lam_deg)   # colat=90
    field = _face_field(coeffs, _YCACHE[key])                        # (P, Nr)

    R, LAM = np.meshgrid(_R_FACE, np.radians(lam_deg))
    X = R * np.cos(LAM); Y = R * np.sin(LAM); Z = np.zeros_like(R)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)
    return verts, rgba


def _meridional_face(coeffs, vmax, lon_deg):
    """Upper-half meridional wall at fixed longitude (lat 0…90)."""
    lat_deg = np.linspace(0.0, 90.0, N_ANG + 1)
    key = ('mer', lon_deg, tuple(np.round(lat_deg, 6)))
    if key not in _YCACHE:
        _YCACHE[key] = _ymat(90.0 - lat_deg, np.full_like(lat_deg, lon_deg))
    field = _face_field(coeffs, _YCACHE[key])                        # (P, Nr)

    R, LA = np.meshgrid(_R_FACE, lat_deg)
    phi = np.radians(LA); lam = np.radians(lon_deg)
    X = R * np.cos(phi) * np.cos(lam)
    Y = R * np.cos(phi) * np.sin(lam)
    Z = R * np.sin(phi)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)
    return verts, rgba


def _inner_core(coeffs, cam, vmax):
    """
    Inner-core sphere (radius R_INNER) = the STABLE radiative interior.  Painted
    with only the large-scale (l ≤ CORE_LMAX) part of the SAME field, continued
    to r = R_INNER by the eigenfunction factor (R_INNER/R_OUTER)^(l/L_REF).  This
    is the large-scale flow that penetrates to the base — a calm tint, honestly
    the only thing a 2-D surface field can say about the deep interior.  Lambert
    shading is kept for 3-D form.
    """
    Lc = CORE_LMAX
    core = np.zeros_like(coeffs)
    fac = (R_INNER / R_OUTER) ** (np.arange(Lc + 1) / L_REF)
    core[:, :Lc + 1, :Lc + 1] = coeffs[:, :Lc + 1, :Lc + 1] * fac[None, :, None]
    cc = pysh.SHCoeffs.from_array(core[:, :Lc + 1, :Lc + 1],
                                  normalization='4pi', csphase=1)
    g = cc.expand(grid='DH2')
    csample = make_sampler(g.data, g.lats(), g.lons())

    u = np.linspace(0, 2 * np.pi, 97)
    v = np.linspace(0, np.pi, 49)
    U, V = np.meshgrid(u, v)
    nx = np.sin(V) * np.cos(U); ny = np.sin(V) * np.sin(U); nz = np.cos(V)
    X = R_INNER * nx; Y = R_INNER * ny; Z = R_INNER * nz

    lat = np.degrees(np.arcsin(np.clip(nz, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(ny, nx))
    field = _node_to_face(csample(lat, lon))
    rgb = _to_rgba(field, vmax)[:, :, :3]

    ndot = _node_to_face(nx) * cam[0] + _node_to_face(ny) * cam[1] \
        + _node_to_face(nz) * cam[2]
    shade = 0.55 + 0.45 * np.clip(ndot, 0.0, 1.0)
    rgb = rgb * shade[:, :, None]
    rgba = np.concatenate([rgb, np.ones((*shade.shape, 1))], axis=-1)

    verts = _mesh_quads(X, Y, Z)
    return verts, rgba.reshape(-1, 4)


# ── boundary curves on the three cut planes ───────────────────────────────
def _arc_equator(radius, n=80):
    lam = np.radians(np.linspace(CUTAWAY_LON_START, CUTAWAY_LON_END, n))
    return radius * np.cos(lam), radius * np.sin(lam), np.zeros(n)


def _arc_meridian(radius, lon_deg, n=80):
    phi = np.radians(np.linspace(0.0, 90.0, n))
    lam = np.radians(lon_deg)
    return (radius * np.cos(phi) * np.cos(lam),
            radius * np.cos(phi) * np.sin(lam),
            radius * np.sin(phi))


def _draw_boundaries(ax):
    zt = 12
    for xyz in (_arc_equator(R_INNER),
                _arc_meridian(R_INNER, CUTAWAY_LON_START),
                _arc_meridian(R_INNER, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='black', lw=2.1, zorder=zt)
    for xyz in (_arc_equator(R_OUTER),
                _arc_meridian(R_OUTER, CUTAWAY_LON_START),
                _arc_meridian(R_OUTER, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='0.1', lw=1.1, zorder=zt)
    for xyz in (_arc_equator(R_MID),
                _arc_meridian(R_MID, CUTAWAY_LON_START),
                _arc_meridian(R_MID, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='0.35', lw=0.7, zorder=zt)
    for lon_d in (CUTAWAY_LON_START, CUTAWAY_LON_END):
        lam = np.radians(lon_d)
        ax.plot([R_INNER * np.cos(lam), R_OUTER * np.cos(lam)],
                [R_INNER * np.sin(lam), R_OUTER * np.sin(lam)],
                [0, 0], color='0.1', lw=1.0, zorder=zt)
    ax.plot([0, 0], [0, 0], [R_INNER, R_OUTER], color='0.1', lw=1.0, zorder=zt)


# ── latitude / longitude graticule (solid = visible, dashed = hidden) ──────
def _draw_graticule(ax, cam):
    n = 400
    solid = dict(color='0.12', lw=0.55, ls='-', zorder=11)
    dash = dict(color='0.45', lw=0.4, ls=(0, (3, 3)), alpha=0.28, zorder=10)

    def emit(x, y, z):
        pos = np.stack([x, y, z], axis=1)
        lon_d = np.degrees(np.arctan2(y, x)) % 360.0
        removed = (lon_d >= CUTAWAY_LON_START) & (lon_d < CUTAWAY_LON_END) & (z > 0)
        visible = (pos @ cam) > 0.0
        state = np.where(removed, 0, np.where(visible, 1, 2))
        i = 0
        while i < len(state):
            s = state[i]; j = i
            while j < len(state) and state[j] == s:
                j += 1
            if s and j - i > 1:
                seg = slice(i, j)
                ax.plot(x[seg], y[seg], z[seg], **(solid if s == 1 else dash))
            i = j

    for lat_line in range(-60, 90, 30):
        phi = np.radians(lat_line)
        lam = np.linspace(0, 2 * np.pi, n)
        emit(R_OUTER * np.cos(phi) * np.cos(lam),
             R_OUTER * np.cos(phi) * np.sin(lam),
             R_OUTER * np.sin(phi) * np.ones(n))
    for lon_line in range(0, 360, 30):
        lam = np.radians(lon_line)
        phi = np.linspace(-np.pi / 2, np.pi / 2, n)
        emit(R_OUTER * np.cos(phi) * np.cos(lam),
             R_OUTER * np.cos(phi) * np.sin(lam),
             R_OUTER * np.sin(phi))


# ── main render function ──────────────────────────────────────────────────
def render_frame(coeffs, frame_idx, total_frames, t_val, figsize_px=None):
    if figsize_px is None:
        figsize_px = IMG_SIZE
    dpi = 100
    fig = plt.figure(figsize=(figsize_px / dpi, figsize_px / dpi),
                     dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white',
                         computed_zorder=False)

    surface_field = coeffs_to_surface(coeffs)
    vmax = np.percentile(np.abs(surface_field), 97) + 1e-12
    lat, lon = latlon_grid()
    sample = make_sampler(surface_field, lat, lon)

    er, ar = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar),
                    np.sin(er)])

    verts_all, rgba_all = [], []
    for v, c in (_outer_surface(sample, vmax),
                 _equatorial_face(coeffs, vmax),
                 _meridional_face(coeffs, vmax, CUTAWAY_LON_START),
                 _meridional_face(coeffs, vmax, CUTAWAY_LON_END),
                 _inner_core(coeffs, cam, vmax)):
        verts_all.append(v); rgba_all.append(c)
    verts = np.concatenate(verts_all, axis=0)
    rgba = np.concatenate(rgba_all, axis=0)

    pc = Poly3DCollection(verts, facecolors=rgba, edgecolors='none',
                          linewidths=0, shade=False, zsort='average')
    pc.set_zorder(1)
    ax.add_collection3d(pc)

    _draw_boundaries(ax)
    _draw_graticule(ax, cam)
    ax.plot([0, 0], [0, 0], [-1.18, 1.18], color='0.4', lw=0.5,
            ls=(0, (4, 4)), zorder=9)

    lim = 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.0, fraction=0.03)
    cbar.set_label(r"$\omega'_z$", fontsize=11)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.46, 0.94, "2-D barotropic vorticity on a rotating sphere",
             ha='center', fontsize=12.5)
    fig.text(0.46, 0.905,
             rf"$\omega'_z$   (not convection — see README_v6)   t = {t_val:5.1f}",
             ha='center', fontsize=9, color='0.25')
    fig.subplots_adjust(left=-0.02, right=0.92, bottom=-0.02, top=0.92)
    return fig


def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    return arr[:, :, :3]
