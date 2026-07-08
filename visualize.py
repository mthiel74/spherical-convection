"""
3-D spherical-shell renderer with an octant cutaway.

The cutaway removes the wedge  { 0° ≤ lon < 90°  AND  lat > 0 }  from the
outer sphere.  Removing that octant exposes three interior faces:

    • an equatorial quarter-annulus      (z = 0 plane, r ∈ [R_INNER, R_OUTER])
    • a meridional face at lon = 0°       (upper half, r ∈ [R_INNER, R_OUTER])
    • a meridional face at lon = 90°      (upper half, r ∈ [R_INNER, R_OUTER])

and reveals the inner core (a smaller sphere of radius R_INNER).

The interior vorticity on those faces is reconstructed from the surface field
with a **spherical-shell mapping**: the field at an interior point (r, θ, φ) is
taken from the surface value at the *same* colatitude/longitude (θ, φ) and
modulated radially — a smooth inward decay plus a couple of radial modes that
vanish at both the inner and outer boundaries.  So the structures curve with
the spherical shell (concentric annuli on the equatorial cut, arcs on the
meridional walls) instead of forming vertical columns, and the cross-sections
still join the surface colours seamlessly at the outer rim (decay → 1, modes
→ 0 there).

Everything that must occlude everything else (outer surface, inner core, the
three cut faces) is emitted into a *single* Poly3DCollection so matplotlib
depth-sorts all polygons together — the only reliable way to get correct
occlusion between separate 3-D surfaces in matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator

from config import (IMG_SIZE, LMAX, R_INNER, R_OUTER, R_MID,
                    CUTAWAY_LON_START, CUTAWAY_LON_END,
                    N_RADIAL, N_ANG, SURFACE_DS, VIEW_ELEV, VIEW_AZIM)


# ── colormap ────────────────────────────────────────────────────────────────
CMAP = plt.cm.RdBu_r                     # red = +ω_z, blue = −ω_z

LIGHT_DIR = None                         # set from the camera each frame


def _to_rgba(vals, vmax):
    """Field values → RGBA via the diverging colormap, clamped to ±vmax."""
    normed = (np.clip(vals / (vmax + 1e-12), -1.0, 1.0) + 1.0) / 2.0
    return CMAP(normed)


# ── inner-core convection field (low degree, l = 2…6) ─────────────────────
CORE_LMAX      = 6      # only large-scale, slow modes in the calmer inner region
CORE_AMP       = 0.60   # colour amplitude relative to the surface (paler = calmer)
CORE_DRIFT_DEG = 0.35   # slow westward drift per frame → gentle inner motion
_CORE_CACHE    = {}     # {'sample', 'vmax'} built lazily on first frame


def _core_field():
    """
    Build (once) a static low-degree vorticity field on the core surface to
    stand in for slower, larger-scale convection in the inner region.  Returns
    a (sample, vmax) pair where sample(lat, lon) interpolates the field.
    """
    if _CORE_CACHE:
        return _CORE_CACHE['sample'], _CORE_CACHE['vmax']

    import pyshtools as pysh
    rng = np.random.default_rng(7)
    arr = np.zeros((2, CORE_LMAX + 1, CORE_LMAX + 1))
    for l in range(2, CORE_LMAX + 1):
        for m in range(l + 1):
            amp = 1.0 / l                      # redder tilt toward the largest scales
            arr[0, l, m] = rng.standard_normal() * amp
            if m > 0:
                arr[1, l, m] = rng.standard_normal() * amp
    coeffs = pysh.SHCoeffs.from_array(arr, normalization='4pi', csphase=1)
    g = coeffs.expand(grid='DH2')
    sample = make_sampler(g.data, g.lats(), g.lons())
    vmax = np.percentile(np.abs(g.data), 97) + 1e-12
    _CORE_CACHE['sample'] = sample
    _CORE_CACHE['vmax'] = vmax
    return sample, vmax


# ── grid coordinates ──────────────────────────────────────────────────────
def latlon_grid():
    """(lats, lons) of the DH2 pyshtools grid at LMAX."""
    import pyshtools as pysh
    dummy = pysh.SHCoeffs.from_zeros(LMAX, normalization='4pi')
    g = dummy.expand(grid='DH2')
    return g.lats(), g.lons()


# ── surface-field sampler (bilinear, longitude-periodic) ──────────────────
def make_sampler(surface_field, lat, lon):
    """
    Return sample(lat_q, lon_q) → interpolated surface value.
    Latitudes in degrees [-90, 90], longitudes in degrees (wrapped mod 360).
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    order = np.argsort(lat)
    lat_s = lat[order]
    fld = surface_field[order, :]
    # pyshtools DH grids carry a redundant 360°(=0°) column — drop it, then
    # re-append a single wrap column so 359°→0° interpolates smoothly.
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
        pts = np.stack([la, lo], axis=-1)
        return interp(pts)

    return sample


def spherical_field(sample, X, Y, Z):
    """
    Spherical-shell interior: sample the surface at the SAME colatitude and
    longitude as the interior point (r, θ, φ), then modulate radially.

    The radial modulation is  decay(r) · (1 + modes(r, θ))  where

        decay  = (r / R_OUTER)**2            — smooth inward fade
        modes  = Σ aₖ sin(kπ ξ) · cos(nₖ θ)  — a few radial cells,
                 ξ = (r − R_INNER)/(R_OUTER − R_INNER) ∈ [0, 1]

    Because sin(kπ ξ) vanishes at ξ = 0 and ξ = 1, the modes die at both the
    inner core and the outer rim: at the rim decay → 1 and modes → 0, so the
    face joins the coloured surface seamlessly, while the mid-shell carries
    extra radial structure so the section is not a mere rescaled copy of the
    surface.  Latitude-dependent mode amplitudes keep the cells from looking
    like uniform rings; the pattern follows the spherical geometry.
    """
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r = np.maximum(r, 1e-9)
    lat = np.degrees(np.arcsin(np.clip(Z / r, -1.0, 1.0)))
    lon_p = np.degrees(np.arctan2(Y, X))
    surf = sample(lat, lon_p)

    xi = np.clip((r - R_INNER) / (R_OUTER - R_INNER), 0.0, 1.0)   # 0 … 1
    decay = (r / R_OUTER) ** 2
    rad = np.radians(lat)
    modes = (0.32 * np.sin(2.0 * np.pi * xi) * np.cos(rad)
             + 0.18 * np.sin(3.0 * np.pi * xi) * np.cos(2.0 * rad))
    return surf * decay * (1.0 + modes)


# ── vectorised mesh → quads ────────────────────────────────────────────────
def _mesh_quads(X, Y, Z):
    """(m,n) vertex mesh → (K,4,3) quad-vertex array, K = (m-1)(n-1)."""
    P = np.stack([X, Y, Z], axis=-1)                 # (m,n,3)
    v0 = P[:-1, :-1]; v1 = P[1:, :-1]
    v2 = P[1:, 1:];   v3 = P[:-1, 1:]
    quads = np.stack([v0, v1, v2, v3], axis=2)       # (m-1,n-1,4,3)
    return quads.reshape(-1, 4, 3)


def _node_to_face(C):
    """(m,n) node field → (m-1,n-1) face field (mean of the 4 corners)."""
    return 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[1:, 1:] + C[:-1, 1:])


# ── individual surfaces (each returns verts (K,4,3), rgba (K,4)) ───────────
def _outer_surface(sample, vmax):
    """Coloured outer sphere with the octant hole-punched out."""
    lat = np.linspace(90.0, -90.0, 2 * (LMAX + 1))[::SURFACE_DS]
    lon = np.linspace(0.0, 360.0, 4 * (LMAX + 1) + 1)[::SURFACE_DS]
    LO, LA = np.meshgrid(lon, lat)                   # (m,n)
    phi = np.radians(LA); lam = np.radians(LO)
    X = R_OUTER * np.cos(phi) * np.cos(lam)
    Y = R_OUTER * np.cos(phi) * np.sin(lam)
    Z = R_OUTER * np.sin(phi)

    field = sample(LA, LO)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)

    # drop faces whose centre lies in the removed octant (lon<90 & lat>0)
    loc = _node_to_face(LO); lac = _node_to_face(LA)
    removed = (loc >= CUTAWAY_LON_START) & (loc < CUTAWAY_LON_END) & (lac > 0.0)
    keep = ~removed.reshape(-1)
    return verts[keep], rgba[keep]


def _equatorial_face(sample, vmax):
    """z = 0 quarter-annulus under the removed wedge, r ∈ [R_INNER, R_OUTER]."""
    r = np.linspace(R_INNER, R_OUTER, N_RADIAL + 1)
    lam = np.radians(np.linspace(CUTAWAY_LON_START, CUTAWAY_LON_END, N_ANG + 1))
    R, LAM = np.meshgrid(r, lam)                     # (n_ang+1, n_r+1)
    X = R * np.cos(LAM); Y = R * np.sin(LAM); Z = np.zeros_like(R)

    field = spherical_field(sample, X, Y, Z)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)
    return verts, rgba


def _meridional_face(sample, vmax, lon_deg):
    """Upper-half meridional wall at fixed longitude, r ∈ [R_INNER, R_OUTER]."""
    r = np.linspace(R_INNER, R_OUTER, N_RADIAL + 1)
    lat = np.linspace(0.0, 90.0, N_ANG + 1)
    R, LA = np.meshgrid(r, lat)                      # (n_ang+1, n_r+1)
    phi = np.radians(LA); lam = np.radians(lon_deg)
    X = R * np.cos(phi) * np.cos(lam)
    Y = R * np.cos(phi) * np.sin(lam)
    Z = R * np.sin(phi)

    field = spherical_field(sample, X, Y, Z)
    verts = _mesh_quads(X, Y, Z)
    rgba = _to_rgba(_node_to_face(field), vmax).reshape(-1, 4)
    return verts, rgba


def _inner_core(cam, frame_idx=0):
    """
    Inner-core sphere (radius R_INNER) coloured by a slow, low-degree
    vorticity field — a smaller, calmer version of the outer convection —
    with lambert shading retained for 3-D form.
    """
    u = np.linspace(0, 2 * np.pi, 97)
    v = np.linspace(0, np.pi, 49)
    U, V = np.meshgrid(u, v)
    nx = np.sin(V) * np.cos(U); ny = np.sin(V) * np.sin(U); nz = np.cos(V)
    X = R_INNER * nx; Y = R_INNER * ny; Z = R_INNER * nz

    # sample the low-degree core field (with a slow longitudinal drift for
    # gentle animation), reduced amplitude → paler, calmer colours
    core_sample, core_vmax = _core_field()
    lat = np.degrees(np.arcsin(np.clip(nz, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(ny, nx)) - frame_idx * CORE_DRIFT_DEG
    field = _node_to_face(core_sample(lat, lon))
    rgb = _to_rgba(field, core_vmax / CORE_AMP)[:, :, :3]

    # cheap lambert shading from the camera direction
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
    """Thick black inner-core + rim edges; thin grey intermediate layer."""
    zt = 12
    # inner-core boundary (thick black) on all three planes
    for xyz in (_arc_equator(R_INNER),
                _arc_meridian(R_INNER, CUTAWAY_LON_START),
                _arc_meridian(R_INNER, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='black', lw=2.1, zorder=zt)

    # outer rim of the opening (medium black)
    for xyz in (_arc_equator(R_OUTER),
                _arc_meridian(R_OUTER, CUTAWAY_LON_START),
                _arc_meridian(R_OUTER, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='0.1', lw=1.1, zorder=zt)

    # one intermediate layer boundary (thin grey)
    for xyz in (_arc_equator(R_MID),
                _arc_meridian(R_MID, CUTAWAY_LON_START),
                _arc_meridian(R_MID, CUTAWAY_LON_END)):
        ax.plot(*xyz, color='0.35', lw=0.7, zorder=zt)

    # radial edges framing the opening (equator, both meridians; + polar axis)
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
        # state: 0 skip(removed), 1 solid(visible), 2 dashed(hidden)
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

    for lat_line in range(-60, 90, 30):               # parallels
        phi = np.radians(lat_line)
        lam = np.linspace(0, 2 * np.pi, n)
        emit(R_OUTER * np.cos(phi) * np.cos(lam),
             R_OUTER * np.cos(phi) * np.sin(lam),
             R_OUTER * np.sin(phi) * np.ones(n))
    for lon_line in range(0, 360, 30):                # meridians
        lam = np.radians(lon_line)
        phi = np.linspace(-np.pi / 2, np.pi / 2, n)
        emit(R_OUTER * np.cos(phi) * np.cos(lam),
             R_OUTER * np.cos(phi) * np.sin(lam),
             R_OUTER * np.sin(phi))


# ── main render function ──────────────────────────────────────────────────
def render_frame(surface_field, frame_idx, total_frames, t_val,
                 figsize_px=None):
    if figsize_px is None:
        figsize_px = IMG_SIZE
    dpi = 100
    fig = plt.figure(figsize=(figsize_px / dpi, figsize_px / dpi),
                     dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white',
                         computed_zorder=False)

    vmax = np.percentile(np.abs(surface_field), 97) + 1e-12
    lat, lon = latlon_grid()
    sample = make_sampler(surface_field, lat, lon)

    # camera direction (origin → camera), for shading + graticule visibility
    er, ar = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])

    # ── one combined Poly3DCollection so everything depth-sorts together ──
    verts_all, rgba_all = [], []
    for v, c in (_outer_surface(sample, vmax),
                 _equatorial_face(sample, vmax),
                 _meridional_face(sample, vmax, CUTAWAY_LON_START),
                 _meridional_face(sample, vmax, CUTAWAY_LON_END),
                 _inner_core(cam, frame_idx)):
        verts_all.append(v); rgba_all.append(c)
    verts = np.concatenate(verts_all, axis=0)
    rgba = np.concatenate(rgba_all, axis=0)

    pc = Poly3DCollection(verts, facecolors=rgba, edgecolors='none',
                          linewidths=0, shade=False, zsort='average')
    pc.set_zorder(1)
    ax.add_collection3d(pc)

    # ── crisp boundary curves + graticule on top ─────────────────────────
    _draw_boundaries(ax)
    _draw_graticule(ax, cam)
    ax.plot([0, 0], [0, 0], [-1.18, 1.18], color='0.4', lw=0.5,
            ls=(0, (4, 4)), zorder=9)                 # rotation axis

    # ── view / framing ────────────────────────────────────────────────────
    lim = 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # colorbar + title
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.0, fraction=0.03)
    cbar.set_label(r"$\omega'_z$", fontsize=11)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.46, 0.94, "Rotating spherical-shell convection",
             ha='center', fontsize=13)
    fig.text(0.46, 0.90, rf"$\omega'_z$      t = {t_val:6.1f}",
             ha='center', fontsize=10, color='0.25')
    fig.subplots_adjust(left=-0.02, right=0.92, bottom=-0.02, top=0.92)
    return fig


# ── figure → RGB ─────────────────────────────────────────────────────────
def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    return arr[:, :, :3]
