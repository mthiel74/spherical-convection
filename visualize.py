"""
3-D sphere renderer with cutaway wedge.

Renders the z-component of vorticity on:
  • the outer spherical surface (with cutaway)
  • the exposed equatorial cross-section
  • the exposed meridional cross-section
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import IMG_SIZE, CUTAWAY_FRACTION, N_RADIAL, LMAX


# ── coordinate helpers ────────────────────────────────────────────────────

def latlon_grid():
    """
    Return lat/lon arrays matching the DH2 grid produced by pyshtools
    at the given LMAX.  Shape: (nlat,) and (nlon,).
    """
    import pyshtools as pysh
    import numpy as np
    dummy = pysh.SHCoeffs.from_zeros(LMAX, normalization='4pi')
    g = dummy.expand(grid='DH2')
    return g.lats(), g.lons()


def _sphere_xyz(lat_deg, lon_deg, r=1.0):
    """Lat/lon (degrees) → Cartesian on unit sphere."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    x = r * np.cos(phi) * np.cos(lam)
    y = r * np.cos(phi) * np.sin(lam)
    z = r * np.sin(phi)
    return x, y, z


# ── cutaway mask ──────────────────────────────────────────────────────────

CUTAWAY_LON_START = 0.0
CUTAWAY_LON_END   = CUTAWAY_FRACTION * 360.0   # e.g. 90° for quarter wedge


def _in_cutaway(lon_deg):
    """True where the longitude is inside the removed wedge."""
    lon = lon_deg % 360.0
    return (lon >= CUTAWAY_LON_START) & (lon < CUTAWAY_LON_END)


# ── colormap ──────────────────────────────────────────────────────────────

CMAP = plt.cm.RdBu_r    # red=positive, blue=negative

def _normalise(field, vmax=None):
    if vmax is None:
        vmax = np.percentile(np.abs(field), 98) + 1e-12
    return np.clip(field / vmax, -1.0, 1.0)


def _field_to_rgba(vals, vmax):
    """vals in [-vmax, vmax] → RGBA array."""
    normed = (np.clip(vals / vmax, -1.0, 1.0) + 1.0) / 2.0   # 0…1
    return CMAP(normed)


# ── Taylor-Proudman interior field ───────────────────────────────────────

def interior_field(surface_field_2d, r_frac):
    """
    Approximate interior vorticity at fractional radius r_frac ∈ [0,1].
    Under the Taylor-Proudman constraint, columnar flow means ω_z is
    constant along the rotation (z) axis.  Map each column to a
    latitude-weighted radial decay.

    surface_field_2d : (nlat, nlon) in lat-major order (north first)
    r_frac           : scalar in (0, 1]
    """
    # Simple model: at radius r, only latitudes |sinφ| < r contribute
    # (outside the tangent cylinder).  Inside the tangent cylinder the
    # field decays to zero.
    lat, lon = latlon_grid()
    phi = np.radians(lat)           # (nlat,)
    sin_phi = np.sin(phi)[:, None]  # (nlat, 1)

    # Cylinder mask: |sinφ| > r_frac → inside tangent cylinder → damp
    mask = np.where(np.abs(sin_phi) > r_frac,
                    np.exp(-5.0 * (np.abs(sin_phi) - r_frac)),
                    1.0)
    # Radial amplitude decay
    amp = r_frac**1.5
    return surface_field_2d * mask * amp


# ── main render function ──────────────────────────────────────────────────

def render_frame(surface_field, frame_idx, total_frames, t_val,
                 figsize_px=None):
    """
    Render one frame.

    Parameters
    ----------
    surface_field : 2-D numpy array (nlat × nlon), vorticity on DH2 grid
    frame_idx     : frame counter (for title)
    total_frames  : total frame count
    t_val         : simulation time value for title
    figsize_px    : output size in pixels (default IMG_SIZE × IMG_SIZE)

    Returns
    -------
    fig : matplotlib Figure
    """
    if figsize_px is None:
        figsize_px = IMG_SIZE

    dpi = 100
    fig_in = figsize_px / dpi

    fig = plt.figure(figsize=(fig_in, fig_in), dpi=dpi, facecolor='white')
    ax  = fig.add_subplot(111, projection='3d',
                           facecolor='white')

    vmax = np.percentile(np.abs(surface_field), 97) + 1e-12

    lat, lon = latlon_grid()
    nlat, nlon = surface_field.shape

    # ── outer spherical shell ────────────────────────────────────────────
    # Draw the surface as a collection of small coloured quads (patches).
    # Only draw patches NOT in the cutaway wedge.

    # For speed, use pcolormesh-on-sphere via surface() with facecolors.
    # Build a fine lat-lon mesh and colour each cell.

    phi_arr  = np.radians(lat)   # (nlat,)
    lam_arr  = np.radians(lon)   # (nlon,)
    PHI, LAM = np.meshgrid(phi_arr, lam_arr, indexing='ij')  # (nlat, nlon)

    X = np.cos(PHI) * np.cos(LAM)
    Y = np.cos(PHI) * np.sin(LAM)
    Z = np.sin(PHI)

    # Mask cutaway
    lon_grid = np.degrees(LAM) % 360.0
    mask_cut = _in_cutaway(lon_grid)   # True = remove

    field_norm = _normalise(surface_field, vmax)
    fcolors = CMAP((field_norm + 1) / 2)  # (nlat, nlon, 4)
    fcolors[mask_cut] = [0, 0, 0, 0]      # transparent in cutaway

    # Downsample for speed (every 2nd point still looks good at 360px)
    ds = 2
    ax.plot_surface(X[::ds, ::ds], Y[::ds, ::ds], Z[::ds, ::ds],
                    facecolors=fcolors[::ds, ::ds],
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False,
                    shade=False, alpha=1.0)

    # ── lat/lon grid lines ───────────────────────────────────────────────
    lw_grid = 0.3
    c_grid  = '0.4'

    # Parallels every 30°
    for lat_line in range(-60, 90, 30):
        phi_l = np.radians(lat_line)
        lam_l = np.linspace(0, 2*np.pi, 360)
        xl = np.cos(phi_l) * np.cos(lam_l)
        yl = np.cos(phi_l) * np.sin(lam_l)
        zl = np.sin(phi_l) * np.ones_like(lam_l)
        lon_l_deg = np.degrees(lam_l) % 360
        in_cut = _in_cutaway(lon_l_deg)
        # Draw in two segments: outside and inside cutaway (dashed)
        _draw_segmented_line(ax, xl, yl, zl, in_cut, c_grid, lw_grid)

    # Meridians every 30° (only the non-cutaway ones)
    for lon_line in range(0, 360, 30):
        if CUTAWAY_LON_START < lon_line < CUTAWAY_LON_END:
            continue
        lam_l = np.radians(lon_line)
        phi_l = np.linspace(-np.pi/2, np.pi/2, 180)
        xl = np.cos(phi_l) * np.cos(lam_l)
        yl = np.cos(phi_l) * np.sin(lam_l)
        zl = np.sin(phi_l)
        ax.plot(xl, yl, zl, color=c_grid, lw=lw_grid, zorder=2)

    # ── equatorial cross-section ─────────────────────────────────────────
    _draw_equatorial_slice(ax, surface_field, vmax, lat, lon)

    # ── meridional cross-section ─────────────────────────────────────────
    _draw_meridional_slice(ax, surface_field, vmax, lat, lon)

    # ── cutaway rim / edge arcs ─────────────────────────────────────────
    _draw_cutaway_edges(ax)

    # ── axes / title ──────────────────────────────────────────────────────
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    ax.view_init(elev=25, azim=45)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=CMAP,
                                norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, fraction=0.03)
    cbar.set_label(r"$\omega'_z$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(r"$\omega'_z$" + f"        t = {t_val:7.1f}",
                 fontsize=11, y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── slice drawing helpers ────────────────────────────────────────────────

def _draw_equatorial_slice(ax, surface_field, vmax, lat, lon):
    """
    Draw a filled disc at z=0 (equatorial plane) for the cutaway region.
    Uses radially-varying field (Taylor-Proudman).
    """
    n_r   = N_RADIAL
    n_lon = len(lon)

    # Build polar grid over the cutaway wedge
    r_vals   = np.linspace(0.0, 1.0, n_r)
    # Longitude over cutaway wedge
    lon_cut  = np.linspace(np.radians(CUTAWAY_LON_START),
                            np.radians(CUTAWAY_LON_END), 60)

    R, LAM_cut = np.meshgrid(r_vals, lon_cut)  # (nlon_cut, n_r)

    # Equatorial latitude = 0 → get surface value by interpolating at lat≈0
    eq_idx = np.argmin(np.abs(lat))   # index closest to equator
    eq_surface = surface_field[eq_idx, :]   # (nlon,)

    # Interpolate onto lon_cut grid
    lon_deg_full = lon % 360.0
    lon_cut_deg  = np.degrees(LAM_cut)   # (nlon_cut, n_r)

    from scipy.interpolate import interp1d
    f_interp = interp1d(lon_deg_full, eq_surface, bounds_error=False,
                         fill_value='extrapolate')
    eq_vals_surface = f_interp(lon_cut_deg)   # (nlon_cut, n_r)

    # Radial decay approximation
    eq_field = eq_vals_surface * R**1.5  # fade to zero at centre

    X_eq = R * np.cos(LAM_cut)
    Y_eq = R * np.sin(LAM_cut)
    Z_eq = np.zeros_like(R)

    field_n = np.clip(eq_field / vmax, -1, 1)
    fc = CMAP((field_n + 1) / 2)

    ax.plot_surface(X_eq, Y_eq, Z_eq,
                    facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)


def _draw_meridional_slice(ax, surface_field, vmax, lat, lon):
    """
    Draw the meridional (lon=0 / lon=cutaway_end) cross-section.
    """
    from scipy.interpolate import interp1d

    for lon_slice_deg in [CUTAWAY_LON_START, CUTAWAY_LON_END]:
        lam_s = np.radians(lon_slice_deg)
        n_r   = N_RADIAL
        n_lat = len(lat)

        r_vals  = np.linspace(0.0, 1.0, n_r)
        lat_deg = np.array(lat)

        # Surface field along this meridian
        lon_deg_full = np.array(lon) % 360.0
        lon_s_deg    = lon_slice_deg % 360.0

        # Interpolate surface field at this longitude
        meridian_surface = np.array([
            interp1d(lon_deg_full, surface_field[i, :],
                     bounds_error=False, fill_value='extrapolate')(lon_s_deg)
            for i in range(n_lat)
        ])   # (nlat,)

        # Build 2-D grid: lat × r
        LAT_g, R_g = np.meshgrid(lat_deg, r_vals, indexing='ij')  # (nlat, nr)

        # Taylor-Proudman interior
        sin_phi = np.sin(np.radians(LAT_g))
        mask = np.where(np.abs(sin_phi) > R_g,
                        np.exp(-5.0 * (np.abs(sin_phi) - R_g)),
                        1.0)
        mer_surface_2d = meridian_surface[:, None] * np.ones((n_lat, n_r))
        mer_field = mer_surface_2d * mask * R_g**1.5

        phi_g = np.radians(LAT_g)
        X_m = R_g * np.cos(phi_g) * np.cos(lam_s)
        Y_m = R_g * np.cos(phi_g) * np.sin(lam_s)
        Z_m = R_g * np.sin(phi_g)

        fn = np.clip(mer_field / vmax, -1, 1)
        fc = CMAP((fn + 1) / 2)

        ax.plot_surface(X_m, Y_m, Z_m,
                        facecolors=fc, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)


def _draw_cutaway_edges(ax):
    """Draw rim arcs and a rotation-axis line for the cutaway region."""
    # Equator arc for cutaway
    lam = np.linspace(np.radians(CUTAWAY_LON_START),
                       np.radians(CUTAWAY_LON_END), 60)
    ax.plot(np.cos(lam), np.sin(lam), np.zeros_like(lam),
            color='0.3', lw=0.6, zorder=5)

    # Rotation axis stub
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='0.3', lw=0.6,
            linestyle='--', zorder=5)


def _draw_segmented_line(ax, x, y, z, mask_dashed, color, lw):
    """Draw a line, solid where mask_dashed=False, dashed where True."""
    n = len(x)
    i = 0
    while i < n:
        j = i
        dashed = mask_dashed[i]
        while j < n and mask_dashed[j] == dashed:
            j += 1
        seg_x = x[i:j]
        seg_y = y[i:j]
        seg_z = z[i:j]
        ls = '--' if dashed else '-'
        ax.plot(seg_x, seg_y, seg_z, color=color, lw=lw, linestyle=ls,
                zorder=2)
        i = j


# ── figure → RGB array ───────────────────────────────────────────────────

def fig_to_rgb(fig):
    """Convert matplotlib figure to H×W×3 uint8 array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    return arr[:, :, :3]
