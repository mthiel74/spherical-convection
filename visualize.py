"""
3-D sphere renderer with cutaway wedge.

Renders the z-component of vorticity on:
  • the outer spherical surface (cutaway region hole-punched with NaN)
  • the exposed equatorial cross-section  (Poly3DCollection)
  • the two meridional cross-section faces (Poly3DCollection)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import IMG_SIZE, CUTAWAY_FRACTION, N_RADIAL, LMAX


# ── grid coordinates ──────────────────────────────────────────────────────

def latlon_grid():
    """Return (lats, lons) arrays matching the DH2 pyshtools grid at LMAX."""
    import pyshtools as pysh
    dummy = pysh.SHCoeffs.from_zeros(LMAX, normalization='4pi')
    g = dummy.expand(grid='DH2')
    return g.lats(), g.lons()


# ── cutaway geometry ──────────────────────────────────────────────────────

CUTAWAY_LON_START = 0.0
CUTAWAY_LON_END   = CUTAWAY_FRACTION * 360.0   # 90° for a quarter wedge


def _in_cutaway(lon_deg):
    lon = np.asarray(lon_deg) % 360.0
    return (lon >= CUTAWAY_LON_START) & (lon < CUTAWAY_LON_END)


# ── colormap helpers ──────────────────────────────────────────────────────

CMAP = plt.cm.RdBu_r   # red = +ω, blue = −ω


def _to_rgba(vals, vmax):
    """vals (any shape) → same-shape RGBA array, range clamped to ±vmax."""
    normed = (np.clip(vals / (vmax + 1e-12), -1.0, 1.0) + 1.0) / 2.0
    return CMAP(normed)


# ── interpolation helper ──────────────────────────────────────────────────

def _interp_lon(lon_arr, field_1d, target_lon_deg):
    """
    Interpolate a 1-D field (defined on lon_arr) at target_lon_deg.
    Wraps around [0, 360).
    """
    from scipy.interpolate import interp1d
    lon_src = np.array(lon_arr) % 360.0
    # Extend to handle wrap-around
    lon_ext = np.concatenate([lon_src - 360.0, lon_src, lon_src + 360.0])
    f_ext   = np.concatenate([field_1d, field_1d, field_1d])
    fn = interp1d(lon_ext, f_ext, kind='linear', bounds_error=False,
                  fill_value=0.0)
    return fn(np.asarray(target_lon_deg))


# ── Poly3DCollection helpers ──────────────────────────────────────────────

def _quads_to_poly(X, Y, Z, fc_rgba):
    """
    Convert a (m,n) mesh of vertex positions + (m-1,n-1,4) face colors
    into a Poly3DCollection of quads.

    X, Y, Z : (m, n) vertex arrays
    fc_rgba : (m-1, n-1, 4) RGBA face colors
    """
    m, n = X.shape
    verts  = []
    colors = []
    for i in range(m - 1):
        for j in range(n - 1):
            quad = [
                (X[i,   j  ], Y[i,   j  ], Z[i,   j  ]),
                (X[i+1, j  ], Y[i+1, j  ], Z[i+1, j  ]),
                (X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]),
                (X[i,   j+1], Y[i,   j+1], Z[i,   j+1]),
            ]
            verts.append(quad)
            colors.append(fc_rgba[i, j])
    pc = Poly3DCollection(verts, linewidths=0)
    pc.set_facecolor(colors)
    pc.set_edgecolor('none')
    return pc


# ── Taylor-Proudman interior ──────────────────────────────────────────────

def _tp_interior(surface_col, lat_deg, r_grid):
    """
    Approximate interior vorticity using Taylor-Proudman columnar flow.

    surface_col : (nlat,)  surface vorticity along one longitude
    lat_deg     : (nlat,)  latitudes in degrees
    r_grid      : (nlat, nr) radius values (0…1)

    Returns field of same shape as r_grid.
    """
    sin_phi = np.sin(np.radians(lat_deg))[:, None]   # (nlat,1)
    # Damp inside tangent cylinder (|sinφ| > r)
    mask = np.where(
        np.abs(sin_phi) > r_grid,
        np.exp(-6.0 * (np.abs(sin_phi) - r_grid)),
        1.0
    )
    field_2d = surface_col[:, None] * mask * r_grid**1.2
    return field_2d


# ── cross-section renderers ───────────────────────────────────────────────

def _draw_equatorial_slice(ax, surface_field, vmax, lat, lon):
    """
    Filled disc at z=0 covering the cutaway wedge, coloured by vorticity.
    Uses Poly3DCollection so depth-sorting artefacts don't hide it.
    """
    n_r    = N_RADIAL
    n_phi  = 64      # angular resolution inside wedge

    r_vals   = np.linspace(0.0, 1.0, n_r + 1)
    lam_vals = np.linspace(np.radians(CUTAWAY_LON_START),
                            np.radians(CUTAWAY_LON_END), n_phi + 1)

    # Equatorial strip from surface field (average the two rows nearest lat=0)
    eq_idx = np.argmin(np.abs(np.array(lat)))
    eq_surface = surface_field[eq_idx, :]   # (nlon,)

    # Build vertex mesh (n_phi+1) × (n_r+1)
    R_v, LAM_v = np.meshgrid(r_vals, lam_vals, indexing='ij')   # (nr+1, nphi+1)
    X_v = R_v * np.cos(LAM_v)
    Y_v = R_v * np.sin(LAM_v)
    Z_v = np.zeros_like(R_v)

    # Face-centre radii and lons
    r_c   = 0.5 * (r_vals[:-1] + r_vals[1:])          # (nr,)
    lam_c = 0.5 * (lam_vals[:-1] + lam_vals[1:])      # (nphi,)
    R_c, LAM_c = np.meshgrid(r_c, lam_c, indexing='ij')   # (nr, nphi)

    lon_c_deg = np.degrees(LAM_c) % 360.0
    # Interpolate surface field to face centres
    surf_c = _interp_lon(lon, eq_surface, lon_c_deg)    # (nr, nphi)
    # Radial decay
    field_c = surf_c * R_c**1.2

    fc_rgba = _to_rgba(field_c, vmax)                   # (nr, nphi, 4)

    pc = _quads_to_poly(X_v.T, Y_v.T, Z_v.T, fc_rgba.transpose(1, 0, 2))
    ax.add_collection3d(pc)


def _draw_meridional_slice(ax, surface_field, vmax, lat, lon):
    """
    Two flat semicircle faces (the boundary planes of the wedge),
    coloured by vorticity (Taylor-Proudman interior).
    """
    n_r   = N_RADIAL
    n_lat = 64

    r_vals  = np.linspace(0.0, 1.0, n_r + 1)
    lat_v   = np.linspace(-90.0, 90.0, n_lat + 1)
    lat_c   = 0.5 * (lat_v[:-1] + lat_v[1:])
    r_c     = 0.5 * (r_vals[:-1] + r_vals[1:])

    for lon_slice_deg in [CUTAWAY_LON_START, CUTAWAY_LON_END]:
        lam_s = np.radians(lon_slice_deg)
        lon_s_deg = lon_slice_deg % 360.0

        # Surface field along this meridian
        mer_surface = np.array([
            _interp_lon(lon, surface_field[i, :], lon_s_deg)
            for i in range(len(lat))
        ])   # (nlat_grid,)

        # Interpolate onto fine lat grid
        from scipy.interpolate import interp1d
        lat_arr = np.array(lat)
        sort_idx = np.argsort(lat_arr)
        f_lat = interp1d(lat_arr[sort_idx], mer_surface[sort_idx],
                         kind='linear', bounds_error=False, fill_value=0.0)
        mer_c_surface = f_lat(lat_c)   # (n_lat,)

        # Build vertex mesh (n_lat+1) × (n_r+1)
        LAT_v, R_v = np.meshgrid(lat_v, r_vals, indexing='ij')  # (nlat+1,nr+1)
        phi_v = np.radians(LAT_v)
        X_v = R_v * np.cos(phi_v) * np.cos(lam_s)
        Y_v = R_v * np.cos(phi_v) * np.sin(lam_s)
        Z_v = R_v * np.sin(phi_v)

        # Face-centre field with Taylor-Proudman decay
        LAT_c2, R_c2 = np.meshgrid(lat_c, r_c, indexing='ij')  # (nlat,nr)
        field_c = _tp_interior(mer_c_surface, lat_c, R_c2)       # (nlat,nr)

        fc_rgba = _to_rgba(field_c, vmax)   # (nlat,nr,4)

        pc = _quads_to_poly(X_v, Y_v, Z_v, fc_rgba)
        ax.add_collection3d(pc)


# ── cutaway edge lines ────────────────────────────────────────────────────

def _draw_cutaway_edges(ax):
    """Rim arcs, bounding radii, and rotation-axis line."""
    c = '0.25'
    lw = 0.7

    # Equator arc along the cutaway wedge boundary
    lam = np.linspace(np.radians(CUTAWAY_LON_START),
                       np.radians(CUTAWAY_LON_END), 90)
    ax.plot(np.cos(lam), np.sin(lam), np.zeros(len(lam)),
            color=c, lw=lw, zorder=6)

    # Two radial lines from centre to sphere at equator (wedge edges)
    for lon_d in [CUTAWAY_LON_START, CUTAWAY_LON_END]:
        lam_d = np.radians(lon_d)
        ax.plot([0, np.cos(lam_d)], [0, np.sin(lam_d)], [0, 0],
                color=c, lw=lw, zorder=6)

    # Rotation axis
    ax.plot([0, 0], [0, 0], [-1.05, 1.05],
            color=c, lw=0.5, linestyle='--', zorder=6)


# ── grid lines ────────────────────────────────────────────────────────────

def _draw_grid_lines(ax):
    lw_s = 0.45   # solid (visible)
    lw_d = 0.25   # dashed (hidden / inside cutaway)
    c_s  = '0.3'
    c_d  = '0.6'

    n_seg = 720

    # Parallels every 30°
    for lat_line in range(-60, 91, 30):
        phi_l = np.radians(lat_line)
        lam_l = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
        xl = np.cos(phi_l) * np.cos(lam_l)
        yl = np.cos(phi_l) * np.sin(lam_l)
        zl = np.sin(phi_l) * np.ones(n_seg)
        lon_l = np.degrees(lam_l) % 360.0
        in_cut = _in_cutaway(lon_l)
        _segmented_line(ax, xl, yl, zl, in_cut, c_s, c_d, lw_s, lw_d)

    # Meridians every 30°
    for lon_line in range(0, 360, 30):
        lam_l = np.radians(lon_line)
        phi_l = np.linspace(-np.pi/2, np.pi/2, 360)
        xl = np.cos(phi_l) * np.cos(lam_l)
        yl = np.cos(phi_l) * np.sin(lam_l)
        zl = np.sin(phi_l)
        if _in_cutaway(lon_line):
            ax.plot(xl, yl, zl, color=c_d, lw=lw_d, linestyle='--', zorder=2)
        else:
            ax.plot(xl, yl, zl, color=c_s, lw=lw_s, linestyle='-', zorder=2)


def _segmented_line(ax, x, y, z, mask_dashed, c_solid, c_dash, lw_s, lw_d):
    n = len(x)
    i = 0
    while i < n:
        j = i
        d = bool(mask_dashed[i])
        while j < n and bool(mask_dashed[j]) == d:
            j += 1
        seg = slice(i, j)
        if d:
            ax.plot(x[seg], y[seg], z[seg], color=c_dash,  lw=lw_d, ls='--', zorder=2)
        else:
            ax.plot(x[seg], y[seg], z[seg], color=c_solid, lw=lw_s, ls='-',  zorder=2)
        i = j


# ── main render function ──────────────────────────────────────────────────

def render_frame(surface_field, frame_idx, total_frames, t_val,
                 figsize_px=None):
    if figsize_px is None:
        figsize_px = IMG_SIZE

    dpi   = 100
    fig   = plt.figure(figsize=(figsize_px/dpi, figsize_px/dpi),
                        dpi=dpi, facecolor='white')
    ax    = fig.add_subplot(111, projection='3d', facecolor='white')

    vmax  = np.percentile(np.abs(surface_field), 97) + 1e-12
    lat, lon = latlon_grid()

    # ── outer sphere (hole-punched with NaN in cutaway) ──────────────────
    phi_arr = np.radians(lat)
    lam_arr = np.radians(lon)
    PHI, LAM = np.meshgrid(phi_arr, lam_arr, indexing='ij')   # (nlat,nlon)

    X = np.cos(PHI) * np.cos(LAM)
    Y = np.cos(PHI) * np.sin(LAM)
    Z = np.sin(PHI)

    lon_grid = np.degrees(LAM) % 360.0
    cut_mask = _in_cutaway(lon_grid)   # True = remove

    # NaN in position arrays → matplotlib skips those quads entirely
    X_plot = np.where(cut_mask, np.nan, X)
    Y_plot = np.where(cut_mask, np.nan, Y)
    Z_plot = np.where(cut_mask, np.nan, Z)

    field_norm = _normalise(surface_field, vmax)
    fcolors    = CMAP((field_norm + 1.0) / 2.0)   # (nlat,nlon,4)

    ds = 2   # downsample factor for speed
    ax.plot_surface(X_plot[::ds, ::ds], Y_plot[::ds, ::ds], Z_plot[::ds, ::ds],
                    facecolors=fcolors[::ds, ::ds],
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)

    # ── cross-section slices ─────────────────────────────────────────────
    _draw_equatorial_slice(ax, surface_field, vmax, lat, lon)
    _draw_meridional_slice(ax, surface_field, vmax, lat, lon)

    # ── cutaway edges + grid ─────────────────────────────────────────────
    _draw_cutaway_edges(ax)
    _draw_grid_lines(ax)

    # ── view / axes ───────────────────────────────────────────────────────
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.48, pad=0.02, fraction=0.03)
    cbar.set_label(r"$\omega'_z$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(r"$\omega'_z$" + f"        t = {t_val:7.1f}",
                 fontsize=11, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── normalise helper (local, not exported) ───────────────────────────────

def _normalise(field, vmax):
    return np.clip(field / vmax, -1.0, 1.0)


# ── figure → RGB ─────────────────────────────────────────────────────────

def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    return arr[:, :, :3]
