# Physical and numerical parameters for rotating spherical shell convection

# Rotation
OMEGA = 15.0          # rotation rate (non-dimensional); high = banana cells

# Spectral resolution
LMAX = 42             # maximum spherical harmonic degree (T42)

# Hyperviscosity  (∇^8 operator coefficient)
NU_HYPER = 1e-12      # hyperviscosity; damps small scales

# Stochastic forcing
FORCE_LMIN = 6        # inject energy at l >= FORCE_LMIN
FORCE_LMAX = 15       # inject energy at l <= FORCE_LMAX
FORCE_AMP  = 0.08     # forcing amplitude per timestep

# Time stepping
DT         = 5e-4     # timestep (non-dimensional)
N_SPINUP   = 4000     # steps before recording
N_FRAMES   = 250      # frames to record
FRAME_SKIP = 20       # simulation steps between frames

# Output
OUTPUT_GIF = "output.gif"
OUTPUT_MP4 = "output.mp4"
FPS        = 12       # animation frame rate

# Rendering
IMG_SIZE   = 360      # pixels (square)
CUTAWAY_FRACTION = 0.25   # fraction of sphere removed (0.25 = quarter wedge)
N_RADIAL   = 30       # radial levels for cross-section
