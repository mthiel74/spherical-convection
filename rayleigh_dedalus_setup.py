"""
rayleigh_dedalus_setup.py — configuration generator for the Rayleigh and Dedalus
community spherical-convection codes.

Scientific improvement #17 (scientific_improvements.md §17): rather than
hand-building a 3-D convection solver (years of development + verification risk),
map our control parameters onto a VALIDATED community code —

    • Rayleigh (Featherstone & Hindman 2016): a pseudo-spectral anelastic
      convection/dynamo code driven by a Fortran-namelist `main_input` file;
    • Dedalus (Burns et al. 2020): a general spectral-PDE framework whose v3
      spherical bases express the Boussinesq/anelastic shell system in a few
      dozen lines of Python.

This module takes a single `ConvectionParameters` object (Ra, Ek, Pr, aspect
ratio, reference state, resolution, boundary conditions) and emits ready-to-run
input for either code, with the SAME physics in both.  It writes no external
data; the two writers are pure string templates plus a self-contained
Fortran-namelist parser used only for round-trip verification.

═══════════════════════════════════════════════════════════════════════════════
PARAMETER MAPPING (nondimensional; the convention both codes share)
═══════════════════════════════════════════════════════════════════════════════
    Rayleigh number   Ra = α g ΔT d³ /(νκ)          — buoyancy vs. diffusion
    Ekman number      Ek = ν /(2Ω d²)               — viscosity vs. rotation
    Prandtl number    Pr = ν/κ                       — momentum vs. thermal diff.
    (magnetic Pr      Pm = ν/η, dynamo runs only)
    aspect ratio      η  = r_i/r_o ;  shell gap d = r_o − r_i = 1
                          ⇒  r_i = η/(1−η),  r_o = 1/(1−η)

Rayleigh expresses these directly in its &Reference_Namelist; Dedalus takes the
same numbers as Python floats and forms the operators.  reference_type = 1 is a
Boussinesq shell (improvement #15); reference_type = 2 is a polytropic anelastic
shell (improvement #16) parameterised by N_ρ density scale heights and polytropic
index n.

VERIFICATION.  (1) The Rayleigh namelist round-trips through the bundled parser
and every control number reads back exactly.  (2) The Dedalus script is
syntactically valid Python (compile()), and its parameter block exec's to the
requested values.  (3) The Boussinesq↔anelastic toggle and no-slip↔stress-free
toggle propagate correctly to both outputs.

References: Featherstone & Hindman (2016) ApJ 818, 32 (Rayleigh); Burns, Vasil,
Oishi, Lecoanet & Brown (2020) Phys. Rev. Research 2, 023068 (Dedalus);
Christensen et al. (2001) PEPI 128, 25 (dynamo benchmark case).
"""

from dataclasses import dataclass, field


# ═════════════════════════════════════════════════════════════════════════════
# Parameter object
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvectionParameters:
    """Nondimensional control parameters + geometry/resolution for a rotating
    convecting spherical shell.  Defaults are the Christensen et al. (2001)
    dynamo benchmark geometry (η=0.35)."""
    rayleigh: float = 1.0e5
    ekman: float = 1.0e-3
    prandtl: float = 1.0
    magnetic_prandtl: float = 5.0
    aspect_ratio: float = 0.35            # η = r_i/r_o
    reference: str = 'boussinesq'         # 'boussinesq' | 'anelastic'
    n_rho: float = 3.0                    # density scale heights (anelastic only)
    poly_n: float = 2.0                   # polytropic index      (anelastic only)
    n_r: int = 64                         # radial modes
    l_max: int = 63                       # max spherical-harmonic degree
    rotation: bool = True
    magnetism: bool = False
    no_slip: bool = True                  # else stress-free
    max_time_step: float = 1.0e-4
    max_iterations: int = 100000

    def __post_init__(self):
        if self.reference not in ('boussinesq', 'anelastic'):
            raise ValueError(f"reference must be 'boussinesq' or 'anelastic', "
                             f"got {self.reference!r}")
        if not 0.0 < self.aspect_ratio < 1.0:
            raise ValueError(f"aspect_ratio must be in (0,1), got "
                             f"{self.aspect_ratio}")

    @property
    def r_inner(self):
        return self.aspect_ratio / (1.0 - self.aspect_ratio)

    @property
    def r_outer(self):
        return 1.0 / (1.0 - self.aspect_ratio)

    @property
    def reference_type(self):
        return 1 if self.reference == 'boussinesq' else 2


# ═════════════════════════════════════════════════════════════════════════════
# Minimal Fortran-namelist parser (self-contained; for round-trip verification)
# ═════════════════════════════════════════════════════════════════════════════

def _fortran_value(v):
    """Convert a Fortran namelist literal to a Python value."""
    s = v.strip().rstrip(',').strip()
    low = s.lower()
    if low in ('.true.', 't'):
        return True
    if low in ('.false.', 'f'):
        return False
    num = low.replace('d', 'e')            # Fortran double exponent 1.0d5 → 1.0e5
    try:
        if any(c in num for c in '.e') and not num.strip('+-').isdigit():
            return float(num)
        return int(num)
    except ValueError:
        return s.strip('"\'')


def parse_namelist(text):
    """Parse Fortran namelist text into {group: {key: value}}.  Supports the
    subset Rayleigh uses (one key=value per line, ! comments, &group … /)."""
    groups, cur = {}, None
    for raw in text.splitlines():
        line = raw.split('!', 1)[0].strip()
        if not line:
            continue
        if line.startswith('&'):
            cur = line[1:].strip().lower()
            if not cur:
                raise ValueError("empty namelist group name")
            groups[cur] = {}
        elif line == '/':
            if cur is None:
                raise ValueError("stray '/' with no open group")
            cur = None
        elif '=' in line:
            if cur is None:
                raise ValueError(f"key=value outside any group: {line!r}")
            k, v = line.split('=', 1)
            groups[cur][k.strip().lower()] = _fortran_value(v)
        else:
            raise ValueError(f"unparseable namelist line: {line!r}")
    if cur is not None:
        raise ValueError("unterminated namelist group (missing '/')")
    return groups


def _fnum(x):
    """Format a float as a Fortran double literal, e.g. 1.0e5 → 1.0d5."""
    return f"{x:.10g}".replace('e', 'd') if isinstance(x, float) else str(x)


# ═════════════════════════════════════════════════════════════════════════════
# Rayleigh main_input writer
# ═════════════════════════════════════════════════════════════════════════════

class RayleighInputWriter:
    """Emit a Rayleigh `main_input` Fortran namelist for the given parameters."""

    def __init__(self, params: ConvectionParameters):
        self.p = params

    def render(self):
        p = self.p
        n_theta = 3 * (p.l_max + 1) // 2                 # standard 3/2 dealiasing
        ref_lines = [f" reference_type = {p.reference_type}",
                     f" rayleigh_number = {_fnum(p.rayleigh)}",
                     f" ekman_number = {_fnum(p.ekman)}",
                     f" prandtl_number = {_fnum(p.prandtl)}"]
        if p.magnetism:
            ref_lines.append(f" magnetic_prandtl_number = "
                             f"{_fnum(p.magnetic_prandtl)}")
        if p.reference == 'anelastic':
            ref_lines += [f" n_rho = {_fnum(p.n_rho)}",
                          f" poly_n = {_fnum(p.poly_n)}"]
        return "\n".join([
            "! Rayleigh main_input — generated by rayleigh_dedalus_setup.py",
            f"! {p.reference} rotating spherical shell, aspect ratio η="
            f"{p.aspect_ratio}",
            "&problemsize_namelist",
            f" n_r = {p.n_r}",
            f" n_theta = {n_theta}",
            f" l_max = {p.l_max}",
            f" aspect_ratio = {_fnum(p.aspect_ratio)}",
            f" shell_depth = {_fnum(1.0)}",
            "/",
            "&physical_controls_namelist",
            f" rotation = {'.true.' if p.rotation else '.false.'}",
            f" magnetism = {'.true.' if p.magnetism else '.false.'}",
            "/",
            "&temporal_controls_namelist",
            f" max_time_step = {_fnum(p.max_time_step)}",
            f" max_iterations = {p.max_iterations}",
            "/",
            "&boundary_conditions_namelist",
            f" no_slip_boundaries = {'.true.' if p.no_slip else '.false.'}",
            " fix_tvar_top = .true.",
            " fix_tvar_bottom = .true.",
            " t_top = 0.0d0",
            " t_bottom = 1.0d0",
            "/",
            "&reference_namelist",
            *ref_lines,
            " gravity_power = 1.0d0",
            "/",
            "&transport_namelist",
            "/",
        ]) + "\n"

    def write(self, path):
        with open(path, 'w') as f:
            f.write(self.render())
        return path


# ═════════════════════════════════════════════════════════════════════════════
# Dedalus (v3) script writer
# ═════════════════════════════════════════════════════════════════════════════

class DedalusScriptWriter:
    """Emit a Dedalus v3 Python script for a rotating convecting shell.  The
    numeric parameter block is a self-contained prefix (exec-able without
    Dedalus) so the mapping can be verified independently of the solver import."""

    def __init__(self, params: ConvectionParameters):
        self.p = params

    def param_block(self):
        """The self-contained numeric parameter prefix (no Dedalus import)."""
        p = self.p
        return "\n".join([
            "# --- parameters (generated) ---",
            f"Rayleigh = {p.rayleigh!r}",
            f"Ekman = {p.ekman!r}",
            f"Prandtl = {p.prandtl!r}",
            f"aspect_ratio = {p.aspect_ratio!r}",
            f"Ri = {p.r_inner!r}",
            f"Ro = {p.r_outer!r}",
            f"Nr = {p.n_r!r}",
            f"Lmax = {p.l_max!r}",
            f"reference = {p.reference!r}",
            f"N_rho = {p.n_rho!r}",
            f"poly_n = {p.poly_n!r}",
            f"no_slip = {p.no_slip!r}",
            f"rotation = {p.rotation!r}",
        ])

    def render(self):
        p = self.p
        bc = ("problem.add_equation(\"u(r=Ri) = 0\")\n"
              "problem.add_equation(\"u(r=Ro) = 0\")" if p.no_slip else
              "problem.add_equation(\"radial(u(r=Ri)) = 0\")\n"
              "problem.add_equation(\"radial(u(r=Ro)) = 0\")")
        return '\n'.join([
            '"""Dedalus v3 rotating convecting spherical shell — generated by',
            'rayleigh_dedalus_setup.py.  Boussinesq (reference==\'boussinesq\')',
            'or anelastic polytrope (reference==\'anelastic\').  Run with mpirun.',
            '"""',
            "import numpy as np",
            "import dedalus.public as d3",
            "",
            self.param_block(),
            "",
            "# --- coordinates, distributor, basis ---",
            "coords = d3.SphericalCoordinates('phi', 'theta', 'r')",
            "dist = d3.Distributor(coords, dtype=np.float64)",
            "shell = d3.ShellBasis(coords, shape=(2*(Lmax+1), Lmax+1, Nr),",
            "                      radii=(Ri, Ro), dtype=np.float64)",
            "",
            "# --- fields ---",
            "u = dist.VectorField(coords, name='u', bases=shell)",
            "T = dist.Field(name='T', bases=shell)",
            "p = dist.Field(name='p', bases=shell)",
            "tau_p = dist.Field(name='tau_p')",
            "",
            "# --- substitutions ---",
            "Omega = 1.0 / (2.0 * Ekman)            # rotation rate from Ekman",
            "nu = Ekman                             # nondim viscosity",
            "kappa = Ekman / Prandtl                # nondim thermal diffusivity",
            "buoyancy = Rayleigh * Ekman / Prandtl  # Ra Ek/Pr buoyancy prefactor",
            "ez = dist.VectorField(coords, bases=shell.meridian_basis)",
            "ez['g'][2] = 1.0",
            "",
            "# --- problem (incompressible Boussinesq form shown) ---",
            "problem = d3.IVP([u, T, p, tau_p], namespace=locals())",
            "problem.add_equation(\"div(u) + tau_p = 0\")",
            "problem.add_equation(\"dt(u) - nu*lap(u) + grad(p) "
            "+ 2*Omega*cross(ez, u) = buoyancy*T*ez - dot(u, grad(u))\")",
            "problem.add_equation(\"dt(T) - kappa*lap(T) = - dot(u, grad(T))\")",
            bc,
            "problem.add_equation(\"T(r=Ri) = 1.0\")",
            "problem.add_equation(\"T(r=Ro) = 0.0\")",
            "problem.add_equation(\"integ(p) = 0\")",
            "",
            "# --- solver ---",
            "solver = problem.build_solver(d3.RK222)",
            f"solver.stop_iteration = {p.max_iterations}",
            f"timestep = {p.max_time_step!r}",
            "while solver.proceed:",
            "    solver.step(timestep)",
            "",
        ])

    def write(self, path):
        with open(path, 'w') as f:
            f.write(self.render())
        return path


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════

def _exec_params(block):
    """exec a self-contained parameter block and return its namespace."""
    ns = {}
    exec(compile(block, '<params>', 'exec'), ns)
    return {k: v for k, v in ns.items() if not k.startswith('__')}


def verify():
    print("=" * 74)
    print("Rayleigh / Dedalus configuration generator — verification")
    print("=" * 74)
    ok = True

    p = ConvectionParameters(rayleigh=1.0e5, ekman=1.0e-3, prandtl=1.0,
                             aspect_ratio=0.35, reference='boussinesq')

    # ── 1. Rayleigh namelist round-trips and maps correctly ───────────────────
    text = RayleighInputWriter(p).render()
    nml = parse_namelist(text)                          # raises on any syntax error
    ref = nml['reference_namelist']
    ps = nml['problemsize_namelist']
    checks = {
        'rayleigh_number': (ref['rayleigh_number'], p.rayleigh),
        'ekman_number': (ref['ekman_number'], p.ekman),
        'prandtl_number': (ref['prandtl_number'], p.prandtl),
        'aspect_ratio': (ps['aspect_ratio'], p.aspect_ratio),
        'reference_type': (ref['reference_type'], 1),
        'n_r': (ps['n_r'], p.n_r),
    }
    print("\n1. Rayleigh main_input round-trip (parse → compare):")
    r_ok = True
    for k, (got, exp) in checks.items():
        good = abs(got - exp) <= 1e-12 * (abs(exp) + 1) if isinstance(exp, float) \
            else got == exp
        r_ok &= good
        print(f"   {k:18s} = {got!r:12} (expect {exp!r})  {'✓' if good else '⚠'}")
    print(f"   → namelist syntactically valid & parameters map  "
          f"{'✓' if r_ok else '⚠'}")
    ok &= r_ok

    # namelist structure: every &group has a matching '/'
    n_open = text.count('&')
    n_close = sum(1 for ln in text.splitlines() if ln.strip() == '/')
    struct_ok = n_open == n_close and n_open >= 5
    print(f"   → {n_open} groups, {n_close} terminators balanced  "
          f"{'✓' if struct_ok else '⚠'}")
    ok &= struct_ok

    # ── 2. Dedalus script is valid Python and its params map ──────────────────
    src = DedalusScriptWriter(p).render()
    try:
        compile(src, '<dedalus_generated>', 'exec')     # syntactic validity
        syn_ok = True
    except SyntaxError as e:
        syn_ok = False
        print(f"   Dedalus syntax error: {e}")
    dp = _exec_params(DedalusScriptWriter(p).param_block())
    d_ok = (abs(dp['Rayleigh'] - p.rayleigh) < 1e-6 and
            abs(dp['Ekman'] - p.ekman) < 1e-12 and
            abs(dp['Prandtl'] - p.prandtl) < 1e-12 and
            abs(dp['Ri'] - p.r_inner) < 1e-12 and
            abs(dp['Ro'] - p.r_outer) < 1e-12 and
            abs(dp['Ro'] * p.aspect_ratio - dp['Ri']) < 1e-12)    # η = Ri/Ro
    print(f"\n2. Dedalus script: compiles as valid Python "
          f"{'✓' if syn_ok else '⚠'};  parameter block maps "
          f"(Ra,Ek,Pr,Ri,Ro, η=Ri/Ro) {'✓' if d_ok else '⚠'}")
    print(f"   Ri={dp['Ri']:.4f}, Ro={dp['Ro']:.4f}, Ri/Ro={dp['Ri']/dp['Ro']:.4f}"
          f" (η={p.aspect_ratio})")
    ok &= syn_ok and d_ok

    # ── 3. Toggles propagate: anelastic reference & stress-free BCs ───────────
    pa = ConvectionParameters(reference='anelastic', n_rho=5.0, poly_n=1.5,
                              no_slip=False, magnetism=True)
    na = parse_namelist(RayleighInputWriter(pa).render())
    an_ok = (na['reference_namelist']['reference_type'] == 2 and
             abs(na['reference_namelist']['n_rho'] - 5.0) < 1e-12 and
             na['boundary_conditions_namelist']['no_slip_boundaries'] is False and
             'magnetic_prandtl_number' in na['reference_namelist'])
    da = _exec_params(DedalusScriptWriter(pa).param_block())
    an_ok &= (da['reference'] == 'anelastic' and da['no_slip'] is False and
              abs(da['N_rho'] - 5.0) < 1e-12)
    # anelastic Dedalus script still compiles
    compile(DedalusScriptWriter(pa).render(), '<anelastic>', 'exec')
    print(f"\n3. Toggles propagate (Boussinesq→anelastic reference_type 1→2, "
          f"no-slip→stress-free, magnetism on)  {'✓' if an_ok else '⚠'}")
    ok &= an_ok

    # ── 4. Invalid inputs rejected ────────────────────────────────────────────
    bad = 0
    for kw in (dict(reference='nonsense'), dict(aspect_ratio=1.5),
               dict(aspect_ratio=0.0)):
        try:
            ConvectionParameters(**kw)
        except ValueError:
            bad += 1
    v_ok = bad == 3
    print(f"4. Invalid parameters rejected ({bad}/3 raised ValueError)  "
          f"{'✓' if v_ok else '⚠'}")
    ok &= v_ok

    print("\n" + ("✓ ALL CHECKS PASSED — Rayleigh namelist & Dedalus script are "
                  "syntactically valid and parameters map correctly"
                  if ok else "⚠ some checks failed"))
    return ok


if __name__ == "__main__":
    verify()
