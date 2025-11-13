import numpy as np

# Optional import guard (rich might not exist)
try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import pytori as pt

# ============================================================
# Helper formatting
# ============================================================
class DisplayFormat:
    """
    Centralized formatting configuration for rich panels and coefficients.
    """
    def __init__(self, fmt=".5e",n_terms=10):
        self.fmt        = fmt
        self.n_terms    = n_terms
    

# GLOBAL used by display helpers
DISPLAY_FMT = DisplayFormat()

def _format_coeff(v, mp, fmt=None):
    """Modular coefficient string formatter."""
    fmt = fmt or DISPLAY_FMT.fmt

    if mp.name == "sympy" and isinstance(v, mp.Basic):
        return mp.sstr(v) 
    try:
        if np.iscomplexobj(v):
            re = float(np.real(v)); 
            im = float(np.imag(v))
            re_sign = " " if im >= 0 else "-"
            im_sign = "+" if im >= 0 else "-"
            return f"{re_sign}{abs(re):{fmt}} {im_sign} {abs(im):{fmt}}j"
        return f"{float(v):{fmt}}"
    except Exception:
        return str(v)
    
def _format_norm(v, mp, fmt=None):
    """Modular coefficient string formatter."""
    fmt = fmt or DISPLAY_FMT.fmt
    if mp.name == "sympy":
        norm_str = '-'
    else:
        try:
            norm_str = f"{float(mp.abs(v)):{fmt}}"
        except Exception:
            norm_str = str(mp.abs(v))
    return norm_str

def _format_index_tuple(k):
    """Format index tuple with fixed-width signed alignment."""
    # Convert tuple of ints into aligned string: pad positives with a space
    parts = []
    for x in k:
        if isinstance(x, (int, np.integer)):
            parts.append(f"{x:+d}".replace("+", " ") if x >= 0 else f"{x:+d}")
        else:
            parts.append(str(x))
    # Join with commas inside parentheses
    return f"({','.join(parts)})"

# ============================================================
# Shared table builder
# ============================================================
def _build_series_table(series):
    """Return a Rich Table for the coefficients of a BaseSeries."""
    coeffs = series.coeffs
    mp = series.mp
    n_terms = len(coeffs)

    # Sorting (by magnitude or symbolic order)
    try:
        if mp.name == "sympy":
            sorted_items = sorted(coeffs.items(), key=lambda kv: (sum(abs(ki) for ki in kv[0]), kv[0]))
        else:
            sorted_items = sorted(coeffs.items(), key=lambda kv: -abs(kv[1]))
    except Exception:
        sorted_items = coeffs.items()

    # Build table
    table = Table(show_header=True, header_style="bright_cyan", box=None, padding=(0, 1))
    table.add_column("Index", style="indian_red", justify="left", no_wrap=True)
    table.add_column(" Value", style="dark_cyan", justify="left")
    table.add_column("Norm", style="royal_blue1", justify="left")

    for k, v in list(sorted_items)[:DISPLAY_FMT.n_terms]:
        k_str = _format_index_tuple(k)
        table.add_row(k_str, _format_coeff(v, mp), _format_norm(v, mp))
    if n_terms > DISPLAY_FMT.n_terms:
        table.add_row("...", "...", "...")

    return table, n_terms


# ============================================================
# Standalone Series renderer (single clean panel)
# ============================================================
def render_series(series):
    """Render a standalone BaseSeries or subclass with a single meta panel."""
    console = Console(record=True, width=100)
    table, n_terms = _build_series_table(series)

    mpname = series.mp.name
    meta = Text(
        f"dim={series.dim}, "
        f"terms={n_terms}, "
        f"max_order={series.max_order}, "
        f"max_terms={series.max_terms}, "
        f"numerical_tol={series.numerical_tol}, "
        f"backend='{mpname}'",
        style="bright_cyan",
    )

    panel = Panel(
        table,
        title=f"[bold bright_cyan]{series.__class__.__name__}[/bold bright_cyan]",
        subtitle=meta,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(panel)
    return ""


# ============================================================
# RICH RENDERERS
# ============================================================
def render_series(series):
    """Render a standalone BaseSeries or subclass with a single meta panel."""
    if not HAS_RICH:
        return render_series_ascii(series)

    from rich.console import Console
    console = Console(record=True, width=100)
    table, n_terms = _build_series_table(series)

    mpname = series.mp.name
    meta = Text(
        f"dim={series.dim}, "
        f"terms={n_terms}, "
        f"max_order={series.max_order}, "
        f"max_terms={series.max_terms}, "
        f"numerical_tol={series.numerical_tol}, "
        f"backend='{mpname}'",
        style="bright_cyan",
    )

    panel = Panel(
        table,
        title=f"[bold bright_cyan]{series.__class__.__name__}[/bold bright_cyan]",
        subtitle=meta,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(panel)
    return ""


def render_torus(torus):
    """Render a Torus with x/y/z series subpanels and metadata."""
    if not HAS_RICH:
        return render_torus_ascii(torus)

    from rich.console import Console
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text

    console = Console(record=True, width=120)

    # Subpanels (reusing table builder)
    subpanels = []
    for p in ["x", "y", "z"]:
        s = getattr(torus, f"_{p}")
        if s is None:
            continue
        table, n_terms = _build_series_table(s)
        subpanels.append(
            Panel(
                table,
                title=f"[bold bright_cyan]{p}[/bold bright_cyan]",
                subtitle=f"[dim]terms={n_terms}[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    layout = Columns(subpanels, expand=True, equal=True, column_first=False)

    # Series class and backend
    clsname = next(
        (s.__class__.__name__ for s in (torus._x, torus._y, torus._z) if s is not None),
        "UnknownSeries",
    )
    mpname = torus.mp.name

    # Safe (Ix, Iy, Iz)
    try:
        if torus.dim == 1:
            Ix = torus.Ix
            I_str = f"(Ix)=({float(Ix):.3g})"
        elif torus.dim == 2:
            Ix, Iy = torus.Ix, torus.Iy
            I_str = f"(Ix,Iy)=({float(Ix):.3g}, {float(Iy):.3g})"
        else:
            Ix, Iy, Iz = torus.Ix, torus.Iy, torus.Iz
            I_str = f"(Ix,Iy,Iz)=({float(Ix):.3g}, {float(Iy):.3g}, {float(Iz):.3g})"
    except Exception:
        I_str = "(I)=(n/a)"

    base_series = torus.x or torus.y or torus.z
    meta = Text(
        f"dim={torus.dim}, "
        f"β₀=({torus.betx0:.3g}, {torus.bety0:.3g}, {torus.betz0:.3g}), "
        f"{I_str}, "
        f"max_order={base_series.max_order if base_series else None}, "
        f"max_terms={base_series.max_terms if base_series else None}, "
        f"numerical_tol={base_series.numerical_tol if base_series else None}, "
        f"backend='{mpname}'",
        style="bright_cyan",
    )

    outer_panel = Panel(
        layout,
        title=f"[bold bright_cyan]Torus ({clsname})[/bold bright_cyan]",
        subtitle=meta,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(outer_panel)
    return ""


# ============================================================
# ASCII FALLBACK RENDERERS
# ============================================================
def render_series_ascii(series):
    """Plain ASCII fallback for standalone Series."""
    coeffs = series.coeffs
    mp = series.mp
    clsname = series.__class__.__name__
    mpname = getattr(mp, "name", "?")
    n_terms = len(coeffs)

    if n_terms == 0:
        return f"{clsname}(dim={series.dim}, terms=0, backend='{mpname}')"

    try:
        sorted_items = sorted(coeffs.items(), key=lambda kv: -abs(kv[1]))
    except Exception:
        sorted_items = list(coeffs.items())

    header = f"{clsname}(dim={series.dim}, terms={n_terms}, backend='{mpname}')"
    lines = [header, "  Index         Value                     Norm"]
    for i, (k, v) in enumerate(sorted_items[:DISPLAY_FMT.n_terms]):
        k_str = str(tuple(k))
        val_str = _format_coeff(v, mp)
        norm_str = _format_norm(v, mp)
        lines.append(f"  {k_str:<10} {val_str:<25} {norm_str:>10}")
    if n_terms > DISPLAY_FMT.n_terms:
        lines.append("  ...")
    return "\n".join(lines)


def render_torus_ascii(torus):
    """Plain ASCII fallback for Torus (summary-level only)."""
    clsname = next(
        (s.__class__.__name__ for s in (torus._x, torus._y, torus._z) if s is not None),
        "UnknownSeries",
    )
    mpname = torus.mp.name
    planes = [p for p in ["x", "y", "z"] if getattr(torus, f"_{p}") is not None]
    return f"Torus ({clsname}) (dim={torus.dim}, planes={planes}, backend='{mpname})'"