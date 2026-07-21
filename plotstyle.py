"""Shared chart style for the toolkit's matplotlib output.

One place for color and chrome so every chart in the repo reads as one system:

- ``SERIES`` is a fixed-order categorical palette (colorblind-validated:
  adjacent-pair CVD deltaE >= 8 and normal-vision deltaE >= 15 in light mode).
  Slots are assigned by *entity* in fixed order, never cycled — a 9th series
  folds into "Other".
- Status colors (``GOOD``/``CRITICAL``) are reserved for buy/sell semantics and
  are never used as series colors; buy/sell marks also differ by shape, so
  color is never the only channel.
- Chrome tokens keep grids/axes recessive and text in ink, not series color.

Call ``apply_style()`` once before building figures.
"""

SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis ticks / de-emphasised context lines
GRID = "#e1e0d9"       # hairline gridlines
BASELINE = "#c3c2b7"   # axis spine / zero line
SURFACE = "#fcfcfb"    # chart surface

GOOD = "#0ca30c"       # status: buy / favourable
CRITICAL = "#d03b3b"   # status: sell / stop


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent; safe to call per figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    })


def series_color(i: int) -> str:
    """Fixed-order categorical slot. Raises past the palette — fold to 'Other'
    (use MUTED) instead of inventing or cycling hues."""
    return SERIES[i]
