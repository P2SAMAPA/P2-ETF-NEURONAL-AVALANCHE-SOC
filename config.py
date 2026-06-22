import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-soc-avalanche-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Avalanche detection ───────────────────────────────────────────────────────
# A "return cascade" (avalanche) is defined as a sequence of consecutive days
# where |r_t| exceeds a threshold (THRESH_SIGMA * rolling_std).
# We record the SIZE of each avalanche (number of days in the cascade).
THRESH_SIGMA = 1.0      # threshold in units of rolling std

# Rolling std window for threshold computation
THRESH_STD_WIN = 21

# ── Power-law fitting ─────────────────────────────────────────────────────────
# We fit P(S ≥ s) ~ s^{-α} via MLE (Clauset et al. 2009).
# Critical exponent at SOC: α ≈ 1.5 (neuronal avalanche result,
# Beggs & Plenz 2003). Markets near criticality show α ≈ 1.5.
# Subcritical (trending): α > 1.5
# Supercritical (chaotic): α < 1.5

SOC_CRITICAL_ALPHA = 1.5   # theoretical critical exponent

# Minimum avalanche size to include in MLE fit
ALPHA_S_MIN = 1

# Minimum number of avalanches needed to fit a power law reliably
MIN_AVALANCHES = 10

# ── Score construction ────────────────────────────────────────────────────────
# Three complementary SOC metrics combined into a composite score:
#
#   criticality_score : how close α is to the critical value 1.5
#                       peak at α=1.5, decays symmetrically
#                       = exp(-0.5 * ((α - 1.5)/σ_α)²)
#
#   predictability_score : near-critical systems have highest predictability
#                          (Sornette 2003). Proxy: coefficient of variation
#                          of avalanche sizes (CV = std/mean). Peak CV at
#                          criticality due to scale-free fluctuations.
#
#   macro_alignment_score : sign agreement between criticality regime and
#                           macro direction (VIX trend). Near-critical + rising
#                           VIX → avoid; near-critical + falling VIX → buy.

WEIGHT_CRITICALITY    = 0.50
WEIGHT_PREDICTABILITY = 0.30
WEIGHT_MACRO_ALIGN    = 0.20

# Width of Gaussian criticality proximity function
ALPHA_SIGMA = 0.5

TOP_N = 3
