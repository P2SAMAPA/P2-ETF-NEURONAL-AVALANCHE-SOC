"""
soc_engine.py — Neuronal Avalanche / Self-Organised Criticality Engine
=======================================================================

Theory
------
**Self-Organised Criticality (SOC, Bak et al. 1987)**

SOC describes systems that spontaneously evolve toward a critical state
between order and disorder — without external tuning of parameters. At the
critical point, the system exhibits:

1. **Scale-free (power-law) dynamics**: fluctuation sizes follow P(S) ~ S^{-α}
2. **Maximum information transmission**: near-critical systems are most sensitive
   to perturbations and most predictable (Beggs & Plenz 2003)
3. **Long-range correlations**: no characteristic timescale — correlations
   decay as power laws rather than exponentially

**Neuronal Avalanches (Beggs & Plenz 2003)**

In neural tissue at criticality, cascades of activation (neuronal avalanches)
follow a power law with exponent α ≈ 3/2 = 1.5. This specific value is the
theoretical prediction for a critical branching process.

**Market Avalanches**

We define a "return cascade" (market avalanche) as a maximal sequence of
consecutive days where the absolute return exceeds a rolling threshold:

    Avalanche starts when: |r_t| > σ_threshold
    Avalanche continues while: |r_t| > σ_threshold
    Avalanche ends when: |r_t| ≤ σ_threshold

    Avalanche SIZE S = number of consecutive days in cascade

The distribution of avalanche sizes P(S) is fitted to a power law:

    P(S ≥ s) ~ s^{−(α−1)}    (complementary CDF form)

**Power-Law MLE (Clauset et al. 2009)**

For a discrete power law P(S=s) ~ s^{−α} with minimum s_min:

    α_MLE = 1 + n · [Σᵢ ln(sᵢ / (s_min − 0.5))]^{−1}

This is the maximum likelihood estimator — unbiased and consistent.

**Score Construction**

Three components:

1. **Criticality score** — how close α is to the theoretical critical value 1.5:
   c(α) = exp(−0.5 · ((α − 1.5) / σ_α)²)
   Peak at α=1.5, decays symmetrically. Near-critical → high score.

2. **Predictability score** — coefficient of variation (CV = std/mean) of
   avalanche sizes. At criticality, CV is maximised due to scale-free
   fluctuations spanning all scales. High CV → near-critical → predictable.

3. **Macro alignment** — sign agreement between criticality state and macro
   direction. Near-critical + falling VIX (risk-on) → positive signal.
   Near-critical + rising VIX (risk-off) → negative signal.

**Distinction from CRITICALITY-ENGINE (in suite)**

CRITICALITY-ENGINE detects criticality via eigenvalue ratios of the
correlation matrix (random matrix theory). This engine uses a completely
different approach: the statistics of return cascade SIZE DISTRIBUTIONS.
They are orthogonal signals — one uses cross-sectional correlation structure,
this one uses the temporal cascade dynamics of individual ETFs.

References
----------
- Bak, P., Tang, C. & Wiesenfeld, K. (1987). Self-organized criticality:
  An explanation of the 1/f noise. Physical Review Letters, 59(4), 381.
- Beggs, J.M. & Plenz, D. (2003). Neuronal avalanches in neocortical circuits.
  Journal of Neuroscience, 23(35), 11167–11177.
- Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009). Power-law distributions
  in empirical data. SIAM Review, 51(4), 661–703.
- Sornette, D. (2003). Why Stock Markets Crash. Princeton University Press.
- Chialvo, D.R. (2010). Emergent complex neural dynamics. Nature Physics, 6(10),
  744–750.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Avalanche detection ───────────────────────────────────────────────────────

def _detect_avalanches(
    log_ret:    np.ndarray,
    thresh_std: float,
    std_win:    int,
) -> List[int]:
    """
    Detect return cascades (avalanches) in a log-return series.

    An avalanche is a maximal run of consecutive days where
    |r_t| > thresh_std * rolling_std(r, std_win).

    Parameters
    ----------
    log_ret    : 1-D array of log returns
    thresh_std : threshold multiplier (σ)
    std_win    : rolling std window

    Returns
    -------
    List of avalanche sizes (integers ≥ 1)
    """
    T = len(log_ret)
    if T < std_win + 5:
        return []

    # Rolling std (forward-looking avoided — use past only)
    roll_std = np.array([
        log_ret[max(0, t-std_win):t].std() if t > std_win else log_ret[:t].std()
        for t in range(1, T+1)
    ])
    roll_std = np.where(roll_std < 1e-10, 1e-10, roll_std)

    # Binary exceedance: 1 if |r_t| > threshold
    exceed = (np.abs(log_ret) > thresh_std * roll_std).astype(int)

    # Find runs of consecutive 1s
    avalanche_sizes = []
    in_avalanche = False
    size = 0

    for e in exceed:
        if e == 1:
            in_avalanche = True
            size += 1
        else:
            if in_avalanche:
                avalanche_sizes.append(size)
                size = 0
                in_avalanche = False

    if in_avalanche and size > 0:
        avalanche_sizes.append(size)

    return avalanche_sizes


# ── Power-law MLE (Clauset et al. 2009) ──────────────────────────────────────

def _fit_powerlaw_mle(sizes: List[int], s_min: int = 1) -> Tuple[float, float]:
    """
    Fit power-law exponent α via MLE for discrete data.

    P(S=s) ~ s^{-α},  s ≥ s_min

    MLE estimator (Clauset et al. 2009, eq. 3.6):
        α = 1 + n · [Σ ln(sᵢ / (s_min − 0.5))]^{-1}

    Also computes Kolmogorov-Smirnov goodness-of-fit p-value proxy
    via the KS statistic between empirical and theoretical CCDFs.

    Returns
    -------
    alpha  : MLE power-law exponent
    ks_stat: KS statistic (lower = better fit)
    """
    s_arr = np.array([s for s in sizes if s >= s_min], dtype=float)
    n     = len(s_arr)

    if n < 2:
        return np.nan, np.nan

    # MLE
    log_sum = np.sum(np.log(s_arr / (s_min - 0.5)))
    if log_sum < 1e-10:
        return np.nan, np.nan
    alpha = 1.0 + n / log_sum

    # KS statistic between empirical CCDF and theoretical power-law CCDF
    s_sorted = np.sort(s_arr)
    n_s      = len(s_sorted)
    emp_ccdf  = 1.0 - np.arange(1, n_s+1) / n_s          # empirical CCDF
    # Theoretical CCDF: P(S ≥ s) = (s/s_min)^{-(α-1)}
    theo_ccdf = (s_sorted / s_min) ** (-(alpha - 1.0))
    theo_ccdf = np.clip(theo_ccdf, 0, 1)
    ks_stat   = float(np.max(np.abs(emp_ccdf - theo_ccdf)))

    return float(alpha), ks_stat


# ── Score components ──────────────────────────────────────────────────────────

def _criticality_score(alpha: float) -> float:
    """
    Gaussian proximity to critical exponent 1.5.
    Peak = 1.0 at α=1.5, decays with width ALPHA_SIGMA.
    """
    if np.isnan(alpha):
        return 0.0
    return float(np.exp(-0.5 * ((alpha - config.SOC_CRITICAL_ALPHA)
                                 / config.ALPHA_SIGMA) ** 2))


def _predictability_score(sizes: List[int]) -> float:
    """
    Coefficient of variation (CV = std/mean) of avalanche sizes.
    Near criticality: CV is maximised (scale-free fluctuations).
    Returns CV clipped to [0, 5].
    """
    if len(sizes) < 3:
        return 0.0
    arr = np.array(sizes, dtype=float)
    mu  = arr.mean()
    if mu < 1e-10:
        return 0.0
    return float(np.clip(arr.std() / mu, 0, 5))


def _macro_alignment(crit_score: float, macro_norm: np.ndarray) -> float:
    """
    Macro direction modifier.
    Near-critical (high crit_score) + risk-on macro → positive.
    Near-critical + risk-off macro → negative.

    macro_norm: latest normalised macro values [VIX, DXY, T10Y2Y, ...]
    VIX direction: negative = risk-on (falling VIX → positive signal)
    T10Y2Y direction: positive = risk-on (steepening)
    """
    if macro_norm.shape[0] == 0:
        return 0.0

    # Simple macro sentiment: -VIX_change + T10Y2Y_change
    # (both normalised, so just use available cols)
    sentiment = 0.0
    n_cols = macro_norm.shape[0]
    if n_cols >= 1:
        sentiment -= macro_norm[0]   # VIX: negative = risk-on
    if n_cols >= 3:
        sentiment += macro_norm[2]   # T10Y2Y: positive = risk-on

    return float(crit_score * np.sign(sentiment) * min(abs(sentiment), 2.0))


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_soc_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Compute SOC / neuronal-avalanche scores for all ETFs in the universe.

    For each ETF over the rolling window:
      1. Detect return avalanches (consecutive |r_t| > threshold)
      2. Fit power-law exponent α via MLE (Clauset 2009)
      3. Score = weighted combination of:
           - criticality (proximity of α to 1.5)
           - predictability (CV of avalanche sizes)
           - macro alignment (regime direction × criticality)

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite SOC z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.THRESH_STD_WIN + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    # Latest normalised macro vector for alignment
    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu   = np.nanmean(macro_vals, axis=0)
        m_std  = np.nanstd(macro_vals,  axis=0) + 1e-8
        # Recent change (last 21d) as direction signal
        recent = macro_vals[-21:] if len(macro_vals) >= 21 else macro_vals
        macro_latest = (recent[-1] - recent[0]) / m_std   # normalised change
    else:
        macro_latest = np.zeros(0)

    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values

        # Use last `window` bars
        ret_win = log_ret[-window:]

        # ── Detect avalanches ─────────────────────────────────────────────────
        avalanches = _detect_avalanches(
            log_ret    = ret_win,
            thresh_std = config.THRESH_SIGMA,
            std_win    = config.THRESH_STD_WIN,
        )

        if len(avalanches) < config.MIN_AVALANCHES:
            print(f"    {ticker}: only {len(avalanches)} avalanches at window={window}d, skipping")
            continue

        # ── Fit power law ─────────────────────────────────────────────────────
        alpha, ks_stat = _fit_powerlaw_mle(avalanches, s_min=config.ALPHA_S_MIN)

        if np.isnan(alpha):
            continue

        print(f"    {ticker}: n_avalanches={len(avalanches)}  "
              f"α={alpha:.3f}  KS={ks_stat:.3f}  "
              f"mean_size={np.mean(avalanches):.2f}")

        # ── Score components ──────────────────────────────────────────────────
        c_score  = _criticality_score(alpha)
        p_score  = _predictability_score(avalanches)
        m_score  = _macro_alignment(c_score, macro_latest)

        composite = (
            config.WEIGHT_CRITICALITY    * c_score
            + config.WEIGHT_PREDICTABILITY * p_score
            + config.WEIGHT_MACRO_ALIGN    * m_score
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
