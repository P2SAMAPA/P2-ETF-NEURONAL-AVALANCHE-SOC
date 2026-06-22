# 🌊 P2-ETF-SOC-AVALANCHE

**Neuronal Avalanche / Self-Organised Criticality Engine**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine detects whether each ETF's return dynamics are operating **near
criticality** — the phase transition between ordered (trending) and disordered
(chaotic) regimes — by analysing the size distribution of return cascades
("market avalanches") using methods from computational neuroscience.

The key insight: near-critical systems have the highest predictability and
the most efficient information transmission. ETFs near criticality with
favourable macro alignment offer the strongest signal.

---

## Theory

### Self-Organised Criticality (Bak, Tang & Wiesenfeld 1987)

SOC describes systems that spontaneously evolve to a critical state without
external parameter tuning. At the critical point:

- Fluctuation sizes follow a **power law**: P(S) ~ S^{−α}
- The system exhibits **maximum sensitivity** to perturbations
- Correlations decay as **power laws** (no characteristic timescale)

### Neuronal Avalanches (Beggs & Plenz 2003)

In neural tissue at criticality, cascade sizes follow P(S) ~ S^{−3/2},
giving the universal critical exponent **α = 1.5**. Markets near criticality
show the same statistic in their return cascade distributions.

### Market Avalanche Definition

```
Avalanche = maximal run of consecutive days where |r_t| > 1.0σ
Avalanche size S = number of days in the cascade
```

The rolling 21-day σ provides an adaptive threshold that adjusts to regime.

### Power-Law MLE (Clauset et al. 2009)

For discrete P(S=s) ~ s^{−α}, the MLE estimator is:

```
α_MLE = 1 + n · [Σᵢ ln(Sᵢ / (s_min − 0.5))]⁻¹
```

Validated via KS statistic between empirical and theoretical CCDFs.

### α Interpretation

| α value | Regime | Market character |
|---------|--------|-----------------|
| α < 1.5 | Supercritical | Large cascades dominate (chaotic/trending) |
| **α ≈ 1.5** | **Critical** | **Maximum predictability** |
| α > 1.5 | Subcritical | Small cascades dominate (mean-reverting) |

---

## Score Construction

```
score = 0.50 · c(α)  +  0.30 · CV(S)  +  0.20 · macro_align
```

| Component | Formula | Meaning |
|-----------|---------|---------|
| Criticality | c(α) = exp(−½·((α−1.5)/0.5)²) | Proximity to critical exponent |
| Predictability | CV = std(S)/mean(S) | Scale-free fluctuation width (max at criticality) |
| Macro alignment | c(α) × sign(macro sentiment) | Direction × confidence |

Final score: **cross-sectional z-score** per universe per window.

---

## Distinction from CRITICALITY-ENGINE

| | CRITICALITY-ENGINE | **SOC-AVALANCHE (this engine)** |
|---|---|---|
| Method | Eigenvalue ratios (RMT) | Return cascade size distribution |
| Domain | Cross-sectional (all ETFs) | Per-ETF temporal dynamics |
| Theory | Random Matrix Theory | Self-Organised Criticality / branching processes |
| Signal | Distance from Marchenko-Pastur | Distance of α from 1.5 |

Orthogonal signals — both can be used simultaneously.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-SOC-AVALANCHE/
├── config.py          # Universes, avalanche threshold, power-law params
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── soc_engine.py      # Core: avalanche detection, power-law MLE, scoring
├── trainer.py         # Orchestrator: load → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (pure numpy, very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-SOC-AVALANCHE
cd P2-ETF-SOC-AVALANCHE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-soc-avalanche-results`

---

## References

- Bak, P., Tang, C. & Wiesenfeld, K. (1987). Self-organized criticality:
  An explanation of the 1/f noise. *Physical Review Letters*, 59(4), 381.
- Beggs, J.M. & Plenz, D. (2003). Neuronal avalanches in neocortical circuits.
  *Journal of Neuroscience*, 23(35), 11167–11177.
- Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009). Power-law distributions
  in empirical data. *SIAM Review*, 51(4), 661–703.
- Sornette, D. (2003). *Why Stock Markets Crash*. Princeton University Press.
- Chialvo, D.R. (2010). Emergent complex neural dynamics.
  *Nature Physics*, 6(10), 744–750.
- Bak, P. (1996). *How Nature Works: The Science of Self-Organised Criticality*.
  Copernicus Press.
