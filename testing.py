"""
S&P 500 PCA — Proper Implementation
=====================================
This is the CORRECT use of PCA for financial data:
  - Many entities (stocks) observed simultaneously each day
  - PCA finds shared patterns of movement ACROSS stocks
  - PC1 = the dominant market-wide factor (systematic risk)
  - PC2, PC3 = sector/style rotations

What this does NOT do (unlike the weak TSLA version):
  - Does NOT apply PCA to a single stock's features
  - Does NOT treat PC1 as "the best indicator"
  - Does NOT confuse compression with prediction

Pipeline:
  1. Download daily prices for S&P 500 constituents
  2. Compute daily returns  → shape: [days × stocks]
  3. Normalise (standardise) each stock's returns
  4. Apply PCA across stocks
  5. Interpret components via loadings
  6. Visualise explained variance + PC time series

pip install yfinance scikit-learn matplotlib seaborn pandas numpy
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

START_DATE   = "2019-11-01"
END_DATE     = "2022-10-31"
MIN_COVERAGE = 0.90   # drop stocks with >10% missing data

# A broad sample of S&P 500 stocks across sectors
# Using ~100 liquid names to keep download time reasonable
# (A real implementation would use all ~500)
TICKERS = [
    # Technology
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","INTC","CSCO",
    "ORCL","IBM","QCOM","TXN","AVGO","NOW","ADBE","CRM","SNOW","PLTR",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","C","AXP","USB","PNC",
    "COF","SCHW","BK","TFC","MTB",
    # Healthcare
    "JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
    "GILD","CVS","CI","HUM","ISRG",
    # Energy
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","HAL","BKR",
    # Consumer Staples
    "PG","KO","PEP","WMT","COST","MO","PM","CL","GIS","K",
    # Consumer Discretionary
    "HD","MCD","NKE","SBUX","TGT","LOW","TJX","BKNG","MAR","YUM",
    # Industrials
    "BA","CAT","GE","HON","UPS","RTX","LMT","DE","EMR","ETN",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","XEL","ES","WEC","ETR",
    # Real Estate
    "AMT","PLD","CCI","EQIX","PSA","SPG","O","WELL","AVB","EQR",
    # Materials
    "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","CF",
]

# ══════════════════════════════════════════════════════════════════════
# STAGE 1: DOWNLOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════

def download_returns(tickers, start, end, min_coverage=0.90):
    """
    Downloads adjusted close prices for all tickers,
    computes daily returns, and drops stocks with too many gaps.

    Returns a DataFrame of shape [trading_days × n_stocks].
    This matrix is the INPUT to PCA — each column is one stock,
    each row is one day.
    """
    print(f"Downloading {len(tickers)} stocks from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    # Daily returns
    returns = raw.pct_change().iloc[1:]

    # Drop stocks with too many missing values
    coverage = returns.notna().mean()
    good = coverage[coverage >= min_coverage].index.tolist()
    dropped = len(tickers) - len(good)
    returns = returns[good].dropna()

    print(f"  Kept {len(good)} stocks ({dropped} dropped for missing data)")
    print(f"  Return matrix shape: {returns.shape}  "
          f"[{returns.shape[0]} days × {returns.shape[1]} stocks]")
    print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}")

    return returns


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: PCA
# ══════════════════════════════════════════════════════════════════════

def run_pca(returns_df, n_components=10):
    """
    The core of the analysis.

    Input:  returns_df  shape [days × stocks]
    Output: pc_scores   shape [days × n_components]
            loadings    shape [stocks × n_components]

    Key distinction vs the weak TSLA version:
      - Here PCA runs ACROSS STOCKS (columns), not across features of one stock
      - Each PC is a daily time series capturing a market-wide pattern
      - PC1 is the dominant shared movement — the "market factor"
    """
    print(f"\nRunning PCA with {n_components} components...")

    # Standardise each stock's returns independently
    # This prevents high-volatility stocks from dominating
    scaler = StandardScaler()
    X = scaler.fit_transform(returns_df.values)

    # Fit PCA — looking for patterns ACROSS the 100 stocks
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\nVariance explained:")
    for i, (v, c) in enumerate(zip(explained, cumulative)):
        bar = "█" * int(v * 200)
        print(f"  PC{i+1:2d}: {v:5.1%}  (cumulative: {c:5.1%})  {bar}")

    # Loadings: how much each STOCK contributes to each PC
    # Shape: [stocks × n_components]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # PC scores: the daily value of each component
    # Shape: [days × n_components]
    pc_scores = pd.DataFrame(
        scores,
        index=returns_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pc_scores, loadings, pca, explained


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: INTERPRET COMPONENTS
# ══════════════════════════════════════════════════════════════════════

def interpret_components(loadings, n_show=5):
    """
    For each PC, print which stocks have the highest loadings.
    This is where a human inspects the output and decides what
    each PC represents — PCA itself gives no labels.

    CRITICAL NOTE: The labels below are HUMAN INTERPRETATIONS.
    PCA just outputs numbers. We look at which stocks cluster
    together and infer a theme.
    """
    print("\n" + "="*60)
    print("COMPONENT INTERPRETATION")
    print("(Labels are human interpretations — PCA gives no names)")
    print("="*60)

    for pc in loadings.columns:
        col = loadings[pc]
        top_pos = col.nlargest(n_show)
        top_neg = col.nsmallest(n_show)

        print(f"\n{pc}:")
        print(f"  Stocks loading positively (move WITH this factor):")
        for ticker, val in top_pos.items():
            print(f"    {ticker:6s}  {val:+.3f}")
        print(f"  Stocks loading negatively (move AGAINST this factor):")
        for ticker, val in top_neg.items():
            print(f"    {ticker:6s}  {val:+.3f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 4: VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def plot_all(returns_df, pc_scores, loadings, explained):
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(4, 2, hspace=0.5, wspace=0.35)

    # ── 1. Explained variance bar chart ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    pcs = [f"PC{i+1}" for i in range(len(explained))]
    ax1.bar(pcs, explained * 100, color="steelblue", edgecolor="white")
    ax1.set_title("Variance Explained by Each PC\n"
                  "(How much of total market movement each factor captures)",
                  fontsize=10)
    ax1.set_ylabel("% Variance Explained")
    ax1.set_xlabel("Principal Component")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(explained):
        ax1.text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", fontsize=8)

    # ── 2. Cumulative explained variance ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative = np.cumsum(explained) * 100
    ax2.plot(pcs, cumulative, marker="o", color="darkorange", lw=2)
    ax2.axhline(80, color="red", ls="--", lw=1, label="80% threshold")
    ax2.fill_between(range(len(pcs)), cumulative, alpha=0.15, color="darkorange")
    ax2.set_title("Cumulative Variance Explained\n"
                  "(How many PCs needed to capture most market behaviour)",
                  fontsize=10)
    ax2.set_ylabel("Cumulative % Explained")
    ax2.set_xlabel("Principal Component")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(range(len(pcs)))
    ax2.set_xticklabels(pcs)

    # ── 3. PC1 time series — the market factor ────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    pc1 = pc_scores["PC1"]
    colors = ["red" if v < 0 else "steelblue" for v in pc1]
    ax3.bar(pc1.index, pc1.values, color=colors, width=1, alpha=0.7)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("PC1 Daily Score — The Market Factor\n"
                  "Blue = market broadly up together | Red = market broadly down together\n"
                  "This is NOT one stock — it's the shared movement of ALL stocks",
                  fontsize=10)
    ax3.set_ylabel("PC1 Score")
    ax3.grid(axis="y", alpha=0.3)

    # Annotate COVID crash
    covid_date = pd.Timestamp("2020-03-16")
    if covid_date in pc1.index:
        ax3.annotate("COVID crash\n(all stocks fall together\n→ huge negative PC1)",
                     xy=(covid_date, pc1.loc[covid_date]),
                     xytext=(covid_date + pd.Timedelta(days=80),
                             pc1.min() * 0.7),
                     arrowprops=dict(arrowstyle="->", color="black"),
                     fontsize=8, color="darkred")

    # ── 4. PC1 loadings — which stocks contribute most ────────────
    ax4 = fig.add_subplot(gs[2, 0])
    pc1_load = loadings["PC1"].sort_values()
    colors_load = ["red" if v < 0 else "steelblue" for v in pc1_load]
    ax4.barh(range(len(pc1_load)), pc1_load.values, color=colors_load, alpha=0.7)
    ax4.set_yticks(range(len(pc1_load)))
    ax4.set_yticklabels(pc1_load.index, fontsize=6)
    ax4.axvline(0, color="black", lw=0.8)
    ax4.set_title("PC1 Loadings by Stock\n"
                  "All positive = all stocks move together\n"
                  "(This confirms PC1 = market-wide factor)",
                  fontsize=9)
    ax4.set_xlabel("Loading")
    ax4.grid(axis="x", alpha=0.3)

    # ── 5. PC2 loadings — sector rotation ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    pc2_load = loadings["PC2"].sort_values()
    colors_load2 = ["red" if v < 0 else "steelblue" for v in pc2_load]
    ax5.barh(range(len(pc2_load)), pc2_load.values, color=colors_load2, alpha=0.7)
    ax5.set_yticks(range(len(pc2_load)))
    ax5.set_yticklabels(pc2_load.index, fontsize=6)
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set_title("PC2 Loadings by Stock\n"
                  "Mix of positive/negative = sector rotation factor\n"
                  "(Some sectors rise while others fall)",
                  fontsize=9)
    ax5.set_xlabel("Loading")
    ax5.grid(axis="x", alpha=0.3)

    # ── 6. Loadings heatmap (top PCs) ─────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    # Show top 30 stocks by absolute PC1 loading
    top30 = loadings["PC1"].abs().nlargest(30).index
    heat_data = loadings.loc[top30, ["PC1","PC2","PC3","PC4","PC5"]].T
    sns.heatmap(heat_data, ax=ax6, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, cbar_kws={"shrink": 0.5})
    ax6.set_title("Loadings Heatmap — Top 30 Stocks by PC1 Contribution\n"
                  "Rows = PCs, Columns = stocks. Colour = how much that stock "
                  "contributes to each PC",
                  fontsize=9)
    ax6.set_xlabel("Stock")
    ax6.set_ylabel("Principal Component")

    plt.suptitle("S&P 500 PCA — Finding Hidden Market Structure\n"
                 "PCA runs ACROSS 100+ stocks simultaneously, not on one stock's features",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig("sp500_pca_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: sp500_pca_results.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 5: CORRELATION CHECK
# ══════════════════════════════════════════════════════════════════════

def correlation_check(returns_df, pc_scores):
    """
    Validates PC1 by checking its correlation with the equal-weighted
    average return across all stocks — a proxy for the market.

    If PC1 truly captures the market factor, it should correlate
    highly (>0.9) with the average stock return each day.
    """
    market_avg = returns_df.mean(axis=1)
    corr = pc_scores["PC1"].corr(market_avg)
    print(f"\nValidation: PC1 correlation with equal-weighted market return: {corr:.3f}")
    if abs(corr) > 0.90:
        print("  ✓ Very high — PC1 genuinely represents the market factor")
    elif abs(corr) > 0.70:
        print("  ~ Moderate — PC1 partly represents the market")
    else:
        print("  ✗ Low — PC1 may not be the market factor; check your data")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*60)
    print("S&P 500 PCA — PROPER IMPLEMENTATION")
    print("="*60)
    print("\nKey difference from single-stock PCA:")
    print("  Input matrix: [trading days × ~100 stocks]")
    print("  PCA finds patterns SHARED ACROSS STOCKS")
    print("  PC1 = the force that moves all stocks together")
    print()

    # 1. Get the return matrix — the correct input for this type of PCA
    returns = download_returns(TICKERS, START_DATE, END_DATE, MIN_COVERAGE)

    # 2. Run PCA across the stock dimension
    pc_scores, loadings, pca_model, explained = run_pca(returns, n_components=10)

    # 3. Interpret — human reads the loadings and assigns meaning
    interpret_components(loadings, n_show=5)

    # 4. Validate that PC1 is genuinely the market factor
    correlation_check(returns, pc_scores)

    # 5. Visualise everything
    print("\nGenerating plots...")
    plot_all(returns, pc_scores, loadings, explained)

    print("\nDone.")
    print("\nSummary of what PCA found:")
    print(f"  PC1 explains {explained[0]:.1%} of all daily variance across stocks")
    print(f"  Top 3 PCs explain {explained[:3].sum():.1%} combined")
    print(f"  Top 5 PCs explain {explained[:5].sum():.1%} combined")
    print(f"\n  This means ~{int(explained[0]*100)}% of why any stock moves on a given")
    print(f"  day is explained by the single market-wide factor (PC1).")
    print(f"  The remaining variance is sector/stock-specific.")