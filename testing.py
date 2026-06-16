"""
S&P 500 PCA — Proper Implementation (Advanced Version)
=========================================================
This is the CORRECT use of PCA for financial data:
  - Many entities (stocks) observed simultaneously each day
  - PCA finds shared patterns of movement ACROSS stocks
  - PC1 = the dominant market-wide factor (systematic risk)
  - PC2, PC3 = sector/style rotations

Advanced Features (suitable for year 2 mathematics project):
  - Kaiser criterion & scree plot for formal component selection
  - Biplot visualization (scores + loadings in same space)
  - Bootstrap confidence intervals on eigenvalues (resampling)
  - Time series analysis: stationarity & autocorrelation tests
  - Variance decomposition: which stocks drive each PC

What this does NOT do (unlike the weak TSLA version):
  - Does NOT apply PCA to a single stock's features
  - Does NOT treat PC1 as "the best indicator"
  - Does NOT confuse compression with prediction

Pipeline:
  1. Download daily prices for S&P 500 constituents
  2. Compute daily returns  → shape: [days × stocks]
  3. Normalise (standardise) each stock's returns
  4. Apply PCA across stocks
  5. Component selection (Kaiser, scree, cumulative variance)
  6. Bootstrap validation (statistical rigor)
  7. Time series properties (mean-reversion, persistence)
  8. Interpret components via loadings & visualization

pip install yfinance scikit-learn matplotlib seaborn pandas numpy statsmodels scipy
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import normaltest
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
# STAGE 6: ADVANCED — COMPONENT SELECTION (Kaiser Criterion, Scree Plot)
# ══════════════════════════════════════════════════════════════════════

def scree_plot_analysis(pca_model, explained):
    """
    Mathematically rigorous approach to choosing number of components.

    Methods:
      1. Kaiser Criterion: keep components with eigenvalue > 1 (variance > 1)
      2. Scree Plot: visual inspection of where the curve flattens (elbow)
      3. Cumulative Variance: keep until reaching 80-90% threshold
    """
    eigenvalues = pca_model.explained_variance_
    n_components_kaiser = (eigenvalues > 1).sum()

    print("\n" + "="*60)
    print("COMPONENT SELECTION ANALYSIS")
    print("="*60)
    print(f"\nKaiser Criterion (eigenvalue > 1):")
    print(f"  Suggested components: {n_components_kaiser}")
    print(f"  Rationale: only keep components explaining more variance")
    print(f"             than a single raw variable")

    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: eigenvalues with Kaiser line
    pcs = range(1, len(eigenvalues) + 1)
    ax1.plot(pcs, eigenvalues, "bo-", linewidth=2, markersize=8)
    ax1.axhline(1, color="red", linestyle="--", linewidth=2, label="Kaiser criterion (λ=1)")
    ax1.fill_between(pcs, eigenvalues, 1, where=(eigenvalues > 1),
                      alpha=0.3, color="green", label="Retained")
    ax1.fill_between(pcs, eigenvalues, 1, where=(eigenvalues <= 1),
                      alpha=0.3, color="red", label="Discarded")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue (λ)")
    ax1.set_title("Scree Plot — Eigenvalue Decay\n"
                  "Elbow indicates where to stop retaining components")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: cumulative variance with thresholds
    cumsum = np.cumsum(explained)
    ax2.plot(pcs, cumsum * 100, "go-", linewidth=2, markersize=8)
    ax2.axhline(80, color="orange", linestyle="--", linewidth=1.5, label="80% threshold")
    ax2.axhline(90, color="red", linestyle="--", linewidth=1.5, label="90% threshold")
    n_80 = (cumsum >= 0.80).argmax() + 1
    n_90 = (cumsum >= 0.90).argmax() + 1
    ax2.plot(n_80, 80, "o", markersize=10, color="orange")
    ax2.plot(n_90, 90, "o", markersize=10, color="red")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance — When to Stop")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig("scree_plot_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nEmpirical thresholds:")
    print(f"  80% variance: {n_80} components")
    print(f"  90% variance: {n_90} components")
    print(f"\nSaved: scree_plot_analysis.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 7: BIPLOT — Scores vs Loadings Combined
# ══════════════════════════════════════════════════════════════════════

def biplot(pc_scores, loadings, ticker_labels, pc_x=0, pc_y=1):
    """
    Biplot visualization: Shows BOTH:
      - Scores: where each observation (day) sits in PC space
      - Loadings: arrows showing how each variable (stock) contributes

    Mathematically: maps both X (observations) and P (loadings) into the same space.
    This reveals both which days were "extreme" and which stocks drove them.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    pc1_name = f"PC{pc_x + 1}"
    pc2_name = f"PC{pc_y + 1}"

    # Plot scores (observations/days)
    scatter = ax.scatter(pc_scores.iloc[:, pc_x],
                         pc_scores.iloc[:, pc_y],
                         alpha=0.4, s=20, c=range(len(pc_scores)),
                         cmap="viridis", label="Trading days")

    # Plot loadings as arrows (variables/stocks)
    scale_factor = 3.5
    for i, ticker in enumerate(loadings.index):
        ax.arrow(0, 0,
                 loadings.iloc[i, pc_x] * scale_factor,
                 loadings.iloc[i, pc_y] * scale_factor,
                 head_width=0.1, head_length=0.1, fc="red", ec="darkred", alpha=0.6)
        ax.text(loadings.iloc[i, pc_x] * scale_factor * 1.1,
                loadings.iloc[i, pc_y] * scale_factor * 1.1,
                ticker, fontsize=7, ha="center")

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel(f"{pc1_name} ({(pc_scores.iloc[:, pc_x].var() / pc_scores.var(axis=0).sum()):.1%} variance)")
    ax.set_ylabel(f"{pc2_name} ({(pc_scores.iloc[:, pc_y].var() / pc_scores.var(axis=0).sum()):.1%} variance)")
    ax.set_title(f"Biplot: {pc1_name} vs {pc2_name}\n"
                 "Blue dots = trading days | Red arrows = stock contributions",
                 fontsize=11)
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Days (earlier → later)")
    plt.tight_layout()
    plt.savefig("biplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: biplot.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 8: BOOTSTRAP CONFIDENCE INTERVALS on Eigenvalues
# ══════════════════════════════════════════════════════════════════════

def bootstrap_eigenvalues(returns_df, n_bootstrap=100, n_components=10):
    """
    Resample trading days WITH REPLACEMENT, recompute PCA each time.
    This gives confidence intervals around eigenvalues — are they statistically stable?

    A robust eigenvalue has a tight CI; a fragile one has a wide CI.
    """
    print("\nBootstrap analysis (resampling days with replacement)...")
    bootstrap_eigenvalues_list = []

    for b in range(n_bootstrap):
        if (b + 1) % 20 == 0:
            print(f"  {b+1}/{n_bootstrap}")

        # Resample rows (days) with replacement
        idx = np.random.choice(len(returns_df), size=len(returns_df), replace=True)
        boot_data = returns_df.iloc[idx]

        # Run PCA
        scaler = StandardScaler()
        X = scaler.fit_transform(boot_data.values)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        bootstrap_eigenvalues_list.append(pca.explained_variance_)

    bootstrap_eigenvalues = np.array(bootstrap_eigenvalues_list)

    # Compute confidence intervals
    ci_lower = np.percentile(bootstrap_eigenvalues, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_eigenvalues, 97.5, axis=0)
    ci_mean = bootstrap_eigenvalues.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pcs = range(1, n_components + 1)

    # Plot with error bars
    ax.errorbar(pcs, ci_mean, 
                yerr=[ci_mean - ci_lower, ci_upper - ci_mean],
                fmt="o-", linewidth=2, markersize=8, capsize=5,
                label="95% CI from bootstrap", color="steelblue")
    ax.axhline(1, color="red", linestyle="--", label="Kaiser criterion")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue (λ)")
    ax.set_title("Bootstrap Confidence Intervals on Eigenvalues\n"
                 "Tight bands = stable components | Wide bands = unstable")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bootstrap_eigenvalues.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: bootstrap_eigenvalues.png")

    # Print widths
    print("\nEigenvalue stability (CI width):")
    for i, (mean, lower, upper) in enumerate(zip(ci_mean, ci_lower, ci_upper)):
        width = upper - lower
        print(f"  PC{i+1}: {mean:.3f}  ±{width/2:.3f}  "
              f"[{lower:.3f}, {upper:.3f}]")


# ══════════════════════════════════════════════════════════════════════
# STAGE 9: TIME SERIES ANALYSIS OF PC Scores
# ══════════════════════════════════════════════════════════════════════

def analyze_pc_timeseries(pc_scores):
    """
    Treat PC scores as time series. Analyse:
      - Autocorrelation (do yesterday's scores predict today?)
      - Stationarity (mean/variance changing over time?)
      - Volatility clustering (do calm periods predict calm?)
    """
    from scipy.stats import normaltest
    from statsmodels.tsa.stattools import adfuller, acf

    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS OF PC SCORES")
    print("="*60)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for pc_idx in range(min(3, len(pc_scores.columns))):
        pc_col = pc_scores.columns[pc_idx]
        series = pc_scores[pc_col]

        # Row for this PC
        row = pc_idx

        # ── Left: Time series plot ─────────────────────────────
        ax = axes[row, 0]
        ax.plot(series.index, series.values, linewidth=0.8, color="steelblue")
        ax.set_title(f"{pc_col} Over Time")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)

        # ── Right: Autocorrelation ─────────────────────────────
        ax = axes[row, 1]
        acf_vals = acf(series.dropna(), nlags=30)
        ax.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax.axhline(1.96 / np.sqrt(len(series)), color="r", linestyle="--", linewidth=1)
        ax.axhline(-1.96 / np.sqrt(len(series)), color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{pc_col} Autocorrelation (95% bounds)")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_ylim([-0.3, 1])

        # Print stats
        print(f"\n{pc_col}:")
        result = adfuller(series.dropna(), autolag="AIC")
        print(f"  ADF test p-value: {result[1]:.4f}", end="")
        if result[1] < 0.05:
            print("  → Stationary (mean-reverting)")
        else:
            print("  → Non-stationary (persistent)")
        print(f"  ACF(1): {acf_vals[1]:.3f} (autocorrelation with yesterday)")

    plt.tight_layout()
    plt.savefig("pc_timeseries_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: pc_timeseries_analysis.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 10: VARIANCE DECOMPOSITION — Which stocks explain PC variance?
# ══════════════════════════════════════════════════════════════════════

def variance_decomposition(loadings, n_show=15):
    """
    For each PC, decompose its variance into contributions from each stock.

    Variance of PC_j = sum over stocks of (loading_ij)^2

    Shows which stocks are most important for explaining each component.
    """
    print("\n" + "="*60)
    print("VARIANCE DECOMPOSITION BY STOCK")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for pc_idx in range(min(4, len(loadings.columns))):
        pc_col = loadings.columns[pc_idx]
        # Squared loadings = variance contribution
        contributions = (loadings[pc_col] ** 2).sort_values(ascending=True)
        top_contrib = contributions.tail(n_show)

        ax = axes[pc_idx]
        colors = ["steelblue" if x > 0 else "coral" for x in loadings[pc_col][top_contrib.index]]
        ax.barh(range(len(top_contrib)), top_contrib.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_contrib)))
        ax.set_yticklabels(top_contrib.index, fontsize=9)
        ax.set_xlabel("Squared Loading (λ²)")
        ax.set_title(f"{pc_col} — Top {n_show} Stock Contributions")
        ax.grid(axis="x", alpha=0.3)

        # Print top 5
        top5 = contributions.tail(5)
        print(f"\n{pc_col} (total variance = {contributions.sum():.3f}):")
        for ticker, val in top5.items():
            print(f"  {ticker}: {val:.4f}  ({val/contributions.sum():.1%})")

    plt.tight_layout()
    plt.savefig("variance_decomposition.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: variance_decomposition.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 11: ROLLING PCA — Time Evolution of Market Structure
# ══════════════════════════════════════════════════════════════════════

def rolling_pca(returns_df, window=60, n_components=5):
    """
    Runs PCA repeatedly on a sliding window of trading days.

    Instead of one static PCA on all data, we ask:
      "What does the market structure look like in THIS 60-day window?"

    Then we slide the window forward one day at a time.

    This turns PCA from a single static result into a time series of
    eigenvalues and loadings — revealing how market structure EVOLVES.

    Mathematical insight:
      The covariance matrix Σ(t) changes over time as new data enters
      and old data leaves the window. Its eigenvectors and eigenvalues
      therefore also change — the market's hidden structure is not fixed.

    Parameters:
        window      : number of trading days per window (default 60 = ~3 months)
        n_components: how many PCs to track per window

    Returns a dict of time series:
        dates           : end date of each window
        pc1_variance    : % variance explained by PC1 in each window
        top3_variance   : % variance explained by PC1-3 combined
        avg_correlation : average pairwise correlation between stocks
        pc1_loadings    : DataFrame of PC1 loadings for each window (how each
                          stock's contribution to PC1 changes over time)
    """
    print(f"\nRolling PCA — window={window} days, "
          f"sliding across {len(returns_df)} total days...")
    print(f"  This runs PCA {len(returns_df) - window} times — one per window position")

    dates           = []
    pc1_variance    = []
    top3_variance   = []
    avg_correlation = []
    pc1_loadings_list = []

    scaler = StandardScaler()

    for i in range(window, len(returns_df)):

        # Slice the window: days i-window to i
        window_data = returns_df.iloc[i - window : i]

        # Standardise within this window only
        # (mean and std computed fresh for each window)
        X = scaler.fit_transform(window_data.values)

        # Run PCA on this window
        pca = PCA(n_components=n_components)
        pca.fit(X)

        ev = pca.explained_variance_ratio_

        # Record the end date of this window
        dates.append(returns_df.index[i])

        # How dominant is PC1 right now?
        pc1_variance.append(ev[0])

        # How much do top 3 PCs capture?
        top3_variance.append(ev[:3].sum())

        # Average pairwise correlation across all stocks in this window
        # High correlation = stocks moving together = market stress
        corr_matrix = pd.DataFrame(X).corr().values
        # Take upper triangle only (avoid double-counting and diagonal)
        upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_correlation.append(upper.mean())

        # Store PC1 loadings (which stocks drive PC1 right now?)
        pc1_loadings_list.append(pca.components_[0])

        if (i - window) % 100 == 0:
            print(f"  Window {i - window}/{len(returns_df) - window}  "
                  f"({returns_df.index[i].date()})  "
                  f"PC1={ev[0]:.1%}")

    # Package everything into a tidy dict
    dates = pd.DatetimeIndex(dates)

    pc1_loadings_df = pd.DataFrame(
        pc1_loadings_list,
        index=dates,
        columns=returns_df.columns
    )

    print(f"\nRolling PCA complete — {len(dates)} windows computed")

    return {
        "dates"          : dates,
        "pc1_variance"   : pd.Series(pc1_variance,    index=dates),
        "top3_variance"  : pd.Series(top3_variance,   index=dates),
        "avg_correlation": pd.Series(avg_correlation, index=dates),
        "pc1_loadings"   : pc1_loadings_df,
    }


def plot_rolling_pca(rolling, returns_df):
    """
    Four panels showing how market structure evolves over time:

    1. PC1 variance over time
       - Spikes = crisis periods (all stocks suddenly correlated)
       - Troughs = calm markets (stocks moving independently)

    2. Average pairwise correlation over time
       - Confirms: when correlation spikes, PC1 also spikes
       - They should move together — a validation check

    3. Top 3 PCs combined variance
       - How well can we describe the market with just 3 factors?
       - Stable line = consistent market structure

    4. PC1 loadings heatmap over time
       - Rows = stocks, Columns = dates
       - Shows which stocks dominate the market factor at each point
       - Bright colour = high loading at that time
    """
    fig = plt.figure(figsize=(16, 18))
    gs  = plt.GridSpec(4, 1, hspace=0.5)

    # ── Panel 1: PC1 variance over time ───────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(rolling["dates"], rolling["pc1_variance"] * 100,
             color="steelblue", lw=1.5, label="PC1 variance explained")
    ax1.fill_between(rolling["dates"], rolling["pc1_variance"] * 100,
                     alpha=0.3, color="steelblue")

    # Mark significant spikes (PC1 > 75th percentile)
    threshold = rolling["pc1_variance"].quantile(0.90)
    spikes = rolling["pc1_variance"] > threshold
    ax1.fill_between(rolling["dates"],
                     rolling["pc1_variance"] * 100,
                     where=spikes,
                     color="red", alpha=0.4, label="Crisis periods (top 10%)")

    # Annotate COVID
    covid = pd.Timestamp("2020-03-16")
    if rolling["dates"].min() <= covid <= rolling["dates"].max():
        ax1.axvline(covid, color="darkred", lw=1.5, ls="--", alpha=0.7)
        ax1.text(covid, ax1.get_ylim()[1] * 0.9, "COVID\ncrash",
                 color="darkred", fontsize=8, ha="center")

    ax1.set_title("PC1 Variance Explained Over Time (Rolling 60-day Window)\n"
                  "Spikes = crisis: all stocks suddenly correlated, "
                  "one factor dominates",
                  fontsize=10)
    ax1.set_ylabel("PC1 Variance (%)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Average pairwise correlation ─────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(rolling["dates"], rolling["avg_correlation"],
             color="darkorange", lw=1.5, label="Avg pairwise correlation")
    ax2.fill_between(rolling["dates"], rolling["avg_correlation"],
                     alpha=0.3, color="darkorange")
    ax2.axhline(rolling["avg_correlation"].mean(), color="black",
                lw=1, ls="--", label="Long-run average")

    if rolling["dates"].min() <= covid <= rolling["dates"].max():
        ax2.axvline(covid, color="darkred", lw=1.5, ls="--", alpha=0.7)

    ax2.set_title("Average Pairwise Stock Correlation Over Time\n"
                  "High correlation = stocks moving in lockstep = "
                  "should match PC1 spikes above (validation check)",
                  fontsize=10)
    ax2.set_ylabel("Avg Correlation")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Top 3 PCs combined variance ──────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(rolling["dates"], rolling["top3_variance"] * 100,
             color="seagreen", lw=1.5, label="PC1+PC2+PC3 combined")
    ax3.plot(rolling["dates"], rolling["pc1_variance"] * 100,
             color="steelblue", lw=1, alpha=0.6, label="PC1 alone")
    ax3.fill_between(rolling["dates"],
                     rolling["pc1_variance"] * 100,
                     rolling["top3_variance"] * 100,
                     alpha=0.3, color="seagreen",
                     label="PC2+PC3 contribution")

    if rolling["dates"].min() <= covid <= rolling["dates"].max():
        ax3.axvline(covid, color="darkred", lw=1.5, ls="--", alpha=0.7)

    ax3.set_title("Top 3 PCs Combined Variance Over Time\n"
                  "Gap between green and blue = how much sector factors "
                  "(PC2, PC3) contribute beyond the market factor",
                  fontsize=10)
    ax3.set_ylabel("Cumulative Variance (%)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── Panel 4: PC1 loadings heatmap ─────────────────────────────
    ax4 = fig.add_subplot(gs[3])

    # Pick top 20 stocks by average absolute PC1 loading over all windows
    mean_abs_loading = rolling["pc1_loadings"].abs().mean()
    top20 = mean_abs_loading.nlargest(20).index

    # Transpose: rows=stocks, columns=dates
    heat = rolling["pc1_loadings"][top20].T

    # Sample every 10th date to avoid overcrowding
    sampled_cols = heat.columns[::10]
    heat_sampled = heat[sampled_cols]

    sns.heatmap(heat_sampled, ax=ax4, cmap="coolwarm", center=0,
                cbar_kws={"label": "PC1 Loading", "shrink": 0.5},
                xticklabels=[d.strftime("%Y-%m") for d in sampled_cols],
                yticklabels=top20)
    ax4.set_title("How PC1 Loadings Change Over Time (Top 20 Stocks)\n"
                  "Each column = one point in time. Colour = how strongly "
                  "that stock drives the market factor",
                  fontsize=10)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Stock")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax4.tick_params(axis="y", labelsize=8)

    plt.suptitle("Rolling PCA — Market Structure Is Not Static\n"
                 "Each point computed from a fresh 60-day window of data",
                 fontsize=13, fontweight="bold")

    plt.savefig("rolling_pca.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rolling_pca.png")

    # Print key findings
    peak_date   = rolling["pc1_variance"].idxmax()
    trough_date = rolling["pc1_variance"].idxmin()
    print(f"\nKey Rolling PCA Findings:")
    print(f"  PC1 most dominant:  {peak_date.date()}  "
          f"({rolling['pc1_variance'][peak_date]:.1%} variance) "
          f"← market most correlated")
    print(f"  PC1 least dominant: {trough_date.date()}  "
          f"({rolling['pc1_variance'][trough_date]:.1%} variance) "
          f"← stocks moving independently")
    print(f"  Avg PC1 variance: "
          f"{rolling['pc1_variance'].mean():.1%}")
    print(f"  Avg pairwise correlation: "
          f"{rolling['avg_correlation'].mean():.3f}")


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

    # 6. ADVANCED: Component selection with Kaiser criterion
    print("\n\nADVANCED ANALYSIS")
    print("="*60)
    scree_plot_analysis(pca_model, explained)

    # 7. ADVANCED: Biplot (scores + loadings)
    print("\nGenerating biplot...")
    biplot(pc_scores, loadings, returns.columns, pc_x=0, pc_y=1)

    # 8. ADVANCED: Bootstrap confidence intervals
    print("\nRunning bootstrap resampling (this may take 30-60 seconds)...")
    bootstrap_eigenvalues(returns, n_bootstrap=200, n_components=10)

    # 9. ADVANCED: Time series analysis
    print("\nAnalyzing PC scores as time series...")
    analyze_pc_timeseries(pc_scores)

    # 10. ADVANCED: Variance decomposition
    print("\nVariance decomposition...")
    variance_decomposition(loadings, n_show=15)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nSummary of what PCA found:")
    print(f"  PC1 explains {explained[0]:.1%} of all daily variance across stocks")
    print(f"  Top 3 PCs explain {explained[:3].sum():.1%} combined")
    print(f"  Top 5 PCs explain {explained[:5].sum():.1%} combined")
    print(f"\n  This means ~{int(explained[0]*100)}% of why any stock moves on a given")
    print(f"  day is explained by the single market-wide factor (PC1).")
    print(f"  The remaining variance is sector/stock-specific.")
    # 11. ROLLING PCA — time evolution of market structure
    print("\nRunning Rolling PCA...")
    rolling_results = rolling_pca(returns, window=60, n_components=5)
    plot_rolling_pca(rolling_results, returns)

    print("\nGenerated plots:")
    print("  1. sp500_pca_results.png       — Main analysis")
    print("  2. scree_plot_analysis.png     — Component selection")
    print("  3. biplot.png                  — Scores & loadings combined")
    print("  4. bootstrap_eigenvalues.png   — Confidence intervals")
    print("  5. pc_timeseries_analysis.png  — Time series properties")
    print("  6. variance_decomposition.png  — Which stocks drive each PC")
    print("  7. rolling_pca.png             — Time evolution of market structure")