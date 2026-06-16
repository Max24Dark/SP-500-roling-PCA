"""
PC1 Early Warning Test
======================

Question:
    Does PC1 provide an early warning of market crisis?

Idea:
    - Download prices for a basket of US stocks.
    - Convert prices to daily returns.
    - Run a rolling PCA on those returns.
    - Track only one number: how much variance PC1 explains.
    - For each crisis date, compare the 20 trading days before the crisis
      with the previous 120 trading days.

Why this is a good student-friendly test:
    - PC1 explained variance is easy to interpret.
    - The setup is an event study, which is a standard economics/statistics idea.
    - A moving block bootstrap is used because rolling PCA values overlap from day to day,
      so a plain t-test would be too optimistic.

The script prints a small table and can also save a plot.

Install:
    pip install yfinance pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "CSCO",
    "ORCL", "IBM", "QCOM", "TXN", "AVGO", "NOW", "ADBE", "CRM", "JPM", "BAC",
    "WFC", "GS", "MS", "BLK", "C", "AXP", "UNH", "PFE", "ABBV", "MRK",
    "TMO", "ABT", "DHR", "XOM", "CVX", "COP", "SLB", "PG", "KO", "PEP",
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "BA", "CAT", "GE", "HON",
    "UPS", "RTX", "LMT", "NEE", "DUK", "SO", "AMT", "PLD", "EQIX", "SPG",
]

CRISIS_EVENTS = [
    {"name": "COVID crash", "date": "2020-03-16"},
    {"name": "2022 inflation / rate shock", "date": "2022-06-13"},
    {"name": "US regional banking stress", "date": "2023-03-13"},
]

DEFAULT_START = "2007-01-01"
DEFAULT_END = "today"
DEFAULT_WINDOW = 100
DEFAULT_WARNING_DAYS = 40
DEFAULT_BASELINE_DAYS = 120
DEFAULT_N_BOOT = 2000
DEFAULT_MIN_COVERAGE = 0.90


# -----------------------------
# Data and PCA helpers
# -----------------------------

def resolve_end_date(end_date: str) -> str:
    if str(end_date).strip().lower() == "today":
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return end_date


def download_returns(
    tickers: List[str],
    start_date: str,
    end_date: str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Download adjusted close prices and convert them to daily returns."""
    print(f"Downloading {len(tickers)} stocks from {start_date} to {end_date}...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    returns = raw.pct_change().iloc[1:]

    coverage = returns.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    returns = returns[keep].dropna()

    print(f"Kept {len(keep)} stocks after coverage filter ({min_coverage:.0%}).")
    print(f"Return matrix: {returns.shape[0]} trading days x {returns.shape[1]} stocks")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    return returns


def rolling_pc1_variance(returns_df: pd.DataFrame, window: int) -> pd.Series:
    """Compute the rolling PC1 explained-variance series."""
    if window > len(returns_df):
        raise ValueError(f"Window {window} is larger than the available sample of {len(returns_df)} days.")

    scaler = StandardScaler()
    dates: List[pd.Timestamp] = []
    pc1_values: List[float] = []

    total_windows = len(returns_df) - window + 1
    print(f"Running rolling PCA: {total_windows} windows of {window} trading days...")

    for end_idx in range(window - 1, len(returns_df)):
        win = returns_df.iloc[end_idx - window + 1 : end_idx + 1]
        x = scaler.fit_transform(win.values)

        pca = PCA(n_components=1)
        pca.fit(x)

        dates.append(returns_df.index[end_idx])
        pc1_values.append(float(pca.explained_variance_ratio_[0]))

    return pd.Series(pc1_values, index=pd.DatetimeIndex(dates), name="pc1_variance")

def plot_pc1_variance(pc1_series, pre_crisis_start=None):
    plt.figure(figsize=(12, 6))

    plt.plot(
        pc1_series.index,
        pc1_series.values,
        linewidth=2,
        label="Rolling PC1 variance"
    )

    plt.xlabel("Date")
    plt.ylabel("PC1 Explained Variance")
    plt.title("Rolling PC1 Explained Variance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# -----------------------------
# Statistics
# -----------------------------

def moving_block_bootstrap_difference(
    pre: np.ndarray,
    baseline: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    block_size: int | None = None,
    seed: int = 42,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Estimate the mean difference with a moving block bootstrap.

    Returns:
        lift      = mean(pre) - mean(baseline)
        boot_stat = lift / bootstrap standard error
        p_one     = one-sided p-value for mean(pre) > mean(baseline)
        ci        = 95% bootstrap interval for the mean difference
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([baseline, pre])
    baseline_len = len(baseline)
    pre_len = len(pre)

    if block_size is None:
        # A simple rule of thumb for overlapping rolling series.
        block_size = max(5, int(round(len(pooled) ** (1.0 / 3.0))))

    block_size = min(block_size, len(pooled))

    def resample_blocks(arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        max_start = max(1, n - block_size + 1)
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])
        return sample[:n]

    lift = float(pre.mean() - baseline.mean())

    # Remove the average level before resampling so the null hypothesis is
    # "no difference between the two periods".
    centered = pooled - pooled.mean()

    boot_lifts = np.empty(n_boot)
    for i in range(n_boot):
        sample = resample_blocks(centered)
        boot_baseline = sample[:baseline_len]
        boot_pre = sample[baseline_len : baseline_len + pre_len]
        boot_lifts[i] = boot_pre.mean() - boot_baseline.mean()

    se = float(boot_lifts.std(ddof=1))
    boot_stat = lift / se if se > 0 else 0.0
    p_one = float(max((boot_lifts >= lift).mean(), 1.0 / n_boot))
    ci_low, ci_high = np.quantile(boot_lifts, [0.025, 0.975])

    return lift, boot_stat, p_one, (float(ci_low), float(ci_high))


def test_crisis_events(
    pc1: pd.Series,
    events: List[dict],
    warning_days: int = DEFAULT_WARNING_DAYS,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    block_size: int | None = None,
) -> pd.DataFrame:
    """Compare pre-crisis PC1 to the earlier baseline for each event."""
    rows = []

    print("\n" + "=" * 72)
    print("CRISIS EVENT TEST")
    print("H0: pre-crisis PC1 <= baseline PC1")
    print("H1: pre-crisis PC1 > baseline PC1")
    print("=" * 72)

    for event in events:
        event_name = event["name"]
        event_date = pd.Timestamp(event["date"])
        hist = pc1.loc[pc1.index < event_date]

        need = warning_days + baseline_days
        print(f"\nEvent: {event_name} ({event_date.date()})")
        if len(hist) < need:
            print("  skipped: not enough history")
            continue

        baseline = hist.iloc[-(warning_days + baseline_days) : -warning_days]
        pre = hist.iloc[-warning_days:]

        lift, boot_stat, p_one, ci = moving_block_bootstrap_difference(
            pre.values,
            baseline.values,
            n_boot=n_boot,
            block_size=block_size,
        )
        warning_mean = float(pre.mean())
        baseline_mean = float(baseline.mean())
        warning_pct = float((baseline < warning_mean).mean() * 100)
        slope = float(np.polyfit(np.arange(len(pre)), pre.values, 1)[0]) if len(pre) >= 2 else 0.0
        significant = bool((lift > 0) and (p_one < alpha))

        verdict = "supports early warning" if significant else "no clear warning"
        print(
            f"  baseline_mean={baseline_mean:.4f}, warning_mean={warning_mean:.4f}, "
            f"lift={lift:+.4f}, p_one={p_one:.4f}, CI=({ci[0]:+.4f}, {ci[1]:+.4f}), "
            f"trend/day={slope:+.6f} -> {verdict}"
        )

        rows.append(
            {
                "event": event_name,
                "event_date": event_date.date(),
                "baseline_mean": baseline_mean,
                "warning_mean": warning_mean,
                "lift": lift,
                "boot_stat": boot_stat,
                "p_one_sided": p_one,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "warning_percentile": warning_pct,
                "warning_slope_per_day": slope,
                "significant": significant,
                "warning_days": warning_days,
                "baseline_days": baseline_days,
                "alpha": alpha,
                "n_boot": n_boot,
                "block_size": int(block_size) if block_size is not None else None,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        support_rate = float(out["significant"].mean())
        print("\nSummary")
        print("-" * 72)
        print(f"Events tested: {len(out)}")
        print(f"Support rate: {support_rate:.1%}")
        out.to_csv("pc1_early_warning_results.csv", index=False)
        print("Saved pc1_early_warning_results.csv")
    else:
        print("\nNo valid event tests were produced.")

    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_pc1(pc1: pd.Series, events: List[dict], window: int, output_file: str = "pc1_early_warning.png") -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(pc1.index, pc1.values * 100, color="steelblue", lw=1.6, label="PC1 variance explained")
    ax.fill_between(pc1.index, pc1.values * 100, color="steelblue", alpha=0.2)
    ax.axhline(float(pc1.quantile(0.90) * 100), color="crimson", ls="--", lw=1.1, label="90th percentile")

    for event in events:
        dt = pd.Timestamp(event["date"])
        if pc1.index.min() <= dt <= pc1.index.max():
            ax.axvline(dt, color="gray", ls=":", lw=1)
            ax.text(dt, ax.get_ylim()[1] * 0.95, event["name"], fontsize=8, ha="center", va="top")

    ax.set_title(f"Rolling PC1 Explained Variance ({window}-day window)")
    ax.set_ylabel("PC1 variance explained (%)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    print(f"Saved {output_file}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether rolling PC1 gives early warning of market crises")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END, help="End date YYYY-MM-DD or 'today'")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Rolling PCA window in trading days")
    parser.add_argument("--warning-days", type=int, default=DEFAULT_WARNING_DAYS, help="Days before crisis to treat as warning period")
    parser.add_argument("--baseline-days", type=int, default=DEFAULT_BASELINE_DAYS, help="Earlier days used as the baseline period")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap replications")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE, help="Drop stocks with lower return coverage")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--block-size", type=int, default=None, help="Optional block size for bootstrap")
    parser.add_argument("--plot", action="store_true", help="Save a chart of rolling PC1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = resolve_end_date(args.end_date)

    print("=" * 72)
    print("PC1 EARLY WARNING TEST")
    print("=" * 72)
    print(f"Universe: {len(TICKERS)} US stocks")
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Rolling window: {args.window} trading days")
    print(f"Warning window: {args.warning_days} days")
    print(f"Baseline window: {args.baseline_days} days")

    returns = download_returns(TICKERS, args.start_date, end_date, min_coverage=args.min_coverage)
    pc1 = rolling_pc1_variance(returns, window=args.window)
    plot_pc1_variance(pc1, pre_crisis_start="2008-08-15")

    results = test_crisis_events(
        pc1,
        CRISIS_EVENTS,
        warning_days=args.warning_days,
        baseline_days=args.baseline_days,
        alpha=args.alpha,
        n_boot=args.n_boot,
        block_size=args.block_size if args.block_size is not None else max(5, args.window // 4),
    )

    print("\nOverall interpretation")
    if results.empty:
        print("  Not enough data to evaluate the crisis windows.")
    else:
        positive = int((results["lift"] > 0).sum())
        significant = int(results["significant"].sum())
        print(f"  Positive lifts: {positive}/{len(results)}")
        print(f"  Statistically significant warnings: {significant}/{len(results)}")
        if significant >= max(1, len(results) // 2):
            print("  PC1 looks like a useful early-warning signal in this sample.")
        else:
            print("  PC1 is not a reliable warning signal in this sample.")

    if args.plot:
        plot_pc1(pc1, CRISIS_EVENTS, window=args.window)


if __name__ == "__main__":
    main()
