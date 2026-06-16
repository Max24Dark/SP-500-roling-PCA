"""
Rolling PCA Systematic-Risk Monitor
==================================
Focused script for one question:
Can rolling PCA provide an early indicator of increasing systematic market risk?

What it does:
1) Downloads daily adjusted-close prices for a broad S&P 500 sample (2019 -> today by default)
2) Computes daily returns [trading_days x stocks] (weekends/holidays naturally excluded)
3) Runs rolling PCA with a configurable window (50-300 trading days)
4) Tracks risk proxies:
   - PC1 variance explained (market factor dominance)
   - Average pairwise cross-stock correlation
5) Tests pre-crisis early-warning hypothesis for US and UK crisis events
6) Reports current regime (z-score, percentile, short-term trend)

Usage examples:
  python rolling_pca_risk.py
  python rolling_pca_risk.py --window 90
  python rolling_pca_risk.py --start-date 2019-01-01 --end-date today --window 180
    python rolling_pca_risk.py --economy uk
    python rolling_pca_risk.py --economy us

Dependencies:
  pip install yfinance pandas numpy scikit-learn scipy matplotlib
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

WINDOW_MIN = 50
WINDOW_MAX = 300
DEFAULT_WINDOW = 100
DEFAULT_START = "2019-01-01"
DEFAULT_END = "today"
MIN_COVERAGE = 0.90

AVAILABLE_ECONOMIES = ["us", "uk"]

# Broad cross-sector set of liquid US names.
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "CSCO",
    "ORCL", "IBM", "QCOM", "TXN", "AVGO", "NOW", "ADBE", "CRM", "SNOW", "PLTR",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB", "PNC",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "HAL", "BKR",
    "PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "CL", "GIS",
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "MAR", "YUM",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "EMR", "ETN",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ES", "WEC", "ETR",
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",
]

ECONOMY_TICKERS = {
    "us": US_TICKERS,
    "uk": [
        "HSBA.L", "SHEL.L", "BP.L", "AZN.L", "ULVR.L", "GSK.L", "DGE.L", "BATS.L", "LSEG.L", "RIO.L",
        "BARC.L", "VOD.L", "AAL.L", "TSCO.L", "REL.L",
    ],
}

SHARED_CRISIS_EVENTS = [
    {"name": "COVID crash", "date": "2020-03-16", "scope": "global"},
    {"name": "2022 inflation / rate shock", "date": "2022-06-13", "scope": "global"},
    {"name": "US regional banking stress", "date": "2023-03-13", "scope": "global"},
]

ECONOMY_CRISIS_EVENTS = {
    "us": [
        {"name": "US inflation peak / Fed tightening shock", "date": "2022-06-13", "scope": "local"},
        {"name": "US regional banking stress", "date": "2023-03-13", "scope": "local"},
    ],
    "uk": [
        {"name": "UK gilt crisis", "date": "2022-09-26", "scope": "local"},
        {"name": "UK mini-budget volatility", "date": "2022-09-23", "scope": "local"},
    ],
}


# =============================================================================
# BLOCK BOOTSTRAP TEST  (replaces ttest_ind throughout)
# =============================================================================

def block_bootstrap_test(
    pre: np.ndarray,
    baseline: np.ndarray,
    n_boot: int = 2000,
    block_size: int = None,
    seed: int = 42,
) -> tuple:
    """
    One-sided block bootstrap test of H1: mean(pre) > mean(baseline).

    Rolling PCA series are autocorrelated (consecutive windows share n-1 days),
    so a standard two-sample t-test would understate uncertainty. The block
    bootstrap preserves serial correlation by resampling contiguous blocks
    from the full event window rather than individual points.

    Parameters
    ----------
    pre        : observations from the pre-crisis window
    baseline   : observations from the baseline window
    n_boot     : number of bootstrap replications (default 2000)
    block_size : block length; defaults to max(3, round(T^(1/3)))
    seed       : random seed for reproducibility

    Returns
    -------
    obs_lift : float  - observed mean(pre) - mean(baseline)
    boot_stat: float  - lift divided by bootstrap standard error
    p_one    : float  - one-sided p-value for H1: mean(pre) > mean(baseline)
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([baseline, pre])
    baseline_len = len(baseline)
    pre_len = len(pre)

    if block_size is None:
        block_size = max(3, int(round(len(pooled) ** (1.0 / 3.0))))

    block_size = min(block_size, len(pooled))

    def resample_blocks(arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        max_start = max(1, n - block_size + 1)
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max_start, size=n_blocks)
        return np.concatenate([arr[s: s + block_size] for s in starts])[:n]

    obs_lift = float(pre.mean() - baseline.mean())

    # Centre the combined event window under H0 so resampling keeps the
    # observed dependence structure but removes the mean shift.
    pooled_c = pooled - pooled.mean()

    boot_lifts = np.empty(n_boot)
    for i in range(n_boot):
        boot_sample = resample_blocks(pooled_c)
        boot_base = boot_sample[:baseline_len]
        boot_pre = boot_sample[baseline_len:baseline_len + pre_len]
        boot_lifts[i] = boot_pre.mean() - boot_base.mean()

    # One-sided p-value: fraction of bootstrap lifts >= observed lift.
    # Apply a continuity correction of 1/n_boot to avoid p=0 exactly.
    p_one = float(max((boot_lifts >= obs_lift).mean(), 1.0 / n_boot))

    se = float(boot_lifts.std(ddof=1))
    boot_stat = obs_lift / se if se > 0 else 0.0

    return obs_lift, boot_stat, p_one



def resolve_end_date(end_date: str) -> str:
    if str(end_date).strip().lower() == "today":
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return end_date


def validate_window(window: int):
    if not (WINDOW_MIN <= window <= WINDOW_MAX):
        raise ValueError(f"Window must be between {WINDOW_MIN} and {WINDOW_MAX} trading days.")


def validate_economy(economy: str):
    if economy not in ECONOMY_TICKERS:
        raise ValueError(f"Unsupported economy '{economy}'. Choose from: {', '.join(AVAILABLE_ECONOMIES)}")


def resolve_universe(economy: str):
    validate_economy(economy)
    return ECONOMY_TICKERS[economy]


def build_event_calendar(economy: str):
    validate_economy(economy)
    events = list(SHARED_CRISIS_EVENTS)
    events.extend(ECONOMY_CRISIS_EVENTS.get(economy, []))
    unique, seen = [], set()
    for event in events:
        key = (event["name"], event["date"])
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


# =============================================================================
# DATA & PCA
# =============================================================================

def download_returns(tickers, start_date, end_date, min_coverage=MIN_COVERAGE):
    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data.index = pd.to_datetime(data.index).tz_localize(None)
    returns = data.pct_change().iloc[1:]

    coverage = returns.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    returns = returns[keep].dropna()

    print(f"Kept {len(keep)} stocks after coverage filter ({min_coverage:.0%}).")
    print(f"Return matrix: {returns.shape[0]} trading days x {returns.shape[1]} stocks")
    print(f"Actual range: {returns.index.min().date()} to {returns.index.max().date()}")

    return returns


def rolling_pca(returns_df: pd.DataFrame, window: int, n_components: int = 5):
    if window > len(returns_df):
        raise ValueError(f"Window {window} exceeds available trading days {len(returns_df)}")

    total_windows = len(returns_df) - window + 1
    print(f"Running rolling PCA: {total_windows} windows of {window} trading days...")

    scaler = StandardScaler()
    dates, pc1_var, top3_var, avg_corr = [], [], [], []

    for end_idx in range(window - 1, len(returns_df)):
        start_idx = end_idx - window + 1
        win = returns_df.iloc[start_idx:end_idx + 1]

        X = scaler.fit_transform(win.values)
        n_fit_components = min(n_components, X.shape[0], X.shape[1])
        if n_fit_components < 1:
            raise ValueError("Not enough data to fit PCA.")
        pca = PCA(n_components=n_fit_components)
        pca.fit(X)
        ev = pca.explained_variance_ratio_

        dates.append(returns_df.index[end_idx])
        pc1_var.append(ev[0])
        top3_var.append(ev[:3].sum())

        corr = pd.DataFrame(X).corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        avg_corr.append(float(np.nanmean(upper)))

    idx = pd.DatetimeIndex(dates)
    return {
        "window": window,
        "dates": idx,
        "pc1_variance": pd.Series(pc1_var, index=idx, name="pc1_variance"),
        "top3_variance": pd.Series(top3_var, index=idx, name="top3_variance"),
        "avg_correlation": pd.Series(avg_corr, index=idx, name="avg_correlation"),
    }


# =============================================================================
# HYPOTHESIS TESTING  (uses block_bootstrap_test, not ttest_ind)
# =============================================================================

def test_early_warning_hypothesis(rolling, events, pre_days=40, baseline_days=120,
                                   alpha=0.05, n_boot=2000):
    print("\n" + "=" * 70)
    print("HYPOTHESIS: rolling PCA can signal increasing systematic risk pre-crisis")
    print("H0: pre-crisis metric <= baseline metric")
    print("H1: pre-crisis metric > baseline metric")
    print(f"Test: block bootstrap (n_boot={n_boot})")
    print("=" * 70)

    metrics = {
        "PC1 variance": rolling["pc1_variance"],
        "Avg correlation": rolling["avg_correlation"],
    }

    rows = []

    for event in events:
        event_name = event["name"]
        event_date = pd.Timestamp(event["date"])
        event_scope = event.get("scope", "global")
        print(f"\nEvent: {event_name} ({event_date.date()}) [{event_scope}]")

        for metric_name, series in metrics.items():
            hist = series.loc[series.index <= event_date]
            need = pre_days + baseline_days + 1
            if len(hist) < need:
                print(f"  {metric_name}: skipped (insufficient data)")
                continue

            baseline = hist.iloc[-(pre_days + baseline_days):-pre_days]
            pre = hist.iloc[-pre_days:]

            block_size = max(3, min(len(hist), int(round(rolling["window"] / 4))))

            # The bootstrap works on the full contiguous event window so the
            # dependence between baseline and pre-crisis periods is preserved.
            lift, boot_stat, p_one = block_bootstrap_test(
                pre.values,
                baseline.values,
                n_boot=n_boot,
                block_size=block_size,
            )

            if np.isnan(boot_stat) or np.isnan(p_one):
                print(f"  {metric_name}: skipped (invalid test output)")
                continue

            sig = bool(p_one < alpha)
            slope = float(np.polyfit(np.arange(len(pre)), pre.values, 1)[0]) if len(pre) >= 2 else 0.0

            verdict = "supports H1" if sig else "fails to reject H0"
            print(
                f"  {metric_name}: lift={lift:+.4f}, boot_stat={boot_stat:+.3f}, "
                f"one-sided p={p_one:.4f}, trend/day={slope:+.6f} -> {verdict}"
            )

            rows.append({
                "event": event_name,
                "scope": event_scope,
                "event_date": event_date.date(),
                "metric": metric_name,
                "pre_mean": float(pre.mean()),
                "baseline_mean": float(baseline.mean()),
                "lift": lift,
                "boot_stat": boot_stat,
                "p_one_sided": p_one,
                "significant": sig,
                "pre_slope_per_day": float(slope),
                "pre_days": int(pre_days),
                "baseline_days": int(baseline_days),
                "alpha": float(alpha),
                "n_boot": int(n_boot),
                "block_size": int(block_size),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv("rolling_pca_hypothesis_tests.csv", index=False)
        support_rate = out["significant"].mean()
        print(f"\nSaved rolling_pca_hypothesis_tests.csv with {len(out)} rows")
        print(f"Support rate across event-metric tests: {support_rate:.1%}")
    else:
        print("\nNo hypothesis-test rows produced.")

    return out


# =============================================================================
# REGIME CHECK & PLOTTING
# =============================================================================

def current_regime_check(rolling, recent_days=20, long_days=252):
    print("\n" + "=" * 70)
    print("CURRENT REGIME CHECK")
    print("=" * 70)

    metrics = {
        "PC1 variance": rolling["pc1_variance"],
        "Avg correlation": rolling["avg_correlation"],
    }

    for name, series in metrics.items():
        recent = series.iloc[-recent_days:]
        long_hist = series.iloc[-min(long_days, len(series)):]
        mu = float(long_hist.mean())
        sd = float(long_hist.std(ddof=1))
        z = 0.0 if sd == 0 else (float(recent.mean()) - mu) / sd
        pct = float((long_hist < float(recent.mean())).mean() * 100)
        slope = float(np.polyfit(np.arange(len(recent)), recent.values, 1)[0]) if len(recent) >= 2 else 0.0

        print(
            f"{name}: recent_mean={recent.mean():.4f}, z={z:+.2f}, "
            f"percentile={pct:.1f}th, trend/day={slope:+.6f}"
        )


def plot_indicators(rolling):
    dates = rolling["dates"]
    pc1 = rolling["pc1_variance"] * 100
    corr = rolling["avg_correlation"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax1.plot(dates, pc1, color="steelblue", lw=1.6)
    ax1.fill_between(dates, pc1, alpha=0.25, color="steelblue")
    q90 = float(rolling["pc1_variance"].quantile(0.90) * 100)
    ax1.axhline(q90, color="crimson", ls="--", lw=1.1, label="90th percentile")
    ax1.set_ylabel("PC1 Variance Explained (%)")
    ax1.set_title(f"Rolling PCA Indicators ({rolling['window']}-day window)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(dates, corr, color="darkorange", lw=1.6)
    ax2.fill_between(dates, corr, alpha=0.25, color="darkorange")
    ax2.axhline(float(corr.mean()), color="black", ls="--", lw=1.0, label="Long-run mean")
    ax2.set_ylabel("Average Pairwise Correlation")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.legend()

    for dt, label in [
        (pd.Timestamp("2020-03-16"), "COVID"),
        (pd.Timestamp("2022-06-13"), "2022 shock"),
        (pd.Timestamp("2023-03-13"), "SVB stress"),
    ]:
        if dates.min() <= dt <= dates.max():
            ax1.axvline(dt, color="gray", ls=":", lw=1)
            ax2.axvline(dt, color="gray", ls=":", lw=1)
            ax1.text(dt, ax1.get_ylim()[1] * 0.95, label, fontsize=8, ha="center", va="top")

    plt.tight_layout()
    plt.savefig("rolling_pca_risk_indicators.png", dpi=160, bbox_inches="tight")
    print("Saved rolling_pca_risk_indicators.png")


# =============================================================================
# PER-ECONOMY RUNNER & MAIN
# =============================================================================

def print_selected_universe(economy, tickers):
    print("\n" + "=" * 70)
    print(f"Selected economy: {economy.upper()}")
    print(f"Selected stock universe: {len(tickers)} tickers")
    print("=" * 70)
    print(", ".join(tickers))


def run_single_economy_analysis(economy, args, end_date):
    tickers = resolve_universe(economy)
    events = build_event_calendar(economy)

    print("\n" + "=" * 70)
    print(f"ROLLING PCA SYSTEMATIC-RISK MONITOR [{economy.upper()}]")
    print("=" * 70)
    print(f"Economy: {economy.upper()}")
    print(f"Window: {args.window} trading days")
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Crisis events loaded: {len(events)}")

    print_selected_universe(economy, tickers)
    returns = download_returns(tickers, args.start_date, end_date, MIN_COVERAGE)
    rolling = rolling_pca(returns, window=args.window, n_components=args.n_components)

    tests = test_early_warning_hypothesis(
        rolling,
        events=events,
        pre_days=args.pre_days,
        baseline_days=args.baseline_days,
        alpha=args.alpha,
        n_boot=args.n_boot,
    )

    current_regime_check(rolling)

    if args.plot:
        plot_indicators(rolling)

    summary = {
        "economy": economy,
        "n_stocks": int(returns.shape[1]),
        "n_days": int(returns.shape[0]),
        "pc1_mean": float(rolling["pc1_variance"].mean()),
        "pc1_recent_mean": float(rolling["pc1_variance"].iloc[-20:].mean()),
        "corr_mean": float(rolling["avg_correlation"].mean()),
        "corr_recent_mean": float(rolling["avg_correlation"].iloc[-20:].mean()),
        "support_rate": float(tests["significant"].mean()) if not tests.empty else np.nan,
        "tests": int(len(tests)),
    }

    safe_name = economy.replace(".", "_")
    tests_file = f"rolling_pca_hypothesis_tests_{safe_name}.csv"
    summary_file = f"rolling_pca_summary_{safe_name}.csv"

    if not tests.empty:
        tests.to_csv(tests_file, index=False)
        print(f"Saved {tests_file}")
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"Saved {summary_file}")

    print("\nDone.")
    if tests.empty:
        print("Note: no valid event tests were produced for this configuration.")

    return summary


def main():

    validate_window(args.window)
    end_date = resolve_end_date(args.end_date)
    validate_economy(args.economy)
    run_single_economy_analysis(args.economy, args, end_date)


if __name__ == "__main__":
    main()