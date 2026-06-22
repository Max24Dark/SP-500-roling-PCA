import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.linalg import subspace_angles

START_DATE = "2008-01-01"
END_DATE = None
DEFAULT_WINDOW = 100
DEFAULT_MIN_COVERAGE = 0.90
DEFAULT_VARIANCE_THRESHOLD = 0.90
DESIRED_PC_NUM = 5

SECTOR_MAP = {
    # Tech
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","META":"Tech",
    "AVGO":"Tech","AMD":"Tech","INTC":"Tech","ORCL":"Tech","ADBE":"Tech",

    # Finance
    "JPM":"Finance","BAC":"Finance","WFC":"Finance","GS":"Finance","MS":"Finance",
    "C":"Finance","BLK":"Finance","AXP":"Finance","SCHW":"Finance","USB":"Finance",

    # Healthcare
    "UNH":"Healthcare","JNJ":"Healthcare","PFE":"Healthcare","MRK":"Healthcare",
    "ABBV":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "BMY":"Healthcare","LLY":"Healthcare",

    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy",
    "PSX":"Energy","MPC":"Energy","VLO":"Energy","HAL":"Energy","OXY":"Energy",

    # Staples
    "PG":"Staples","KO":"Staples","PEP":"Staples","WMT":"Staples","COST":"Staples",
    "PM":"Staples","MO":"Staples","MDLZ":"Staples","CL":"Staples","KMB":"Staples",

    # Industrial
    "BA":"Industrial","CAT":"Industrial","GE":"Industrial","HON":"Industrial",
    "UPS":"Industrial","RTX":"Industrial","LMT":"Industrial",

    # Real Estate
    "AMT":"RealEstate","PLD":"RealEstate","EQIX":"RealEstate","SPG":"RealEstate",
    "DLR":"RealEstate","PSA":"RealEstate",

    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "AEP":"Utilities","EXC":"Utilities","NGG":"Utilities"
}

TICKERS = list(SECTOR_MAP.keys())

crisis_periods = {
    "Euro_Debt_2011": ("2011-07-01", "2012-01-31"),
    "China_Selloff_2015": ("2015-08-01", "2016-02-01"),
    "COVID_Crash": ("2020-02-15", "2020-05-15"),
    "Inflation_2022": ("2022-01-01", "2022-10-01")
}

def download_returns(tickers, start_date, end_date, min_coverage):
    """Download adjusted close prices and convert them to daily returns."""
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]

    if isinstance(raw_data, pd.Series):
        raw_data = raw_data.to_frame()

    raw_data.index = pd.to_datetime(raw_data.index).tz_localize(None)
    returns = raw_data.pct_change().iloc[1:]

    coverage = returns.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    returns = returns[keep].dropna()

    print(f"Kept {len(keep)}/{len(tickers)} stocks after coverage filter ({min_coverage:.0%}).")
    print(f"Returns shape: {returns.shape[0]} trading days x {returns.shape[1]} stocks")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    volatility = returns.rolling(20).std().mean(axis=1)
    return returns, volatility

def rolling_pca_summary(returns, window_length, variance_threshold, desired_components):
    """Compute the rolling PCA of the returns dataframe."""
    dates = returns.index
    num_windows = len(returns) - window_length + 1
    n_components_90_variance = []
    eigenvalue_capture = []
    pc1_variance = np.empty(num_windows)
    sector_sum_history, sector_abs_history, pc1_direction_history, subspace_history = [], [], [], []
    previous_pc1 = None

    print(f"Running rolling PCA: {num_windows} windows of {window_length} trading days...")

    for i in range(window_length - 1, len(returns)):
        window = returns.iloc[i - window_length + 1 : i + 1]
        x = StandardScaler().fit_transform(window.values)
        pca = PCA()
        pca.fit(x)

        explained_variance_ratio = pca.explained_variance_ratio_
        eigvals = pca.explained_variance_
        pc1_loadings = pca.components_[0]
        subspace = pca.components_[:3].T

        if previous_pc1 is not None and np.dot(pc1_loadings, previous_pc1) < 0:
            pc1_loadings = -pc1_loadings
        previous_pc1 = pc1_loadings

        cumvar = np.cumsum(explained_variance_ratio)
        n_components_to_90 = np.searchsorted(cumvar, variance_threshold) + 1
        n_components_90_variance.append(n_components_to_90)
        pc1_direction_history.append(pc1_loadings)
        subspace_history.append(subspace)

        eigenvalue_capture.append({"eigenvalues": eigvals[:desired_components], "explained_variance_ratio": explained_variance_ratio[:desired_components]})
        pc1_variance[i - window_length + 1] = float(explained_variance_ratio[0])
        sector_sum_window, sector_abs_window = sector_contribution_pc1(returns.columns, SECTOR_MAP, pc1_loadings)

        sector_sum_history.append(sector_sum_window)
        sector_abs_history.append(sector_abs_window)

    pc1_direction_history = np.array(pc1_direction_history)
    result = pd.DataFrame({
        "pc1_variance": pc1_variance,
        "ncomponents_90variance": n_components_90_variance,
        "eigenvalue_capture": eigenvalue_capture},
        index=dates[window_length - 1:]
    )
    sector_sum_history_contrib = pd.DataFrame(
        sector_sum_history,
        index=dates[window_length - 1:]
    )
    sector_abs_history_contrib = pd.DataFrame(
        sector_abs_history,
        index=dates[window_length - 1:]
    )
    return result, sector_sum_history_contrib, sector_abs_history_contrib, pc1_direction_history, subspace_history

def sector_contribution_pc1(tickers, sector_map, pc1_loadings):
    if len(tickers) != len(pc1_loadings):
        raise ValueError("Tickers and loadings must have the same length.")
    df = pd.DataFrame({
        "ticker": tickers,
        "loading": pc1_loadings
    })

    df["sector"] = df["ticker"].map(sector_map)
    df["contrib"] = df["loading"] ** 2

    sector_sum = df.groupby("sector")["contrib"].sum().to_dict()
    sector_abs = df.groupby("sector")["loading"].apply(lambda x: np.abs(x).mean()).to_dict()

    return sector_sum, sector_abs

def plot_dual_components(df):
    fig, ax1 = plt.subplots()

    ax1.plot(df.index, df["pc1_variance"], color="red")
    ax1.set_ylabel("PC1 variance", color="red")

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["ncomponents_90variance"], color="blue")
    ax2.set_ylabel("N components (90%)", color="blue")

    add_crisis_shading(ax1)

    plt.title("Market Concentration vs Effective Dimension")
    plt.show()

def plot_dual_volatility(rolling_pca_output, volatility):
    volatility = volatility.loc[rolling_pca_output.index]
    fig, ax1 = plt.subplots()

    ax1.plot(rolling_pca_output.index, rolling_pca_output["pc1_variance"], color="red")
    ax1.set_ylabel("PC1 variance", color="red")

    ax2 = ax1.twinx()
    ax2.plot(volatility.index, volatility, color="green")
    ax2.set_ylabel("Volatility", color="green")

    add_crisis_shading(ax1)

    plt.title("Market Concentration vs Volatility")
    plt.show()

def plot_sector_contribution_to_pc1(sector_sum_history_contrib, sector_abs_history_contrib):
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    sns.heatmap(sector_sum_history_contrib.T, cmap="RdBu_r", center=0, ax=ax[0])
    ax[0].set_title("Sector Squared Loading Contribution")

    sns.heatmap(sector_abs_history_contrib.T, cmap="viridis", ax=ax[1])
    ax[1].set_title("Sector Importance (Abs Mean)")

    plt.tight_layout()
    plt.show()

def pc1_direction_vs_volatility(pc1_direction_history, volatility, dates):
    #volatility has values from index 19 onwards from startdate to enddate

    norms = np.linalg.norm(pc1_direction_history, axis=1, keepdims=True)
    pc1_norm = np.where(norms > 1e-12, pc1_direction_history / norms, 0)
    cosine_angle = np.sum(pc1_norm[1:] * pc1_norm[:-1], axis=1)
    angles = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    pc1_angles = np.concatenate([[0], angles])

    pc1_angles = pd.Series(pc1_angles, index=dates)
    volatility = volatility.loc[pc1_angles.index]

    lags = range(-20, 120)
    corrs = []

    for lag in lags:
        shifted_volatility = volatility.shift(lag)
        corrs.append(pc1_angles.corr(shifted_volatility))

    plt.plot(lags, corrs)
    plt.axvline(0, color='black')
    plt.title("Lag correlation: PC1 angle vs volatility")
    plt.show()

def subspace_changes(subspace_history, dates):
    angles = []

    for i in range(1, len(subspace_history)):
        a = subspace_history[i - 1]
        b = subspace_history[i]

        theta = subspace_angles(a, b)

        angles.append(theta.mean())
    angles = pd.Series(angles, index=dates[1:])

    return angles

def plot_subspace_angles(angles):
    fig, ax = plt.subplots()

    ax.plot(angles.index, angles.values, label="Subspace rotation")
    ax.set_ylabel("Subspace angle")

    add_crisis_shading(ax)

    plt.title("Subspace rotation over time")
    plt.show()

def crisis_regime_analysis(angles, pc1_variance, eff_dim, volatility):
    summary = pd.DataFrame({
        "angle": angles,
        "pc1variance": pc1_variance,
        "effdim": eff_dim,
        "volatility": volatility
    }).dropna()

    summary["crisis"] = False

    # Mark crisis periods
    for _, (start, end) in crisis_periods.items():
        mask = (summary.index >= start) & (summary.index <= end)
        summary.loc[mask, "crisis"] = True

    crisis = summary[summary["crisis"]]
    normal = summary[~summary["crisis"]]

    results = pd.DataFrame({
        "crisis_mean": crisis.mean(),
        "normal_mean": normal.mean(),
        "difference": crisis.mean() - normal.mean(),
        "ratio": crisis.mean() / normal.mean()
    })

    print("\n=== REGIME COMPARISON ===")
    print(results)

def add_crisis_shading(ax, alpha=0.2):
    for _, (start, end) in crisis_periods.items():
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=alpha)

def main() -> None:
    returns, volatility = download_returns(TICKERS, START_DATE, END_DATE, DEFAULT_MIN_COVERAGE)
    rolling_pca_output, sector_sum_history_contrib, sector_abs_history_contrib, pc1_direction_history, subspace_history = rolling_pca_summary(returns, DEFAULT_WINDOW, DEFAULT_VARIANCE_THRESHOLD, DESIRED_PC_NUM)

    plot_dual_components(rolling_pca_output)
    plot_dual_volatility(rolling_pca_output, volatility)
    plot_sector_contribution_to_pc1(sector_sum_history_contrib, sector_abs_history_contrib)
    pc1_direction_vs_volatility(pc1_direction_history, volatility, returns.index[DEFAULT_WINDOW - 1:])
    angles = subspace_changes(subspace_history, returns.index[DEFAULT_WINDOW - 1:])
    plot_subspace_angles(angles)
    crisis_regime_analysis(angles, rolling_pca_output["pc1_variance"], rolling_pca_output["ncomponents_90variance"], volatility.loc[rolling_pca_output.index])

if __name__ == "__main__":
    main()
