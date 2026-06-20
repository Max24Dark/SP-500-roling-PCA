import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.linalg import subspace_angles

STARTDATE = "2008-01-01"
ENDDATE = None
DEFAULTWINDOW = 100
DEFAULTMINCOVERAGE = 0.90
DEFAULTVARIANCETHRESHOLD = 0.90
DESIREDPCNUM = 5

SECTORMAP = {
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

TICKERS = list(SECTORMAP.keys())

crisisperiods = {
    "Euro_Debt_2011": ("2011-07-01", "2012-01-31"),
    "China_Selloff_2015": ("2015-08-01", "2016-02-01"),
    "COVID_Crash": ("2020-02-15", "2020-05-15"),
    "Inflation_2022": ("2022-01-01", "2022-10-01")
}

def downloadreturns(tickers, startdate, enddate, mincoverage):
    """Download adjusted close prices and convert them to daily returns."""
    rawdata = yf.download(tickers, start=startdate, end=enddate, auto_adjust=True, progress=False)["Close"]

    if isinstance(rawdata, pd.Series):
        rawdata = rawdata.to_frame()

    rawdata.index = pd.to_datetime(rawdata.index).tz_localize(None)
    returns = rawdata.pct_change().iloc[1:]

    coverage = returns.notna().mean()
    keep = coverage[coverage >= mincoverage].index.tolist()
    returns = returns[keep].dropna()

    print(f"Kept {len(keep)}/{len(tickers)} stocks after coverage filter ({mincoverage:.0%}).")
    print(f"Returns shape: {returns.shape[0]} trading days x {returns.shape[1]} stocks")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    volatility = returns.rolling(20).std().mean(axis=1)
    return returns, volatility

def rollingpca(returns, windowlength, thresholdvar, desiredcomponents):
    """Compute the rolling PCA of the returns dataframe."""
    dates = returns.index
    noofwindows = len(returns) - windowlength + 1
    ninetyvariancevalues = []
    eigenvaluecapture = []
    pc1values= np.empty(noofwindows)
    sectorsumhistory, sectorabshistory, pc1dirhistory, subspacehistory = [], [], [], []
    previouspc1 = None
    
    print(f"Running rolling PCA: {noofwindows} windows of {windowlength} trading days...")

    for i in range(windowlength - 1, len(returns)):
        window = returns.iloc[i - windowlength + 1 : i + 1]
        x = StandardScaler().fit_transform(window.values)
        pca = PCA()
        pca.fit(x)

        evr = pca.explained_variance_ratio_
        eigvals = pca.explained_variance_
        pc1loadings = pca.components_[0]
        subspace = pca.components_[:3].T

        if previouspc1 is not None and np.dot(pc1loadings, previouspc1) < 0:
            pc1loadings = -pc1loadings
        previouspc1 = pc1loadings

        cumvar = np.cumsum(evr)
        ncomponentsto90 = np.searchsorted(cumvar, thresholdvar) + 1
        ninetyvariancevalues.append(ncomponentsto90)
        pc1dirhistory.append(pc1loadings)
        subspacehistory.append(subspace)

        eigenvaluecapture.append({"eigenvalues": eigvals[:desiredcomponents], "explained_variance_ratio": evr[:desiredcomponents]})
        pc1values[i - windowlength + 1] = float(evr[0])   # Store the first principal component's explained variance ratio not direction
        sectorsumwindow, sectorabswindow = sectorcontributionpc1(returns.columns, SECTORMAP, pc1loadings)

        sectorsumhistory.append(sectorsumwindow)
        sectorabshistory.append(sectorabswindow)

    pc1dirhistory = np.array(pc1dirhistory)
    result = pd.DataFrame({
        "pc1variance": pc1values,
        "ncomponents90variance": ninetyvariancevalues,
        "eigenvaluecapture": eigenvaluecapture},
        index=dates[windowlength - 1:]
    )
    sectorsumhistorycontrib = pd.DataFrame(
        sectorsumhistory,
        index=dates[windowlength - 1:]
    )
    sectorabshistorycontrib = pd.DataFrame(
        sectorabshistory,
        index=dates[windowlength - 1:]
    )
    return result, sectorsumhistorycontrib, sectorabshistorycontrib, pc1dirhistory, subspacehistory

def sectorcontributionpc1(tickers, sectormap, pc1loadings):
    if len(tickers) != len(pc1loadings):
        raise ValueError("Tickers and loadings must have the same length.")
    df = pd.DataFrame({
        "ticker": tickers,
        "loading": pc1loadings
    })

    df["sector"] = df["ticker"].map(sectormap)
    df["contrib"] = df["loading"] ** 2


    sectorsum = df.groupby("sector")["contrib"].sum().to_dict()
    sectorabs = df.groupby("sector")["loading"].apply(lambda x: np.abs(x).mean()).to_dict()

    return sectorsum, sectorabs

def plotdualcomponents(df):
    fig, ax1 = plt.subplots()

    ax1.plot(df.index, df["pc1variance"], color="red")
    ax1.set_ylabel("PC1 variance", color="red")

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["ncomponents90variance"], color="blue")
    ax2.set_ylabel("N components (90%)", color="blue")

    addcrisisshading(ax1)

    plt.title("Market Concentration vs Effective Dimension")
    plt.show()

def plotdualvolatility(rollingpcaoutput, volatility):
    volatility = volatility.loc[rollingpcaoutput.index]
    fig, ax1 = plt.subplots()

    ax1.plot(rollingpcaoutput.index, rollingpcaoutput["pc1variance"], color="red")
    ax1.set_ylabel("PC1 variance", color="red")

    ax2 = ax1.twinx()
    ax2.plot(volatility.index, volatility, color="green")
    ax2.set_ylabel("Volatility", color="green")

    addcrisisshading(ax1)

    plt.title("Market Concentration vs Volatility")
    plt.show()

def plotsectorcontributiontopc1(sectorsumhistorycontrib, sectorabshistorycontrib):
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    sns.heatmap(sectorsumhistorycontrib.T, cmap="RdBu_r", center=0, ax=ax[0])
    ax[0].set_title("Sector Squared Loading Contribution")

    sns.heatmap(sectorabshistorycontrib.T, cmap="viridis", ax=ax[1])
    ax[1].set_title("Sector Importance (Abs Mean)")

    plt.tight_layout()
    plt.show()

def pc1directionvsvolatility(pc1dirhistory, volatility, dates):
    #volatility has values from index 19 onwards from startdate to enddate

    norms = np.linalg.norm(pc1dirhistory, axis=1, keepdims=True)
    pc1norm = np.where(norms > 1e-12, pc1dirhistory / norms, 0)
    sim = np.sum(pc1norm[1:] * pc1norm[:-1], axis=1)
    angles = np.arccos(np.clip(sim, -1.0, 1.0))
    pc1angles = np.concatenate([[0], angles])

    pc1angles = pd.Series(pc1angles, index=dates)
    volatility = volatility.loc[pc1angles.index]

    lags = range(-20, 120)
    corrs = []

    for lag in lags:
        shiftedvolatility = volatility.shift(lag)
        corrs.append(pc1angles.corr(shiftedvolatility))

    plt.plot(lags, corrs)
    plt.axvline(0, color='black')
    plt.title("Lag correlation: PC1 angle vs volatility")
    plt.show()
    
def subspacechanges(subspacehistory, dates):
    angles = []

    for i in range(1, len(subspacehistory)):
        A = subspacehistory[i-1]
        B = subspacehistory[i]

        theta = subspace_angles(A, B)

        angles.append(theta.mean())
    angles = pd.Series(angles, index=dates[1:])

    return angles

def plotsubspaceangles(angles):
    fig, ax = plt.subplots()

    ax.plot(angles.index, angles.values, label="Subspace rotation")
    ax.set_ylabel("Subspace angle")

    addcrisisshading(ax)

    plt.title("Subspace rotation over time")
    plt.show()

def crisisregimeanalysis(angles, pc1variance, effdim, volatility):
    summary = pd.DataFrame({
        "angle": angles,
        "pc1variance": pc1variance,
        "effdim": effdim,
        "volatility": volatility
    }).dropna()

    summary["crisis"] = False

    # Mark crisis periods
    for _, (start, end) in crisisperiods.items():
        mask = (summary.index >= start) & (summary.index <= end)
        summary.loc[mask, "crisis"] = True

    crisis = summary[summary["crisis"]]
    normal = summary[~summary["crisis"]]

    results = pd.DataFrame({
        "crisismean": crisis.mean(),
        "normalmean": normal.mean(),
        "difference": crisis.mean() - normal.mean(),
        "ratio": crisis.mean() / normal.mean()
    })

    print("\n=== REGIME COMPARISON ===")
    print(results)

def addcrisisshading(ax, alpha=0.2):
    for _, (start, end) in crisisperiods.items():
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=alpha)

def main() -> None:
    returns, volatility = downloadreturns(TICKERS, STARTDATE, ENDDATE, DEFAULTMINCOVERAGE)
    rollingpcaoutput, sectorsumhistorycontrib, sectorabshistorycontrib, pc1dirhistory, subspacehistory = rollingpca(returns, DEFAULTWINDOW, DEFAULTVARIANCETHRESHOLD, DESIREDPCNUM)

    plotdualcomponents(rollingpcaoutput)
    plotdualvolatility(rollingpcaoutput, volatility)
    plotsectorcontributiontopc1(sectorsumhistorycontrib, sectorabshistorycontrib)
    pc1directionvsvolatility(pc1dirhistory, volatility, returns.index[DEFAULTWINDOW-1:])
    angles = subspacechanges(subspacehistory, returns.index[DEFAULTWINDOW-1:])
    plotsubspaceangles(angles)
    crisisregimeanalysis(angles, rollingpcaoutput["pc1variance"], rollingpcaoutput["ncomponents90variance"], volatility.loc[rollingpcaoutput.index])

if __name__ == "__main__":
    main()
