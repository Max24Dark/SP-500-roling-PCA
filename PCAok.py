import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

STARTDATE = "2019-01-01"
ENDDATE = None
DEFAULTWINDOW = 200
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
    sectorsumhistory, sectorabshistory = [], []
    print(returns)
    
    print(f"Running rolling PCA: {noofwindows} windows of {windowlength} trading days...")

    for i in range(windowlength - 1, len(returns)):
        window = returns.iloc[i - windowlength + 1 : i + 1]
        x = StandardScaler().fit_transform(window.values)
        pca = PCA()
        pca.fit(x)

        evr = pca.explained_variance_ratio_
        eigvals = pca.explained_variance_
        pc1loadings = pca.components_[0]

        cumvar = np.cumsum(evr)
        ncomponentsto90 = np.searchsorted(cumvar, thresholdvar) + 1
        ninetyvariancevalues.append(ncomponentsto90)

        eigenvaluecapture.append({"eigenvalues": eigvals[:desiredcomponents], "explained_variance_ratio": evr[:desiredcomponents]})
        pc1values[i - windowlength + 1] = float(evr[0])
        sectorsumwindow, sectorabswindow = sectorcontributionpc1(returns.columns, SECTORMAP, pc1loadings)

        sectorsumhistory.append(sectorsumwindow)
        sectorabshistory.append(sectorabswindow)

    
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
    return result, sectorsumhistorycontrib, sectorabshistorycontrib

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

    

def main() -> None:
    returns, volatility = downloadreturns(TICKERS, STARTDATE, ENDDATE, DEFAULTMINCOVERAGE)
    rollingpcaoutput, sectorsumhistorycontrib, sectorabshistorycontrib = rollingpca(returns, DEFAULTWINDOW, DEFAULTVARIANCETHRESHOLD, DESIREDPCNUM)

    plotdualcomponents(rollingpcaoutput)
    plotdualvolatility(rollingpcaoutput, volatility)
    plotsectorcontributiontopc1(sectorsumhistorycontrib, sectorabshistorycontrib)





if __name__ == "__main__":
    main()
