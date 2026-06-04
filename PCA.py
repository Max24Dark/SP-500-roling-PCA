import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from collections import Counter
import statsmodels.api as sm
from scipy.stats import pearsonr

current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# Parse date and format it the same way
date = pd.to_datetime("2025-01-06T20:45").strftime("%Y-%m-%d %H:%M:%S")



API_KEY = "QRL5CIUMC157I31J"
print(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

def get_quarterly_ticker_data(value):
    tickerdata = yf.Ticker(value)

    # Quarterly financials
    qincomestm = tickerdata.quarterly_financials
    qbalancesheet = tickerdata.quarterly_balance_sheet
    qcashflow = tickerdata.quarterly_cashflow

    # Raw data

    # Quarterly
    qtotalrevenue = qincomestm.loc['Total Revenue']
    qcostofrevenue = qincomestm.loc['Cost Of Revenue']
    qcogspercent = ((qcostofrevenue / qtotalrevenue) * 100).reindex(qtotalrevenue.index)

    qgrossprofit = qincomestm.loc['Gross Profit']

    qsganda = qincomestm.loc['Selling General And Administration']
    qsgandapercent = ((qsganda / qtotalrevenue) * 100).reindex(qtotalrevenue.index)

    qebt = qincomestm.loc['Pretax Income']
    qebtpercent = ((qebt / qtotalrevenue) * 100).reindex(qtotalrevenue.index)

    qtax = qincomestm.loc['Tax Provision']
    qtaxpercent = ((qtax / qtotalrevenue) * 100).reindex(qtotalrevenue.index)

    qnetincome = qincomestm.loc['Net Income']
    qnetprofitmargin = ((qnetincome / qtotalrevenue) * 100).reindex(qtotalrevenue.index)

    # Liquidity Ratios
    qcurrentassets = qbalancesheet.loc['Current Assets']
    qcurrentliabilities = qbalancesheet.loc['Current Liabilities']
    qinventory = qbalancesheet.loc['Inventory']

    qquickratio = ((qcurrentassets - qinventory) / qcurrentliabilities).reindex(qinventory.index)
    qcurrentratio = (qcurrentassets / qcurrentliabilities).reindex(qcurrentliabilities.index)
    qnetworkingcapital = (qcurrentassets - qcurrentliabilities).reindex(qcurrentliabilities.index)

    # Leverage Ratios
    qtotalliab = qbalancesheet.loc['Total Liabilities Net Minority Interest']
    qstockholderequity = qbalancesheet.loc['Stockholders Equity']

    qdebttoequity = (qtotalliab / qstockholderequity).reindex(qstockholderequity.index)
    qdebttocapital = (qtotalliab / (qtotalliab + qstockholderequity)).reindex(qtotalliab.index)

    qebitda = qincomestm.loc['EBITDA']
    qdebtoebitda = (qtotalliab / qebitda).reindex(qtotalliab.index)
    qebit = qincomestm.loc['EBIT']

    # Operating Efficiency Ratios
    qinventoryturnover = (qcostofrevenue / qinventory).reindex(qinventory.index)

    qaccountsreceivable = qbalancesheet.loc['Receivables']
    qaccountsreceivabledays = ((qaccountsreceivable / qtotalrevenue) * 365).reindex(qtotalrevenue.index)

    qaccountspayable = qbalancesheet.loc['Accounts Payable']
    qaccountspayabledays = ((qaccountspayable / qcostofrevenue) * 365).reindex(qcostofrevenue.index)

    qtotalassets = qbalancesheet.loc['Total Assets']
    qtotalassetturnover = (qtotalrevenue / qtotalassets).reindex(qtotalrevenue.index)

    qnetassets = qtotalassets - qcurrentliabilities
    qnetassetturnover = (qtotalrevenue / qnetassets).reindex(qtotalrevenue.index)

    # Cash Conversion Cycle
    qaverageinventory = (qinventory.iloc[0] + qinventory.iloc[1]) / 2
    qaverageaccountsreceivable = (qaccountsreceivable.iloc[0] + qaccountsreceivable.iloc[1]) / 2
    qaverageaccountspayable = (qaccountspayable.iloc[0] + qaccountspayable.iloc[1]) / 2

    qdio = (qaverageinventory / qcostofrevenue.iloc[0]) * 365
    qdso = (qaverageaccountsreceivable / qtotalrevenue.iloc[0]) * 365
    qdpo = (qaverageaccountspayable / qcostofrevenue.iloc[0]) * 365

    qccc = qdio + qdso - qdpo

    # Rates of Return and Profitability
    qreturnonequity = (qnetincome / qstockholderequity).reindex(qnetincome.index)
    qreturnonassets = (qnetincome / qtotalassets).reindex(qnetincome.index)

    # DuPont components
    qequitymultiplier = (qtotalassets / qstockholderequity).reindex(qtotalassets.index)
    qroedupont = (qnetprofitmargin * qtotalassetturnover * qequitymultiplier).reindex(qequitymultiplier.index)


    features = pd.DataFrame({
    'Net Profit Margin': qnetprofitmargin,
    'Quick Ratio': qquickratio,
    'ROA': qreturnonassets,
    'ROE': qreturnonequity,
    'Asset Turnover': qtotalassetturnover,
    'Debt-to-Equity': qdebttoequity,
    'Dupont ROE': qroedupont
    }).dropna(axis=0)   
    return features

def get_quarterly_ticker_data_av(ticker):
    base_url = 'https://www.alphavantage.co/query'

    def fetch_data(function):
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': API_KEY
        }
        response = requests.get(base_url, params=params)
        return response.json()

    income_data = fetch_data('INCOME_STATEMENT')
    balance_data = fetch_data('BALANCE_SHEET')

    # Parse quarterly data
    q_income = income_data['quarterlyReports']
    q_balance = balance_data['quarterlyReports']

    # Use most recent quarters only
    #df_income = pd.DataFrame(q_income).set_index('fiscalDateEnding').astype(float)
    df_income = pd.DataFrame(q_income).set_index('fiscalDateEnding')
    df_income = df_income.apply(pd.to_numeric, errors='coerce')
    #df_balance = pd.DataFrame(q_balance).set_index('fiscalDateEnding').astype(float)
    df_balance = pd.DataFrame(q_balance).set_index('fiscalDateEnding')
    df_balance = df_balance.apply(pd.to_numeric, errors='coerce')

    # Align and keep only matching dates
    common_dates = df_income.index.intersection(df_balance.index)
    df_income = df_income.loc[common_dates]
    df_balance = df_balance.loc[common_dates]
    #print(df_income)
    #print(df_balance)
    # Calculate metrics
    total_revenue = df_income['totalRevenue']
    net_income = df_income['netIncome']
    ebit = df_income['ebit']
    total_assets = df_balance['totalAssets']
    total_liab = df_balance['totalLiabilities']
    total_equity = df_balance['totalShareholderEquity']
    inventory = df_balance['inventory']
    current_assets = df_balance['totalCurrentAssets']
    current_liab = df_balance['totalCurrentLiabilities']
    receivables = df_balance['currentNetReceivables']
    accounts_payable = df_balance['currentAccountsPayable']

    # Ratios
    net_profit_margin = (net_income / total_revenue) * 100
    quick_ratio = ((current_assets - inventory) / current_liab)
    roa = (net_income / total_assets)
    roe = (net_income / total_equity)
    asset_turnover = total_revenue / total_assets
    debt_to_equity = total_liab / total_equity
    equity_multiplier = total_assets / total_equity
    roedupont = net_profit_margin * asset_turnover * equity_multiplier

    # Create features DataFrame
    features = pd.DataFrame({
        'Net Profit Margin': net_profit_margin,
        'Quick Ratio': quick_ratio,
        'ROA': roa,
        'ROE': roe,
        'Asset Turnover': asset_turnover,
        'Debt-to-Equity': debt_to_equity,
        'Dupont ROE': roedupont
    }).dropna()

    return features

def get_yearly_ticker_data_av(ticker):
    base_url = 'https://www.alphavantage.co/query'

    def fetch_data(function):
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': API_KEY
        }
        response = requests.get(base_url, params=params)
        return response.json()

    income_data = fetch_data('INCOME_STATEMENT')
    balance_data = fetch_data('BALANCE_SHEET')
    # Parse annual data (instead of quarterlyReports)
    y_income = income_data['annualReports']
    y_balance = balance_data['annualReports']

    # Convert to DataFrame and set index as fiscalDateEnding
    df_income = pd.DataFrame(y_income).set_index('fiscalDateEnding')
    df_income = df_income.apply(pd.to_numeric, errors='coerce')

    df_balance = pd.DataFrame(y_balance).set_index('fiscalDateEnding')
    df_balance = df_balance.apply(pd.to_numeric, errors='coerce')

    # Align and keep only matching dates
    common_dates = df_income.index.union(df_balance.index)
    df_income = df_income.reindex(common_dates)
    df_balance = df_balance.reindex(common_dates)
    #print(df_income)
    #print(df_balance)
    # Extract relevant financials
    total_revenue = df_income['totalRevenue']
    net_income = df_income['netIncome']
    ebit = df_income['ebit']
    ebitda = df_income['ebitda']

    total_assets = df_balance['totalAssets']
    total_debt = df_balance['shortLongTermDebtTotal']
    total_equity = df_balance['totalShareholderEquity']
    current_assets = df_balance['totalCurrentAssets']
    current_liab = df_balance['totalCurrentLiabilities']
    inventory = df_balance['inventory']
    accounts_receivable = df_balance['currentNetReceivables']
    accounts_payable = df_balance['currentAccountsPayable']

    interest_expense = df_income['interestExpense']
    pretax_income = df_income['incomeBeforeTax']
    tax_provision = df_income['incomeTaxExpense']
    cost_of_revenue = df_income['costOfRevenue']
    sg_and_a = df_income['sellingGeneralAndAdministrative']

    # Now calculate the additional ratios:

    # Basic ratios you already have
    net_profit_margin = (net_income / total_revenue) * 100
    quick_ratio = ((current_assets - inventory) / current_liab)
    roa = (net_income / total_assets)
    roe = (net_income / total_equity)
    asset_turnover = total_revenue / total_assets
    debt_to_equity = total_debt / total_equity
    equity_multiplier = total_assets / total_equity
    roedupont = net_profit_margin * asset_turnover * equity_multiplier

    # Additional ratios from Alpha Vantage data:

    debt_to_capital = total_debt / (total_debt + total_equity)

    debt_to_ebitda = total_debt / ebitda  # check None

    interest_coverage = ebit / interest_expense

    cogs_percent = (cost_of_revenue / total_revenue) * 100
    sganda_percent = (sg_and_a / total_revenue) * 100
    interest_expense_percent = (interest_expense / total_revenue) * 100
    ebt_percent = (pretax_income / total_revenue) * 100
    tax_percent = (tax_provision / total_revenue) * 100

    inventory_turnover = cost_of_revenue / inventory

    accounts_receivable_days = (accounts_receivable / total_revenue) * 365

    accounts_payable_days = (accounts_payable / cost_of_revenue) * 365

    total_asset_turnover = total_revenue / total_assets

    net_assets = total_assets - current_liab
    net_asset_turnover = total_revenue / net_assets

    # Cash Conversion Cycle (using last two years average)
    yaverage_inventory = (inventory + inventory.shift(-1)) / 2
    yaverage_accounts_receivable = (accounts_receivable + accounts_receivable.shift(-1)) / 2
    yaverage_accounts_payable = (accounts_payable + accounts_payable.shift(-1)) / 2
    print(yaverage_inventory)
    print(yaverage_accounts_receivable)
    print(yaverage_accounts_payable)
    dio = (yaverage_inventory / cost_of_revenue.shift(-1)) * 365
    dso = (yaverage_accounts_receivable / total_revenue.shift(-1)) * 365
    dpo = (yaverage_accounts_payable / cost_of_revenue.shift(-1)) * 365
    print(dio)
    print(dso)
    print(dpo)
    cash_conversion_cycle = dio + dso - dpo

    # Build your extended DataFrame

    features = pd.DataFrame({
        'Net Profit Margin': net_profit_margin,
        'Quick Ratio': quick_ratio,
        'ROA': roa,
        'ROE': roe,
        'Asset Turnover': asset_turnover,
        'Debt-to-Equity': debt_to_equity,
        'Equity Multiplier': equity_multiplier,
        'Dupont ROE': roedupont,
        'Debt-to-Capital': debt_to_capital,
        'Debt-to-EBITDA': debt_to_ebitda,
        'Interest Coverage': interest_coverage,
        'COGS Percent': cogs_percent,
        'SG&A Percent': sganda_percent,
        'Interest Expense Percent': interest_expense_percent,
        'EBT Percent': ebt_percent,
        'Tax Percent': tax_percent,
        'Inventory Turnover': inventory_turnover,
        'Accounts Receivable Days': accounts_receivable_days,
        'Accounts Payable Days': accounts_payable_days,
        'Total Asset Turnover': total_asset_turnover,
        'Net Asset Turnover': net_asset_turnover,
        'Cash Conversion Cycle': cash_conversion_cycle
    })

    return features

ratiocategories = {
    # Liquidity Ratios
    'Quick Ratio': 'Liquidity',

    # Leverage Ratios
    'Debt-to-Equity': 'Leverage',
    'Equity Multiplier': 'Leverage',
    'Dupont ROE': 'Leverage',
    'Debt-to-Capital': 'Leverage',
    'Debt-to-EBITDA': 'Leverage',
    'Interest Coverage': 'Leverage',

    # Efficiency Ratios
    'Asset Turnover': 'Efficiency',
    'Inventory Turnover': 'Efficiency',
    'Accounts Receivable Days': 'Efficiency',
    'Accounts Payable Days': 'Efficiency',
    'Total Asset Turnover': 'Efficiency',
    'Net Asset Turnover': 'Efficiency',
    'Cash Conversion Cycle': 'Efficiency',

    # Profitability Ratios
    'Net Profit Margin': 'Profitability',
    'ROA': 'Profitability',
    'ROE': 'Profitability',
    'EBT Percent': 'Profitability',
    'Tax Percent': 'Profitability',
    'COGS Percent': 'Profitability',
    'SG&A Percent': 'Profitability',
    'Interest Expense Percent': 'Profitability',

    # Market Value Ratios - affected by the ratios above
    # None of the provided ratios fall under this category
}

ratiocategoriescount = {
    "Liquidity" : 0,
    "Leverage" : 0,
    "Efficiency" : 0,
    "Profitability" : 0,
}
features = get_yearly_ticker_data_av("BARC")
features = features.replace([np.inf, -np.inf], np.nan)
features = features.dropna()
print(features)
features.sort_index()

scaler = StandardScaler()
X = scaler.fit_transform(features)

pca = PCA(n_components=min(X.shape))
Xpca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_
cum_explained = explained.cumsum()

plt.figure(figsize=(8,4))
plt.plot(np.arange(1, len(explained)+1), explained, 'o-', label='Individual')
plt.plot(np.arange(1, len(explained)+1), cum_explained, 's--', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree / Explained Variance')
plt.grid(True)
plt.legend()
plt.show()

# 7. Component loadings (interpretation)
loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])])
print("Loadings:\n", loadings.round(3))

# 8. Create PCA scores time series (DataFrame)
pcs = pd.DataFrame(Xpca, index=features.index, columns=[f'PC{i+1}' for i in range(Xpca.shape[1])])
print(pcs.head())

# 9. Plot first 2 PC time-series
pcs[['PC1','PC2']].plot(subplots=True, figsize=(10,4), title='PC scores over time')
plt.show()

# 10. Biplot-ish visualization: loadings vs PC scores for first two PCs
plt.figure(figsize=(8,6))
plt.scatter(pcs['PC1'], pcs['PC2'])
for i, col in enumerate(loadings.index):
    plt.arrow(0, 0, loadings.iloc[i,0]*3, loadings.iloc[i,1]*3,
              head_width=0.05, head_length=0.05, linewidth=1, alpha=0.8)
    plt.text(loadings.iloc[i,0]*3.2, loadings.iloc[i,1]*3.2, col)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 vs PC2 (arrows show variable loadings)')
plt.grid(True)
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    index=features.columns,
    columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
)
print("Loadings:\n", loadings.round(3))

categories = pd.DataFrame(
    index=loadings.columns,
    columns=["Category1", "Category2", "Category3"]
)


for pc in loadings.columns:
    print(f"\nTop variables for {pc}:")
    print(loadings[pc].sort_values(key=abs, ascending=False).head(5))
    topratiosperpc = loadings[pc].abs().sort_values(ascending=False).head(10)
    print(topratiosperpc)
    for ratio, loading in topratiosperpc.items():
        ratiocategoriescount[ratiocategories[ratio]] += loading
    print(ratiocategoriescount)

#With each top ratio, measure correlation between themselves and the companies profits, revenues or any success indicator
#Analyse strength of correlation
#Using LinearRegression for each PC and performance metric and possibly predict the companies performance in the future

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
plt.title("PCA Loadings")
plt.show()

pc_to_plot = "PC1"
loadings[pc_to_plot].sort_values().plot(kind='barh', figsize=(8,6))
plt.title(f"Loadings for {pc_to_plot}")
plt.show()


ticker_df = features.copy()
for i in range(pca.n_components_):
    ticker_df[f'PC{i+1}'] = Xpca[:, i]

# Add stock returns
price = yf.download("BARC.L", start="2015-01-01", end="2025-01-01", interval="3mo")
price['Return'] = price['Adj Close'].pct_change()

# Merge PCA with returns
merged = ticker_df.merge(price[['Return']], left_index=True, right_index=True, how='inner')

# Check correlation
corr_pc1, _ = pearsonr(merged['PC1'], merged['Return'])
print(f"Correlation between PC1 and returns: {corr_pc1:.3f}")