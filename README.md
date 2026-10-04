Analysed time-varying structure of US stock equity returns using rolling PCA, subspace angles, and sector loadings to study changes in market dimensionality during financial crises.

# Overview
- Collected daily adjusted closing prices for a diversified set of US stocks across major sectors.
- Computed daily percentage returns and standardised them for comparability.
- Applied rolling window PCA to capture time-varying factor structure in the cross-section of returns.
- Analysed the evolution of the dominant market factor and higher-order components.

# Metrics
- Principal Component Variance (PC1 dominance) – measures how much of market variation is explained by the main factor.
- Effective Dimensionality – number of principal components required to explain 90% of variance.
- PC1 Loadings (Factor Exposures) – contribution of each stock and sector to the main market factor and direction of PC1.
- Sector Contributions – decomposition of PC1 into sector-level influence.
- Subspace Rotation (Principal Angles) – measures structural changes in the factor space over time.
- PC1 Direction Stability – tracks temporal consistency of the dominant market factor.
- Market Volatility – rolling realised volatility for comparison with structural changes.

```bash
git clone https://github.com/Max24Dark/SP-500-roling-PCA.git
cd SP-500-roling-PCA
pip install -r requirements.txt
python rolling_pca.py
```


# Some Metric Diagrams

<img width="613" height="445" alt="image" src="https://github.com/user-attachments/assets/f0132148-1935-420f-93d9-2ae2a940a53d" />

<img width="626" height="453" alt="image" src="https://github.com/user-attachments/assets/ee060136-2d6e-4282-9abd-7b116fbd1e88" />

<img width="626" height="457" alt="image" src="https://github.com/user-attachments/assets/f254120f-a074-4717-8bc9-db209c1a59dd" />



# Insights
Most market crises are characterised by a more concentrated and lower-dimensional structure, where a single dominant factor explains a larger share of variance leading to more synchronised movements of the stocks.
In times of crisis, I noticed that the subspace produced by the top 3 principal components are more stable
