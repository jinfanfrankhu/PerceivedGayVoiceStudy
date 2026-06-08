# Gay Voice Perception Research - Analysis Summary

## Sample: 85 listeners, 36 speakers
Missing ratings: 12 / 3060 (0.4%)

## 1A. Overall Accuracy
- Spearman ρ = 0.2769, p = 0.102110
- Regression: predicted = 0.150 * actual + 2.339

## 1B. Accuracy by Gender
- Male: ρ = 0.3196, N = 31
- Female: ρ = 0.2446, N = 54

## 1C. Accuracy by Orientation
- Gay/Lesbian: insufficient data (N=2)
- Straight: ρ = 0.2467, N = 70
- Bi/Pan: ρ = 0.1450, N = 4
- Other/Prefer not to say: ρ = 0.4084, N = 8

## 1D. Accuracy by Familiarity
- Low: ρ = 0.2362, N = 30
- Medium: ρ = 0.2728, N = 44
- High: ρ = 0.3317, N = 10

## 2A. Individual Listener Accuracy
- Mean ρ = 0.1608, Median = 0.1638
- SD = 0.1692, Range = [-0.2228, 0.5611]
- N listeners with valid r: 85

## 2B. Accuracy vs Listener Characteristics
- Gender: Male M=0.172 vs Female M=0.154, t=0.46, p=0.6439
- Orientation ANOVA: F=0.92, p=0.4039
- Religiosity vs accuracy: r=0.027, p=0.8105
- Familiarity vs accuracy: r=0.154, p=0.1606

## 3A. Inter-Rater Reliability (ICC)
- ICC(2,1) single measures = 0.2380
- ICC(2,k) average measures = 0.9637
- Computed on 27 speakers with complete data across 85 raters
- (Mean-imputed, all 36 speakers) ICC(2,1) = 0.2457, ICC(2,k) = 0.9651

## 3B. Per-Speaker Agreement
- Highest agreement (lowest SD): AI9 (SD=0.900)
- Lowest agreement (highest SD): RU35 (SD=1.658)
- Mean SD across speakers: 1.368

## 3C. ICC by Demographic Group
- Male (N=31): ICC(2,1) = 0.2312, ICC(2,k) = 0.9031
- Female (N=54): ICC(2,1) = 0.2710, ICC(2,k) = 0.9525
- Fam: Low (N=30): ICC(2,1) = 0.1947, ICC(2,k) = 0.8789
- Fam: Medium (N=44): ICC(2,1) = 0.2895, ICC(2,k) = 0.9472
- Fam: High (N=10): ICC(2,1) = 0.2677, ICC(2,k) = 0.7852

## 4A. Consensus vs Accuracy
- Correlation between SD and absolute error: r = 0.2691, p = 0.1125
- Interpretation: Higher agreement associated with higher accuracy

## 4B. Mean-Based Error vs Individual-Level MAE
- Individual-level MAE = mean(|rating_i - actual|) per speaker
- Points above diagonal: mean metric was overly optimistic (bimodal cancellation)
- Top 5 divergers (IndivMAE - MeanBased):
  TY28: indiv MAE = 1.268, mean-based = 0.147, diff = +1.122
  LS20: indiv MAE = 0.932, mean-based = 0.149, diff = +0.783
  PS25: indiv MAE = 1.202, mean-based = 0.458, diff = +0.744
  WCH30: indiv MAE = 1.548, mean-based = 0.842, diff = +0.706
  BO10: indiv MAE = 1.633, mean-based = 0.991, diff = +0.642

## 5A. Speaker Readability
(See quadrant plot for detailed speaker positions)

## 5B. Accuracy at Extremes
- Kruskal-Wallis: H=6.05, p=0.1952
  - 1 (Straight): N=15, Mean error=1.316
  - 2: N=8, Mean error=1.057
  - 3 (Middle): N=4, Mean error=0.822
  - 4: N=1, Mean error=2.033
  - 5 (Gay): N=8, Mean error=1.766

## 5C1. Hierarchical Clustering
- Ward linkage on standardized 85-dimensional listener rating vectors
- See dendrogram plot for speaker groupings

## 5C2. PCA Variance Diagnostics
- PC1 = 30.9%, PC2 = 6.1%, 2D total = 37.0%
- Components needed for 50% variance: 5 (PC1–PC5: 51.3%)
- Components needed for 70% variance: 11 (PC1–PC11: 71.2%)

## 5C3. K-means Diagnostics (2D PCA)
- Best k by silhouette: k=4 (score=0.518)
- k=3 silhouette: 0.446
- k=4 silhouette: 0.518

## 5C4. DBSCAN on 2D PCA
- Kneedle eps=2.048: 2 cluster(s), 5 noise
  - Noise: PP3, DM7, GE15, SI26, XX34
  - Cluster 0: JC1, LQ2, HZ5, AB8, AI9, BR11, DH13, GO16, HY17, JF18, LS20, PE22, PR24, TU27, TY28, VI29, WCL31, WH32, WT33, RU35, QI36
  - Cluster 1: MM4, GA6, BO10, CA12, EX14, LA19, NO21, PL23, PS25, WCH30
- Conservative eps=1.520: 6 cluster(s), 14 noise
  - Noise: PP3, MM4, DM7, EX14, GE15, JF18, PL23, SI26, TU27, WCH30, WH32, WT33, XX34, RU35
  - Cluster 0: JC1, DH13, PE22, PR24, VI29
  - Cluster 1: LQ2, HZ5, AI9, BR11, LS20
  - Cluster 2: GA6, CA12, NO21
  - Cluster 3: AB8, HY17, WCL31
  - Cluster 4: GO16, TY28, QI36
  - Cluster 5: BO10, LA19, PS25
- K-means k=4 cluster members:
  - Cluster 0: MM4, GA6, BO10, CA12, EX14, LA19, NO21, PL23, PS25, WCH30
  - Cluster 1: JC1, LQ2, HZ5, AI9, BR11, DH13, JF18, LS20, PE22, PR24, VI29, WH32, WT33, XX34
  - Cluster 2: PP3, DM7, GE15, SI26
  - Cluster 3: AB8, GO16, HY17, TU27, TY28, WCL31, RU35, QI36

## 5C5. DBSCAN on 5-Component PCA (51.3% variance)
- Kneedle eps=5.135: 2 cluster(s), 3 noise
  - Noise: PL23, TU27, RU35
  - Cluster 0: JC1, LQ2, MM4, HZ5, GA6, AB8, AI9, BO10, BR11, CA12, DH13, EX14, GO16, HY17, JF18, LA19, LS20, NO21, PE22, PR24, PS25, TY28, VI29, WCH30, WCL31, WH32, WT33, XX34, QI36
  - Cluster 1: PP3, DM7, GE15, SI26
- Conservative eps=3.869: 3 cluster(s), 16 noise
  - Noise: MM4, GA6, AB8, BO10, CA12, GE15, GO16, HY17, NO21, PL23, TU27, TY28, WCH30, WCL31, RU35, QI36
  - Cluster 0: JC1, LQ2, HZ5, AI9, BR11, DH13, JF18, LS20, PE22, PR24, VI29, WH32, WT33, XX34
  - Cluster 1: PP3, DM7, SI26
  - Cluster 2: EX14, LA19, PS25

## 5C6. DBSCAN on 11-Component PCA (71.2% variance)
- Kneedle eps=5.081: 1 cluster(s), 30 noise
  - Noise: JC1, PP3, MM4, GA6, DM7, AB8, BO10, CA12, DH13, EX14, GE15, GO16, HY17, JF18, LA19, NO21, PE22, PL23, PR24, PS25, SI26, TU27, TY28, VI29, WCH30, WCL31, WH32, WT33, RU35, QI36
  - Cluster 0: LQ2, HZ5, AI9, BR11, LS20, XX34
- Conservative eps=6.572: 4 cluster(s), 14 noise
  - Noise: MM4, GA6, BO10, DH13, GE15, GO16, HY17, NO21, PL23, TU27, TY28, WCH30, WT33, RU35
  - Cluster 0: JC1, LQ2, HZ5, AI9, BR11, JF18, LS20, PE22, PR24, VI29, WH32, XX34
  - Cluster 1: PP3, DM7, SI26
  - Cluster 2: AB8, WCL31, QI36
  - Cluster 3: CA12, EX14, LA19, PS25

## 5D1. PC1 vs Actual Orientation
- Pearson r = 0.2727, p = 0.1077
- R² = 0.0743
- PC1 range: [-7.534, 11.670]

## 5D2. PC1 vs Mean Rating
- Pearson r = 0.9977, p = 0.0000
- R² = 0.9953
- High r² validates PC1 as the consensus perceived-gayness dimension

## 5D3. |PC1| vs Absolute Error
- Pearson r = -0.1720, p = 0.3158
- Negative r = speakers at PC1 extremes are rated MORE accurately
- Positive r = speakers at PC1 extremes are rated LESS accurately

## 5E1. PC2 vs Mean Listener Rating
- Pearson r = -0.0131, p = 0.9393
- R² = 0.0002
- Variance explained by PC2: 6.1%

## 5E2. PC3 vs Mean Listener Rating
- Pearson r = 0.0257, p = 0.8817
- R² = 0.0007
- Variance explained by PC3: 5.6%

## 5E3. PC4 vs Mean Listener Rating
- Pearson r = 0.0208, p = 0.9041
- R² = 0.0004
- Variance explained by PC4: 4.6%

## 5E4. PC5 vs Mean Listener Rating
- Pearson r = 0.0040, p = 0.9816
- R² = 0.0000
- Variance explained by PC5: 4.1%

## 5F. Speaker Rating SD vs |PC2|
- Pearson r = 0.3559, p = 0.0332
- R² = 0.1266
- r ~= 0  => PC2 is NOT simply capturing disagreement magnitude
- r > 0   => high-SD speakers sit further from the PC2 origin

## 5G. Rating SD vs Non-Consensus PCA Distance (PC2-PC5)
- Pearson r = 0.7788, p = 0.0000
- R2 = 0.6066
- Non-consensus distance = sqrt(PC2^2 + PC3^2 + PC4^2 + PC5^2)
- Variance captured by PC2-PC5: 20.3%

## 6A. Listener Response Bias
- Grand mean rating: 2.726
- SD of listener means: 0.392
- Range: [1.94, 3.58]

## 6B. Religiosity vs Rating Bias
- r = 0.0398, p = 0.7212
- More religious listeners rate speakers as gayer

## 6C. Listener Response Range
- Mean listener SD: 1.550
- Range of SDs: [1.09, 1.89]

## 7A. Gender x Orientation Interaction
- statsmodels not available; skipping formal ANOVA

## 7B. Match Effects
- Gay speakers (actual >= 4): 8 speakers
- Straight speakers (actual <= 2): 20 speakers
- Gay listeners: MAE for gay speakers = 1.675, for straight speakers = 0.860
- Straight listeners: MAE for gay speakers = 1.810, for straight speakers = 1.465
- Bi/Pan listeners: MAE for gay speakers = 1.556, for straight speakers = 1.685

## 8A. Signal Detection Theory
- Threshold for dichotomization: 3.0
- Mean d' = 0.1127 (SD = 0.5039)
- Mean criterion c = -0.0771 (SD = 0.3144)
- d' > 0 indicates above-chance sensitivity

## 8B. d' by Listener Group
- Gender=Male: Mean d'=0.178, N=31
- Gender=Female: Mean d'=0.075, N=54
- Orientation=Straight: Mean d'=0.103, N=70
- Orientation=Bi/Pan: Mean d'=-0.033, N=4
- Orientation=Gay: Mean d'=0.416, N=2
- Familiarity=Low: Mean d'=0.033, N=30
- Familiarity=Medium: Mean d'=0.130, N=44
- Familiarity=High: Mean d'=0.213, N=10

## 9A. Per-Speaker Mean-Based Error (sorted)
- Range: 0.147 (TY28) to 3.047 (XX34)
- Mean across speakers: 1.324

## 9B. Per-Speaker Individual-Level MAE (sorted)
- Range: 0.619 (SI26) to 3.035 (XX34)
- Mean across speakers: 1.571

## Key Findings - Speaker Rankings

### Top 5 Most Accurately Perceived Speakers (lowest absolute error)
- TY28: error = 0.147, actual = 2.6, predicted = 2.45
- LS20: error = 0.149, actual = 1.8, predicted = 1.65
- DM7: error = 0.379, actual = 4.6, predicted = 4.22
- PS25: error = 0.458, actual = 2.6, predicted = 3.06
- JC1: error = 0.553, actual = 1.4, predicted = 1.95

### Top 5 Least Accurately Perceived Speakers (highest absolute error)
- XX34: error = 3.047, actual = 5.0, predicted = 1.95
- WCL31: error = 2.536, actual = 5.0, predicted = 2.46
- QI36: error = 2.419, actual = 5.0, predicted = 2.58
- PP3: error = 2.300, actual = 2.2, predicted = 4.50
- PL23: error = 2.240, actual = 1.4, predicted = 3.64

### Overall Metrics
- Mean Absolute Error (MAE): 1.3236
- RMSE: 1.5012