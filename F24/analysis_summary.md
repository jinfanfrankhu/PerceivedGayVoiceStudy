# Gay Voice Perception Research - Analysis Summary

## Sample: 86 listeners, 36 speakers
Missing ratings: 12 / 3096 (0.4%)

## 1A. Overall Accuracy
- Pearson r = 0.2875, R² = 0.0826, p = 0.089102
- Regression: predicted = 0.150 * actual + 2.339

## 1B. Accuracy by Gender
- Male: r = 0.3138, R² = 0.0985, N = 32
- Female: r = 0.2581, R² = 0.0666, N = 54

## 1C. Accuracy by Orientation
- Gay/Lesbian: insufficient data (N=2)
- Straight: r = 0.2685, R² = 0.0721, N = 71
- Bi/Pan: r = 0.1523, R² = 0.0232, N = 4
- Other/Prefer not to say: r = 0.3396, R² = 0.1153, N = 8

## 1D. Accuracy by Familiarity
- Low: r = 0.2441, R² = 0.0596, N = 31
- Medium: r = 0.2740, R² = 0.0751, N = 44
- High: r = 0.3412, R² = 0.1164, N = 10

## 2A. Individual Listener Accuracy
- Mean r = 0.1482, Median = 0.1558
- SD = 0.1767, Range = [-0.2759, 0.5843]
- N listeners with valid r: 85

## 2B. Accuracy vs Listener Characteristics
- Gender: Male M=0.161 vs Female M=0.141, t=0.49, p=0.6271
- Orientation ANOVA: F=1.79, p=0.1735
- Religiosity vs accuracy: r=-0.001, p=0.9903
- Familiarity vs accuracy: r=0.177, p=0.1043

## 3A. Inter-Rater Reliability (ICC)
- ICC(2,1) single measures = 0.2317
- ICC(2,k) average measures = 0.9629
- Computed on 27 speakers with complete data across 86 raters

## 3B. Per-Speaker Agreement
- Highest agreement (lowest SD): AI9 (SD=0.896)
- Lowest agreement (highest SD): RU35 (SD=1.662)
- Mean SD across speakers: 1.376

## 4A. Consensus vs Accuracy
- Correlation between SD and absolute error: r = 0.2740, p = 0.1059
- Interpretation: Higher agreement associated with higher accuracy

## 5A. Speaker Readability
(See quadrant plot for detailed speaker positions)

## 5B. Accuracy at Extremes
- Kruskal-Wallis: H=6.05, p=0.1952
  - 1 (Straight): N=15, Mean error=1.316
  - 2: N=8, Mean error=1.057
  - 3 (Middle): N=4, Mean error=0.822
  - 4: N=1, Mean error=2.033
  - 5 (Gay): N=8, Mean error=1.766

## 5C. Speaker Clustering
- PCA: PC1 explains 31.4%, PC2 explains 6.3%
- Cluster 0: PP3, DM7, EX14, GE15, LA19, PL23, PS25, SI26, WCH30
- Cluster 1: JC1, LQ2, HZ5, AI9, BR11, DH13, JF18, LS20, PE22, PR24, VI29, XX34
- Cluster 2: MM4, GA6, AB8, BO10, CA12, GO16, HY17, NO21, TU27, TY28, WCL31, WH32, WT33, RU35, QI36

## 6A. Listener Response Bias
- Grand mean rating: 2.706
- SD of listener means: 0.432
- Range: [1.00, 3.58]

## 6B. Religiosity vs Rating Bias
- r = 0.0578, p = 0.6013
- More religious listeners rate speakers as gayer

## 6C. Listener Response Range
- Mean listener SD: 1.532
- Range of SDs: [0.00, 1.89]

## 7A. Gender x Orientation Interaction
- statsmodels not available; skipping formal ANOVA

## 7B. Match Effects
- Gay speakers (actual >= 4): 8 speakers
- Straight speakers (actual <= 2): 20 speakers
- Gay listeners: MAE for gay speakers = 1.675, for straight speakers = 0.860
- Straight listeners: MAE for gay speakers = 1.840, for straight speakers = 1.449
- Bi/Pan listeners: MAE for gay speakers = 1.556, for straight speakers = 1.685

## 8A. Signal Detection Theory
- Threshold for dichotomization: 3.0
- Mean d' = 0.1164 (SD = 0.5021)
- Mean criterion c = -0.0546 (SD = 0.3758)
- d' > 0 indicates above-chance sensitivity

## 8B. d' by Listener Group
- Gender=Male: Mean d'=0.186, N=32
- Gender=Female: Mean d'=0.075, N=54
- Orientation=Straight: Mean d'=0.107, N=71
- Orientation=Bi/Pan: Mean d'=-0.033, N=4
- Orientation=Gay: Mean d'=0.416, N=2
- Familiarity=Low: Mean d'=0.046, N=31
- Familiarity=Medium: Mean d'=0.130, N=44
- Familiarity=High: Mean d'=0.213, N=10

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