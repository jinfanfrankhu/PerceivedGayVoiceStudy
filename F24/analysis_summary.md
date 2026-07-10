# Gay Voice Perception Research - Analysis Summary

## Sample: 85 listeners, 36 speakers
Missing ratings: 12 / 3060 (0.4%)

## 3C. ICC by Demographic Group
- Male (N=31): ICC(2,1) = 0.2312, ICC(2,k) = 0.9031
- Female (N=54): ICC(2,1) = 0.2710, ICC(2,k) = 0.9525
- Fam: Low (N=30): ICC(2,1) = 0.1947, ICC(2,k) = 0.8789
- Fam: Medium (N=44): ICC(2,1) = 0.2895, ICC(2,k) = 0.9472
- Fam: High (N=10): ICC(2,1) = 0.2677, ICC(2,k) = 0.7852

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

## 5G. Rating SD vs Non-Consensus PCA Distance (PC2-PC5)
- Pearson r = 0.7788, p = 0.0000
- R2 = 0.6066
- Non-consensus distance = sqrt(PC2^2 + PC3^2 + PC4^2 + PC5^2)
- Variance captured by PC2-PC5: 20.3%

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