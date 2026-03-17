"""
Gay Voice Perception Research - Comprehensive Statistical Analysis
Analyzes relationship between perceived and actual sexual orientation from voice.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'figure.dpi': 300})

# ============================================================
# DATA LOADING
# ============================================================
INPUT_CSV = r"C:\Users\jinfa\Desktop\GayStudy\F24\Results.csv"
OUTPUT_DIR = r"C:\Users\jinfa\Desktop\GayStudy\F24\Dataplots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw = pd.read_csv(INPUT_CSV, header=None)

# Column 0 is row labels (empty for data rows). Speakers in cols 1-36, demographics in cols 37-44.
# Speaker codes from row 0, columns 1-36
speaker_codes = [str(raw.iloc[0, c]).strip().replace('\xa0', '') for c in range(1, 37)]

# Listener ratings: rows 2-87 (0-indexed), columns 1-36
ratings = raw.iloc[2:88, 1:37].apply(pd.to_numeric, errors='coerce')
ratings.index = range(86)
ratings.columns = speaker_codes

# Listener demographics: rows 2-87, columns 37-44
demo_cols = ['YOB', 'Gender', 'Orientation', 'English', 'GayFriends', 'GayAcquaint', 'Religiosity', 'Score']
demographics = raw.iloc[2:88, 37:45].apply(pd.to_numeric, errors='coerce')
demographics.index = range(86)
demographics.columns = demo_cols

# Converted self-reported scores (row 90, 0-indexed), columns 1-36
actual_scores = raw.iloc[90, 1:37].apply(pd.to_numeric, errors='coerce')
actual_scores.index = speaker_codes

# Average predicted scores (row 91, 0-indexed), columns 1-36
avg_predicted = raw.iloc[91, 1:37].apply(pd.to_numeric, errors='coerce')
avg_predicted.index = speaker_codes

print(f"Loaded {ratings.shape[0]} listeners x {ratings.shape[1]} speakers")
print(f"Missing ratings: {ratings.isna().sum().sum()} / {ratings.size}")
print(f"Actual scores range: {actual_scores.min():.1f} - {actual_scores.max():.1f}")
print(f"Avg predicted range: {avg_predicted.min():.2f} - {avg_predicted.max():.2f}")

summary_lines = []
def log(msg):
    print(msg)
    summary_lines.append(msg)

log("# Gay Voice Perception Research - Analysis Summary\n")
log(f"## Sample: {ratings.shape[0]} listeners, {ratings.shape[1]} speakers")
log(f"Missing ratings: {ratings.isna().sum().sum()} / {ratings.size} ({100*ratings.isna().sum().sum()/ratings.size:.1f}%)\n")

# ============================================================
# 1A. Overall R² - Perception Accuracy
# ============================================================
print("\n--- 1A: Overall Accuracy Scatter ---")
mask = actual_scores.notna() & avg_predicted.notna()
x_all, y_all = actual_scores[mask].values, avg_predicted[mask].values
r_val, p_val = stats.pearsonr(x_all, y_all)
slope, intercept = np.polyfit(x_all, y_all, 1)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x_all, y_all, s=80, alpha=0.7, edgecolors='k', zorder=3)
ax.plot([1, 5], [1, 5], 'k--', alpha=0.4, label='Perfect accuracy')
xs = np.linspace(0.5, 5.5, 100)
ax.plot(xs, slope * xs + intercept, 'r-', linewidth=2, label=f'Regression (R²={r_val**2:.3f})')
ax.set_xlabel('Actual Converted Score (1-5)')
ax.set_ylabel('Average Predicted Score (1-5)')
ax.set_title(f'Overall Perception Accuracy\nr = {r_val:.3f}, R² = {r_val**2:.3f}, p = {p_val:.4f}')
ax.legend()
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
for i, s in enumerate(speaker_codes):
    if mask.iloc[i]:
        ax.annotate(s, (x_all[list(mask[mask].index).index(s)], y_all[list(mask[mask].index).index(s)]),
                    fontsize=6, alpha=0.6) if False else None
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1A_overall_accuracy_scatter.png'))
plt.close()
log(f"## 1A. Overall Accuracy")
log(f"- Pearson r = {r_val:.4f}, R² = {r_val**2:.4f}, p = {p_val:.6f}")
log(f"- Regression: predicted = {slope:.3f} * actual + {intercept:.3f}\n")

# ============================================================
# Helper: compute group-level avg predicted scores and R²
# ============================================================
def group_accuracy(listener_mask, label=""):
    """Compute avg predicted per speaker for a subset of listeners, return r, R², n."""
    sub = ratings.loc[listener_mask]
    n = sub.shape[0]
    if n < 3:
        return None, None, n, None
    group_avg = sub.mean(axis=0)
    m = actual_scores.notna() & group_avg.notna()
    if m.sum() < 3:
        return None, None, n, None
    r, p = stats.pearsonr(actual_scores[m], group_avg[m])
    return r, r**2, n, group_avg

# ============================================================
# 1B. R² by Listener Gender
# ============================================================
print("--- 1B: Accuracy by Gender ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
gender_labels = {1: 'Male', 2: 'Female'}
gender_results = {}
for idx, (gval, glabel) in enumerate(gender_labels.items()):
    mask_g = demographics['Gender'] == gval
    r_g, r2_g, n_g, gavg = group_accuracy(mask_g, glabel)
    gender_results[glabel] = (r_g, r2_g, n_g)
    ax = axes[idx]
    if gavg is not None:
        m = actual_scores.notna() & gavg.notna()
        ax.scatter(actual_scores[m], gavg[m], s=60, alpha=0.7, edgecolors='k')
        sl, ic = np.polyfit(actual_scores[m].values, gavg[m].values, 1)
        ax.plot(xs, sl*xs+ic, 'r-', lw=2)
        ax.plot([1,5],[1,5],'k--',alpha=0.3)
    ax.set_title(f'{glabel} Listeners (N={n_g})\nR²={r2_g:.3f}' if r2_g else f'{glabel} (N={n_g})')
    ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
    ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
plt.suptitle('Perception Accuracy by Listener Gender', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1B_accuracy_by_gender.png'), bbox_inches='tight')
plt.close()
log("## 1B. Accuracy by Gender")
for g, (r, r2, n) in gender_results.items():
    log(f"- {g}: r = {r:.4f}, R² = {r2:.4f}, N = {n}" if r else f"- {g}: insufficient data (N={n})")
log("")

# ============================================================
# 1C. R² by Listener Sexual Orientation
# ============================================================
print("--- 1C: Accuracy by Orientation ---")
orient_labels = {1: 'Gay/Lesbian', 2: 'Straight', 3: 'Bi/Pan'}
orient_results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (oval, olabel) in enumerate(orient_labels.items()):
    mask_o = demographics['Orientation'] == oval
    r_o, r2_o, n_o, oavg = group_accuracy(mask_o, olabel)
    orient_results[olabel] = (r_o, r2_o, n_o)
    ax = axes[idx]
    if oavg is not None and r_o is not None:
        m = actual_scores.notna() & oavg.notna()
        ax.scatter(actual_scores[m], oavg[m], s=60, alpha=0.7, edgecolors='k')
        sl, ic = np.polyfit(actual_scores[m].values, oavg[m].values, 1)
        ax.plot(xs, sl*xs+ic, 'r-', lw=2)
        ax.plot([1,5],[1,5],'k--',alpha=0.3)
    ax.set_title(f'{olabel} (N={n_o})\nR²={r2_o:.3f}' if r2_o else f'{olabel} (N={n_o})')
    ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
    ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
plt.suptitle('Perception Accuracy by Listener Sexual Orientation', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1C_accuracy_by_orientation.png'), bbox_inches='tight')
plt.close()
log("## 1C. Accuracy by Orientation")
for o, (r, r2, n) in orient_results.items():
    log(f"- {o}: r = {r:.4f}, R² = {r2:.4f}, N = {n}" if r else f"- {o}: insufficient data (N={n})")
# Also check "other/prefer not to say"
mask_other = demographics['Orientation'].isin([4, 5])
r_oth, r2_oth, n_oth, _ = group_accuracy(mask_other, "Other/PNS")
orient_results["Other/PNS"] = (r_oth, r2_oth, n_oth)
log(f"- Other/Prefer not to say: r = {r_oth:.4f}, R² = {r2_oth:.4f}, N = {n_oth}" if r_oth else f"- Other/PNS: insufficient data (N={n_oth})")
log("")

# ============================================================
# 1D. R² by Familiarity with Gay Men
# ============================================================
print("--- 1D: Accuracy by Familiarity ---")
# Composite familiarity
friends = demographics['GayFriends']
acquaint = demographics['GayAcquaint']
fam = pd.Series(index=range(86), dtype=str)
for i in range(86):
    f, a = friends.iloc[i], acquaint.iloc[i]
    if pd.isna(f) or pd.isna(a):
        fam.iloc[i] = np.nan
    elif f == 1 and a == 1:
        fam.iloc[i] = 'Low'
    elif f == 3 or a == 3:
        fam.iloc[i] = 'High'
    else:
        fam.iloc[i] = 'Medium'

fam_results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, flabel in enumerate(['Low', 'Medium', 'High']):
    mask_f = fam == flabel
    r_f, r2_f, n_f, favg = group_accuracy(mask_f, flabel)
    fam_results[flabel] = (r_f, r2_f, n_f)
    ax = axes[idx]
    if favg is not None and r_f is not None:
        m = actual_scores.notna() & favg.notna()
        ax.scatter(actual_scores[m], favg[m], s=60, alpha=0.7, edgecolors='k')
        sl, ic = np.polyfit(actual_scores[m].values, favg[m].values, 1)
        ax.plot(xs, sl*xs+ic, 'r-', lw=2)
        ax.plot([1,5],[1,5],'k--',alpha=0.3)
    ax.set_title(f'{flabel} Familiarity (N={n_f})\nR²={r2_f:.3f}' if r2_f else f'{flabel} (N={n_f})')
    ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
    ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
plt.suptitle('Perception Accuracy by Familiarity with Gay Men', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1D_accuracy_by_familiarity.png'), bbox_inches='tight')
plt.close()
log("## 1D. Accuracy by Familiarity")
for fl, (r, r2, n) in fam_results.items():
    log(f"- {fl}: r = {r:.4f}, R² = {r2:.4f}, N = {n}" if r else f"- {fl}: insufficient data (N={n})")
log("")

# ============================================================
# 1E. Pearson r Comparison Bar Chart
# ============================================================
print("--- 1E: Correlation Comparison Bar Chart ---")
all_groups = {'Overall': (r_val, ratings.shape[0])}
all_groups.update({k: (v[0], v[2]) for k, v in gender_results.items()})
all_groups.update({k: (v[0], v[2]) for k, v in orient_results.items()})
all_groups.update({f'Fam: {k}': (v[0], v[2]) for k, v in fam_results.items()})

labels = [k for k in all_groups if all_groups[k][0] is not None]
r_vals = [all_groups[k][0] for k in labels]
n_vals = [all_groups[k][1] for k in labels]

fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette('tab10', len(labels))
bars = ax.bar(range(len(labels)), r_vals, color=colors, edgecolor='k', alpha=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f'{l}\n(N={n_vals[i]})' for i, l in enumerate(labels)], rotation=30, ha='right')
ax.set_ylabel('Pearson r')
ax.set_title('Pearson r (Perception Accuracy) Across Listener Groups')
ax.axhline(0, color='k', lw=0.5)
for i, v in enumerate(r_vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1E_correlation_comparison_barplot.png'))
plt.close()

# ============================================================
# 2A. Per-Listener Correlation with Actual Scores
# ============================================================
print("--- 2A: Individual Accuracy Distribution ---")
indiv_r = []
for i in range(86):
    listener_ratings = ratings.iloc[i]
    m = listener_ratings.notna() & actual_scores.notna()
    if m.sum() >= 5:
        r_i, _ = stats.pearsonr(listener_ratings[m], actual_scores[m])
        indiv_r.append(r_i)
    else:
        indiv_r.append(np.nan)
indiv_r = pd.Series(indiv_r)

fig, ax = plt.subplots(figsize=(10, 7))
valid_r = indiv_r.dropna()
ax.hist(valid_r, bins=20, edgecolor='k', alpha=0.7, color='steelblue')
ax.axvline(valid_r.mean(), color='red', lw=2, ls='--', label=f'Mean = {valid_r.mean():.3f}')
ax.axvline(valid_r.median(), color='orange', lw=2, ls='-.', label=f'Median = {valid_r.median():.3f}')
ax.set_xlabel('Individual Pearson r (with actual scores)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Individual Listener Accuracy')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2A_individual_accuracy_distribution.png'))
plt.close()
log("## 2A. Individual Listener Accuracy")
log(f"- Mean r = {valid_r.mean():.4f}, Median = {valid_r.median():.4f}")
log(f"- SD = {valid_r.std():.4f}, Range = [{valid_r.min():.4f}, {valid_r.max():.4f}]")
log(f"- N listeners with valid r: {valid_r.shape[0]}\n")

# ============================================================
# 2B. Accuracy vs. Listener Characteristics
# ============================================================
print("--- 2B: Accuracy vs Characteristics ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Gender boxplot
ax = axes[0, 0]
df_plot = pd.DataFrame({'Accuracy': indiv_r, 'Gender': demographics['Gender'].map({1: 'Male', 2: 'Female'})})
df_valid = df_plot.dropna()
sns.boxplot(x='Gender', y='Accuracy', data=df_valid, ax=ax, palette='Set2')
ax.set_title('Accuracy by Gender')
# t-test
male_acc = df_valid[df_valid['Gender']=='Male']['Accuracy']
female_acc = df_valid[df_valid['Gender']=='Female']['Accuracy']
if len(male_acc) > 1 and len(female_acc) > 1:
    t_stat, t_p = stats.ttest_ind(male_acc, female_acc)
    ax.set_xlabel(f't={t_stat:.2f}, p={t_p:.4f}')
    log("## 2B. Accuracy vs Listener Characteristics")
    log(f"- Gender: Male M={male_acc.mean():.3f} vs Female M={female_acc.mean():.3f}, t={t_stat:.2f}, p={t_p:.4f}")

# Orientation boxplot
ax = axes[0, 1]
orient_map = {1: 'Gay', 2: 'Straight', 3: 'Bi/Pan', 4: 'Other', 5: 'PNS'}
df_plot2 = pd.DataFrame({'Accuracy': indiv_r, 'Orientation': demographics['Orientation'].map(orient_map)})
df_valid2 = df_plot2.dropna()
sns.boxplot(x='Orientation', y='Accuracy', data=df_valid2, ax=ax, palette='Set3')
ax.set_title('Accuracy by Orientation')
# ANOVA for main groups
groups_anova = [df_valid2[df_valid2['Orientation']==g]['Accuracy'].values for g in ['Gay', 'Straight', 'Bi/Pan']
                if len(df_valid2[df_valid2['Orientation']==g]) > 0]
if len(groups_anova) >= 2:
    f_stat, f_p = stats.f_oneway(*[g for g in groups_anova if len(g) > 1])
    ax.set_xlabel(f'F={f_stat:.2f}, p={f_p:.4f}')
    log(f"- Orientation ANOVA: F={f_stat:.2f}, p={f_p:.4f}")

# Religiosity scatter
ax = axes[1, 0]
df_plot3 = pd.DataFrame({'Accuracy': indiv_r, 'Religiosity': demographics['Religiosity']}).dropna()
ax.scatter(df_plot3['Religiosity'], df_plot3['Accuracy'], alpha=0.6, edgecolors='k')
if len(df_plot3) > 3:
    sl, ic = np.polyfit(df_plot3['Religiosity'], df_plot3['Accuracy'], 1)
    ax.plot(np.linspace(0, 10, 50), sl*np.linspace(0, 10, 50)+ic, 'r-', lw=2)
    r_rel, p_rel = stats.pearsonr(df_plot3['Religiosity'], df_plot3['Accuracy'])
    ax.set_xlabel(f'Religiosity (r={r_rel:.3f}, p={p_rel:.4f})')
    log(f"- Religiosity vs accuracy: r={r_rel:.3f}, p={p_rel:.4f}")
ax.set_ylabel('Accuracy (r)')
ax.set_title('Accuracy vs Religiosity')

# Familiarity scatter
ax = axes[1, 1]
fam_numeric = friends.fillna(0) + acquaint.fillna(0)
df_plot4 = pd.DataFrame({'Accuracy': indiv_r, 'Familiarity': fam_numeric}).dropna()
ax.scatter(df_plot4['Familiarity'], df_plot4['Accuracy'], alpha=0.6, edgecolors='k')
if len(df_plot4) > 3:
    sl, ic = np.polyfit(df_plot4['Familiarity'], df_plot4['Accuracy'], 1)
    ax.plot(np.linspace(df_plot4['Familiarity'].min(), df_plot4['Familiarity'].max(), 50),
            sl*np.linspace(df_plot4['Familiarity'].min(), df_plot4['Familiarity'].max(), 50)+ic, 'r-', lw=2)
    r_fam, p_fam = stats.pearsonr(df_plot4['Familiarity'], df_plot4['Accuracy'])
    ax.set_xlabel(f'Familiarity composite (r={r_fam:.3f}, p={p_fam:.4f})')
    log(f"- Familiarity vs accuracy: r={r_fam:.3f}, p={p_fam:.4f}")
ax.set_ylabel('Accuracy (r)')
ax.set_title('Accuracy vs Familiarity (composite)')
log("")

plt.suptitle('Listener Accuracy vs Demographics', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2B_accuracy_vs_characteristics.png'), bbox_inches='tight')
plt.close()

# ============================================================
# 3A. Inter-Rater Reliability (ICC)
# ============================================================
print("--- 3A: Inter-Rater Reliability ---")
# ICC(2,k) - two-way random, average measures
# Using manual computation
ratings_clean = ratings.dropna(how='any', axis=1)  # speakers with no missing
n_subj = ratings_clean.shape[1]  # speakers
n_raters = ratings_clean.shape[0]  # listeners

grand_mean = ratings_clean.values.mean()
ss_between = n_raters * ((ratings_clean.mean(axis=0) - grand_mean)**2).sum()
ss_within = ((ratings_clean - ratings_clean.mean(axis=0))**2).sum().sum()
ms_between = ss_between / (n_subj - 1)
ms_within = ss_within / (n_subj * (n_raters - 1))

# Also compute using two-way model
row_means = ratings_clean.mean(axis=1)
col_means = ratings_clean.mean(axis=0)
ss_rows = n_subj * ((row_means - grand_mean)**2).sum()  # raters
ss_cols = n_raters * ((col_means - grand_mean)**2).sum()  # speakers
ss_total = ((ratings_clean.values - grand_mean)**2).sum()
ss_error = ss_total - ss_rows - ss_cols

ms_rows = ss_rows / (n_raters - 1)
ms_cols = ss_cols / (n_subj - 1)
ms_error = ss_error / ((n_raters - 1) * (n_subj - 1))

# ICC(2,1) single measures
icc_21 = (ms_cols - ms_error) / (ms_cols + (n_raters - 1)*ms_error + n_raters*(ms_rows - ms_error)/n_subj)
# ICC(2,k) average measures
icc_2k = (ms_cols - ms_error) / (ms_cols + (ms_rows - ms_error)/n_subj)

log("## 3A. Inter-Rater Reliability (ICC)")
log(f"- ICC(2,1) single measures = {icc_21:.4f}")
log(f"- ICC(2,k) average measures = {icc_2k:.4f}")
log(f"- Computed on {n_subj} speakers with complete data across {n_raters} raters\n")

# Optional: correlation heatmap of a subset of raters
fig, ax = plt.subplots(figsize=(10, 8))
corr_mat = ratings.T.corr()
# Show a random sample of 20 raters for readability
sample_idx = sorted(np.random.RandomState(42).choice(86, min(20, 86), replace=False))
sub_corr = corr_mat.iloc[sample_idx, sample_idx]
sns.heatmap(sub_corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=[f'L{i}' for i in sample_idx],
            yticklabels=[f'L{i}' for i in sample_idx])
ax.set_title(f'Inter-Rater Correlation (20 sampled listeners)\nICC(2,k) = {icc_2k:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3A_interrater_reliability.png'))
plt.close()

# ============================================================
# 3B. Per-Speaker Agreement (SD of ratings)
# ============================================================
print("--- 3B: Per-Speaker Agreement ---")
speaker_sd = ratings.std(axis=0).sort_values()

fig, ax = plt.subplots(figsize=(14, 7))
colors_sd = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(speaker_sd)))
ax.bar(range(len(speaker_sd)), speaker_sd.values, color=colors_sd, edgecolor='k')
ax.set_xticks(range(len(speaker_sd)))
ax.set_xticklabels(speaker_sd.index, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Speaker')
ax.set_ylabel('SD of Ratings')
ax.set_title('Rating Agreement per Speaker (lower SD = higher agreement)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3B_speaker_agreement.png'))
plt.close()
log("## 3B. Per-Speaker Agreement")
log(f"- Highest agreement (lowest SD): {speaker_sd.index[0]} (SD={speaker_sd.iloc[0]:.3f})")
log(f"- Lowest agreement (highest SD): {speaker_sd.index[-1]} (SD={speaker_sd.iloc[-1]:.3f})")
log(f"- Mean SD across speakers: {speaker_sd.mean():.3f}\n")

# ============================================================
# 4A. Consensus vs Accuracy
# ============================================================
print("--- 4A: Consensus vs Accuracy ---")
abs_error = (avg_predicted - actual_scores).abs()
speaker_sd_all = ratings.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(speaker_sd_all, abs_error, s=80, alpha=0.7, edgecolors='k')
for sp in speaker_codes:
    ax.annotate(sp, (speaker_sd_all[sp], abs_error[sp]), fontsize=7, alpha=0.7)
r_ca, p_ca = stats.pearsonr(speaker_sd_all.dropna(), abs_error[speaker_sd_all.notna()])
ax.set_xlabel('SD of Ratings (low = high agreement)')
ax.set_ylabel('Absolute Error |predicted - actual|')
ax.set_title(f'Consensus vs Accuracy\nr = {r_ca:.3f}, p = {p_ca:.4f}')
sl, ic = np.polyfit(speaker_sd_all.dropna().values, abs_error[speaker_sd_all.notna()].values, 1)
xr = np.linspace(speaker_sd_all.min(), speaker_sd_all.max(), 50)
ax.plot(xr, sl*xr+ic, 'r-', lw=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4A_consensus_vs_accuracy.png'))
plt.close()
log("## 4A. Consensus vs Accuracy")
log(f"- Correlation between SD and absolute error: r = {r_ca:.4f}, p = {p_ca:.4f}")
log(f"- Interpretation: {'Higher agreement associated with higher accuracy' if r_ca > 0 else 'Higher agreement associated with lower accuracy'}\n")

# ============================================================
# 5A. Speaker Readability Quadrants
# ============================================================
print("--- 5A: Speaker Readability Quadrants ---")
med_sd = speaker_sd_all.median()
med_err = abs_error.median()

fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(speaker_sd_all, abs_error, s=100, alpha=0.7, edgecolors='k', zorder=3)
for sp in speaker_codes:
    ax.annotate(sp, (speaker_sd_all[sp], abs_error[sp]), fontsize=8, alpha=0.8,
                xytext=(5, 5), textcoords='offset points')
ax.axvline(med_sd, color='gray', ls='--', alpha=0.5)
ax.axhline(med_err, color='gray', ls='--', alpha=0.5)
ax.text(speaker_sd_all.min(), abs_error.min(), 'High Agreement\nHigh Accuracy', fontsize=10, color='green', va='bottom')
ax.text(speaker_sd_all.max(), abs_error.min(), 'Low Agreement\nHigh Accuracy\n(Polarizing)', fontsize=10, color='blue', ha='right', va='bottom')
ax.text(speaker_sd_all.min(), abs_error.max(), 'High Agreement\nLow Accuracy\n(Confidently Wrong)', fontsize=10, color='orange', va='top')
ax.text(speaker_sd_all.max(), abs_error.max(), 'Low Agreement\nLow Accuracy\n(Ambiguous)', fontsize=10, color='red', ha='right', va='top')
ax.set_xlabel('SD of Ratings (low = high agreement)')
ax.set_ylabel('Absolute Error')
ax.set_title('Speaker Readability Quadrants')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5A_speaker_readability_quadrants.png'))
plt.close()

# Categorize speakers
log("## 5A. Speaker Readability")
for sp in speaker_codes:
    sd_val, err_val = speaker_sd_all[sp], abs_error[sp]
    if sd_val <= med_sd and err_val <= med_err:
        cat = "Readable (high agree, high acc)"
    elif sd_val <= med_sd and err_val > med_err:
        cat = "Confidently Wrong"
    elif sd_val > med_sd and err_val <= med_err:
        cat = "Polarizing but Correct"
    else:
        cat = "Ambiguous"
log("(See quadrant plot for detailed speaker positions)\n")

# ============================================================
# 5B. Accuracy at Extremes
# ============================================================
print("--- 5B: Accuracy by Actual Score ---")
# Bin actual scores
score_bins = {}
for sp in speaker_codes:
    sc = actual_scores[sp]
    if pd.isna(sc):
        continue
    if sc <= 1.5:
        cat = '1 (Straight)'
    elif sc <= 2.5:
        cat = '2'
    elif sc <= 3.5:
        cat = '3 (Middle)'
    elif sc <= 4.5:
        cat = '4'
    else:
        cat = '5 (Gay)'
    if cat not in score_bins:
        score_bins[cat] = []
    score_bins[cat].append(abs_error[sp])

fig, ax = plt.subplots(figsize=(10, 7))
bin_labels = sorted(score_bins.keys())
data_for_box = [score_bins[bl] for bl in bin_labels]
bp = ax.boxplot(data_for_box, labels=bin_labels, patch_artist=True)
colors_box = sns.color_palette('coolwarm', len(bin_labels))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_xlabel('Actual Score Category')
ax.set_ylabel('Absolute Error')
ax.set_title('Prediction Error by Actual Score Category')
# Kruskal-Wallis if enough groups
if len([d for d in data_for_box if len(d) > 0]) >= 2:
    valid_data = [d for d in data_for_box if len(d) > 0]
    h_stat, h_p = stats.kruskal(*valid_data)
    ax.text(0.95, 0.95, f'Kruskal-Wallis: H={h_stat:.2f}, p={h_p:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    log("## 5B. Accuracy at Extremes")
    log(f"- Kruskal-Wallis: H={h_stat:.2f}, p={h_p:.4f}")
    for bl in bin_labels:
        vals = score_bins[bl]
        log(f"  - {bl}: N={len(vals)}, Mean error={np.mean(vals):.3f}")
    log("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5B_accuracy_by_actual_score.png'))
plt.close()

# ============================================================
# SHARED SETUP FOR ALL 5C SUBSECTIONS
# ============================================================
print("--- 5C: Speaker Clustering (all subsections) ---")

# Use rating vectors (transpose: speakers as rows, listeners as columns)
ratings_T = ratings.T.copy()
ratings_filled = ratings_T.apply(lambda row: row.fillna(row.mean()), axis=1)
ratings_filled = ratings_filled.fillna(ratings_T.stack().mean())

# Standardize: critical so DBSCAN distances aren't dominated by high-variance listeners
scaler = StandardScaler()
ratings_scaled = scaler.fit_transform(ratings_filled.values)

# Full PCA (all components) — used to find 50%/70% thresholds in 5C2
pca_full = PCA(n_components=min(ratings_scaled.shape))
pca_full.fit(ratings_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_50 = int(np.searchsorted(cumvar, 50.0)) + 1   # components needed for >=50%
n_70 = int(np.searchsorted(cumvar, 70.0)) + 1   # components needed for >=70%

# 2D projection (for visualization in all panels)
pca2d = PCA(n_components=2)
pca_coords = pca2d.fit_transform(ratings_scaled)
pc1_var = pca2d.explained_variance_ratio_[0] * 100
pc2_var = pca2d.explained_variance_ratio_[1] * 100

# Higher-dimensional projections for 5C5/5C6
coords_50 = PCA(n_components=n_50).fit_transform(ratings_scaled)
coords_70 = PCA(n_components=n_70).fit_transform(ratings_scaled)

# ---- Shared helper: kneedle eps selection from a coordinate array ----
def kneedle_eps(coords, k=3):
    """Return (eps_elbow, eps_conservative, kth_distances, elbow_idx)."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    kd = np.sort(dists[:, k - 1])[::-1]
    n = len(kd)
    xn = np.arange(n, dtype=float) / (n - 1)
    yn = (kd - kd.min()) / (kd.max() - kd.min())
    perp = np.abs(xn + yn - 1) / np.sqrt(2)
    ei = int(np.argmax(perp))
    return float(kd[ei]), float(np.percentile(kd, 50)), kd, ei

# ---- Shared helper: draw a DBSCAN result onto an existing axes ----
def plot_dbscan(ax, plot_coords, labels, speaker_names, pc1v, pc2v, title_extra=''):
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_no = list(labels).count(-1)
    unique_lbl = sorted(set(labels))
    pal = sns.color_palette('Set1', max(n_cl, 1))
    ci = 0
    cmap = {}
    for lbl in unique_lbl:
        cmap[lbl] = 'lightgray' if lbl == -1 else pal[ci]; ci += (lbl != -1)
    pt_colors = [cmap[lbl] for lbl in labels]
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
               c=pt_colors, s=120, edgecolors='k', alpha=0.9, zorder=3)
    for i, sp in enumerate(speaker_names):
        ax.annotate(sp, (plot_coords[i, 0], plot_coords[i, 1]),
                    fontsize=7.5, xytext=(5, 5), textcoords='offset points')
    legend_els = [Patch(facecolor=cmap[lbl], edgecolor='k',
                        label='Noise (ambiguous)' if lbl == -1 else f'Cluster {lbl}')
                  for lbl in unique_lbl]
    ax.legend(handles=legend_els, loc='best', fontsize=8)
    ax.set_xlabel(f'PC1 ({pc1v:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pc2v:.1f}% var)')
    ax.set_title(f'{n_cl} cluster(s), {n_no}/{len(speaker_names)} noise\n{title_extra}')
    return labels, n_cl, n_no

# ============================================================
# 5C1. Hierarchical Clustering Dendrogram
# ============================================================
print("--- 5C1: Hierarchical Clustering ---")
Z = linkage(ratings_scaled, method='ward')
fig, ax = plt.subplots(figsize=(14, 8))
dendrogram(Z, labels=speaker_codes, leaf_rotation=45, leaf_font_size=9, ax=ax)
ax.set_title('5C1: Hierarchical Clustering of Speakers\n'
             'by Listener Perception Profiles (Ward linkage, standardized)')
ax.set_ylabel('Ward Distance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C1_hierarchical_dendrogram.png'))
plt.close()
log("## 5C1. Hierarchical Clustering")
log("- Ward linkage on standardized 86-dimensional listener rating vectors")
log("- See dendrogram plot for speaker groupings\n")

# ============================================================
# 5C2. PCA Variance Diagnostics — Scree + Cumulative
# ============================================================
print("--- 5C2: PCA Variance Diagnostics ---")
n_show = min(20, len(cumvar))  # show first 20 components
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scree plot
ax = axes[0]
ax.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show] * 100,
       color='steelblue', edgecolor='k', alpha=0.8)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Scree Plot\n(each bar = one PC)')
ax.grid(axis='y', alpha=0.3)

# Cumulative variance
ax = axes[1]
ax.plot(range(1, n_show + 1), cumvar[:n_show], 'b-o', markersize=6)
ax.axhline(50, color='orange', ls='--', lw=1.5, label=f'50% threshold → {n_50} components')
ax.axhline(70, color='red',    ls='--', lw=1.5, label=f'70% threshold → {n_70} components')
ax.axvline(n_50, color='orange', ls=':', lw=1, alpha=0.7)
ax.axvline(n_70, color='red',    ls=':', lw=1, alpha=0.7)
ax.axvline(2,    color='gray',   ls=':', lw=1, alpha=0.7)
ax.text(2.2, 5, f'2D PCA\n({pc1_var+pc2_var:.1f}%)', color='gray', fontsize=8)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('Cumulative Variance Explained\n(how many PCs needed?)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.suptitle('5C2: PCA Variance Diagnostics', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C2_pca_variance_diagnostics.png'), bbox_inches='tight')
plt.close()
log("## 5C2. PCA Variance Diagnostics")
log(f"- PC1 = {pc1_var:.1f}%, PC2 = {pc2_var:.1f}%, 2D total = {pc1_var+pc2_var:.1f}%")
log(f"- Components needed for 50% variance: {n_50} (PC1–PC{n_50}: {cumvar[n_50-1]:.1f}%)")
log(f"- Components needed for 70% variance: {n_70} (PC1–PC{n_70}: {cumvar[n_70-1]:.1f}%)\n")

# ============================================================
# 5C3. K-means Diagnostics (Elbow + Silhouette)
# ============================================================
print("--- 5C3: K-means Diagnostics ---")
k_range = range(2, min(10, len(speaker_codes) - 1))
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls = km.fit_predict(pca_coords)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(pca_coords, lbls) if len(set(lbls)) > 1 else np.nan)

best_k = list(k_range)[np.nanargmax(sil_scores)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, 'bo-', markersize=7)
axes[0].set_xlabel('Number of clusters k')
axes[0].set_ylabel('Inertia (within-cluster sum of squares)')
axes[0].set_title('K-means Elbow Plot\n(look for bend to choose k)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 'ro-', markersize=7)
axes[1].axvline(best_k, color='green', ls='--', lw=1.5, label=f'Best k={best_k}')
axes[1].set_xlabel('Number of clusters k')
axes[1].set_ylabel('Silhouette Score (higher = better)')
axes[1].set_title('K-means Silhouette Scores\n(higher = clusters are more distinct)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('5C3: K-means Diagnostics (run on 2D PCA coords)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C3_kmeans_diagnostics.png'), bbox_inches='tight')
plt.close()
log("## 5C3. K-means Diagnostics (2D PCA)")
log(f"- Best k by silhouette: k={best_k} (score={sil_scores[list(k_range).index(best_k)]:.3f})")
log(f"- k=3 silhouette: {sil_scores[list(k_range).index(3)]:.3f}" if 3 in k_range else "")
log(f"- k=4 silhouette: {sil_scores[list(k_range).index(4)]:.3f}" if 4 in k_range else "")
log("")

# ============================================================
# 5C4. DBSCAN on 2D PCA + K-means k=4 Comparison
# ============================================================
print("--- 5C4: DBSCAN on 2D PCA ---")
eps_elbow_2d, eps_cons_2d, kd_2d, ei_2d = kneedle_eps(pca_coords, k=3)

# k-distance plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(kd_2d) + 1), kd_2d, 'b-o', markersize=5, label='k-distance')
ax.axhline(eps_elbow_2d, color='red',    ls='--', lw=1.8,
           label=f'Kneedle eps = {eps_elbow_2d:.2f} (point {ei_2d+1})')
ax.axhline(eps_cons_2d,  color='orange', ls=':',  lw=1.8,
           label=f'Conservative eps = {eps_cons_2d:.2f} (50th pct)')
ax.axvline(ei_2d + 1, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Speakers ranked: most isolated → most surrounded')
ax.set_ylabel('Distance to 3rd nearest neighbor')
ax.set_title('5C4: DBSCAN Eps Selection — k-Distance Plot (2D PCA space)\n'
             'Steep left = isolated/noise; flat right = cluster members; elbow = eps')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C4_dbscan_kdistance_2d.png'))
plt.close()

# Three-panel: DBSCAN kneedle | DBSCAN conservative | K-means k=4
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

db2d_elbow = DBSCAN(eps=eps_elbow_2d, min_samples=3).fit_predict(pca_coords)
db_labels, n_cl_e, n_no_e = plot_dbscan(
    axes[0], pca_coords, db2d_elbow, speaker_codes, pc1_var, pc2_var,
    title_extra=f'DBSCAN kneedle eps={eps_elbow_2d:.2f}')

db2d_cons = DBSCAN(eps=eps_cons_2d, min_samples=3).fit_predict(pca_coords)
db_labels_cons, n_cl_c, n_no_c = plot_dbscan(
    axes[1], pca_coords, db2d_cons, speaker_codes, pc1_var, pc2_var,
    title_extra=f'DBSCAN conservative eps={eps_cons_2d:.2f}')

kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
km4_labels = kmeans4.fit_predict(pca_coords)
ax = axes[2]
sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                c=km4_labels, cmap='Set1', s=120, edgecolors='k', alpha=0.8)
for i, sp in enumerate(speaker_codes):
    ax.annotate(sp, (pca_coords[i, 0], pca_coords[i, 1]),
                fontsize=7.5, xytext=(5, 5), textcoords='offset points')
ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8)
ax.set_xlabel(f'PC1 ({pc1_var:.1f}% var)')
ax.set_ylabel(f'PC2 ({pc2_var:.1f}% var)')
ax.set_title(f'K-means k=4 (best silhouette)\nAll speakers forced into a cluster — no noise')

plt.suptitle(f'5C4: Speaker Clustering on 2D PCA ({pc1_var+pc2_var:.1f}% variance)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C4_clustering_2d_comparison.png'), bbox_inches='tight')
plt.close()

log("## 5C4. DBSCAN on 2D PCA")
log(f"- Kneedle eps={eps_elbow_2d:.3f}: {n_cl_e} cluster(s), {n_no_e} noise")
for lbl in sorted(set(db2d_elbow)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db2d_elbow[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log(f"- Conservative eps={eps_cons_2d:.3f}: {n_cl_c} cluster(s), {n_no_c} noise")
for lbl in sorted(set(db2d_cons)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db2d_cons[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log(f"- K-means k=4 cluster members:")
for c in range(4):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if km4_labels[i] == c]
    log(f"  - Cluster {c}: {', '.join(members)}")
log("")

# ============================================================
# 5C5. DBSCAN on First N Components → 50% Variance
# ============================================================
print(f"--- 5C5: DBSCAN on {n_50} components (>= 50% variance) ---")
eps_elbow_50, eps_cons_50, kd_50, ei_50 = kneedle_eps(coords_50, k=3)

# k-distance plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(kd_50) + 1), kd_50, 'b-o', markersize=5, label='k-distance')
ax.axhline(eps_elbow_50, color='red',    ls='--', lw=1.8,
           label=f'Kneedle eps = {eps_elbow_50:.2f}')
ax.axhline(eps_cons_50,  color='orange', ls=':',  lw=1.8,
           label=f'Conservative eps = {eps_cons_50:.2f}')
ax.axvline(ei_50 + 1, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Speakers ranked: most isolated → most surrounded')
ax.set_ylabel(f'Distance to 3rd nearest neighbor ({n_50}D space)')
ax.set_title(f'5C5: DBSCAN Eps Selection — k-Distance Plot ({n_50}-component PCA space)\n'
             f'Clustering on {n_50} PCs = {cumvar[n_50-1]:.1f}% variance retained')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C5_dbscan_kdistance_50pct.png'))
plt.close()

# Clustering panels (plot projected back to 2D for visibility)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

db50_elbow = DBSCAN(eps=eps_elbow_50, min_samples=3).fit_predict(coords_50)
db50_cons  = DBSCAN(eps=eps_cons_50,  min_samples=3).fit_predict(coords_50)

_, n_cl_50e, n_no_50e = plot_dbscan(
    axes[0], pca_coords, db50_elbow, speaker_codes, pc1_var, pc2_var,
    title_extra=f'kneedle eps={eps_elbow_50:.2f} | clustered in {n_50}D')
_, n_cl_50c, n_no_50c = plot_dbscan(
    axes[1], pca_coords, db50_cons, speaker_codes, pc1_var, pc2_var,
    title_extra=f'conservative eps={eps_cons_50:.2f} | clustered in {n_50}D')

plt.suptitle(f'5C5: DBSCAN on {n_50}-Component PCA ({cumvar[n_50-1]:.1f}% variance)\n'
             f'Points plotted in 2D for readability — clusters assigned in {n_50}D',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C5_clustering_50pct_variance.png'), bbox_inches='tight')
plt.close()

log(f"## 5C5. DBSCAN on {n_50}-Component PCA ({cumvar[n_50-1]:.1f}% variance)")
log(f"- Kneedle eps={eps_elbow_50:.3f}: {n_cl_50e} cluster(s), {n_no_50e} noise")
for lbl in sorted(set(db50_elbow)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db50_elbow[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log(f"- Conservative eps={eps_cons_50:.3f}: {n_cl_50c} cluster(s), {n_no_50c} noise")
for lbl in sorted(set(db50_cons)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db50_cons[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log("")

# ============================================================
# 5C6. DBSCAN on First N Components → 70% Variance
# ============================================================
print(f"--- 5C6: DBSCAN on {n_70} components (>= 70% variance) ---")
eps_elbow_70, eps_cons_70, kd_70, ei_70 = kneedle_eps(coords_70, k=3)

# k-distance plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(kd_70) + 1), kd_70, 'b-o', markersize=5, label='k-distance')
ax.axhline(eps_elbow_70, color='red',    ls='--', lw=1.8,
           label=f'Kneedle eps = {eps_elbow_70:.2f}')
ax.axhline(eps_cons_70,  color='orange', ls=':',  lw=1.8,
           label=f'Conservative eps = {eps_cons_70:.2f}')
ax.axvline(ei_70 + 1, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Speakers ranked: most isolated → most surrounded')
ax.set_ylabel(f'Distance to 3rd nearest neighbor ({n_70}D space)')
ax.set_title(f'5C6: DBSCAN Eps Selection — k-Distance Plot ({n_70}-component PCA space)\n'
             f'Clustering on {n_70} PCs = {cumvar[n_70-1]:.1f}% variance retained')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C6_dbscan_kdistance_70pct.png'))
plt.close()

# Clustering panels
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

db70_elbow = DBSCAN(eps=eps_elbow_70, min_samples=3).fit_predict(coords_70)
db70_cons  = DBSCAN(eps=eps_cons_70,  min_samples=3).fit_predict(coords_70)

_, n_cl_70e, n_no_70e = plot_dbscan(
    axes[0], pca_coords, db70_elbow, speaker_codes, pc1_var, pc2_var,
    title_extra=f'kneedle eps={eps_elbow_70:.2f} | clustered in {n_70}D')
_, n_cl_70c, n_no_70c = plot_dbscan(
    axes[1], pca_coords, db70_cons, speaker_codes, pc1_var, pc2_var,
    title_extra=f'conservative eps={eps_cons_70:.2f} | clustered in {n_70}D')

plt.suptitle(f'5C6: DBSCAN on {n_70}-Component PCA ({cumvar[n_70-1]:.1f}% variance)\n'
             f'Points plotted in 2D for readability — clusters assigned in {n_70}D',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5C6_clustering_70pct_variance.png'), bbox_inches='tight')
plt.close()

log(f"## 5C6. DBSCAN on {n_70}-Component PCA ({cumvar[n_70-1]:.1f}% variance)")
log(f"- Kneedle eps={eps_elbow_70:.3f}: {n_cl_70e} cluster(s), {n_no_70e} noise")
for lbl in sorted(set(db70_elbow)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db70_elbow[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log(f"- Conservative eps={eps_cons_70:.3f}: {n_cl_70c} cluster(s), {n_no_70c} noise")
for lbl in sorted(set(db70_cons)):
    members = [speaker_codes[i] for i in range(len(speaker_codes)) if db70_cons[i] == lbl]
    log(f"  - {'Noise' if lbl==-1 else f'Cluster {lbl}'}: {', '.join(members)}")
log("")

# ============================================================
# 5D. PC1 as Perceived Gayness Axis
# ============================================================
print("--- 5D: PC1 Analyses ---")

# PC1 score for each speaker (signed: negative = perceived straight, positive = perceived gay)
pc1_scores = pd.Series(pca_coords[:, 0], index=speaker_codes)

# Align with actual/predicted where available
df_5d = pd.DataFrame({
    'PC1': pc1_scores,
    'Actual': actual_scores,
    'MeanRating': avg_predicted,
    'AbsError': abs_error,
}).dropna()

# ---- 5D1. PC1 vs Actual Orientation ----
fig, ax = plt.subplots(figsize=(10, 8))
r_5d1, p_5d1 = stats.pearsonr(df_5d['Actual'], df_5d['PC1'])
xs = np.linspace(df_5d['Actual'].min(), df_5d['Actual'].max(), 100)
sl_5d1, ic_5d1 = np.polyfit(df_5d['Actual'], df_5d['PC1'], 1)
ax.scatter(df_5d['Actual'], df_5d['PC1'], s=80, alpha=0.7, edgecolors='k', color='steelblue')
ax.plot(xs, sl_5d1 * xs + ic_5d1, 'r-', lw=2)
for sp in df_5d.index:
    ax.annotate(sp, (df_5d.loc[sp, 'Actual'], df_5d.loc[sp, 'PC1']),
                fontsize=7.5, xytext=(5, 3), textcoords='offset points')
ax.set_xlabel('Actual Sexuality (1=Straight, 5=Gay)')
ax.set_ylabel(f'PC1 Score ({pc1_var:.1f}% variance)')
ax.set_title(f'5D1: PC1 (Perceived Gayness Axis) vs Actual Orientation\n'
             f'r = {r_5d1:.3f}, p = {p_5d1:.4f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5D1_pc1_vs_actual.png'))
plt.close()
log("## 5D1. PC1 vs Actual Orientation")
log(f"- Pearson r = {r_5d1:.4f}, p = {p_5d1:.4f}")
log(f"- R² = {r_5d1**2:.4f}")
log(f"- PC1 range: [{df_5d['PC1'].min():.3f}, {df_5d['PC1'].max():.3f}]\n")

# ---- 5D2. PC1 vs Mean Rating ----
fig, ax = plt.subplots(figsize=(10, 8))
r_5d2, p_5d2 = stats.pearsonr(df_5d['MeanRating'], df_5d['PC1'])
xs2 = np.linspace(df_5d['MeanRating'].min(), df_5d['MeanRating'].max(), 100)
sl_5d2, ic_5d2 = np.polyfit(df_5d['MeanRating'], df_5d['PC1'], 1)
ax.scatter(df_5d['MeanRating'], df_5d['PC1'], s=80, alpha=0.7, edgecolors='k', color='mediumseagreen')
ax.plot(xs2, sl_5d2 * xs2 + ic_5d2, 'r-', lw=2)
for sp in df_5d.index:
    ax.annotate(sp, (df_5d.loc[sp, 'MeanRating'], df_5d.loc[sp, 'PC1']),
                fontsize=7.5, xytext=(5, 3), textcoords='offset points')
ax.set_xlabel('Mean Listener Rating (1=Straight, 5=Gay)')
ax.set_ylabel(f'PC1 Score ({pc1_var:.1f}% variance)')
ax.set_title(f'5D2: PC1 vs Mean Listener Rating\n'
             f'r = {r_5d2:.3f}, p = {p_5d2:.4f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5D2_pc1_vs_mean_rating.png'))
plt.close()
log("## 5D2. PC1 vs Mean Rating")
log(f"- Pearson r = {r_5d2:.4f}, p = {p_5d2:.4f}")
log(f"- R² = {r_5d2**2:.4f}")
log("- High r² validates PC1 as the consensus perceived-gayness dimension\n")

# ---- 5D3. |PC1| vs Absolute Error ----
# Hypothesis: speakers at PC1 extremes (clearly perceived as gay or straight)
# are rated more accurately (lower abs error)
df_5d['AbsPC1'] = df_5d['PC1'].abs()
r_5d3, p_5d3 = stats.pearsonr(df_5d['AbsPC1'], df_5d['AbsError'])
fig, ax = plt.subplots(figsize=(10, 8))
xs3 = np.linspace(df_5d['AbsPC1'].min(), df_5d['AbsPC1'].max(), 100)
sl_5d3, ic_5d3 = np.polyfit(df_5d['AbsPC1'], df_5d['AbsError'], 1)
sc = ax.scatter(df_5d['AbsPC1'], df_5d['AbsError'], s=80, alpha=0.7, edgecolors='k',
                c=df_5d['Actual'], cmap='RdYlBu_r', vmin=1, vmax=5)
ax.plot(xs3, sl_5d3 * xs3 + ic_5d3, 'r-', lw=2)
for sp in df_5d.index:
    ax.annotate(sp, (df_5d.loc[sp, 'AbsPC1'], df_5d.loc[sp, 'AbsError']),
                fontsize=7.5, xytext=(5, 3), textcoords='offset points')
plt.colorbar(sc, ax=ax, label='Actual Sexuality (1=Straight, 5=Gay)')
ax.set_xlabel(f'|PC1| — Distance from Perceptual Center ({pc1_var:.1f}% var)')
ax.set_ylabel('Absolute Error (|mean rating − actual|)')
ax.set_title(f'5D3: Perceptual Extremity (|PC1|) vs Rating Accuracy\n'
             f'r = {r_5d3:.3f}, p = {p_5d3:.4f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5D3_pc1_extremity_vs_error.png'))
plt.close()
log("## 5D3. |PC1| vs Absolute Error")
log(f"- Pearson r = {r_5d3:.4f}, p = {p_5d3:.4f}")
log(f"- Negative r = speakers at PC1 extremes are rated MORE accurately")
log(f"- Positive r = speakers at PC1 extremes are rated LESS accurately\n")

# ============================================================
# 6A. Response Bias - Mean Rating per Listener
# ============================================================
print("--- 6A: Listener Response Bias ---")
listener_means = ratings.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(listener_means.dropna(), bins=20, edgecolor='k', alpha=0.7, color='coral')
ax.axvline(listener_means.mean(), color='red', lw=2, ls='--', label=f'Mean = {listener_means.mean():.2f}')
ax.set_xlabel('Mean Rating Across All Speakers')
ax.set_ylabel('Count')
ax.set_title('Listener Response Bias Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6A_listener_response_bias.png'))
plt.close()
log("## 6A. Listener Response Bias")
log(f"- Grand mean rating: {listener_means.mean():.3f}")
log(f"- SD of listener means: {listener_means.std():.3f}")
log(f"- Range: [{listener_means.min():.2f}, {listener_means.max():.2f}]\n")

# ============================================================
# 6B. Religiosity vs Rating Bias
# ============================================================
print("--- 6B: Religiosity vs Bias ---")
fig, ax = plt.subplots(figsize=(10, 7))
df_rb = pd.DataFrame({'MeanRating': listener_means, 'Religiosity': demographics['Religiosity']}).dropna()
ax.scatter(df_rb['Religiosity'], df_rb['MeanRating'], alpha=0.6, edgecolors='k', s=60)
if len(df_rb) > 3:
    sl, ic = np.polyfit(df_rb['Religiosity'], df_rb['MeanRating'], 1)
    ax.plot(np.linspace(0, 10, 50), sl*np.linspace(0, 10, 50)+ic, 'r-', lw=2)
    r_rb, p_rb = stats.pearsonr(df_rb['Religiosity'], df_rb['MeanRating'])
    ax.set_title(f'Religiosity vs Rating Bias\nr = {r_rb:.3f}, p = {p_rb:.4f}')
    log("## 6B. Religiosity vs Rating Bias")
    log(f"- r = {r_rb:.4f}, p = {p_rb:.4f}")
    log(f"- {'More religious listeners rate speakers as straighter' if sl < 0 else 'More religious listeners rate speakers as gayer'}\n")
ax.set_xlabel('Religiosity (0-10)')
ax.set_ylabel('Mean Rating (lower = straighter bias)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6B_religiosity_vs_bias.png'))
plt.close()

# ============================================================
# 6C. Response Range per Listener
# ============================================================
print("--- 6C: Listener Response Range ---")
listener_sds = ratings.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(listener_sds.dropna(), bins=20, edgecolor='k', alpha=0.7, color='mediumpurple')
ax.axvline(listener_sds.mean(), color='red', lw=2, ls='--', label=f'Mean SD = {listener_sds.mean():.2f}')
ax.set_xlabel('SD of Ratings per Listener')
ax.set_ylabel('Count')
ax.set_title('Listener Response Range Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6C_listener_response_range.png'))
plt.close()
log("## 6C. Listener Response Range")
log(f"- Mean listener SD: {listener_sds.mean():.3f}")
log(f"- Range of SDs: [{listener_sds.min():.2f}, {listener_sds.max():.2f}]\n")

# ============================================================
# 7A. Gender x Orientation Interaction on Accuracy
# ============================================================
print("--- 7A: Gender x Orientation Interaction ---")
df_int = pd.DataFrame({
    'Accuracy': indiv_r,
    'Gender': demographics['Gender'].map({1: 'Male', 2: 'Female'}),
    'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'})
}).dropna()

fig, ax = plt.subplots(figsize=(10, 7))
# Interaction plot
for orient in ['Gay', 'Straight', 'Bi/Pan']:
    sub = df_int[df_int['Orientation'] == orient]
    means = sub.groupby('Gender')['Accuracy'].mean()
    if len(means) >= 2:
        ax.plot(means.index, means.values, 'o-', label=orient, markersize=8, lw=2)
    elif len(means) == 1:
        ax.plot(means.index, means.values, 'o', label=orient, markersize=8)

ax.set_xlabel('Gender')
ax.set_ylabel('Mean Accuracy (Pearson r)')
ax.set_title('Gender x Orientation Interaction on Accuracy')
ax.legend(title='Orientation')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '7A_gender_orientation_interaction.png'))
plt.close()

# Two-way ANOVA using OLS
log("## 7A. Gender x Orientation Interaction")
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    model = ols('Accuracy ~ C(Gender) * C(Orientation)', data=df_int).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    log(f"- Two-way ANOVA results:")
    for idx_row in anova_table.index:
        row = anova_table.loc[idx_row]
        if 'PR(>F)' in row.index:
            log(f"  - {idx_row}: F={row['F']:.3f}, p={row['PR(>F)']:.4f}" if not pd.isna(row['F']) else f"  - {idx_row}: (residual)")
except ImportError:
    log("- statsmodels not available; skipping formal ANOVA")
log("")

# ============================================================
# 7B. Listener-Speaker Match Effects
# ============================================================
print("--- 7B: Match Effects ---")
# Gay speakers (actual >= 4), Straight speakers (actual <= 2), Middle (2 < actual < 4)
gay_speakers = [sp for sp in speaker_codes if actual_scores[sp] >= 4]
straight_speakers = [sp for sp in speaker_codes if actual_scores[sp] <= 2]
middle_speakers = [sp for sp in speaker_codes if 2 < actual_scores[sp] < 4]

# For each listener, compute accuracy (r) for gay vs straight speakers
listener_gay_acc = []
listener_str_acc = []
for i in range(86):
    lr = ratings.iloc[i]
    # Gay speakers accuracy
    m_g = lr[gay_speakers].notna()
    if m_g.sum() >= 3:
        r_g, _ = stats.pearsonr(lr[gay_speakers][m_g], actual_scores[gay_speakers][m_g])
        listener_gay_acc.append(r_g)
    else:
        listener_gay_acc.append(np.nan)
    # Straight speakers accuracy
    m_s = lr[straight_speakers].notna()
    if m_s.sum() >= 3:
        r_s, _ = stats.pearsonr(lr[straight_speakers][m_s], actual_scores[straight_speakers][m_s])
        listener_str_acc.append(r_s)
    else:
        listener_str_acc.append(np.nan)

# Mean absolute error instead (more intuitive for subgroups)
listener_gay_mae = []
listener_str_mae = []
for i in range(86):
    lr = ratings.iloc[i]
    m_g = lr[gay_speakers].notna()
    if m_g.sum() > 0:
        listener_gay_mae.append((lr[gay_speakers][m_g] - actual_scores[gay_speakers][m_g]).abs().mean())
    else:
        listener_gay_mae.append(np.nan)
    m_s = lr[straight_speakers].notna()
    if m_s.sum() > 0:
        listener_str_mae.append((lr[straight_speakers][m_s] - actual_scores[straight_speakers][m_s]).abs().mean())
    else:
        listener_str_mae.append(np.nan)

df_match = pd.DataFrame({
    'MAE_Gay_Speakers': listener_gay_mae,
    'MAE_Straight_Speakers': listener_str_mae,
    'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'})
}).dropna()

fig, ax = plt.subplots(figsize=(10, 7))
df_melt = df_match.melt(id_vars='Orientation', value_vars=['MAE_Gay_Speakers', 'MAE_Straight_Speakers'],
                        var_name='Speaker Type', value_name='MAE')
df_melt['Speaker Type'] = df_melt['Speaker Type'].map({'MAE_Gay_Speakers': 'Gay Speakers', 'MAE_Straight_Speakers': 'Straight Speakers'})
sns.barplot(x='Orientation', y='MAE', hue='Speaker Type', data=df_melt, ax=ax, palette='Set2', errorbar=('ci', 95))
ax.set_ylabel('Mean Absolute Error (lower = more accurate)')
ax.set_title('Accuracy for Gay vs Straight Speakers\nby Listener Orientation')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '7B_match_effects.png'))
plt.close()
log("## 7B. Match Effects")
log(f"- Gay speakers (actual >= 4): {len(gay_speakers)} speakers")
log(f"- Straight speakers (actual <= 2): {len(straight_speakers)} speakers")
for orient in ['Gay', 'Straight', 'Bi/Pan']:
    sub = df_match[df_match['Orientation'] == orient]
    if len(sub) > 0:
        log(f"- {orient} listeners: MAE for gay speakers = {sub['MAE_Gay_Speakers'].mean():.3f}, for straight speakers = {sub['MAE_Straight_Speakers'].mean():.3f}")
log("")

# ============================================================
# 8A. Signal Detection Theory
# ============================================================
print("--- 8A: Signal Detection Theory ---")
threshold = 3.0  # Use 3.0 as midpoint for dichotomization
actual_binary = (actual_scores >= threshold).astype(int)  # 1 = gay

listener_dprime = []
listener_criterion = []
for i in range(86):
    lr = ratings.iloc[i]
    hits = 0; misses = 0; fas = 0; crs = 0
    for sp in speaker_codes:
        if pd.isna(lr[sp]) or pd.isna(actual_binary[sp]):
            continue
        pred = 1 if lr[sp] >= threshold else 0
        actual = actual_binary[sp]
        if actual == 1 and pred == 1: hits += 1
        elif actual == 1 and pred == 0: misses += 1
        elif actual == 0 and pred == 1: fas += 1
        else: crs += 1

    n_signal = hits + misses
    n_noise = fas + crs
    if n_signal > 0 and n_noise > 0:
        hr = hits / n_signal
        far = fas / n_noise
        # Apply correction for extreme values
        hr = np.clip(hr, 0.5/n_signal, 1 - 0.5/n_signal)
        far = np.clip(far, 0.5/n_noise, 1 - 0.5/n_noise)
        d = stats.norm.ppf(hr) - stats.norm.ppf(far)
        c = -0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(far))
        listener_dprime.append(d)
        listener_criterion.append(c)
    else:
        listener_dprime.append(np.nan)
        listener_criterion.append(np.nan)

dprime_series = pd.Series(listener_dprime)
criterion_series = pd.Series(listener_criterion)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
valid_dp = dprime_series.dropna()
ax.hist(valid_dp, bins=15, edgecolor='k', alpha=0.7, color='teal')
ax.axvline(valid_dp.mean(), color='red', lw=2, ls='--', label=f"Mean d' = {valid_dp.mean():.2f}")
ax.set_xlabel("d' (sensitivity)")
ax.set_ylabel('Count')
ax.set_title("Distribution of d' Across Listeners")
ax.legend()

ax = axes[1]
valid_c = criterion_series.dropna()
ax.hist(valid_c, bins=15, edgecolor='k', alpha=0.7, color='salmon')
ax.axvline(valid_c.mean(), color='red', lw=2, ls='--', label=f"Mean c = {valid_c.mean():.2f}")
ax.set_xlabel("c (criterion)")
ax.set_ylabel('Count')
ax.set_title("Distribution of Criterion (c) Across Listeners")
ax.legend()

plt.suptitle('Signal Detection Theory Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '8A_signal_detection.png'), bbox_inches='tight')
plt.close()

log("## 8A. Signal Detection Theory")
log(f"- Threshold for dichotomization: {threshold}")
log(f"- Mean d' = {valid_dp.mean():.4f} (SD = {valid_dp.std():.4f})")
log(f"- Mean criterion c = {valid_c.mean():.4f} (SD = {valid_c.std():.4f})")
log(f"- d' > 0 indicates above-chance sensitivity\n")

# ============================================================
# 8B. d' by Listener Group
# ============================================================
print("--- 8B: d' by Group ---")
df_sdt = pd.DataFrame({
    'dprime': dprime_series,
    'Gender': demographics['Gender'].map({1: 'Male', 2: 'Female'}),
    'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'}),
    'Familiarity': fam
}).dropna(subset=['dprime'])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
sns.boxplot(x='Gender', y='dprime', data=df_sdt.dropna(subset=['Gender']), ax=ax, palette='Set2')
ax.set_title("d' by Gender")
ax.set_ylabel("d'")

ax = axes[1]
sns.boxplot(x='Orientation', y='dprime', data=df_sdt.dropna(subset=['Orientation']), ax=ax, palette='Set3')
ax.set_title("d' by Orientation")
ax.set_ylabel("d'")

ax = axes[2]
sns.boxplot(x='Familiarity', y='dprime', data=df_sdt.dropna(subset=['Familiarity']),
            order=['Low', 'Medium', 'High'], ax=ax, palette='Set1')
ax.set_title("d' by Familiarity")
ax.set_ylabel("d'")

plt.suptitle("Signal Detection Sensitivity by Listener Group", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '8B_dprime_by_group.png'), bbox_inches='tight')
plt.close()

log("## 8B. d' by Listener Group")
for grp_col in ['Gender', 'Orientation', 'Familiarity']:
    sub = df_sdt.dropna(subset=[grp_col])
    for gval in sub[grp_col].unique():
        vals = sub[sub[grp_col]==gval]['dprime']
        log(f"- {grp_col}={gval}: Mean d'={vals.mean():.3f}, N={len(vals)}")
log("")

# ============================================================
# Summary: Top/Bottom speakers
# ============================================================
log("## Key Findings - Speaker Rankings")
log("\n### Top 5 Most Accurately Perceived Speakers (lowest absolute error)")
sorted_err = abs_error.sort_values()
for i in range(min(5, len(sorted_err))):
    sp = sorted_err.index[i]
    log(f"- {sp}: error = {sorted_err.iloc[i]:.3f}, actual = {actual_scores[sp]:.1f}, predicted = {avg_predicted[sp]:.2f}")

log("\n### Top 5 Least Accurately Perceived Speakers (highest absolute error)")
sorted_err_desc = abs_error.sort_values(ascending=False)
for i in range(min(5, len(sorted_err_desc))):
    sp = sorted_err_desc.index[i]
    log(f"- {sp}: error = {sorted_err_desc.iloc[i]:.3f}, actual = {actual_scores[sp]:.1f}, predicted = {avg_predicted[sp]:.2f}")

# MAE and RMSE
mae = abs_error.mean()
rmse = np.sqrt(((avg_predicted - actual_scores)**2).mean())
log(f"\n### Overall Metrics")
log(f"- Mean Absolute Error (MAE): {mae:.4f}")
log(f"- RMSE: {rmse:.4f}")

# ============================================================
# Write summary markdown
# ============================================================
summary_path = os.path.join(os.path.dirname(INPUT_CSV), 'analysis_summary.md')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f"\nSummary saved to {summary_path}")
print(f"All plots saved to {OUTPUT_DIR}")
print("Analysis complete!")
