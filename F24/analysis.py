"""
Gay Voice Perception Research - Comprehensive Statistical Analysis
Analyzes relationship between perceived and actual sexual orientation from voice.
"""

import os
import warnings
from pathlib import Path
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

script_dir = Path(__file__).resolve().parent
INPUT_CSV  = script_dir / "Results.csv"
OUTPUT_DIR = script_dir / "Dataplots"


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = pd.read_csv(INPUT_CSV, header=None)

    speaker_codes = [str(raw.iloc[0, c]).strip().replace('\xa0', '') for c in range(1, 37)]

    ratings = raw.iloc[2:87, 1:37].apply(pd.to_numeric, errors='coerce')
    ratings.index = range(85)
    ratings.columns = speaker_codes

    demo_cols = ['YOB', 'Gender', 'Orientation', 'English', 'GayFriends', 'GayAcquaint', 'Religiosity', 'Score']
    demographics = raw.iloc[2:87, 37:45].apply(pd.to_numeric, errors='coerce')
    demographics.index = range(85)
    demographics.columns = demo_cols

    actual_scores = raw.iloc[89, 1:37].apply(pd.to_numeric, errors='coerce')
    actual_scores.index = speaker_codes

    avg_predicted = raw.iloc[90, 1:37].apply(pd.to_numeric, errors='coerce')
    avg_predicted.index = speaker_codes

    abs_error     = (avg_predicted - actual_scores).abs()
    speaker_sd_all = ratings.std(axis=0)
    listener_means = ratings.mean(axis=1)
    listener_sds   = ratings.std(axis=1)

    # Familiarity composite
    friends  = demographics['GayFriends']
    acquaint = demographics['GayAcquaint']
    fam = pd.Series(index=range(85), dtype=str)
    for i in range(85):
        f, a = friends.iloc[i], acquaint.iloc[i]
        if pd.isna(f) or pd.isna(a):
            fam.iloc[i] = np.nan
        elif f == 1 and a == 1:
            fam.iloc[i] = 'Low'
        elif f == 3 or a == 3:
            fam.iloc[i] = 'High'
        else:
            fam.iloc[i] = 'Medium'

    # Per-listener Pearson r with actual scores
    indiv_r = []
    for i in range(85):
        lr = ratings.iloc[i]
        m = lr.notna() & actual_scores.notna()
        if m.sum() >= 5:
            r_i, _ = stats.spearmanr(lr[m], actual_scores[m])
            indiv_r.append(r_i)
        else:
            indiv_r.append(np.nan)
    indiv_r = pd.Series(indiv_r)

    # Overall r (used as fallback when 1A is skipped but 1E is run)
    mask = actual_scores.notna() & avg_predicted.notna()
    r_val_overall, _ = stats.spearmanr(actual_scores[mask], avg_predicted[mask])

    print(f"Loaded {ratings.shape[0]} listeners x {ratings.shape[1]} speakers")
    print(f"Missing ratings: {ratings.isna().sum().sum()} / {ratings.size}")

    return {
        'INPUT_CSV': INPUT_CSV, 'OUTPUT_DIR': OUTPUT_DIR,
        'speaker_codes': speaker_codes,
        'ratings': ratings, 'demographics': demographics,
        'actual_scores': actual_scores, 'avg_predicted': avg_predicted,
        'abs_error': abs_error,
        'speaker_sd_all': speaker_sd_all,
        'listener_means': listener_means,
        'listener_sds': listener_sds,
        'fam': fam,
        'indiv_r': indiv_r,
        'r_val_overall': r_val_overall,
        'xs': np.linspace(0.5, 5.5, 100),
    }


# ============================================================
# SHARED HELPERS
# ============================================================

def group_accuracy(ratings, actual_scores, listener_mask):
    """Avg predicted per speaker for a listener subset; returns r, n, group_avg."""
    sub = ratings.loc[listener_mask]
    n = sub.shape[0]
    if n < 3:
        return None, n, None
    group_avg = sub.mean(axis=0)
    m = actual_scores.notna() & group_avg.notna()
    if m.sum() < 3:
        return None, n, None
    r, p = stats.spearmanr(actual_scores[m], group_avg[m])
    return r, n, group_avg


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
# CLUSTERING SETUP (shared by 5C* and 5D)
# ============================================================

def setup_clustering(data):
    ratings        = data['ratings']
    speaker_codes  = data['speaker_codes']
    actual_scores  = data['actual_scores']
    avg_predicted  = data['avg_predicted']
    abs_error      = data['abs_error']

    ratings_T      = ratings.T.copy()
    ratings_filled = ratings_T.apply(lambda row: row.fillna(row.mean()), axis=1)
    ratings_filled = ratings_filled.fillna(ratings_T.stack().mean())

    scaler         = StandardScaler()
    ratings_scaled = scaler.fit_transform(ratings_filled.values)

    pca_full = PCA(n_components=min(ratings_scaled.shape))
    pca_full.fit(ratings_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_50   = int(np.searchsorted(cumvar, 50.0)) + 1
    n_70   = int(np.searchsorted(cumvar, 70.0)) + 1

    pca2d      = PCA(n_components=2)
    pca_coords = pca2d.fit_transform(ratings_scaled)
    pc1_var    = pca2d.explained_variance_ratio_[0] * 100
    pc2_var    = pca2d.explained_variance_ratio_[1] * 100

    coords_50 = PCA(n_components=n_50).fit_transform(ratings_scaled)
    coords_70 = PCA(n_components=n_70).fit_transform(ratings_scaled)

    # K-means diagnostics (needed for 5C3 plot and 5C4 best_k)
    k_range   = range(2, min(10, len(speaker_codes) - 1))
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(pca_coords)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(pca_coords, lbls) if len(set(lbls)) > 1 else np.nan)
    best_k = list(k_range)[np.nanargmax(sil_scores)]

    # 5D data
    pc1_scores = pd.Series(pca_coords[:, 0], index=speaker_codes)
    df_5d = pd.DataFrame({
        'PC1':        pc1_scores,
        'Actual':     actual_scores,
        'MeanRating': avg_predicted,
        'AbsError':   abs_error,
    }).dropna()

    return {
        'ratings_scaled': ratings_scaled,
        'pca_coords':     pca_coords,
        'pc1_var': pc1_var, 'pc2_var': pc2_var,
        'pca_full': pca_full, 'cumvar': cumvar,
        'n_50': n_50, 'n_70': n_70,
        'coords_50': coords_50, 'coords_70': coords_70,
        'k_range': k_range, 'inertias': inertias,
        'sil_scores': sil_scores, 'best_k': best_k,
        'df_5d': df_5d,
        'speaker_codes': speaker_codes,
    }


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analysis_1a(data, log):
    print("\n--- 1A: Overall Accuracy Scatter ---")
    actual_scores = data['actual_scores']
    avg_predicted = data['avg_predicted']
    xs            = data['xs']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    mask = actual_scores.notna() & avg_predicted.notna()
    x_all, y_all = actual_scores[mask].values, avg_predicted[mask].values
    r_val, p_val = stats.spearmanr(x_all, y_all)
    slope, intercept = np.polyfit(x_all, y_all, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_all, y_all, s=80, alpha=0.7, edgecolors='k', zorder=3)
    ax.plot([1, 5], [1, 5], 'k--', alpha=0.4, label='Perfect accuracy')
    ax.plot(xs, slope * xs + intercept, 'r-', linewidth=2, label='Regression')
    ax.set_xlabel('Actual Converted Score (1-5)')
    ax.set_ylabel('Average Predicted Score (1-5)')
    ax.set_title(f'Overall Perception Accuracy\nρ = {r_val:.3f}, p = {p_val:.4f}')
    ax.legend()
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1A_overall_accuracy_scatter.png'))
    plt.close()

    log(f"## 1A. Overall Accuracy")
    log(f"- Spearman ρ = {r_val:.4f}, p = {p_val:.6f}")
    log(f"- Regression: predicted = {slope:.3f} * actual + {intercept:.3f}\n")
    return r_val


def analysis_1b(data, log):
    print("--- 1B: Accuracy by Gender ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    demographics  = data['demographics']
    xs            = data['xs']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    gender_labels  = {1: 'Male', 2: 'Female'}
    gender_results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (gval, glabel) in enumerate(gender_labels.items()):
        mask_g = demographics['Gender'] == gval
        r_g, n_g, gavg = group_accuracy(ratings, actual_scores, mask_g)
        gender_results[glabel] = (r_g, n_g)
        ax = axes[idx]
        if gavg is not None:
            m = actual_scores.notna() & gavg.notna()
            ax.scatter(actual_scores[m], gavg[m], s=60, alpha=0.7, edgecolors='k')
            sl, ic = np.polyfit(actual_scores[m].values, gavg[m].values, 1)
            ax.plot(xs, sl*xs+ic, 'r-', lw=2)
            ax.plot([1,5],[1,5],'k--',alpha=0.3)
        ax.set_title(f'{glabel} Listeners (N={n_g})\nρ={r_g:.3f}' if r_g is not None else f'{glabel} (N={n_g})')
        ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
        ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
    plt.suptitle('Perception Accuracy by Listener Gender', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1B_accuracy_by_gender.png'), bbox_inches='tight')
    plt.close()

    log("## 1B. Accuracy by Gender")
    for g, (r, n) in gender_results.items():
        log(f"- {g}: ρ = {r:.4f}, N = {n}" if r is not None else f"- {g}: insufficient data (N={n})")
    log("")
    return gender_results


def analysis_1c(data, log):
    print("--- 1C: Accuracy by Orientation ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    demographics  = data['demographics']
    xs            = data['xs']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    orient_labels  = {1: 'Gay/Lesbian', 2: 'Straight', 3: 'Bi/Pan'}
    orient_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (oval, olabel) in enumerate(orient_labels.items()):
        mask_o = demographics['Orientation'] == oval
        r_o, n_o, oavg = group_accuracy(ratings, actual_scores, mask_o)
        orient_results[olabel] = (r_o, n_o)
        ax = axes[idx]
        if oavg is not None and r_o is not None:
            m = actual_scores.notna() & oavg.notna()
            ax.scatter(actual_scores[m], oavg[m], s=60, alpha=0.7, edgecolors='k')
            sl, ic = np.polyfit(actual_scores[m].values, oavg[m].values, 1)
            ax.plot(xs, sl*xs+ic, 'r-', lw=2)
            ax.plot([1,5],[1,5],'k--',alpha=0.3)
        ax.set_title(f'{olabel} (N={n_o})\nρ={r_o:.3f}' if r_o is not None else f'{olabel} (N={n_o})')
        ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
        ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
    plt.suptitle('Perception Accuracy by Listener Sexual Orientation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1C_accuracy_by_orientation.png'), bbox_inches='tight')
    plt.close()

    log("## 1C. Accuracy by Orientation")
    for o, (r, n) in orient_results.items():
        log(f"- {o}: ρ = {r:.4f}, N = {n}" if r is not None else f"- {o}: insufficient data (N={n})")
    mask_other = demographics['Orientation'].isin([4, 5])
    r_oth, n_oth, _ = group_accuracy(ratings, actual_scores, mask_other)
    orient_results["Other/PNS"] = (r_oth, n_oth)
    log(f"- Other/Prefer not to say: ρ = {r_oth:.4f}, N = {n_oth}" if r_oth is not None else f"- Other/PNS: insufficient data (N={n_oth})")
    log("")
    return orient_results


def analysis_1d(data, log):
    print("--- 1D: Accuracy by Familiarity ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    fam           = data['fam']
    xs            = data['xs']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    fam_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, flabel in enumerate(['Low', 'Medium', 'High']):
        mask_f = fam == flabel
        r_f, n_f, favg = group_accuracy(ratings, actual_scores, mask_f)
        fam_results[flabel] = (r_f, n_f)
        ax = axes[idx]
        if favg is not None and r_f is not None:
            m = actual_scores.notna() & favg.notna()
            ax.scatter(actual_scores[m], favg[m], s=60, alpha=0.7, edgecolors='k')
            sl, ic = np.polyfit(actual_scores[m].values, favg[m].values, 1)
            ax.plot(xs, sl*xs+ic, 'r-', lw=2)
            ax.plot([1,5],[1,5],'k--',alpha=0.3)
        ax.set_title(f'{flabel} Familiarity (N={n_f})\nρ={r_f:.3f}' if r_f is not None else f'{flabel} (N={n_f})')
        ax.set_xlabel('Actual Score'); ax.set_ylabel('Avg Predicted Score')
        ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
    plt.suptitle('Perception Accuracy by Familiarity with Gay Men', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1D_accuracy_by_familiarity.png'), bbox_inches='tight')
    plt.close()

    log("## 1D. Accuracy by Familiarity")
    for fl, (r, n) in fam_results.items():
        log(f"- {fl}: ρ = {r:.4f}, N = {n}" if r is not None else f"- {fl}: insufficient data (N={n})")
    log("")
    return fam_results


def analysis_1e(data, log, r_val, gender_results, orient_results, fam_results):
    print("--- 1E: Correlation Comparison Bar Chart ---")
    ratings    = data['ratings']
    OUTPUT_DIR = data['OUTPUT_DIR']

    all_groups = {'Overall': (r_val, ratings.shape[0])}
    all_groups.update({k: (v[0], v[1]) for k, v in gender_results.items()})
    all_groups.update({k: (v[0], v[1]) for k, v in orient_results.items()})
    all_groups.update({f'Fam: {k}': (v[0], v[1]) for k, v in fam_results.items()})

    labels = [k for k in all_groups if all_groups[k][0] is not None]
    r_vals = [all_groups[k][0] for k in labels]
    n_vals = [all_groups[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette('tab10', len(labels))
    ax.bar(range(len(labels)), r_vals, color=colors, edgecolor='k', alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{l}\n(N={n_vals[i]})' for i, l in enumerate(labels)], rotation=30, ha='right')
    ax.set_ylabel("Spearman's ρ")
    ax.set_title("Spearman's ρ (Perception Accuracy) Across Listener Groups")
    ax.axhline(0, color='k', lw=0.5)
    for i, v in enumerate(r_vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1E_correlation_comparison_barplot.png'))
    plt.close()


def analysis_2a(data, log):
    print("--- 2A: Individual Accuracy Distribution ---")
    indiv_r    = data['indiv_r']
    OUTPUT_DIR = data['OUTPUT_DIR']
    ratings    = data['ratings']

    fig, ax = plt.subplots(figsize=(10, 7))
    valid_r = indiv_r.dropna()
    ax.hist(valid_r, bins=20, edgecolor='k', alpha=0.7, color='steelblue')
    ax.axvline(valid_r.mean(), color='red', lw=2, ls='--', label=f'Mean = {valid_r.mean():.3f}')
    ax.axvline(valid_r.median(), color='orange', lw=2, ls='-.', label=f'Median = {valid_r.median():.3f}')
    ax.set_xlabel("Individual Spearman's ρ (with actual scores)")
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Individual Listener Accuracy')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2A_individual_accuracy_distribution.png'))
    plt.close()

    log("## 2A. Individual Listener Accuracy")
    log(f"- Mean ρ = {valid_r.mean():.4f}, Median = {valid_r.median():.4f}")
    log(f"- SD = {valid_r.std():.4f}, Range = [{valid_r.min():.4f}, {valid_r.max():.4f}]")
    log(f"- N listeners with valid r: {valid_r.shape[0]}\n")


def analysis_2b(data, log):
    print("--- 2B: Accuracy vs Characteristics ---")
    indiv_r      = data['indiv_r']
    demographics = data['demographics']
    fam          = data['fam']
    OUTPUT_DIR   = data['OUTPUT_DIR']

    friends  = demographics['GayFriends']
    acquaint = demographics['GayAcquaint']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Gender boxplot
    ax = axes[0, 0]
    df_plot = pd.DataFrame({'Accuracy': indiv_r, 'Gender': demographics['Gender'].map({1: 'Male', 2: 'Female'})})
    df_valid = df_plot.dropna()
    sns.boxplot(x='Gender', y='Accuracy', data=df_valid, ax=ax, palette='Set2')
    ax.set_title('Accuracy by Gender')
    male_acc   = df_valid[df_valid['Gender']=='Male']['Accuracy']
    female_acc = df_valid[df_valid['Gender']=='Female']['Accuracy']
    if len(male_acc) > 1 and len(female_acc) > 1:
        t_stat, t_p = stats.ttest_ind(male_acc, female_acc)
        ax.set_xlabel(f't={t_stat:.2f}, p={t_p:.4f}')
        log("## 2B. Accuracy vs Listener Characteristics")
        log(f"- Gender: Male M={male_acc.mean():.3f} vs Female M={female_acc.mean():.3f}, t={t_stat:.2f}, p={t_p:.4f}")

    # Orientation boxplot
    ax = axes[0, 1]
    orient_map = {1: 'Gay', 2: 'Straight', 3: 'Bi/Pan', 4: 'Other', 5: 'PNS'}
    df_plot2  = pd.DataFrame({'Accuracy': indiv_r, 'Orientation': demographics['Orientation'].map(orient_map)})
    df_valid2 = df_plot2.dropna()
    sns.boxplot(x='Orientation', y='Accuracy', data=df_valid2, ax=ax, palette='Set3')
    ax.set_title('Accuracy by Orientation')
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


def _compute_icc(mat):
    """Compute ICC(2,1) and ICC(2,k) from a complete (no-NaN) ratings matrix (raters x subjects)."""
    n_raters, n_subj = mat.shape
    grand_mean = mat.mean()
    row_means  = mat.mean(axis=1)
    col_means  = mat.mean(axis=0)
    ss_rows    = n_subj   * ((row_means - grand_mean)**2).sum()
    ss_cols    = n_raters * ((col_means - grand_mean)**2).sum()
    ss_total   = ((mat - grand_mean)**2).sum()
    ss_error   = ss_total - ss_rows - ss_cols
    ms_rows    = ss_rows  / (n_raters - 1)
    ms_cols    = ss_cols  / (n_subj - 1)
    ms_error   = ss_error / ((n_raters - 1) * (n_subj - 1))
    icc_21 = (ms_cols - ms_error) / (ms_cols + (n_raters-1)*ms_error + n_raters*(ms_rows-ms_error)/n_subj)
    icc_2k = (ms_cols - ms_error) / (ms_cols + (ms_rows - ms_error)/n_subj)
    return icc_21, icc_2k


def analysis_3a(data, log):
    print("--- 3A: Inter-Rater Reliability ---")
    ratings    = data['ratings']
    OUTPUT_DIR = data['OUTPUT_DIR']

    # --- Complete-cases ICC (original) ---
    ratings_clean = ratings.dropna(how='any', axis=1)
    n_subj_clean  = ratings_clean.shape[1]
    n_raters      = ratings_clean.shape[0]
    icc_21, icc_2k = _compute_icc(ratings_clean.values)

    # --- Mean-imputed ICC (all speakers) ---
    ratings_imputed = ratings.apply(lambda col: col.fillna(col.mean()), axis=0)
    n_subj_full     = ratings_imputed.shape[1]
    icc_21_full, icc_2k_full = _compute_icc(ratings_imputed.values)

    log("## 3A. Inter-Rater Reliability (ICC)")
    log(f"- ICC(2,1) single measures = {icc_21:.4f}")
    log(f"- ICC(2,k) average measures = {icc_2k:.4f}")
    log(f"- Computed on {n_subj_clean} speakers with complete data across {n_raters} raters")
    log(f"- (Mean-imputed, all {n_subj_full} speakers) ICC(2,1) = {icc_21_full:.4f}, ICC(2,k) = {icc_2k_full:.4f}\n")

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_mat = ratings.T.corr()
    sns.heatmap(corr_mat, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.set_title(f'Inter-Rater Correlation (all 85 listeners)\nICC(2,k) = {icc_2k:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3A_interrater_reliability.png'))
    plt.close()


def analysis_3b(data, log):
    print("--- 3B: Per-Speaker Agreement ---")
    ratings    = data['ratings']
    OUTPUT_DIR = data['OUTPUT_DIR']

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


def analysis_3c(data, log):
    """3C: ICC(2,1) and ICC(2,k) stratified by listener gender and gay familiarity."""
    print("--- 3C: ICC by Demographics ---")
    ratings      = data['ratings']
    demographics = data['demographics']
    fam          = data['fam']
    OUTPUT_DIR   = data['OUTPUT_DIR']

    groups = {
        'Male':       demographics['Gender'] == 1,
        'Female':     demographics['Gender'] == 2,
        'Fam: Low':   fam == 'Low',
        'Fam: Medium':fam == 'Medium',
        'Fam: High':  fam == 'High',
    }

    results = {}
    for label, mask in groups.items():
        sub = ratings.loc[mask.values]
        n   = sub.shape[0]
        if n < 3:
            results[label] = (np.nan, np.nan, n)
            continue
        sub_imp = sub.apply(lambda col: col.fillna(col.mean()), axis=0)
        sub_imp = sub_imp.fillna(sub_imp.stack().mean())
        icc21, icc2k = _compute_icc(sub_imp.values)
        results[label] = (icc21, icc2k, n)

    log("## 3C. ICC by Demographic Group")
    for label, (icc21, icc2k, n) in results.items():
        if np.isnan(icc21):
            log(f"- {label} (N={n}): insufficient data")
        else:
            log(f"- {label} (N={n}): ICC(2,1) = {icc21:.4f}, ICC(2,k) = {icc2k:.4f}")
    log("")

    labels = list(results.keys())
    icc21s = [results[l][0] for l in labels]
    icc2ks = [results[l][1] for l in labels]
    ns     = [results[l][2] for l in labels]
    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, icc21s, width, label='ICC(2,1) single',
                   color='steelblue', edgecolor='k', alpha=0.85)
    bars2 = ax.bar(x + width/2, icc2ks, width, label='ICC(2,k) average',
                   color='salmon',    edgecolor='k', alpha=0.85)
    for bar, val in zip(bars1, icc21s):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, icc2ks):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(N={ns[i]})' for i, l in enumerate(labels)], fontsize=10)
    ax.set_ylabel('ICC Value')
    ax.set_ylim(0, 1.1)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('3C: Inter-Rater Reliability by Demographic Group')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3C_icc_by_demographics.png'))
    plt.close()


def analysis_4a(data, log):
    print("--- 4A: Consensus vs Accuracy ---")
    speaker_codes  = data['speaker_codes']
    avg_predicted  = data['avg_predicted']
    actual_scores  = data['actual_scores']
    abs_error      = data['abs_error']
    speaker_sd_all = data['speaker_sd_all']
    OUTPUT_DIR     = data['OUTPUT_DIR']

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


def analysis_4b(data, log):
    """4B: Mean-based error vs individual-level MAE — where does the mean metric mislead?"""
    print("--- 4B: Mean-based Error vs Individual-level MAE ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    abs_error     = data['abs_error']
    speaker_codes = data['speaker_codes']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    # Per-speaker mean(|rating_i - actual|)
    indiv_mae = {}
    for sp in speaker_codes:
        actual = actual_scores[sp]
        if pd.isna(actual):
            continue
        col = ratings[sp].dropna()
        if len(col) > 0:
            indiv_mae[sp] = (col - actual).abs().mean()
    indiv_mae = pd.Series(indiv_mae)

    df = pd.DataFrame({
        'MeanBased': abs_error.reindex(speaker_codes),
        'IndivMAE':  indiv_mae.reindex(speaker_codes),
        'Actual':    actual_scores.reindex(speaker_codes),
    }).dropna()

    divergence = (df['IndivMAE'] - df['MeanBased']).sort_values(ascending=False)
    top_divergers = divergence.head(5)

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(df['MeanBased'], df['IndivMAE'], s=80, alpha=0.8, edgecolors='k',
                    c=df['Actual'], cmap='RdYlBu_r', vmin=1, vmax=5, zorder=3)
    plt.colorbar(sc, ax=ax, label='Actual Sexuality (1=Straight, 5=Gay)')

    lim_max = max(df['MeanBased'].max(), df['IndivMAE'].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, lw=1.5, label='y = x (metrics agree)')

    for sp in df.index:
        ax.annotate(sp, (df.loc[sp, 'MeanBased'], df.loc[sp, 'IndivMAE']),
                    fontsize=7.5, xytext=(5, 3), textcoords='offset points')

    ax.set_xlabel('Mean-Based Error  |mean(ratings) - actual|')
    ax.set_ylabel('Individual-Level MAE  mean(|rating_i - actual|)')
    ax.set_title('4B: Mean-Based Error vs Individual-Level MAE\n'
                 'Points above diagonal = mean metric was overly optimistic')
    ax.legend(loc='upper left')
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4B_meanbased_vs_indivmae.png'))
    plt.close()

    log("## 4B. Mean-Based Error vs Individual-Level MAE")
    log("- Individual-level MAE = mean(|rating_i - actual|) per speaker")
    log("- Points above diagonal: mean metric was overly optimistic (bimodal cancellation)")
    log("- Top 5 divergers (IndivMAE - MeanBased):")
    for sp, div in top_divergers.items():
        log(f"  {sp}: indiv MAE = {indiv_mae[sp]:.3f}, mean-based = {abs_error[sp]:.3f}, "
            f"diff = +{div:.3f}")
    log("")


def analysis_5a(data, log):
    print("--- 5A: Speaker Readability Quadrants ---")
    speaker_codes  = data['speaker_codes']
    abs_error      = data['abs_error']
    speaker_sd_all = data['speaker_sd_all']
    OUTPUT_DIR     = data['OUTPUT_DIR']

    med_sd  = speaker_sd_all.median()
    med_err = abs_error.median()

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(speaker_sd_all, abs_error, s=100, alpha=0.7, edgecolors='k', zorder=3)
    for sp in speaker_codes:
        ax.annotate(sp, (speaker_sd_all[sp], abs_error[sp]), fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    ax.axvline(med_sd,  color='gray', ls='--', alpha=0.5)
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

    log("## 5A. Speaker Readability")
    log("(See quadrant plot for detailed speaker positions)\n")


def analysis_5b(data, log):
    print("--- 5B: Accuracy by Actual Score ---")
    speaker_codes = data['speaker_codes']
    actual_scores = data['actual_scores']
    abs_error     = data['abs_error']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    score_bins = {}
    for sp in speaker_codes:
        sc = actual_scores[sp]
        if pd.isna(sc):
            continue
        if sc <= 1.5:   cat = '1 (Straight)'
        elif sc <= 2.5: cat = '2'
        elif sc <= 3.5: cat = '3 (Middle)'
        elif sc <= 4.5: cat = '4'
        else:           cat = '5 (Gay)'
        score_bins.setdefault(cat, []).append(abs_error[sp])

    fig, ax = plt.subplots(figsize=(10, 7))
    bin_labels   = sorted(score_bins.keys())
    data_for_box = [score_bins[bl] for bl in bin_labels]
    bp = ax.boxplot(data_for_box, labels=bin_labels, patch_artist=True)
    colors_box = sns.color_palette('coolwarm', len(bin_labels))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    ax.set_xlabel('Actual Score Category')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Prediction Error by Actual Score Category')
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


def analysis_5c1(data, log, cl):
    print("--- 5C1: Hierarchical Clustering ---")
    OUTPUT_DIR    = data['OUTPUT_DIR']
    speaker_codes = cl['speaker_codes']
    ratings_scaled = cl['ratings_scaled']

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
    log("- Ward linkage on standardized 85-dimensional listener rating vectors")
    log("- See dendrogram plot for speaker groupings\n")


def analysis_5c2(data, log, cl):
    print("--- 5C2: PCA Variance Diagnostics ---")
    OUTPUT_DIR = data['OUTPUT_DIR']
    cumvar     = cl['cumvar']
    pca_full   = cl['pca_full']
    pc1_var    = cl['pc1_var']
    pc2_var    = cl['pc2_var']
    n_50       = cl['n_50']
    n_70       = cl['n_70']
    n_show     = min(20, len(cumvar))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show] * 100,
           color='steelblue', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot\n(each bar = one PC)')
    ax.grid(axis='y', alpha=0.3)

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


def analysis_5c3(data, log, cl):
    print("--- 5C3: K-means Diagnostics ---")
    OUTPUT_DIR  = data['OUTPUT_DIR']
    k_range     = cl['k_range']
    inertias    = cl['inertias']
    sil_scores  = cl['sil_scores']
    best_k      = cl['best_k']

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
    if 3 in k_range: log(f"- k=3 silhouette: {sil_scores[list(k_range).index(3)]:.3f}")
    if 4 in k_range: log(f"- k=4 silhouette: {sil_scores[list(k_range).index(4)]:.3f}")
    log("")


def analysis_5c4(data, log, cl):
    print("--- 5C4: DBSCAN on 2D PCA ---")
    OUTPUT_DIR    = data['OUTPUT_DIR']
    pca_coords    = cl['pca_coords']
    pc1_var       = cl['pc1_var']
    pc2_var       = cl['pc2_var']
    best_k        = cl['best_k']
    speaker_codes = cl['speaker_codes']

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

    # Three-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    db2d_elbow = DBSCAN(eps=eps_elbow_2d, min_samples=3).fit_predict(pca_coords)
    _, n_cl_e, n_no_e = plot_dbscan(
        axes[0], pca_coords, db2d_elbow, speaker_codes, pc1_var, pc2_var,
        title_extra=f'DBSCAN kneedle eps={eps_elbow_2d:.2f}')

    db2d_cons = DBSCAN(eps=eps_cons_2d, min_samples=3).fit_predict(pca_coords)
    _, n_cl_c, n_no_c = plot_dbscan(
        axes[1], pca_coords, db2d_cons, speaker_codes, pc1_var, pc2_var,
        title_extra=f'DBSCAN conservative eps={eps_cons_2d:.2f}')

    kmeans4    = KMeans(n_clusters=4, random_state=42, n_init=10)
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


def analysis_5c5(data, log, cl):
    print(f"--- 5C5: DBSCAN on {cl['n_50']} components (>= 50% variance) ---")
    OUTPUT_DIR    = data['OUTPUT_DIR']
    pca_coords    = cl['pca_coords']
    pc1_var       = cl['pc1_var']
    pc2_var       = cl['pc2_var']
    coords_50     = cl['coords_50']
    n_50          = cl['n_50']
    cumvar        = cl['cumvar']
    speaker_codes = cl['speaker_codes']

    eps_elbow_50, eps_cons_50, kd_50, ei_50 = kneedle_eps(coords_50, k=3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(kd_50) + 1), kd_50, 'b-o', markersize=5, label='k-distance')
    ax.axhline(eps_elbow_50, color='red',    ls='--', lw=1.8, label=f'Kneedle eps = {eps_elbow_50:.2f}')
    ax.axhline(eps_cons_50,  color='orange', ls=':',  lw=1.8, label=f'Conservative eps = {eps_cons_50:.2f}')
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


def analysis_5c6(data, log, cl):
    print(f"--- 5C6: DBSCAN on {cl['n_70']} components (>= 70% variance) ---")
    OUTPUT_DIR    = data['OUTPUT_DIR']
    pca_coords    = cl['pca_coords']
    pc1_var       = cl['pc1_var']
    pc2_var       = cl['pc2_var']
    coords_70     = cl['coords_70']
    n_70          = cl['n_70']
    cumvar        = cl['cumvar']
    speaker_codes = cl['speaker_codes']

    eps_elbow_70, eps_cons_70, kd_70, ei_70 = kneedle_eps(coords_70, k=3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(kd_70) + 1), kd_70, 'b-o', markersize=5, label='k-distance')
    ax.axhline(eps_elbow_70, color='red',    ls='--', lw=1.8, label=f'Kneedle eps = {eps_elbow_70:.2f}')
    ax.axhline(eps_cons_70,  color='orange', ls=':',  lw=1.8, label=f'Conservative eps = {eps_cons_70:.2f}')
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


def analysis_5d(data, log, cl):
    """5D1, 5D2, 5D3: PC1 as perceived-gayness axis."""
    print("--- 5D: PC1 Analyses ---")
    OUTPUT_DIR = data['OUTPUT_DIR']
    pc1_var    = cl['pc1_var']
    df_5d      = cl['df_5d'].copy()

    # 5D1: PC1 vs Actual
    fig, ax = plt.subplots(figsize=(10, 8))
    r_5d1, p_5d1 = stats.pearsonr(df_5d['Actual'], df_5d['PC1'])
    xs = np.linspace(df_5d['Actual'].min(), df_5d['Actual'].max(), 100)
    sl, ic = np.polyfit(df_5d['Actual'], df_5d['PC1'], 1)
    ax.scatter(df_5d['Actual'], df_5d['PC1'], s=80, alpha=0.7, edgecolors='k', color='steelblue')
    ax.plot(xs, sl*xs+ic, 'r-', lw=2)
    for sp in df_5d.index:
        ax.annotate(sp, (df_5d.loc[sp,'Actual'], df_5d.loc[sp,'PC1']),
                    fontsize=7.5, xytext=(5,3), textcoords='offset points')
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

    # 5D2: PC1 vs Mean Rating
    fig, ax = plt.subplots(figsize=(10, 8))
    r_5d2, p_5d2 = stats.pearsonr(df_5d['MeanRating'], df_5d['PC1'])
    xs2 = np.linspace(df_5d['MeanRating'].min(), df_5d['MeanRating'].max(), 100)
    sl2, ic2 = np.polyfit(df_5d['MeanRating'], df_5d['PC1'], 1)
    ax.scatter(df_5d['MeanRating'], df_5d['PC1'], s=80, alpha=0.7, edgecolors='k', color='mediumseagreen')
    ax.plot(xs2, sl2*xs2+ic2, 'r-', lw=2)
    for sp in df_5d.index:
        ax.annotate(sp, (df_5d.loc[sp,'MeanRating'], df_5d.loc[sp,'PC1']),
                    fontsize=7.5, xytext=(5,3), textcoords='offset points')
    ax.set_xlabel('Mean Listener Rating (1=Straight, 5=Gay)')
    ax.set_ylabel(f'PC1 Score ({pc1_var:.1f}% variance)')
    ax.set_title(f'5D2: PC1 vs Mean Listener Rating\nr = {r_5d2:.3f}, p = {p_5d2:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5D2_pc1_vs_mean_rating.png'))
    plt.close()
    log("## 5D2. PC1 vs Mean Rating")
    log(f"- Pearson r = {r_5d2:.4f}, p = {p_5d2:.4f}")
    log(f"- R² = {r_5d2**2:.4f}")
    log("- High r² validates PC1 as the consensus perceived-gayness dimension\n")

    # 5D3: |PC1| vs Absolute Error
    df_5d['AbsPC1'] = df_5d['PC1'].abs()
    r_5d3, p_5d3 = stats.pearsonr(df_5d['AbsPC1'], df_5d['AbsError'])
    xs3 = np.linspace(df_5d['AbsPC1'].min(), df_5d['AbsPC1'].max(), 100)
    sl3, ic3 = np.polyfit(df_5d['AbsPC1'], df_5d['AbsError'], 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df_5d['AbsPC1'], df_5d['AbsError'], s=80, alpha=0.7, edgecolors='k',
                    c=df_5d['Actual'], cmap='RdYlBu_r', vmin=1, vmax=5)
    ax.plot(xs3, sl3*xs3+ic3, 'r-', lw=2)
    for sp in df_5d.index:
        ax.annotate(sp, (df_5d.loc[sp,'AbsPC1'], df_5d.loc[sp,'AbsError']),
                    fontsize=7.5, xytext=(5,3), textcoords='offset points')
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


def analysis_6a(data, log):
    print("--- 6A: Listener Response Bias ---")
    listener_means = data['listener_means']
    OUTPUT_DIR     = data['OUTPUT_DIR']

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


def analysis_6b(data, log):
    print("--- 6B: Religiosity vs Bias ---")
    listener_means = data['listener_means']
    demographics   = data['demographics']
    OUTPUT_DIR     = data['OUTPUT_DIR']

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


def analysis_6c(data, log):
    print("--- 6C: Listener Response Range ---")
    listener_sds = data['listener_sds']
    OUTPUT_DIR   = data['OUTPUT_DIR']

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


def analysis_7a(data, log):
    print("--- 7A: Gender x Orientation Interaction ---")
    indiv_r      = data['indiv_r']
    demographics = data['demographics']
    OUTPUT_DIR   = data['OUTPUT_DIR']

    df_int = pd.DataFrame({
        'Accuracy':    indiv_r,
        'Gender':      demographics['Gender'].map({1: 'Male', 2: 'Female'}),
        'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'})
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 7))
    for orient in ['Gay', 'Straight', 'Bi/Pan']:
        sub   = df_int[df_int['Orientation'] == orient]
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

    log("## 7A. Gender x Orientation Interaction")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        model       = ols('Accuracy ~ C(Gender) * C(Orientation)', data=df_int).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        log("- Two-way ANOVA results:")
        for idx_row in anova_table.index:
            row = anova_table.loc[idx_row]
            if 'PR(>F)' in row.index:
                log(f"  - {idx_row}: F={row['F']:.3f}, p={row['PR(>F)']:.4f}" if not pd.isna(row['F']) else f"  - {idx_row}: (residual)")
    except ImportError:
        log("- statsmodels not available; skipping formal ANOVA")
    log("")


def analysis_7b(data, log):
    print("--- 7B: Match Effects ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    speaker_codes = data['speaker_codes']
    indiv_r       = data['indiv_r']
    demographics  = data['demographics']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    gay_speakers      = [sp for sp in speaker_codes if actual_scores[sp] >= 4]
    straight_speakers = [sp for sp in speaker_codes if actual_scores[sp] <= 2]

    listener_gay_mae = []
    listener_str_mae = []
    for i in range(85):
        lr  = ratings.iloc[i]
        m_g = lr[gay_speakers].notna()
        listener_gay_mae.append(
            (lr[gay_speakers][m_g] - actual_scores[gay_speakers][m_g]).abs().mean()
            if m_g.sum() > 0 else np.nan)
        m_s = lr[straight_speakers].notna()
        listener_str_mae.append(
            (lr[straight_speakers][m_s] - actual_scores[straight_speakers][m_s]).abs().mean()
            if m_s.sum() > 0 else np.nan)

    df_match = pd.DataFrame({
        'MAE_Gay_Speakers':      listener_gay_mae,
        'MAE_Straight_Speakers': listener_str_mae,
        'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'})
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 7))
    df_melt = df_match.melt(id_vars='Orientation',
                            value_vars=['MAE_Gay_Speakers', 'MAE_Straight_Speakers'],
                            var_name='Speaker Type', value_name='MAE')
    df_melt['Speaker Type'] = df_melt['Speaker Type'].map(
        {'MAE_Gay_Speakers': 'Gay Speakers', 'MAE_Straight_Speakers': 'Straight Speakers'})
    sns.barplot(x='Orientation', y='MAE', hue='Speaker Type', data=df_melt, ax=ax,
                palette='Set2', errorbar=('ci', 95))
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
            log(f"- {orient} listeners: MAE for gay speakers = {sub['MAE_Gay_Speakers'].mean():.3f}, "
                f"for straight speakers = {sub['MAE_Straight_Speakers'].mean():.3f}")
    log("")


def analysis_8a(data, log):
    print("--- 8A: Signal Detection Theory ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    speaker_codes = data['speaker_codes']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    threshold     = 3.0
    actual_binary = (actual_scores >= threshold).astype(int)

    listener_dprime    = []
    listener_criterion = []
    for i in range(85):
        lr = ratings.iloc[i]
        hits = misses = fas = crs = 0
        for sp in speaker_codes:
            if pd.isna(lr[sp]) or pd.isna(actual_binary[sp]):
                continue
            pred   = 1 if lr[sp] >= threshold else 0
            actual = actual_binary[sp]
            if   actual == 1 and pred == 1: hits   += 1
            elif actual == 1 and pred == 0: misses += 1
            elif actual == 0 and pred == 1: fas    += 1
            else:                           crs    += 1
        n_signal = hits + misses
        n_noise  = fas  + crs
        if n_signal > 0 and n_noise > 0:
            hr  = np.clip(hits/n_signal, 0.5/n_signal, 1 - 0.5/n_signal)
            far = np.clip(fas/n_noise,   0.5/n_noise,  1 - 0.5/n_noise)
            listener_dprime.append(stats.norm.ppf(hr) - stats.norm.ppf(far))
            listener_criterion.append(-0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(far)))
        else:
            listener_dprime.append(np.nan)
            listener_criterion.append(np.nan)

    dprime_series    = pd.Series(listener_dprime)
    criterion_series = pd.Series(listener_criterion)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    valid_dp = dprime_series.dropna()
    axes[0].hist(valid_dp, bins=15, edgecolor='k', alpha=0.7, color='teal')
    axes[0].axvline(valid_dp.mean(), color='red', lw=2, ls='--', label=f"Mean d' = {valid_dp.mean():.2f}")
    axes[0].set_xlabel("d' (sensitivity)")
    axes[0].set_ylabel('Count')
    axes[0].set_title("Distribution of d' Across Listeners")
    axes[0].legend()

    valid_c = criterion_series.dropna()
    axes[1].hist(valid_c, bins=15, edgecolor='k', alpha=0.7, color='salmon')
    axes[1].axvline(valid_c.mean(), color='red', lw=2, ls='--', label=f"Mean c = {valid_c.mean():.2f}")
    axes[1].set_xlabel("c (criterion)")
    axes[1].set_ylabel('Count')
    axes[1].set_title("Distribution of Criterion (c) Across Listeners")
    axes[1].legend()

    plt.suptitle('Signal Detection Theory Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8A_signal_detection.png'), bbox_inches='tight')
    plt.close()

    log("## 8A. Signal Detection Theory")
    log(f"- Threshold for dichotomization: {threshold}")
    log(f"- Mean d' = {valid_dp.mean():.4f} (SD = {valid_dp.std():.4f})")
    log(f"- Mean criterion c = {valid_c.mean():.4f} (SD = {valid_c.std():.4f})")
    log(f"- d' > 0 indicates above-chance sensitivity\n")

    return dprime_series


def analysis_8b(data, log, dprime_series):
    print("--- 8B: d' by Group ---")
    demographics = data['demographics']
    fam          = data['fam']
    OUTPUT_DIR   = data['OUTPUT_DIR']

    df_sdt = pd.DataFrame({
        'dprime':      dprime_series,
        'Gender':      demographics['Gender'].map({1: 'Male', 2: 'Female'}),
        'Orientation': demographics['Orientation'].map({1: 'Gay', 2: 'Straight', 3: 'Bi/Pan'}),
        'Familiarity': fam
    }).dropna(subset=['dprime'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(x='Gender',      y='dprime', data=df_sdt.dropna(subset=['Gender']),
                ax=axes[0], palette='Set2')
    axes[0].set_title("d' by Gender");      axes[0].set_ylabel("d'")
    sns.boxplot(x='Orientation', y='dprime', data=df_sdt.dropna(subset=['Orientation']),
                ax=axes[1], palette='Set3')
    axes[1].set_title("d' by Orientation"); axes[1].set_ylabel("d'")
    sns.boxplot(x='Familiarity', y='dprime', data=df_sdt.dropna(subset=['Familiarity']),
                order=['Low', 'Medium', 'High'], ax=axes[2], palette='Set1')
    axes[2].set_title("d' by Familiarity"); axes[2].set_ylabel("d'")

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


def analysis_9a(data, log):
    """9A: Bar plot of mean-based error per speaker, sorted max to min."""
    print("--- 9A: Mean-Based Error by Speaker ---")
    abs_error     = data['abs_error'].sort_values(ascending=False)
    actual_scores = data['actual_scores']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(abs_error)))
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(range(len(abs_error)), abs_error.values, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(abs_error)))
    ax.set_xticklabels(abs_error.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean-Based Error  |mean(ratings) - actual|')
    ax.set_title('9A: Per-Speaker Error — Mean-Based  (high to low)')
    ax.axhline(abs_error.mean(), color='steelblue', lw=1.5, ls='--',
               label=f'Mean = {abs_error.mean():.3f}')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '9A_meanbased_error_by_speaker.png'))
    plt.close()

    log("## 9A. Per-Speaker Mean-Based Error (sorted)")
    log(f"- Range: {abs_error.iloc[-1]:.3f} ({abs_error.index[-1]}) "
        f"to {abs_error.iloc[0]:.3f} ({abs_error.index[0]})")
    log(f"- Mean across speakers: {abs_error.mean():.3f}\n")


def analysis_9b(data, log):
    """9B: Bar plot of individual-level MAE per speaker, sorted max to min."""
    print("--- 9B: Individual-Level MAE by Speaker ---")
    ratings       = data['ratings']
    actual_scores = data['actual_scores']
    speaker_codes = data['speaker_codes']
    OUTPUT_DIR    = data['OUTPUT_DIR']

    indiv_mae = {}
    for sp in speaker_codes:
        actual = actual_scores[sp]
        if pd.isna(actual):
            continue
        col = ratings[sp].dropna()
        if len(col) > 0:
            indiv_mae[sp] = (col - actual).abs().mean()
    indiv_mae = pd.Series(indiv_mae).sort_values(ascending=False)

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(indiv_mae)))
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(range(len(indiv_mae)), indiv_mae.values, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(indiv_mae)))
    ax.set_xticklabels(indiv_mae.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Individual-Level MAE  mean(|rating_i - actual|)')
    ax.set_title('9B: Per-Speaker Error — Individual-Level MAE  (high to low)')
    ax.axhline(indiv_mae.mean(), color='steelblue', lw=1.5, ls='--',
               label=f'Mean = {indiv_mae.mean():.3f}')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '9B_indivmae_by_speaker.png'))
    plt.close()

    log("## 9B. Per-Speaker Individual-Level MAE (sorted)")
    log(f"- Range: {indiv_mae.iloc[-1]:.3f} ({indiv_mae.index[-1]}) "
        f"to {indiv_mae.iloc[0]:.3f} ({indiv_mae.index[0]})")
    log(f"- Mean across speakers: {indiv_mae.mean():.3f}\n")


def write_summary(data, summary_lines):
    actual_scores = data['actual_scores']
    avg_predicted = data['avg_predicted']
    abs_error     = data['abs_error']
    speaker_codes = data['speaker_codes']

    def log(msg): summary_lines.append(msg)

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

    mae  = abs_error.mean()
    rmse = np.sqrt(((avg_predicted - actual_scores)**2).mean())
    log(f"\n### Overall Metrics")
    log(f"- Mean Absolute Error (MAE): {mae:.4f}")
    log(f"- RMSE: {rmse:.4f}")

    summary_path = os.path.join(os.path.dirname(data['INPUT_CSV']), 'analysis_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to {summary_path}")
    print(f"All plots saved to {data['OUTPUT_DIR']}")
    print("Analysis complete!")


# ============================================================
# MAIN
def analysis_5e(data, log, cl):
    """5E: PC2–PC5 vs Mean Listener Rating (are higher PCs orthogonal to consensus?)"""
    print("--- 5E: PC2-PC5 vs Mean Listener Rating ---")
    OUTPUT_DIR     = data['OUTPUT_DIR']
    avg_predicted  = data['avg_predicted']
    ratings_scaled = cl['ratings_scaled']
    speaker_codes  = cl['speaker_codes']

    pca5    = PCA(n_components=5)
    coords5 = pca5.fit_transform(ratings_scaled)          # (n_speakers, 5)
    pc_vars = pca5.explained_variance_ratio_ * 100

    mean_rating = avg_predicted.reindex(speaker_codes).values
    valid       = ~np.isnan(mean_rating)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        pc_idx    = i + 1                                 # PC2=1, PC3=2, PC4=3, PC5=4
        pc_scores = coords5[valid, pc_idx]
        mr        = mean_rating[valid]
        names     = [speaker_codes[j] for j in range(len(speaker_codes)) if valid[j]]

        r, p  = stats.pearsonr(mr, pc_scores)
        xs    = np.linspace(mr.min(), mr.max(), 100)
        sl, ic = np.polyfit(mr, pc_scores, 1)

        ax.scatter(mr, pc_scores, s=70, alpha=0.7, edgecolors='k', color='steelblue')
        ax.plot(xs, sl * xs + ic, 'r-', lw=2)
        for j, sp in enumerate(names):
            ax.annotate(sp, (mr[j], pc_scores[j]), fontsize=7,
                        xytext=(4, 3), textcoords='offset points')
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('Mean Listener Rating (1=Straight, 5=Gay)')
        ax.set_ylabel(f'PC{pc_idx + 1} Score ({pc_vars[pc_idx]:.1f}% var)')
        ax.set_title(f'PC{pc_idx + 1} vs Mean Rating\nr = {r:.3f}, p = {p:.4f}')
        ax.grid(True, alpha=0.3)

        log(f"## 5E{i + 1}. PC{pc_idx + 1} vs Mean Listener Rating")
        log(f"- Pearson r = {r:.4f}, p = {p:.4f}")
        log(f"- R² = {r ** 2:.4f}")
        log(f"- Variance explained by PC{pc_idx + 1}: {pc_vars[pc_idx]:.1f}%\n")

    plt.suptitle('5E: Higher PCs vs Mean Listener Rating\n'
                 '(PC1 ≈ consensus; do PC2–PC5 capture structured disagreement?)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5E_higher_pcs_vs_mean_rating.png'),
                bbox_inches='tight')
    plt.close()


def analysis_5f(data, log, cl):
    """5F: SD of listener ratings per speaker vs |PC2| score."""
    print("--- 5F: Speaker Rating SD vs |PC2| ---")
    OUTPUT_DIR    = data['OUTPUT_DIR']
    speaker_sd    = data['speaker_sd_all']
    pca_coords    = cl['pca_coords']
    pc2_var       = cl['pc2_var']
    speaker_codes = cl['speaker_codes']

    pc2_scores = pd.Series(pca_coords[:, 1], index=speaker_codes)

    df = pd.DataFrame({
        'SD':   speaker_sd.reindex(speaker_codes),
        'AbsPC2': pc2_scores.abs(),
    }).dropna()

    r, p   = stats.pearsonr(df['SD'], df['AbsPC2'])
    xs     = np.linspace(df['SD'].min(), df['SD'].max(), 100)
    sl, ic = np.polyfit(df['SD'], df['AbsPC2'], 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['SD'], df['AbsPC2'], s=80, alpha=0.7, edgecolors='k', color='darkorange')
    ax.plot(xs, sl * xs + ic, 'r-', lw=2)
    for sp in df.index:
        ax.annotate(sp, (df.loc[sp, 'SD'], df.loc[sp, 'AbsPC2']),
                    fontsize=7.5, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel('SD of Listener Ratings (low = high agreement)')
    ax.set_ylabel(f'|PC2 Score| ({pc2_var:.1f}% var)')
    ax.set_title(f'5F: Speaker Rating SD vs |PC2|\nr = {r:.3f}, p = {p:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5F_sd_vs_pc2.png'))
    plt.close()

    log("## 5F. Speaker Rating SD vs |PC2|")
    log(f"- Pearson r = {r:.4f}, p = {p:.4f}")
    log(f"- R² = {r ** 2:.4f}")
    log("- r ~= 0  => PC2 is NOT simply capturing disagreement magnitude")
    log("- r > 0   => high-SD speakers sit further from the PC2 origin\n")


def analysis_5g(data, log, cl):
    """5G: Total non-consensus PCA distance (PC2-PC5) vs speaker rating SD."""
    print("--- 5G: Non-Consensus PCA Distance vs SD ---")
    OUTPUT_DIR     = data['OUTPUT_DIR']
    speaker_sd     = data['speaker_sd_all']
    ratings_scaled = cl['ratings_scaled']
    speaker_codes  = cl['speaker_codes']

    pca5    = PCA(n_components=5)
    coords5 = pca5.fit_transform(ratings_scaled)
    pc_vars = pca5.explained_variance_ratio_ * 100

    # Euclidean norm of PC2-PC5 per speaker
    nc_dist = pd.Series(
        np.sqrt((coords5[:, 1:] ** 2).sum(axis=1)),
        index=speaker_codes
    )

    df = pd.DataFrame({
        'SD':     speaker_sd.reindex(speaker_codes),
        'NCDist': nc_dist,
    }).dropna()

    r, p   = stats.pearsonr(df['SD'], df['NCDist'])
    xs     = np.linspace(df['SD'].min(), df['SD'].max(), 100)
    sl, ic = np.polyfit(df['SD'], df['NCDist'], 1)
    var_pct = sum(pc_vars[1:5])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['SD'], df['NCDist'], s=80, alpha=0.7, edgecolors='k', color='mediumpurple')
    ax.plot(xs, sl * xs + ic, 'r-', lw=2)
    for sp in df.index:
        ax.annotate(sp, (df.loc[sp, 'SD'], df.loc[sp, 'NCDist']),
                    fontsize=7.5, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel('SD of Listener Ratings (low = high agreement)')
    ax.set_ylabel(f'Non-Consensus Distance sqrt(PC2^2+...+PC5^2) ({var_pct:.1f}% var)')
    ax.set_title(f'5G: Rating Disagreement vs Non-Consensus PCA Distance\nr = {r:.3f}, p = {p:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5G_sd_vs_nonconsensus_dist.png'))
    plt.close()

    log("## 5G. Rating SD vs Non-Consensus PCA Distance (PC2-PC5)")
    log(f"- Pearson r = {r:.4f}, p = {p:.4f}")
    log(f"- R2 = {r ** 2:.4f}")
    log(f"- Non-consensus distance = sqrt(PC2^2 + PC3^2 + PC4^2 + PC5^2)")
    log(f"- Variance captured by PC2-PC5: {var_pct:.1f}%\n")


# ============================================================

def main():
    # ── Edit RUN to run only specific analyses. Empty set = run all. ──
    # Valid keys: '1A','1B','1C','1D','1E','2A','2B','3A','3B','4A',
    #             '4B','5A','5B','5C1','5C2','5C3','5C4','5C5','5C6','5D',
    #             '5E','5F','5G','6A','6B','6C','7A','7B','8A','8B',
    #             '9A','9B'
    RUN = set()

    run_all = not RUN
    def should_run(name): return run_all or name in RUN

    data = load_data()
    summary_lines = []
    def log(msg):
        print(msg)
        summary_lines.append(msg)

    log("# Gay Voice Perception Research - Analysis Summary\n")
    log(f"## Sample: {data['ratings'].shape[0]} listeners, {data['ratings'].shape[1]} speakers")
    log(f"Missing ratings: {data['ratings'].isna().sum().sum()} / {data['ratings'].size} "
        f"({100*data['ratings'].isna().sum().sum()/data['ratings'].size:.1f}%)\n")

    # ---- Section 1: Perception Accuracy ----
    r_val = analysis_1a(data, log) if should_run('1A') else data['r_val_overall']

    gender_results = analysis_1b(data, log) if should_run('1B') else {}
    orient_results = analysis_1c(data, log) if should_run('1C') else {}
    fam_results    = analysis_1d(data, log) if should_run('1D') else {}

    if should_run('1E'):
        analysis_1e(data, log, r_val, gender_results, orient_results, fam_results)

    # ---- Section 2: Individual Differences ----
    if should_run('2A'): analysis_2a(data, log)
    if should_run('2B'): analysis_2b(data, log)

    # ---- Section 3: Reliability ----
    if should_run('3A'): analysis_3a(data, log)
    if should_run('3B'): analysis_3b(data, log)
    if should_run('3C'): analysis_3c(data, log)

    # ---- Section 4: Consensus ----
    if should_run('4A'): analysis_4a(data, log)
    if should_run('4B'): analysis_4b(data, log)

    # ---- Section 5: Speaker Analyses ----
    if should_run('5A'): analysis_5a(data, log)
    if should_run('5B'): analysis_5b(data, log)

    # Clustering setup — only when needed (expensive)
    CLUSTER_IDS = {'5C1','5C2','5C3','5C4','5C5','5C6','5D','5E','5F','5G'}
    cl = None
    if run_all or RUN & CLUSTER_IDS:
        print("\n--- Setting up clustering (PCA + K-means) ---")
        cl = setup_clustering(data)

    if cl is not None:
        if should_run('5C1'): analysis_5c1(data, log, cl)
        if should_run('5C2'): analysis_5c2(data, log, cl)
        if should_run('5C3'): analysis_5c3(data, log, cl)
        if should_run('5C4'): analysis_5c4(data, log, cl)
        if should_run('5C5'): analysis_5c5(data, log, cl)
        if should_run('5C6'): analysis_5c6(data, log, cl)
        if should_run('5D'):  analysis_5d(data, log, cl)
        if should_run('5E'):  analysis_5e(data, log, cl)
        if should_run('5F'):  analysis_5f(data, log, cl)
        if should_run('5G'):  analysis_5g(data, log, cl)

    # ---- Section 6: Response Bias ----
    if should_run('6A'): analysis_6a(data, log)
    if should_run('6B'): analysis_6b(data, log)
    if should_run('6C'): analysis_6c(data, log)

    # ---- Section 7: Interactions ----
    if should_run('7A'): analysis_7a(data, log)
    if should_run('7B'): analysis_7b(data, log)

    # ---- Section 8: Signal Detection ----
    dprime_series = None
    if should_run('8A'):
        dprime_series = analysis_8a(data, log)
    if should_run('8B'):
        if dprime_series is None:
            # Compute 8A stats without saving to get dprime_series
            dprime_series = analysis_8a(data, log)
        analysis_8b(data, log, dprime_series)

    # ---- Section 9: Per-Speaker Error Rankings ----
    if should_run('9A'): analysis_9a(data, log)
    if should_run('9B'): analysis_9b(data, log)

    write_summary(data, summary_lines)


if __name__ == '__main__':
    main()
