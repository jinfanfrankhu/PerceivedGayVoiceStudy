import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

features = pd.read_csv('features.csv')
meta = pd.read_csv('Master Spreadsheet.csv')
meta = meta.rename(columns={'Initials': 'ID'})
df = features.merge(meta[['ID', 'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation']], on='ID')

feature_cols = [c for c in features.columns if c != 'ID']
X_raw = df[feature_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# --- Label vectors ---
binary_labels = df['Self-Described Sexual Orientation'].str.lower().str.strip()
y_binary = (binary_labels == 'straight').astype(int)

def three_class(row):
    label = str(row['Self-Described Sexual Orientation']).lower().strip()
    if label == 'straight': return 0
    elif label == 'gay': return 2
    else: return 1

y_three = df.apply(three_class, axis=1).values
y_ordinal = df['Kinsey Scale (1-5)'].values

labels = {
    'Binary (straight vs. non-straight)': y_binary,
    '3-class (straight / bi / gay)':      y_three,
    'Ordinal (1-5 attraction scale)':      y_ordinal,
}

# ============================================================
# 1. SelectKBest — ranks by F-statistic against each label set
# ============================================================
print('\n' + '='*60)
print('SELECT K BEST — all features ranked by F-statistic')
print('(higher F = stronger univariate relationship with label)')
print('='*60)

skb_results = {}
for label_name, y in labels.items():
    f_scores, p_values = f_classif(X, y)
    ranked_idx = np.argsort(f_scores)[::-1]
    print(f'\n--- {label_name} ---')
    print(f'{"Rank":<5} {"Feature":<45} {"F-score":>8} {"p-value":>10}')
    print('-' * 72)
    rows = []
    for rank, idx in enumerate(ranked_idx, 1):
        fname = feature_cols[idx]
        fscore = f_scores[idx]
        pval = p_values[idx]
        sig = '**' if pval < 0.05 else ('*' if pval < 0.1 else '')
        print(f'{rank:<5} {fname:<45} {fscore:>8.3f} {pval:>10.4f} {sig}')
        rows.append({'rank': rank, 'feature': fname, 'f_score': fscore, 'p_value': pval})
    skb_results[label_name] = pd.DataFrame(rows)

# ============================================================
# 2. PCA — ranks original features by weighted loading magnitude
# ============================================================
print('\n\n' + '='*60)
print('PCA — top original features by weighted component loading')
print('(how much each feature drives the top principal components,')
print(' weighted by how much variance each component explains)')
print('='*60)

pca = PCA(n_components=min(X.shape))
pca.fit(X)

print(f'\nVariance explained by top 20 components:')
cumvar = np.cumsum(pca.explained_variance_ratio_)
for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar), 1):
    bar = '█' * int(var * 200)
    print(f'  PC{i:<3} {var*100:5.1f}%  (cumulative {cum*100:5.1f}%)  {bar}')

# Weight each component's loadings by its explained variance
weights = pca.explained_variance_ratio_           # shape: (20,)
loadings = np.abs(pca.components_)               # shape: (20, 88)
weighted_importance = (loadings * weights[:, None]).sum(axis=0)  # shape: (88,)

ranked_pca = np.argsort(weighted_importance)[::-1]

print(f'\n--- Overall feature importance from PCA ---')
print(f'{"Rank":<5} {"Feature":<45} {"Weighted loading":>16}')
print('-' * 68)
pca_rows = []
for rank, idx in enumerate(ranked_pca, 1):
    fname = feature_cols[idx]
    importance = weighted_importance[idx]
    print(f'{rank:<5} {fname:<45} {importance:>16.4f}')
    pca_rows.append({'rank': rank, 'feature': fname, 'weighted_loading': importance})

pca_df = pd.DataFrame(pca_rows)

# ============================================================
# 3. Save all results
# ============================================================
os.makedirs('feature_rankings', exist_ok=True)
for label_name, df_skb in skb_results.items():
    fname = label_name.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
    df_skb.to_csv(f'feature_rankings/{fname}.csv', index=False)
pca_df.to_csv('feature_rankings/PCA_all_labels.csv', index=False)

print('\n\nSaved to feature_rankings/')

# ============================================================
# 4. Features appearing in top 10 across multiple methods
# ============================================================
print('\n' + '='*60)
print('OVERLAP — features in top 10 across 2+ methods')
print('='*60)

all_top10 = {}
for label_name, df_skb in skb_results.items():
    for feat in df_skb.head(10)['feature']:
        all_top10[feat] = all_top10.get(feat, []) + [label_name[:10]]

pca_top10 = set(pca_df.head(10)['feature'])
for feat in pca_top10:
    all_top10[feat] = all_top10.get(feat, []) + ['PCA']

consistent = {f: v for f, v in all_top10.items() if len(v) >= 2}
for feat, sources in sorted(consistent.items(), key=lambda x: -len(x[1])):
    print(f'  {feat:<45} appears in {len(sources)} methods: {", ".join(sources)}')

# ============================================================
# 5. Generate summary.csv cross-referencing all 4 methods
# ============================================================
binary_saved  = pd.read_csv('feature_rankings/Binary_straight_vs._non-straight.csv')
three_saved   = pd.read_csv('feature_rankings/3-class_straight_-_bi_-_gay.csv')
ordinal_saved = pd.read_csv('feature_rankings/Ordinal_1-5_attraction_scale.csv')
pca_saved     = pd.read_csv('feature_rankings/PCA_all_labels.csv')

def lookup(df, feat, col):
    row = df[df['feature'] == feat]
    return row[col].values[0] if len(row) else None

summary_rows = []
for feat in feature_cols:
    row = {'feature': feat}
    row['binary_rank']          = lookup(binary_saved,  feat, 'rank')
    row['binary_f_score']       = lookup(binary_saved,  feat, 'f_score')
    row['binary_p_value']       = lookup(binary_saved,  feat, 'p_value')
    row['threeclass_rank']      = lookup(three_saved,   feat, 'rank')
    row['threeclass_f_score']   = lookup(three_saved,   feat, 'f_score')
    row['threeclass_p_value']   = lookup(three_saved,   feat, 'p_value')
    row['ordinal_rank']         = lookup(ordinal_saved, feat, 'rank')
    row['ordinal_f_score']      = lookup(ordinal_saved, feat, 'f_score')
    row['ordinal_p_value']      = lookup(ordinal_saved, feat, 'p_value')
    row['pca_rank']             = lookup(pca_saved,     feat, 'rank')
    row['pca_weighted_loading'] = lookup(pca_saved,     feat, 'weighted_loading')
    ranks = [r for r in [row['binary_rank'], row['threeclass_rank'],
                         row['ordinal_rank'], row['pca_rank']] if r is not None]
    row['avg_rank']    = round(sum(ranks) / len(ranks), 2) if ranks else None
    row['top10_count'] = sum(1 for r in ranks if r <= 10)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values('avg_rank').reset_index(drop=True)
summary_df.to_csv('feature_rankings/summary.csv', index=False)
print('\nSaved feature_rankings/summary.csv')