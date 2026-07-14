import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from common import SPEAKERS_CSV, PROC

ALPHAS = np.logspace(-1, 3, 12)


def model(kind):
    est = (RidgeCV(alphas=ALPHAS) if kind == 'ridge'
           else ElasticNetCV(l1_ratio=[0.5, 0.9, 0.99], n_alphas=30, cv=5,
                             max_iter=5000, random_state=0))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def oof(kind, X, y):
    n = len(y); o = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        o[i] = model(kind).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return o


sp = pd.read_csv(SPEAKERS_CSV)
seg = pd.read_csv(PROC / 'segmental_speaker.csv')
df = sp.merge(seg, on='file_id', how='left', suffixes=('', '_seg'))

print("S_cog ALONE (single feature):")
for tgt, col in [('perceived', 'perceived_mean'), ('actual', 'Kinsey Scale (1-5)')]:
    d = df.dropna(subset=[col]); y = d[col].to_numpy(float)
    X = d[['S_cog']].to_numpy(float)
    for kind in ['ridge', 'enet']:
        o = oof(kind, X, y); rho = spearmanr(o, y).statistic
        r2 = 1 - ((y - o) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rng = np.random.default_rng(0); nulls = np.empty(1000)
        for k in range(1000):
            yp = rng.permutation(y)
            nulls[k] = spearmanr(oof(kind, X, yp), yp).statistic
        p = (1 + int((nulls >= rho).sum())) / 1001
        print(f"  {tgt:10} {kind:5}  rho={rho:+.3f}  R2={r2:+.3f}  p={p:.4f}")

# also: raw Spearman of S_cog vs target (no model at all)
print("raw Spearman(S_cog, target):")
for tgt, col in [('perceived', 'perceived_mean'), ('actual', 'Kinsey Scale (1-5)')]:
    d = df.dropna(subset=[col])
    r = spearmanr(d['S_cog'], d[col])
    print(f"  {tgt:10} rho={r.statistic:+.3f}  p={r.pvalue:.4f}")
