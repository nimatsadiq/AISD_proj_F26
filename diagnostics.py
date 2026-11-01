"""Statistical diagnostics backing each preprocessing decision.

Every function here produces a number that a written claim can cite. Nothing
is estimated by eye: the skewness that justifies a log transform, the VIF that
justifies dropping a feature, and the MCAR test that justifies treating gaps
as informative are all computed on the training split and reported.

Training-split-only is not a stylistic choice. A diagnostic computed on the
full dataset is a decision informed by the test period, which is leakage by a
slower route than fitting a scaler on it.
"""

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Missingness
# --------------------------------------------------------------------------


def littles_mcar_test(df, columns):
    """Little's (1988) test of the MCAR hypothesis.

    Groups rows by their missingness pattern and asks whether the observed
    means differ across patterns more than chance allows. A low p-value
    rejects MCAR, meaning missingness is related to the data itself.

    The statistic is
        d2 = sum over patterns j of  n_j * (xbar_j - mu)' S^-1 (xbar_j - mu)
    restricted to the variables observed in pattern j, with
        df = sum_j (observed vars in j) - (number of vars)

    Returns a dict with the statistic, degrees of freedom, p-value and the
    pattern count. Rejecting MCAR does not by itself distinguish MAR from
    MNAR; that argument has to come from how the data was generated.
    """
    X = df[columns].copy()
    n_vars = len(columns)

    mu = X.mean()
    S = X.cov()
    # Ridge the covariance so it inverts even when features are collinear.
    S_inv = np.linalg.pinv(S.values + np.eye(n_vars) * 1e-8)

    pattern = X.isna().astype(int).astype(str).agg("".join, axis=1)

    d2 = 0.0
    df_total = 0
    n_patterns = 0
    for _, idx in X.groupby(pattern).groups.items():
        block = X.loc[idx]
        observed = [i for i, c in enumerate(columns) if block[c].notna().all()]
        if not observed or len(block) < 2:
            continue
        n_j = len(block)
        diff = (block.iloc[:, observed].mean() - mu.iloc[observed]).values
        S_sub = np.linalg.pinv(S.values[np.ix_(observed, observed)]
                               + np.eye(len(observed)) * 1e-8)
        d2 += n_j * float(diff @ S_sub @ diff)
        df_total += len(observed)
        n_patterns += 1

    dof = max(df_total - n_vars, 1)
    p = float(stats.chi2.sf(d2, dof))
    return {
        "statistic": round(float(d2), 2),
        "df": int(dof),
        "p_value": p,
        "n_patterns": int(n_patterns),
        "n_variables": int(n_vars),
        "mcar_rejected": bool(p < 0.05),
    }


def cramers_v(x, y):
    """Cramer's V for association between two categorical variables.

    Reported alongside chi-square because chi-square significance is inflated
    at large n: with 4,956 training rows almost any association is
    "significant". V is an effect size on [0, 1] and says whether it matters.
    """
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    # Bias correction (Bergsma, 2013)
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    v = float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
    p = float(stats.chi2_contingency(table)[1])
    return {"cramers_v": round(v, 4), "chi2": round(float(chi2), 2), "p_value": p}


def missingness_association(panel, gap_col="is_gap"):
    """Is the gap pattern associated with state, commodity or season?

    An association supports structured rather than completely random
    missingness, which is the empirical half of the MNAR argument.
    """
    out = {}
    d = panel.copy()
    d["month_num"] = pd.to_datetime(d["month"]).dt.month
    for col in ["state", "commodity", "food_type", "month_num"]:
        if col in d.columns:
            out[col] = cramers_v(d[col], d[gap_col])
    return out


# --------------------------------------------------------------------------
# Distribution shape and outliers
# --------------------------------------------------------------------------


def distribution_shape(series, label=""):
    """Skewness and kurtosis, the numbers that decide IQR versus Z-score."""
    s = pd.Series(series).dropna()
    return {
        "feature": label,
        "n": int(len(s)),
        "skewness": round(float(stats.skew(s)), 3),
        "kurtosis": round(float(stats.kurtosis(s)), 3),
        "mean": round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "std": round(float(s.std()), 3),
    }


def iqr_fence(series, k=1.5):
    """Tukey fence bounds plus how many values fall outside them."""
    s = pd.Series(series).dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    flagged = int(((s < lo) | (s > hi)).sum())
    return {
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(iqr), 3),
        "lower_fence": round(float(lo), 3),
        "upper_fence": round(float(hi), 3),
        "n_flagged": flagged,
        "pct_flagged": round(100 * flagged / len(s), 2),
    }


def transformation_effect(series):
    """Skewness through the transformation chain, for the Figure 4.1 claim."""
    s = pd.Series(series).dropna()
    s = s[s > 0]
    log_s = np.log1p(s)
    return {
        "skew_raw": round(float(stats.skew(s)), 3),
        "kurtosis_raw": round(float(stats.kurtosis(s)), 3),
        "skew_log1p": round(float(stats.skew(log_s)), 3),
        "kurtosis_log1p": round(float(stats.kurtosis(log_s)), 3),
        "skew_standardised": round(
            float(stats.skew((log_s - log_s.mean()) / log_s.std())), 3
        ),
    }


# --------------------------------------------------------------------------
# Collinearity
# --------------------------------------------------------------------------


def variance_inflation(df, columns):
    """VIF per feature: 1 / (1 - R^2) from regressing it on the others.

    Implemented from the correlation matrix inverse rather than by fitting
    k regressions, which is equivalent and far faster. Values above 10 are
    the conventional drop threshold.
    """
    X = df[columns].dropna()
    corr = X.corr().values
    inv = np.linalg.pinv(corr)
    vifs = np.diag(inv)
    return (
        pd.DataFrame({"feature": columns, "VIF": np.round(vifs, 2)})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def collinear_pairs(df, columns, threshold=0.85, method="spearman"):
    """Feature pairs above the correlation threshold."""
    corr = df[columns].corr(method=method)
    out = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = corr.values[i, j]
            if abs(r) >= threshold:
                out.append({"feature_a": columns[i], "feature_b": columns[j],
                            "rho": round(float(r), 4)})
    return out


# --------------------------------------------------------------------------
# Feature-target association
# --------------------------------------------------------------------------


def target_association(df, columns, target):
    """Point-biserial r, Mann-Whitney U and mutual information per feature.

    Three tests because one is not enough: point-biserial assumes a linear
    relationship, Mann-Whitney assumes only a monotonic shift, and mutual
    information assumes nothing. A feature that matters to the trees but not
    to the logistic model shows up as low r and high MI.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer

    y = df[target].astype(int)
    rows = []
    for c in columns:
        sub = df[[c, target]].dropna()
        if sub[c].nunique() < 2:
            continue
        yy = sub[target].astype(int)
        r, p_r = stats.pointbiserialr(yy, sub[c])
        g1, g0 = sub.loc[yy == 1, c], sub.loc[yy == 0, c]
        if len(g1) and len(g0):
            u, p_u = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        else:
            u, p_u = np.nan, np.nan
        rows.append({"feature": c, "point_biserial_r": round(float(r), 4),
                     "r_p_value": float(p_r), "mannwhitney_U": float(u),
                     "u_p_value": float(p_u)})
    res = pd.DataFrame(rows)

    usable = [c for c in columns if df[c].nunique() >= 2]
    Xi = SimpleImputer(strategy="median").fit_transform(df[usable])
    mi = pd.Series(mutual_info_classif(Xi, y, random_state=42), index=usable)
    res["mutual_information"] = res["feature"].map(mi).round(4)
    return res.sort_values("mutual_information", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Dimensionality
# --------------------------------------------------------------------------


def pca_variance(df, columns, cutoffs=(0.90, 0.95)):
    """Eigenvalues, variance ratios and the components needed per cutoff."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X = SimpleImputer(strategy="median").fit_transform(df[columns])
    X = StandardScaler().fit_transform(X)
    pca = PCA().fit(X)
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    return {
        "n_features": len(columns),
        "eigenvalues": [round(float(v), 4) for v in pca.explained_variance_],
        "variance_ratios": [round(float(v), 4) for v in ratios],
        "cumulative": [round(float(v), 4) for v in cum],
        "components_for": {f"{int(c*100)}%": int(np.searchsorted(cum, c) + 1)
                           for c in cutoffs},
        "pc1_share": round(float(ratios[0]), 4),
    }
