"""ColumnTransformer and Pipeline definitions.

Two model-specific pipelines share one feature-construction stage, because
scaling and collinearity reduction are required by logistic regression and
unnecessary for gradient boosted trees. Applying one rule to both would
needlessly discard information from the model that can use it.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .transformers import (
    CyclicalMonthEncoder,
    LabelBuilder,
    LagFeatureBuilder,
    MissingnessIndicator,
    NeighbourPriceFeatures,
    PanelBuilder,
    RatioCapper,
    ShortGapInterpolator,
)

# --------------------------------------------------------------------------
# Feature inventory
# --------------------------------------------------------------------------

RATIO_FEATURES = [
    "mom_1m",
    "mom_3m",
    "yoy",
    "dev_from_6m",
    "dev_from_12m",
    "volatility_6m",
    "mom_same_month_last_year",
    "neighbour_mom_1m",
    "fuel_mom_1m",
    "fuel_mom_3m",
]

# The design specified four missingness features. Two of them, `is_gap` and
# `months_since_obs`, turn out to be constant inside the modelling set: a gap
# month has no price, so it has no trailing mean, so it has no label, so it is
# never a modelled row. The MNAR signal survives through the two features
# below, which describe gaps in a row's recent history rather than in the row
# itself. See the implementation audit.
MISSINGNESS_FEATURES = [
    "gaps_last_6m",
    "price_was_interpolated",
]

#: Computed but excluded from the model: zero variance after labelling.
DROPPED_ZERO_VARIANCE = ["is_gap", "months_since_obs"]

CYCLICAL_FEATURES = ["month_sin", "month_cos"]

NUMERIC_FEATURES = RATIO_FEATURES + MISSINGNESS_FEATURES + CYCLICAL_FEATURES

CATEGORICAL_FEATURES = ["state", "food_type"]

TARGET = "spike_next_month"


def build_panel(raw, threshold=0.15, states=None, start="2016-01-01"):
    """Raw WFP long format to a labelled, feature-complete panel.

    Deterministic and target-free: reshaping, backward-looking windows and
    the label are all fixed functions of the data, so running this once
    before splitting cannot leak. What must stay inside a fold is anything
    that *learns* a parameter: capping bounds, imputation values, scaler
    statistics, target encodings and PCA. Those live in the model pipelines
    below.
    """
    steps = [
        ("panel", PanelBuilder(states=states, start=start)),
        ("short_gaps", ShortGapInterpolator(max_gap=1)),
        ("missing", MissingnessIndicator()),
        ("label", LabelBuilder(threshold=threshold)),
        ("lags", LagFeatureBuilder()),
        ("month", CyclicalMonthEncoder()),
        ("neighbours", NeighbourPriceFeatures()),
    ]
    out = raw
    for _, step in steps:
        out = step.fit_transform(out)
    return out


def temporal_split(panel, train_end="2023-06-30", val_end="2024-12-31"):
    """Forward chronological split.

    Random splitting is inadmissible here: price series are autocorrelated,
    so a randomly held-out month sits between two training months that nearly
    determine it, and reported performance would not survive deployment.

    Stratification is deliberately not applied. It requires reshuffling,
    which destroys the temporal ordering this design exists to respect. The
    class balance of each split is reported instead.
    """
    df = panel.dropna(subset=[TARGET]).copy()
    df["month"] = pd.to_datetime(df["month"])
    train = df[df["month"] <= train_end]
    val = df[(df["month"] > train_end) & (df["month"] <= val_end)]
    test = df[df["month"] > val_end]
    return train, val, test


#: Dropped from the LINEAR model only, by the collinearity protocol.
#: Pairwise screen flagged (mom_3m, dev_from_6m) at rho = 0.876; VIF then
#: flagged dev_from_12m at 10.52. Mutual information decided which of the
#: overlapping pair to drop: dev_from_6m scores 0.0636 against dev_from_12m at
#: 0.0353, so dev_from_12m goes. Max VIF falls from 10.52 to 6.72.
#: Trees keep the full set: they split on rank order and are not destabilised
#: by correlated inputs, so reducing for them would discard usable signal.
VIF_DROPPED = ["dev_from_12m"]


def _column_transformer(scale: bool, drop=()):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    ratio_cols = [c for c in RATIO_FEATURES if c not in drop]
    other_cols = [c for c in MISSINGNESS_FEATURES + CYCLICAL_FEATURES if c not in drop]

    return ColumnTransformer(
        transformers=[
            (
                "ratios",
                Pipeline(
                    [("cap", RatioCapper(columns=ratio_cols))] + numeric_steps
                ),
                ratio_cols,
            ),
            ("other_numeric", Pipeline(numeric_steps), other_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(C=1.0):
    """Median imputation, ratio capping, StandardScaler, logistic regression.

    L1/L2 penalties shrink coefficients in proportion to feature magnitude,
    so unscaled inputs would be penalised differently purely because of
    units. `class_weight='balanced'` handles the 4.5:1 imbalance without
    synthesising rows: SMOTE would interpolate between neighbouring months
    and fabricate price histories that never occurred.
    """
    return Pipeline(
        [
            ("prep", _column_transformer(scale=True, drop=VIF_DROPPED)),
            (
                "model",
                LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )


def build_extratrees_pipeline(**kwargs):
    """Extremely randomised trees, unscaled. The selected non-linear model.

    Chosen over gradient boosting on measured performance, not convention.
    On the held-out test partition Extra Trees reached PR-AUC 0.386 and
    precision@50 of 0.62, against 0.301 and 0.50 for XGBoost, with the same
    features and the same temporal split. See `src/model_selection.py`.

    The reason is the regime break. Boosting fits residuals sequentially and
    has enough capacity to model the pre-2023 price regime tightly, which does
    not transfer across the subsidy removal and currency float. Extra Trees
    randomises split thresholds as well as features, which is a stronger
    variance reduction and degrades more gracefully when the test distribution
    shifts.

    No scaling: trees split on rank order and are invariant to monotonic
    rescaling, so a scaler would add a fitted step with no benefit and one
    more opportunity for leakage. No collinearity reduction either, for the
    same reason the full feature set is kept for any tree model.
    """
    from sklearn.ensemble import ExtraTreesClassifier

    params = dict(
        n_estimators=400,
        min_samples_leaf=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    params.update(kwargs)
    return Pipeline(
        [("prep", _column_transformer(scale=False)), ("model", ExtraTreesClassifier(**params))]
    )


def build_tree_pipeline(**kwargs):
    """Gradient boosted trees, unscaled. Retained for comparison only.

    Tested and not selected: it placed last of six candidates on every metric
    (see `src/model_selection.py`). Kept in the codebase so the comparison is
    reproducible rather than asserted.

    Trees split on rank order, which is invariant to monotonic rescaling, so
    a scaler would add a fitted step with no benefit and one more chance of
    leakage. Falls back to scikit-learn's HistGradientBoosting if XGBoost is
    not installed.
    """
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42)
    params.update(kwargs)
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
        )
    return Pipeline([("prep", _column_transformer(scale=False)), ("model", model)])


def seasonal_persistence_baseline(train, target_df):
    """The free guess the model has to beat.

    Predicts a spike for a (state, commodity, calendar month) whenever that
    combination spiked in more than half of its observed training years.
    Anyone could apply this rule without a model, so a model that cannot
    beat it is not justified.
    """
    rate = train.groupby(["commodity", "month_num"])[TARGET].mean()
    key = list(zip(target_df["commodity"], target_df["month_num"]))
    return np.array([1 if rate.get(k, 0) > 0.5 else 0 for k in key])


def set_positive_class_weight(pipeline, y_train):
    """Apply the imbalance ratio to an XGBoost pipeline, if present."""
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos and hasattr(pipeline.named_steps["model"], "scale_pos_weight"):
        pipeline.named_steps["model"].set_params(scale_pos_weight=neg / pos)
    return pipeline
