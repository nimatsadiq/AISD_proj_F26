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

# Phase 1 planned four missingness features. Two of them, `is_gap` and
# `months_since_obs`, turn out to be constant inside the modelling set: a gap
# month has no price, so it has no trailing mean, so it has no label, so it is
# never a modelled row. The MNAR signal survives through the two features
# below, which describe gaps in a row's recent history rather than in the row
# itself. See the Phase 2 decision log.
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


def _column_transformer(scale: bool):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "ratios",
                Pipeline(
                    [("cap", RatioCapper(columns=RATIO_FEATURES))] + numeric_steps
                ),
                RATIO_FEATURES,
            ),
            ("other_numeric", Pipeline(numeric_steps), MISSINGNESS_FEATURES + CYCLICAL_FEATURES),
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
            ("prep", _column_transformer(scale=True)),
            (
                "model",
                LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )


def build_tree_pipeline(**kwargs):
    """Gradient boosted trees, unscaled.

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
