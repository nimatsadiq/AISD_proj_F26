"""Custom transformers for the food price spike early warning pipeline.

Every transformer is scikit-learn compatible (fit / transform) so it can sit
inside a Pipeline and be fitted strictly on training folds.

Design notes carried over from the Phase 1 strategy report:

* Price levels are never capped. An extreme upward movement is the target
  event, so winsorising prices would delete positive cases from the data.
  Only engineered ratio features are capped (RatioCapper), because an
  absurd ratio is an artefact of construction, not market behaviour.
* Gaps in the monthly grid are treated as MNAR and encoded as features
  rather than imputed away (MissingnessIndicator).
* Target encoding is fitted on training data only (OutOfFoldTargetEncoder).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------

IN_SCOPE_STATES = ["Borno", "Yobe", "Adamawa"]

#: WFP categories whose commodities spoil within days or weeks. Perishables
#: respond to immediate local supply; storables respond to harvests, stored
#: reserves and the exchange rate.
PERISHABLE_CATEGORIES = {"vegetables and fruits", "meat, fish and eggs"}

#: Fuel rows are not food. They are removed from the prediction target and
#: re-joined as a state-level transport-cost feature instead.
FUEL_COMMODITIES = {"Fuel (diesel)", "Fuel (petrol-gasoline)"}


def classify_food_type(category: str) -> str:
    """Return 'perishable' or 'storable' for a WFP category."""
    return "perishable" if category in PERISHABLE_CATEGORIES else "storable"


# --------------------------------------------------------------------------
# Stage 1: reshape
# --------------------------------------------------------------------------


class PanelBuilder(BaseEstimator, TransformerMixin):
    """Reshape long WFP records into a (state, commodity, month) panel.

    Three things happen here:

    1. Filter to retail prices, in-scope states, and the modelling period.
    2. Collapse markets within a state to the **median**, not the mean.
       Market-level prices within a state are right-skewed: one market in a
       besieged area can print several times the state norm, and a mean would
       transmit that excursion into the state series and manufacture a spike
       most of the state never experienced.
    3. Reindex each state-commodity series onto a continuous monthly grid, so
       a month with no observation becomes an explicit gap rather than an
       absent row. This is what makes missingness visible at all: the raw
       file has no null cells because WFP publishes only observed records.
    """

    def __init__(self, states=None, start="2016-01-01", pricetype="Retail"):
        self.states = states
        self.start = start
        self.pricetype = pricetype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        states = self.states if self.states is not None else IN_SCOPE_STATES
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])

        df = df[
            (df["pricetype"] == self.pricetype)
            & (df["admin1"].isin(states))
            & (df["date"] >= pd.Timestamp(self.start))
        ]

        # Fuel is held back as a covariate, not predicted.
        fuel = df[df["commodity"].isin(FUEL_COMMODITIES)]
        food = df[~df["commodity"].isin(FUEL_COMMODITIES)]

        panel = (
            food.groupby(
                ["admin1", "commodity", "category", pd.Grouper(key="date", freq="MS")]
            )["price"]
            .median()
            .reset_index()
            .rename(columns={"date": "month", "admin1": "state"})
        )

        # Continuous monthly grid per series.
        frames = []
        for (state, commodity), g in panel.groupby(["state", "commodity"]):
            category = g["category"].iloc[0]
            g = g.set_index("month").asfreq("MS")
            g["state"] = state
            g["commodity"] = commodity
            g["category"] = category
            frames.append(g.reset_index())
        out = pd.concat(frames, ignore_index=True)

        out["food_type"] = out["category"].map(classify_food_type)

        # State-level fuel price, median across markets and fuel types.
        if len(fuel):
            fuel_m = (
                fuel.groupby(["admin1", pd.Grouper(key="date", freq="MS")])["price"]
                .median()
                .reset_index()
                .rename(columns={"date": "month", "admin1": "state", "price": "fuel_price"})
            )
            out = out.merge(fuel_m, on=["state", "month"], how="left")
        else:
            out["fuel_price"] = np.nan

        return out.sort_values(["state", "commodity", "month"]).reset_index(drop=True)


class LabelBuilder(BaseEstimator, TransformerMixin):
    """Attach the binary target: a spike in month t+1.

    y = 1 when the price in month t+1 exceeds its trailing three-month mean
    (months t-2 .. t) by more than `threshold`.

    The trend-based definition catches sustained climbs that no single
    month-on-month change would flag: a household is squeezed by three months
    of 6% rises just as surely as by one month of 18%.
    """

    def __init__(self, threshold=0.15, window=3):
        self.threshold = threshold
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        grp = df.groupby(["state", "commodity"])["price"]

        # Trailing mean of months t-2..t, evaluated at t.
        df["trail_mean"] = grp.transform(
            lambda s: s.rolling(self.window, min_periods=self.window).mean()
        )
        # Price in t+1 and the trailing mean as at t.
        df["price_next"] = grp.shift(-1)

        ratio_next = df["price_next"] / df["trail_mean"]
        df["spike_next_month"] = np.where(
            ratio_next.notna(), (ratio_next > 1 + self.threshold).astype(float), np.nan
        )
        return df


# --------------------------------------------------------------------------
# Stage 2: features (all strictly backward-looking from month t)
# --------------------------------------------------------------------------


class LagFeatureBuilder(BaseEstimator, TransformerMixin):
    """Backward-looking price features.

    Every window looks strictly backwards from month t, and the label is
    built from month t+1. Combined with a forward chronological split this
    means a training row can never draw on a test-period price: a row at
    time t only touches months <= t, all of which precede the cutoff.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        g = df.groupby(["state", "commodity"])

        df["price_lag1"] = g["price"].shift(1)
        df["price_lag3"] = g["price"].shift(3)
        df["price_lag12"] = g["price"].shift(12)

        # Momentum
        df["mom_1m"] = df["price"] / df["price_lag1"] - 1
        df["mom_3m"] = df["price"] / df["price_lag3"] - 1
        df["yoy"] = df["price"] / df["price_lag12"] - 1

        # Position against trend
        df["roll_mean_6"] = g["price"].transform(lambda s: s.rolling(6, min_periods=3).mean())
        df["roll_mean_12"] = g["price"].transform(lambda s: s.rolling(12, min_periods=6).mean())
        df["dev_from_6m"] = df["price"] / df["roll_mean_6"] - 1
        df["dev_from_12m"] = df["price"] / df["roll_mean_12"] - 1

        # Recent volatility of monthly changes
        df["volatility_6m"] = g["mom_1m"].transform(
            lambda s: s.rolling(6, min_periods=3).std()
        )

        # Same month last year, to carry the planting and harvest cycle
        df["mom_same_month_last_year"] = g["mom_1m"].shift(12)

        # Transport cost
        df["fuel_mom_1m"] = g["fuel_price"].transform(lambda s: s / s.shift(1) - 1)
        df["fuel_mom_3m"] = g["fuel_price"].transform(lambda s: s / s.shift(3) - 1)

        return df


class CyclicalMonthEncoder(BaseEstimator, TransformerMixin):
    """Encode calendar month on a circle.

    Ordinal encoding would assert that December (11) is maximally distant
    from January (0) when they are adjacent. Sine and cosine preserve the
    wrap-around, which matters because the harvest cycle is the dominant
    seasonal driver.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        m = pd.to_datetime(df["month"]).dt.month
        df["month_num"] = m
        df["month_sin"] = np.sin(2 * np.pi * m / 12)
        df["month_cos"] = np.cos(2 * np.pi * m / 12)
        return df


class NeighbourPriceFeatures(BaseEstimator, TransformerMixin):
    """Mean recent price change for the same commodity in other in-scope states.

    Price shocks propagate along trade routes, so a move next door is a clue
    about the month ahead. Computed from month t only, like every other
    feature, and the state's own value is excluded.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = (
            df.groupby(["commodity", "month"])["mom_1m"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "_s", "count": "_n"})
            .reset_index()
        )
        df = df.merge(agg, on=["commodity", "month"], how="left")
        own = df["mom_1m"].fillna(0)
        own_n = df["mom_1m"].notna().astype(int)
        denom = (df["_n"] - own_n).replace(0, np.nan)
        df["neighbour_mom_1m"] = (df["_s"] - own) / denom
        return df.drop(columns=["_s", "_n"])


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Flag gap months and distance from the last observation.

    Gaps are MNAR: a market that stops reporting is often a market under
    stress, and stress is when prices spike. So absence carries signal and is
    retained as a feature instead of being silently imputed away.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["is_gap"] = df["price"].isna().astype(int)

        def months_since(s):
            out, last = [], np.nan
            for i, v in enumerate(s):
                if not np.isnan(v):
                    last = i
                out.append(np.nan if np.isnan(last) else i - last)
            return pd.Series(out, index=s.index)

        df["months_since_obs"] = (
            df.groupby(["state", "commodity"])["price"].transform(months_since)
        )
        df["gaps_last_6m"] = df.groupby(["state", "commodity"])["is_gap"].transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
        return df


class ShortGapInterpolator(BaseEstimator, TransformerMixin):
    """Interpolate single-month gaps only, and record where it happened.

    A one-month hole between two observations is unlikely to conceal a whole
    spike cycle. A longer silence is left alone: bridging it would invent a
    smooth price path through exactly the period most likely to contain a
    spike, destroying the label and fabricating the features.
    """

    def __init__(self, max_gap=1):
        self.max_gap = max_gap

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        filled = df.groupby(["state", "commodity"])["price"].transform(
            lambda s: s.interpolate(limit=self.max_gap, limit_area="inside")
        )
        df["price_was_interpolated"] = (df["price"].isna() & filled.notna()).astype(int)
        df["price"] = filled
        return df


# --------------------------------------------------------------------------
# Stage 3: fitted transformers (these learn parameters, so fold discipline)
# --------------------------------------------------------------------------


class RatioCapper(BaseEstimator, TransformerMixin):
    """Cap engineered ratio features at training percentiles.

    This is the one place capping applies. A month-on-month ratio computed
    across a long gap can produce an absurd multiple; that is an artefact of
    construction, not observed market behaviour. Bounds are learned on the
    training fold only.
    """

    def __init__(self, columns=None, lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        self.bounds_ = {
            c: (X[c].quantile(self.lower), X[c].quantile(self.upper))
            for c in cols
            if c in X.columns
        }
        return self

    def transform(self, X):
        df = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            df[c] = df[c].clip(lo, hi)
        return df

    def get_feature_names_out(self, input_features=None):
        """Capping does not rename or add columns."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(list(self.bounds_), dtype=object)


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding for high-cardinality commodity.

    One-hot on 37 commodities against ~6,700 rows would add 37 sparse columns
    and overfit rare commodities. Baseline spike rates differ sharply by
    commodity, so a single encoded column carries that signal compactly.

    Fitted on the training fold only. Fitting on the full dataset would embed
    test-period spike rates into a training feature, which is the most likely
    way leakage enters this pipeline unnoticed.
    """

    def __init__(self, column="commodity", smoothing=20.0):
        self.column = column
        self.smoothing = smoothing

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = y.mean()
        stats = y.groupby(X[self.column]).agg(["mean", "count"])
        self.mapping_ = (
            (stats["mean"] * stats["count"] + self.prior_ * self.smoothing)
            / (stats["count"] + self.smoothing)
        ).to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df[f"{self.column}_te"] = (
            df[self.column].map(self.mapping_).fillna(self.prior_)
        )
        return df
