"""Post-hoc probability calibration under weighted loss.

Why this is needed. Both models correct the 4.3:1 imbalance by reweighting the
loss (`class_weight='balanced'`, `scale_pos_weight`). That deliberately shifts
the decision boundary toward the minority class, which is the intended effect on
*ranking* and the unintended effect on *probabilities*: a reported score of 0.70
no longer means "70% of such months spike". It is systematically inflated.

Two consequences for this project:

1. The dashboard shows a probability to a state official. An uncalibrated 0.70
   is not a number anyone should act on as a probability.
2. A decision threshold chosen on uncalibrated scores has no absolute meaning
   and will not transfer if the model is retrained under a different class
   weight.

What calibration does *not* change: isotonic and Platt scaling are monotonic, so
the ranking is identical and precision@k is unchanged. Calibration changes what
a displayed number means, not which months are ranked highest.

The calibrator is fitted on a held-out split and never on training data, for the
same reason every other fitted transformer is.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate(fitted_model, X_cal, y_cal, method="isotonic"):
    """Wrap an already-fitted model with a calibrator fitted on held-out data.

    `cv="prefit"` tells scikit-learn the base estimator is already trained and
    that `X_cal` is to be used only for learning the probability mapping. This
    is the out-of-fold requirement: the calibration split must not be any part
    of what the model was trained on.

    isotonic is the default because it is non-parametric and the distortion
    from weighted loss is not guaranteed to be the sigmoid shape Platt scaling
    assumes. With fewer than ~1,000 calibration rows, prefer method="sigmoid"
    (Platt), which has fewer degrees of freedom and overfits less.
    """
    try:  # scikit-learn >= 1.6 replaced cv="prefit" with FrozenEstimator
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def calibration_report(y_true, p_raw, p_cal, n_bins=10):
    """Compare raw and calibrated scores: Brier plus a reliability table.

    Perfect calibration means that among months scored near 0.3, close to 30%
    actually spike. The table below makes that directly checkable.
    """
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        for label, p in (("raw", p_raw), ("calibrated", p_cal)):
            m = (p >= lo) & (p < hi)
            if m.sum() == 0:
                continue
            rows.append({
                "version": label,
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_predicted": round(float(p[m].mean()), 3),
                "observed_rate": round(float(np.asarray(y_true)[m].mean()), 3),
            })
    summary = {
        "brier_raw": round(brier_score_loss(y_true, p_raw), 4),
        "brier_calibrated": round(brier_score_loss(y_true, p_cal), 4),
        "mean_predicted_raw": round(float(np.mean(p_raw)), 3),
        "mean_predicted_calibrated": round(float(np.mean(p_cal)), 3),
        "actual_base_rate": round(float(np.mean(y_true)), 3),
    }
    return summary, rows
