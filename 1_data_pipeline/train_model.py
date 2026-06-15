"""
train_model.py — Feature engineering + XGBoost passage probability model.

Reads bills_clean.csv (produced by rag_pipeline.py) and trains an XGBoost
classifier to predict whether a bill will pass. Saves the model to
passage_predictor.pkl for use in the Streamlit app.

Steps:
  1. Load data
  2. Feature engineering
  3. Train/test split
  4. Handle class imbalance (scale_pos_weight)
  5. Train XGBoost
  6. Evaluate (F1, ROC-AUC, confusion matrix)
  7. 5-fold cross-validation
  8. SHAP feature importance
  9. Retrain on full dataset + save model

Usage:
  python train_model.py
  python train_model.py --data data/bills_clean.csv --output data/passage_predictor.pkl
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from config import (
    CLEAN_CSV_FILE,
    MODEL_OUTPUT_FILE,
    FEATURES,
    BILL_TYPE_MAP,
    RESOLUTION_TYPES,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features to the bills dataframe.

    Features cover text length, bill type, sponsor politics, committee
    engagement, and topic keyword flags.

    Args:
        df: Raw bills dataframe with at minimum 'clean_text', 'title',
            'bill_type', 'sponsor_chamber', 'sponsor_party',
            'num_committees', 'bypassed_committee', 'is_naming_bill' columns.

    Returns:
        DataFrame with all FEATURES columns populated.
    """
    text = df["clean_text"].fillna("")

    # Text-based structural features
    df["text_length"]      = text.apply(len)
    df["word_count"]       = text.apply(lambda x: len(x.split()))
    df["title_word_count"] = df["title"].fillna("").apply(lambda x: len(x.split()))

    # Bill type features
    df["is_resolution"] = df["bill_type"].isin(RESOLUTION_TYPES).astype(int)
    df["bill_type_enc"] = df["bill_type"].map(BILL_TYPE_MAP).fillna(8).astype(int)

    # Sponsor political features
    df["is_senate"]     = (df["sponsor_chamber"] == "Senate").astype(int)
    df["is_democrat"]   = (df["sponsor_party"] == "D").astype(int)
    df["is_republican"] = (df["sponsor_party"] == "R").astype(int)

    # Topic keyword flags (does bill text mention these domains?)
    keyword_flags = {
        "has_appropriation": r'approp|fund|billion|million|\$\s*\d',
        "has_health":        r'health|medical|medicare|medicaid|hospital',
        "has_defense":       r'defense|military|armed forces|veteran|national security',
        "has_education":     r'education|school|student|university|college',
    }
    for col, pattern in keyword_flags.items():
        df[col] = text.str.lower().str.contains(pattern, regex=True).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print classification report, ROC-AUC, and confusion matrix."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    log.info("\n=== CLASSIFICATION REPORT ===\n%s",
             classification_report(y_test, y_pred, target_names=["Not Passed", "Passed"]))
    log.info("ROC-AUC: %.3f", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    log.info("\n=== CONFUSION MATRIX ===")
    log.info("                  Pred Not Passed  Pred Passed")
    log.info("Actual Not Passed      %6d      %6d", cm[0][0], cm[0][1])
    log.info("Actual Passed          %6d      %6d", cm[1][0], cm[1][1])


def print_shap_importance(model, X_test: pd.DataFrame) -> None:
    """Print SHAP-based feature importance as a bar chart in the terminal."""
    log.info("\n=== FEATURE IMPORTANCE (SHAP) ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEATURES,
    ).sort_values(ascending=False)

    for feat, val in mean_shap.items():
        bar = "█" * int(val * 100)
        log.info("  %-25s %.4f  %s", feat, val, bar)

    log.info("\nKey insight: word_count and text_length typically rank above party — "
             "structural features outweigh politics for bill passage prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train(data_path: Path, output_path: Path) -> None:
    """
    Full training pipeline: load → features → split → train → evaluate → save.

    Args:
        data_path:   Path to bills_clean.csv.
        output_path: Where to save the passage_predictor.pkl file.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Loading data from %s ...", data_path)
    df = pd.read_csv(data_path)
    log.info("Loaded %d bills.", len(df))

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("Engineering features...")
    df = engineer_features(df)

    X = df[FEATURES]
    y = df["passed"]  # binary: 1 = passed, 0 = not passed

    log.info("Feature matrix: %s", X.shape)
    log.info("Passed: %d | Not passed: %d", y.sum(), (y == 0).sum())

    # ── 3. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── 4. Class imbalance weight ─────────────────────────────────────────────
    # Without this, XGBoost predicts "not passed" almost every time.
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    log.info("scale_pos_weight: %.2f", scale_pos_weight)

    # ── 5. Train XGBoost ──────────────────────────────────────────────────────
    log.info("Training XGBoost (%d estimators)...", XGB_N_ESTIMATORS)
    model = XGBClassifier(
        n_estimators     = XGB_N_ESTIMATORS,
        max_depth        = XGB_MAX_DEPTH,
        learning_rate    = XGB_LEARNING_RATE,
        scale_pos_weight = scale_pos_weight,
        random_state     = RANDOM_STATE,
        eval_metric      = "logloss",
        verbosity        = 0,
    )
    model.fit(X_train, y_train)
    log.info("Training complete.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print_evaluation(model, X_test, y_test)

    # ── 7. Cross-validation ───────────────────────────────────────────────────
    log.info("Running %d-fold cross-validation...", CV_FOLDS)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    log.info("F1 per fold: %s", cv_scores.round(3))
    log.info("Mean F1: %.3f (+/- %.3f)", cv_scores.mean(), cv_scores.std())

    # ── 8. SHAP importance ────────────────────────────────────────────────────
    print_shap_importance(model, X_test)

    # ── 9. Retrain on full dataset + save ─────────────────────────────────────
    log.info("Retraining on full dataset for production model...")
    model.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "features": FEATURES}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    log.info("Model saved to: %s", output_path)
    log.info("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost bill passage predictor."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=CLEAN_CSV_FILE,
        help="Path to bills_clean.csv (produced by rag_pipeline.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_OUTPUT_FILE,
        help="Where to save the trained model .pkl file",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            "Run rag_pipeline.py first to produce bills_clean.csv."
        )

    train(args.data, args.output)


if __name__ == "__main__":
    main()
