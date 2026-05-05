"""
Phishing Email Detection System — Training Pipeline
====================================================
Group ID  : 2026/CSE/SEC-B/G14
Members   : Vandit Jain · Rishabh Kumar · Siddharth Panchal · Purnima Ahalawat
Supervisor: Mr. Santosh Upadhyay — AKGEC, Ghaziabad
AKTU      : Dr. A.P.J. Abdul Kalam Technical University, Lucknow

Features
--------
* Enron spam/ham dataset with automatic download via kaggle CLI (falls back
  to rich synthetic data when Kaggle credentials are not available)
* Strict text cleaning: HTML / URL / email / number normalisation
* 12 hand-crafted heuristic features + TF-IDF (1-2 grams) via FeatureUnion
* SMOTEENN-style class balancing with lexical perturbation + ENN pruning
* Logistic Regression (L2, C=1.0, balanced)  &  Random Forest (200 trees)
* 5-fold stratified cross-validation with F1-macro scoring
* Full metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  per-class report, CV mean/std — written to model_metrics.json

Run:  python train_model.py
      python train_model.py --dataset /path/to/enron_spam_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# Shared feature engineering — imported so train and app share the same class
from features import HeuristicFeatures, URGENCY_WORDS  # noqa: F401

# Candidate CSV file names searched automatically
_ENRON_CANDIDATES = [
    "enron_spam_data.csv",
    "enron-spam-data.csv",
    "spam.csv",
    "emails.csv",
    "email.csv",
    "phishing_email.csv",
    "dataset.csv",
]


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING — Enron first, synthetic fallback
# ══════════════════════════════════════════════════════════════

def _try_kaggle_download() -> Path | None:
    """
    Attempt to download the Enron spam dataset via the Kaggle CLI.
    Returns path to the CSV if successful, None otherwise.
    Requires ~/.kaggle/kaggle.json with valid credentials.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "wanderfj/enron-spam",
                "--unzip", "--quiet",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            for name in _ENRON_CANDIDATES:
                p = Path(name)
                if p.exists():
                    log.info(f"Kaggle download succeeded → {p}")
                    return p
    except Exception as exc:
        log.debug(f"Kaggle download skipped: {exc}")
    return None


def _load_csv_as_dataframe(filepath: str | Path) -> pd.DataFrame:
    """
    Robustly load an Enron-format CSV.
    Handles common column naming variations and label encodings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Normalise text column ──────────────────────────────────────
    for alias in ("message", "body", "content", "email_body", "email", "mail"):
        if alias in df.columns and "text" not in df.columns:
            df.rename(columns={alias: "text"}, inplace=True)
            break

    # ── Normalise label column ────────────────────────────────────
    for alias in ("spam/ham", "spam", "class", "category", "type", "is_spam", "label_num"):
        if alias in df.columns and "label" not in df.columns:
            df.rename(columns={alias: "label"}, inplace=True)
            break

    if "text" not in df.columns:
        raise ValueError(
            f"Cannot find text column. Available columns: {df.columns.tolist()}"
        )
    if "label" not in df.columns:
        raise ValueError(
            f"Cannot find label column. Available columns: {df.columns.tolist()}"
        )

    # ── Encode labels ─────────────────────────────────────────────
    def encode_label(x) -> str:
        s = str(x).strip().lower()
        if s in ("spam", "1", "phishing", "yes", "true") and s not in ("ham",):
            return "Phishing"
        return "Safe"

    df["label"] = df["label"].apply(encode_label)
    df = df[["text", "label"]].dropna()

    phish_n = (df["label"] == "Phishing").sum()
    safe_n  = (df["label"] == "Safe").sum()
    log.info(
        f"Loaded {path.name}: {len(df):,} rows  |  "
        f"Phishing={phish_n:,}  Safe={safe_n:,}"
    )
    return df


def generate_synthetic_data(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Rich synthetic dataset used when no real data is available.
    Covers broad vocabulary so TF-IDF and heuristics have meaningful signal.
    Intentionally NOT identical copies — each sample gets lexical variation.
    """
    phishing_templates = [
        "URGENT: Your {kw} requires immediate verification to avoid permanent suspension.",
        "Congratulations! You have won a {kw}. Click here now to claim your reward.",
        "Security alert: Unauthorised login detected on your {kw}. Verify immediately.",
        "Your {kw} has been LIMITED. Update your information to restore access.",
        "Dear customer, your {kw} expires today. Act now to avoid unexpected charges.",
        "ACTION REQUIRED: Confirm your {kw} within 24 hours or it will be closed.",
        "We noticed suspicious activity on your {kw}. Click to secure your account.",
        "Your {kw} is about to be suspended. Verify billing to continue using our service.",
        "FINAL WARNING: Update your {kw} password immediately to avoid loss of access.",
        "You qualify for a FREE {kw}! Claim your prize before the offer expires today.",
        "IMPORTANT: Unusual sign-in to your {kw} detected. Confirm it was you now.",
        "Your {kw} payment failed. Update your card details or service will be cut off.",
        "Security update required for {kw}. Failure to comply will result in suspension.",
        "Lucky draw winner: Claim your guaranteed {kw} reward by clicking the link below.",
        "Alert: Your {kw} will expire in 2 hours. Renew now to avoid service disruption.",
        "We detected unusual access to your {kw}. Login now to review suspicious activity.",
        "NOTICE: {kw} account locked due to multiple failed login attempts. Verify now.",
        "Your {kw} refund of £312.50 is ready. Update bank details to receive it immediately.",
    ]
    safe_templates = [
        "Hi, just checking if you are free for the {kw} on Friday afternoon.",
        "Please find the {kw} attached for your review and approval.",
        "Reminder: the team {kw} is scheduled for 3 pm in the main conference room.",
        "Hope you enjoyed the {kw} last weekend! See you at the next one.",
        "Could you send over the {kw} notes from our last session when you get a chance?",
        "Following up on the {kw} we discussed — let me know if you have any questions.",
        "The {kw} report is ready. I have shared it with the whole team on the drive.",
        "Just wanted to say the {kw} yesterday was excellent — great job everyone.",
        "Can we reschedule the {kw}? I have a conflict on Tuesday afternoon.",
        "Thanks for your help with the {kw} — it went really smoothly.",
        "Looking forward to seeing you at the {kw} next week!",
        "I have completed the {kw} review — no major issues found, we are good to go.",
        "Quick update on the {kw}: everything is on track for delivery this Friday.",
        "The {kw} budget has been approved — we can proceed with phase two.",
        "Let's grab coffee after the {kw} to catch up properly.",
        "The {kw} is attached. Please review and send back your feedback by Thursday.",
        "We are planning the {kw} for next Tuesday at 10am, hope you can join.",
    ]
    phishing_kw = [
        "account", "bank card", "PayPal login", "password", "billing info",
        "credit card", "Apple ID", "Amazon account", "NHS number", "tax refund",
        "Netflix subscription", "Microsoft account", "Google account", "HMRC refund",
        "Dropbox storage", "bank account", "debit card", "email address",
        "iTunes account", "Facebook login", "Instagram account", "eBay account",
    ]
    safe_kw = [
        "meeting", "lunch", "presentation", "project update", "invoice",
        "schedule", "report", "training session", "call", "document",
        "quarterly review", "team standup", "sprint planning", "code review",
        "budget proposal", "client demo", "workshop", "onboarding session",
        "annual leave form", "performance review", "project proposal",
    ]

    rng = random.Random(random_state)
    np.random.seed(random_state)
    data: dict[str, list] = {"text": [], "label": []}

    for _ in range(n // 2):
        tmpl = rng.choice(phishing_templates)
        kw   = rng.choice(phishing_kw)
        text = tmpl.replace("{kw}", kw)
        # Randomly inject a malicious URL (~65 % of samples)
        if rng.random() < 0.65:
            domain = rng.choice(["secure-login", "verify-account", "confirm-now",
                                  "update-info", "billing-portal"])
            tld    = rng.choice(["net", "xyz", "info", "tk"])
            text  += f" http://{domain}-{rng.randint(100,999)}.{tld}/validate"
        # Randomly ALL-CAPS some words (~30 %)
        if rng.random() < 0.30:
            words  = text.split()
            n_caps = max(1, int(len(words) * 0.15))
            idxs   = rng.sample(range(len(words)), n_caps)
            for i in idxs:
                words[i] = words[i].upper()
            text = " ".join(words)
        data["text"].append(text)
        data["label"].append("Phishing")

    for _ in range(n // 2):
        text = rng.choice(safe_templates).replace("{kw}", rng.choice(safe_kw))
        data["text"].append(text)
        data["label"].append("Safe")

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    log.info(f"Synthetic dataset: {len(df):,} samples generated.")
    return df


def load_data(cli_path: str | None = None) -> tuple[pd.DataFrame, str]:
    """
    Data source priority:
    1. Path passed via --dataset CLI argument
    2. Any known CSV file present in the working directory
    3. Kaggle auto-download (requires ~/.kaggle/kaggle.json)
    4. Synthetic fallback

    Returns (dataframe, source_label).
    """
    # 1. CLI path
    if cli_path:
        df = _load_csv_as_dataframe(cli_path)
        return df, f"CSV ({cli_path})"

    # 2. Local file scan
    for fname in _ENRON_CANDIDATES:
        if Path(fname).exists():
            try:
                df = _load_csv_as_dataframe(fname)
                return df, f"Enron CSV ({fname})"
            except Exception as exc:
                log.warning(f"Skipping {fname}: {exc}")

    # 3. Kaggle download
    log.info(
        "No local dataset found.  "
        "Attempting Kaggle download (requires ~/.kaggle/kaggle.json)…"
    )
    kaggle_path = _try_kaggle_download()
    if kaggle_path:
        df = _load_csv_as_dataframe(kaggle_path)
        return df, f"Enron via Kaggle ({kaggle_path.name})"

    # 4. Synthetic fallback
    log.warning(
        "No Enron dataset available.\n"
        "  → To use real data, place enron_spam_data.csv in the working directory,\n"
        "    or install the Kaggle CLI: pip install kaggle  and add your API key.\n"
        "  → Falling back to synthetic data for demonstration purposes."
    )
    df = generate_synthetic_data()
    return df, "Synthetic (no real dataset found)"


# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════

_URL_RE        = re.compile(r"https?://\S+|www\.\S+", re.I)
_HTML_RE       = re.compile(r"<[^>]+>")
_EMAIL_RE      = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_NUM_RE        = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_RE    = re.compile(r"[^\w\s]")   # remove punctuation after placeholders


def clean_text(text: str) -> str:
    """
    Normalise raw email text for feature extraction.
    Order matters: HTML → URL → email → numeric → punctuation → whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = _EMAIL_RE.sub(" EMAIL ", text)
    text = _NUM_RE.sub(" NUM ", text)
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, drop empty/duplicate rows, enforce minimum length."""
    df = df.copy()
    df["text"]  = df["text"].fillna("").apply(clean_text)
    df["label"] = df["label"].str.strip()

    before = len(df)
    df = df[df["text"].str.len() >= 15].reset_index(drop=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    after = len(df)

    if before - after:
        log.info(f"Removed {before - after:,} short/duplicate rows → {after:,} remaining.")
    return df


# ══════════════════════════════════════════════════════════════
# 3. CLASS BALANCING (SMOTEENN-style, text domain)
# ══════════════════════════════════════════════════════════════

def smoteenn_balance(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    SMOTEENN-style balancing for text corpora:

    Step 1 — SMOTE analogue (oversample minority):
        Generate synthetic minority-class samples by randomly dropping 10–25 %
        of words from existing minority texts.  This creates diverse near-
        duplicates rather than exact copies, preventing the classifier from
        memorising single sentences.

    Step 2 — ENN analogue (clean noisy majority):
        Remove majority-class samples that contain no urgency vocabulary AND
        no URL token.  These samples lie near the class boundary and add noise
        to the decision boundary.

    Only activates when the imbalance ratio exceeds 1.25 : 1.
    """
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Dataset must contain at least two classes (Phishing and Safe).")

    maj_label = counts.idxmax()
    min_label = counts.idxmin()
    ratio     = counts[maj_label] / counts[min_label]

    if ratio <= 1.25:
        log.info(f"Classes balanced (ratio={ratio:.2f}) — SMOTEENN skipped.")
        return df

    log.info(
        f"Imbalance ratio={ratio:.2f}  "
        f"{maj_label}={counts[maj_label]:,}  {min_label}={counts[min_label]:,}"
    )

    rng          = random.Random(random_state)
    minority_txts = df.loc[df["label"] == min_label, "text"].tolist()

    # ── Step 1: Minority oversampling ─────────────────────────────
    needed   = counts[maj_label] - len(minority_txts)
    synthetics: list[str] = []

    for _ in range(needed):
        base  = rng.choice(minority_txts)
        words = base.split()
        if len(words) > 4:
            drop_n  = max(1, int(len(words) * rng.uniform(0.10, 0.25)))
            drop_set = set(rng.sample(range(len(words)), drop_n))
            words   = [w for i, w in enumerate(words) if i not in drop_set]
        synthetics.append(" ".join(words) if words else base)

    synthetic_df = pd.DataFrame({"text": synthetics, "label": min_label})

    # ── Step 2: Noisy majority removal ───────────────────────────
    maj_mask = df["label"] == maj_label
    noisy    = df[maj_mask]["text"].apply(
        lambda t: (
            not any(w in URGENCY_WORDS for w in t.split())
            and "url" not in t
        )
    )
    n_removed = noisy.sum()
    log.info(f"ENN step: removing {n_removed:,} noisy majority samples.")
    cleaned_maj = df[maj_mask][~noisy]

    balanced = (
        pd.concat(
            [cleaned_maj, df[df["label"] == min_label], synthetic_df],
            ignore_index=True,
        )
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    new_counts = balanced["label"].value_counts()
    log.info(f"Post-SMOTEENN counts: {new_counts.to_dict()}")
    return balanced


# ══════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════


def build_feature_pipeline() -> FeatureUnion:
    """
    TF-IDF (unigrams + bigrams, sublinear TF, up to 30 000 features)
    fused with 12 hand-crafted heuristic features via FeatureUnion.
    """
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        analyzer="word",
        stop_words="english",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
    )
    return FeatureUnion([
        ("tfidf",      tfidf),
        ("heuristics", HeuristicFeatures()),
    ])


# ══════════════════════════════════════════════════════════════
# 5. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

def build_models(feature_pipeline: FeatureUnion) -> dict[str, Pipeline]:
    """
    Two production-quality pipelines sharing the same FeatureUnion.

    Logistic Regression — L2 regularised, lbfgs solver, class_weight balanced.
        C=1.0 is a sensible default; increase for less regularisation on large
        Enron corpora, decrease if overfitting on synthetic data.

    Random Forest — 200 trees, max_depth=None (full trees), balanced weights,
        bootstrap=True, n_jobs=-1 (all CPU cores).  oob_score gives a free
        out-of-bag estimate without extra CV overhead.
    """
    return {
        "Logistic Regression": Pipeline([
            ("features", feature_pipeline),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
                tol=1e-4,
            )),
        ]),
        "Random Forest": Pipeline([
            ("features", feature_pipeline),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,         # full trees — regularised by min_samples_leaf
                min_samples_leaf=2,
                class_weight="balanced",
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ══════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(name: str, model: Pipeline, X_test, y_test) -> dict:
    """Full evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix."""
    y_pred = model.predict(X_test)

    # Probability of 'Phishing' class for ROC-AUC
    classes    = list(model.classes_)
    phish_idx  = classes.index("Phishing")
    y_prob     = model.predict_proba(X_test)[:, phish_idx]
    y_bin      = (y_test == "Phishing").astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Phishing", zero_division=0)
    rec  = recall_score(y_test, y_pred,    pos_label="Phishing", zero_division=0)
    f1   = f1_score(y_test, y_pred,        pos_label="Phishing", zero_division=0)
    auc  = roc_auc_score(y_bin, y_prob)
    cm   = confusion_matrix(y_test, y_pred, labels=["Phishing", "Safe"]).tolist()

    sep = "═" * 60
    log.info(f"\n{sep}\n{name}\n{classification_report(y_test, y_pred)}\n{sep}")

    # OOB score for Random Forest (free estimate, no extra cost)
    oob_score = None
    clf = model.named_steps.get("clf")
    if hasattr(clf, "oob_score_"):
        oob_score = round(float(clf.oob_score_), 4)
        log.info(f"  OOB accuracy (RF): {oob_score:.4f}")

    result: dict = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "roc_auc":          round(auc,  4),
        "confusion_matrix": cm,
    }
    if oob_score is not None:
        result["oob_score"] = oob_score

    return result


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhishGuard phishing detection models."
    )
    parser.add_argument(
        "--dataset", "-d",
        metavar="CSV",
        default=None,
        help="Path to Enron spam CSV (optional; auto-detected if omitted).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data reserved for testing (default: 0.20).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    return parser.parse_args()


def train_and_save(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    log.info("=" * 60)
    log.info("PhishGuard Training Pipeline — Group G13 · AKGEC 2026")
    log.info("=" * 60)

    # ── Load data ────────────────────────────────────────────
    df, source_label = load_data(args.dataset)
    log.info(f"Dataset source: {source_label}")

    # ── Preprocess ───────────────────────────────────────────
    df = preprocess_dataframe(df)
    log.info(f"After preprocessing: {len(df):,} samples")

    # ── Validate minimum dataset size ────────────────────────
    if len(df) < 100:
        raise RuntimeError(
            f"Dataset too small ({len(df)} samples). "
            "Need at least 100 samples for reliable training."
        )

    # ── SMOTEENN balancing ───────────────────────────────────
    df = smoteenn_balance(df)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    log.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Build & train models ─────────────────────────────────
    features = build_feature_pipeline()
    models   = build_models(features)
    metrics: dict = {}

    for name, pipeline in models.items():
        log.info(f"\n── Training: {name} ──")
        pipeline.fit(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())
        log.info(
            f"  {args.cv_folds}-Fold CV F1-macro: "
            f"{cv_mean:.4f} ± {cv_std:.4f}  "
            f"(min={cv_scores.min():.4f}, max={cv_scores.max():.4f})"
        )

        m             = evaluate(name, pipeline, X_test, y_test)
        m["cv_f1_mean"]     = round(cv_mean, 4)
        m["cv_f1_std"]      = round(cv_std,  4)
        m["cv_f1_per_fold"] = [round(s, 4) for s in cv_scores.tolist()]
        m["dataset_source"] = source_label
        m["train_samples"]  = int(len(X_train))
        m["test_samples"]   = int(len(X_test))
        metrics[name]       = m

        # Persist model
        fname = f"model_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipeline, fname, compress=3)
        log.info(f"  Saved → {fname}")

    # ── Save metrics ─────────────────────────────────────────
    with open("model_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("\n✅  All models trained.  Metrics written to model_metrics.json.")
    log.info("    Launch app: streamlit run app.py")


if __name__ == "__main__":
    train_and_save()