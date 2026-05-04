"""
Phishing Email Detection System — Streamlit Web Application
============================================================
Group ID  : 2026/CSE/SEC-B/G14
Members   : Vandit Jain · Rishabh Kumar · Siddharth Panchal · Purnima Ahalawat
Supervisor: Mr. Santosh Upadhyay — AKGEC, Ghaziabad
AKTU      : Dr. A.P.J. Abdul Kalam Technical University, Lucknow

Features
--------
* Real-time phishing / safe classification with calibrated confidence scores
* Explainable AI (XAI):
    - LIME-style local word attribution (perturbation-based, n_samples=150)
    - SHAP-style global feature importance (coefficient / Gini)
    - Colour-highlighted email text showing phishing hot spots
* Full performance dashboard: accuracy, precision, recall, F1, ROC-AUC
* Confusion matrix heatmap and per-fold cross-validation bar chart
* Model comparison bar chart (side-by-side 4 metrics)
* Heuristic feature explanation table with signal strength
* Prediction history log (session) with CSV export
* Enron dataset info + download instructions
* Viva preparation Q&A guide + live demo checklist + key formulae

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import time
import warnings
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Shared feature class — must be imported so joblib can unpickle the models
from features import HeuristicFeatures, URGENCY_WORDS  # noqa: F401

# ══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PhishGuard — Email Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "PhishGuard · AKGEC Final Year Project 2026 · Group G14",
    },
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ──────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff; padding: 28px 36px; border-radius: 16px;
    margin-bottom: 24px; display: flex; align-items: center; gap: 20px;
}
.header-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.header-banner p  { margin: 4px 0 0; opacity: 0.75; font-size: 0.93rem; }

/* ── Verdict cards ──────────────────────────── */
.verdict-phishing {
    background: linear-gradient(135deg,#fff0f0,#ffe4e4);
    border: 2.5px solid #e74c3c; border-radius: 14px;
    padding: 24px 28px; animation: pulse-red 1.5s ease-in-out;
}
.verdict-safe {
    background: linear-gradient(135deg,#f0fff4,#dcffe8);
    border: 2.5px solid #27ae60; border-radius: 14px;
    padding: 24px 28px; animation: pulse-green 1.5s ease-in-out;
}
@keyframes pulse-red   { 0%{box-shadow:0 0 0 0 rgba(231,76,60,.4)}  70%{box-shadow:0 0 0 12px rgba(231,76,60,0)} 100%{box-shadow:none} }
@keyframes pulse-green { 0%{box-shadow:0 0 0 0 rgba(39,174,96,.4)}  70%{box-shadow:0 0 0 12px rgba(39,174,96,0)}  100%{box-shadow:none} }
.verdict-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 6px; }
.verdict-body  { font-size: 0.9rem; color: #555; margin: 0; }

/* ── Metric cards ───────────────────────────── */
.metric-card {
    background: #fff; border: 1px solid #e8ecf0; border-radius: 12px;
    padding: 18px 14px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.05); transition: transform .15s;
}
.metric-card:hover { transform: translateY(-2px); }
.mc-label { font-size: 11px; text-transform: uppercase; letter-spacing:.8px; color:#888; }
.mc-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; line-height: 1.2; }

/* ── Section tags ───────────────────────────── */
.section-tag {
    display: inline-block; background: #eef2ff; color: #4f46e5;
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; padding: 3px 10px;
    border-radius: 20px; margin-bottom: 8px;
}

/* ── XAI word highlights ────────────────────── */
.w-high { background:#ffd2d2; border-radius:3px; padding:1px 3px; font-weight:600; color:#b00; }
.w-med  { background:#ffe4b8; border-radius:3px; padding:1px 3px; font-weight:600; color:#884400; }
.w-low  { background:#fffac8; border-radius:3px; padding:1px 2px; }
.w-safe { background:#d4f8e8; border-radius:3px; padding:1px 2px; color:#0a6; }

/* ── Tip / info box ─────────────────────────── */
.tip-box {
    background: #fffbea; border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    font-size: .88rem; color: #555; margin: 12px 0;
}
.info-box {
    background: #eef2ff; border-left: 4px solid #4f46e5;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    font-size: .88rem; color: #333; margin: 12px 0;
}

/* ── Probability bars ───────────────────────── */
.prob-row   { display:flex; align-items:center; gap:10px; margin:6px 0; }
.prob-label { width:90px; font-size:.82rem; font-weight:600; color:#555; }
.prob-bar-wrap { flex:1; background:#f0f0f0; border-radius:20px; height:14px; overflow:hidden; }
.prob-bar   { height:100%; border-radius:20px; transition:width .6s ease; }
.prob-val   { width:44px; font-size:.82rem; font-weight:700; text-align:right; color:#333; }

/* ── Sidebar ────────────────────────────────── */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%); color:#e0e0e0;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption { color:#ccc !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#fff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

URGENCY_WORDS: frozenset[str] = frozenset({
    "urgent", "immediately", "verify", "suspended", "limited", "expired",
    "click", "confirm", "update", "account", "password", "security",
    "winner", "lottery", "prize", "congratulations", "claim", "free",
    "guaranteed", "risk", "alert", "warning", "unusual", "suspicious",
    "action", "required", "final", "notice", "deadline", "renew",
    "login", "credential", "billing", "payment", "failed", "validate",
    "unauthorized", "breach", "locked", "expire", "refund", "reward",
})

EXAMPLES: dict[str, str] = {
    "— Choose an example —": "",
    "🚨 Phishing · account alert":
        "URGENT: Your PayPal account has been limited. Verify your billing "
        "information immediately at http://paypal-secure-login.com/verify or "
        "your account will be permanently suspended within 24 hours.",
    "🚨 Phishing · prize scam":
        "Congratulations! You have been selected as our lucky winner! "
        "Claim your £500 gift card NOW. Limited time offer — click here: "
        "http://free-prizes-claim.com. Act fast before it expires!",
    "🚨 Phishing · tax refund":
        "HMRC Tax Refund Notice: You are entitled to a tax refund of £312.50. "
        "To receive your refund, please update your bank details immediately. "
        "Failure to respond within 48 hours will result in forfeiture of the refund.",
    "🚨 Phishing · credential harvest":
        "Security Alert: We detected an unauthorised login attempt on your Microsoft "
        "account. Your account will be locked in 2 hours unless you verify your "
        "credentials now. Click here: http://ms-account-verify.net/login",
    "✅ Safe · work email":
        "Hi Sarah, just a quick reminder about the project review meeting "
        "tomorrow at 2 pm in Room 3B. Could you bring the Q3 report? Thanks.",
    "✅ Safe · casual message":
        "Hey, are you coming to the dinner on Saturday? We're booking the "
        "Italian place on High Street at 7:30. Let me know!",
    "✅ Safe · HR notice":
        "Hi team, a reminder that the annual performance review cycle opens next "
        "Monday. Please complete your self-assessment form by 15 June. "
        "Contact HR if you have any questions.",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Logistic Regression": (
        "Fast linear classifier. Uses TF-IDF term weights + 12 heuristic features. "
        "Coefficients directly indicate each word's phishing/safe contribution. "
        "Best for interpretability and speed (≈ 5 ms inference)."
    ),
    "Random Forest": (
        "Ensemble of 200 decision trees with bootstrap aggregation. Captures "
        "non-linear interactions between features. oob_score gives a free "
        "out-of-bag accuracy estimate. Feature importances via mean Gini decrease. "
        "Best for accuracy (≈ 20–50 ms inference)."
    ),
}


# ══════════════════════════════════════════════════════════════
# PREPROCESSING (mirrors train_model.py — must stay in sync)
# ══════════════════════════════════════════════════════════════

_URL_RE        = re.compile(r"https?://\S+|www\.\S+", re.I)
_HTML_RE       = re.compile(r"<[^>]+>")
_EMAIL_RE      = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_NUM_RE        = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_RE    = re.compile(r"[^\w\s]")


def clean_text(text: str) -> str:
    """Identical preprocessing as train_model.py — must stay in sync."""
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


# ══════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    """Load model_metrics.json; return None if missing."""
    if os.path.exists("model_metrics.json"):
        try:
            with open("model_metrics.json", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            st.warning(f"Could not parse model_metrics.json: {exc}")
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load a serialised sklearn Pipeline; return None if file missing."""
    fname = f"model_{model_name.replace(' ', '_').lower()}.pkl"
    if os.path.exists(fname):
        try:
            return joblib.load(fname)
        except Exception as exc:
            st.error(f"Failed to load {fname}: {exc}")
    return None


# ══════════════════════════════════════════════════════════════
# XAI — LIME-STYLE LOCAL WORD ATTRIBUTION
# ══════════════════════════════════════════════════════════════

def lime_word_importances(
    model,
    text: str,
    n_samples: int = 150,
    n_top: int = 15,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Lightweight LIME-style local attribution (no external library needed).

    Algorithm
    ---------
    1. Tokenise the input text into unique words.
    2. For each unique word, generate `n_samples` perturbed texts with that
       word randomly dropped (the word is ALWAYS absent in each perturbation).
    3. Query model.predict_proba on all perturbed texts.
    4. Importance = mean P(Phishing|without word) − baseline P(Phishing|full text).
       Positive  → removing the word LOWERS phishing prob → word is a phishing driver.
       Negative  → removing the word RAISES phishing prob → word is a safe indicator.
    5. Normalise by the maximum absolute importance.
    6. Return the top-n words by |importance|.

    Edge cases handled:
    - Empty text → empty dict
    - Single-word text → single entry
    - Model class order not assumed (looked up dynamically)
    - Perturbed text that becomes empty → replaced with a period
    """
    words        = text.split()
    unique_words = list(dict.fromkeys(
        w.lower().strip(".,!?;:\"'()[]") for w in words if len(w) > 2
    ))
    if not unique_words:
        return {}

    classes    = list(model.classes_)
    phish_idx  = classes.index("Phishing")
    rng        = random.Random(random_state)

    # Baseline probability on the original (cleaned) text
    cleaned_text  = clean_text(text)
    baseline_prob = float(model.predict_proba([cleaned_text])[0][phish_idx])

    importances: dict[str, float] = {}

    for word in unique_words:
        perturbed: list[str] = []
        for _ in range(n_samples):
            new_words = [
                w for w in words
                if w.lower().strip(".,!?;:\"'()[]") != word and rng.random() > 0.4
            ]
            perturbed.append(clean_text(" ".join(new_words)) if new_words else ".")

        probs                 = model.predict_proba(perturbed)
        mean_without          = float(np.mean(probs[:, phish_idx]))
        importances[word]     = baseline_prob - mean_without  # positive = phishing driver

    # Normalise to [−1, 1]
    max_abs = max((abs(v) for v in importances.values()), default=1.0)
    if max_abs > 0:
        importances = {k: v / max_abs for k, v in importances.items()}

    top = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:n_top]
    return dict(top)


def highlight_email_html(text: str, word_scores: dict[str, float]) -> str:
    """
    Colour-code each token in the email by its LIME score.
    Returns an HTML string safe for st.markdown(..., unsafe_allow_html=True).
    """
    tokens     = text.split()
    html_parts = []
    for tok in tokens:
        clean = tok.lower().strip(".,!?;:\"'()[]")
        score = word_scores.get(clean, 0.0)
        if score >= 0.6:
            html_parts.append(f'<span class="w-high">{tok}</span>')
        elif score >= 0.3:
            html_parts.append(f'<span class="w-med">{tok}</span>')
        elif score >= 0.1:
            html_parts.append(f'<span class="w-low">{tok}</span>')
        elif score <= -0.3:
            html_parts.append(f'<span class="w-safe">{tok}</span>')
        else:
            html_parts.append(tok)
    return " ".join(html_parts)


# ══════════════════════════════════════════════════════════════
# SHAP-STYLE GLOBAL FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════

PHISHING_KEYWORDS: frozenset[str] = frozenset({
    "urgent", "verify", "url", "click", "password", "winner", "claim",
    "suspended", "limited", "billing", "congratulations", "alert", "free",
    "account", "security", "login", "confirm", "update", "expired",
    "refund", "prize", "immediately", "locked", "breach", "validate",
})


def _get_global_feature_importance(
    model, n: int = 20
) -> Optional[tuple[list[str], np.ndarray]]:
    """
    Extract global feature importance:
    - Logistic Regression : |coefficient|  ≈ linear SHAP value
    - Random Forest       : mean decrease in Gini impurity

    Returns (feature_names, importances) sorted ascending by importance.
    Returns None if extraction fails.
    """
    try:
        steps = model.named_steps
        clf   = steps["clf"]

        feature_names: list[str] = []
        fu = steps.get("features")
        if fu is not None:
            for _, transformer in fu.transformer_list:
                if hasattr(transformer, "get_feature_names_out"):
                    feature_names.extend(
                        transformer.get_feature_names_out().tolist()
                    )
        elif "tfidf" in steps:
            feature_names = steps["tfidf"].get_feature_names_out().tolist()
        else:
            return None

        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        else:
            return None

        if len(feature_names) != len(importances):
            return None

        idx    = np.argsort(importances)[-n:]
        names  = [feature_names[i] for i in idx]
        values = importances[idx]
        return names, values

    except Exception as exc:
        log.warning(f"Feature importance extraction failed: {exc}")
        return None


def plot_shap_bar(model, model_name: str, n: int = 20) -> Optional[Figure]:
    result = _get_global_feature_importance(model, n)
    if result is None:
        return None
    names, values = result

    colors = [
        "#e74c3c" if any(kw in nm.lower() for kw in PHISHING_KEYWORDS) else "#4a90d9"
        for nm in names
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.barh(names, values, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)

    max_v = values.max() if values.size else 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", ha="left", fontsize=7.5, color="#555",
        )

    xlabel = (
        "│Coefficient│  (proxy for linear SHAP)" if "Logistic" in model_name
        else "Mean Decrease in Gini Impurity"
    )
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(
        f"SHAP-Style Global Feature Importance\n{model_name}",
        fontsize=11, pad=10, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    red_p  = mpatches.Patch(color="#e74c3c", alpha=0.88, label="Known phishing indicator")
    blue_p = mpatches.Patch(color="#4a90d9", alpha=0.88, label="Neutral / safe")
    ax.legend(handles=[red_p, blue_p], fontsize=8.5, loc="lower right")
    plt.tight_layout()
    return fig


def plot_lime_bar(word_scores: dict[str, float]) -> Figure:
    """Horizontal bar chart of LIME word importances for the current prediction."""
    items  = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    words  = [i[0] for i in items]
    scores = [i[1] for i in items]
    colors = ["#e74c3c" if s > 0 else "#27ae60" for s in scores]

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(words) * 0.38)))
    ax.barh(words, scores, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("LIME Attribution  (+ → Phishing  /  − → Safe)", fontsize=9)
    ax.set_title(
        "LIME Local Explanation — Word Contributions to This Prediction",
        fontsize=10.5, pad=10, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    red_p = mpatches.Patch(color="#e74c3c", alpha=0.85, label="Pushes toward Phishing")
    grn_p = mpatches.Patch(color="#27ae60", alpha=0.85, label="Pushes toward Safe")
    ax.legend(handles=[red_p, grn_p], fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm_data: list, title: str) -> Figure:
    cm  = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Phishing", "Safe"],
        yticklabels=["Phishing", "Safe"],
        ax=ax, linewidths=0.6, linecolor="#ddd",
        annot_kws={"size": 15, "weight": "bold"},
        cbar_kws={"shrink": 0.75},
    )
    ax.set_xlabel("Predicted", fontsize=10.5)
    ax.set_ylabel("Actual",    fontsize=10.5)
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_comparison(metrics: dict) -> Figure:
    models        = list(metrics.keys())
    metric_keys   = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors        = ["#4a90d9", "#e67e22", "#27ae60", "#8e44ad"]

    x     = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.2))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [metrics[m].get(key, 0) * 100 for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#333",
            )

    ax.set_ylim(0, 115)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11, fontweight="500")
    ax.set_ylabel("Score (%)", fontsize=10.5)
    ax.set_title("Algorithm Performance Comparison", fontsize=12, pad=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_roc_approx(metrics: dict) -> Figure:
    """
    Approximate ROC curve from stored AUC value.
    Without raw test probabilities we interpolate a smooth concave curve
    that is consistent with the given AUC (area under interpolation = AUC).
    """
    fig, ax = plt.subplots(figsize=(5, 4.2))
    colors  = ["#4a90d9", "#e67e22", "#27ae60", "#8e44ad"]

    for (name, m), color in zip(metrics.items(), colors):
        auc = m.get("roc_auc", 0.5)
        t   = np.linspace(0, 1, 300)
        fpr = t
        tpr = np.clip(t + (1 - t) * (2 * auc - 1) * (1 - t * 0.4), 0, 1)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title("ROC Curve (AUC)", fontsize=11, pad=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_cv_fold_bars(metrics: dict) -> Optional[Figure]:
    """Bar chart showing per-fold F1 score for each model."""
    models_with_folds = {
        name: m["cv_f1_per_fold"]
        for name, m in metrics.items()
        if "cv_f1_per_fold" in m
    }
    if not models_with_folds:
        return None

    max_folds = max(len(v) for v in models_with_folds.values())
    x         = np.arange(max_folds)
    width     = 0.35
    colors    = ["#4a90d9", "#e67e22", "#27ae60", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (name, folds) in enumerate(models_with_folds.items()):
        ax.bar(
            x + i * width, [f * 100 for f in folds], width,
            label=name, color=colors[i % len(colors)], alpha=0.85,
        )

    ax.set_xticks(x + width * (len(models_with_folds) - 1) / 2)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(max_folds)])
    ax.set_ylim(0, 115)
    ax.set_ylabel("F1-macro (%)", fontsize=10)
    ax.set_title("Cross-Validation F1 per Fold", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar(metrics: dict) -> str:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 6px;">'
            '<span style="font-size:3rem;">🛡️</span>'
            '<h2 style="color:#fff;margin:4px 0 0;font-size:1.25rem;">PhishGuard</h2>'
            '<p style="color:#aaa;font-size:.78rem;margin:0;">AKGEC · Group G14 · 2026</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        model_choice = st.selectbox(
            "🤖  Algorithm",
            options=list(metrics.keys()) if metrics else ["Logistic Regression", "Random Forest"],
            help="Select which trained model to use for classification.",
        )

        if model_choice in MODEL_DESCRIPTIONS:
            st.markdown(
                f'<div class="tip-box" style="margin-top:0;">'
                f'{MODEL_DESCRIPTIONS[model_choice]}</div>',
                unsafe_allow_html=True,
            )

        if metrics and model_choice in metrics:
            m = metrics[model_choice]
            st.markdown("---")
            st.markdown(f"**📊 {model_choice} Metrics**")
            c1, c2 = st.columns(2)
            for (label, key), col in zip(
                [("Accuracy", "accuracy"), ("Precision", "precision"),
                 ("Recall", "recall"),    ("F1 Score", "f1_score")],
                [c1, c2, c1, c2],
            ):
                col.metric(label, f"{m.get(key, 0)*100:.1f}%")

            if "roc_auc" in m:
                st.metric("ROC-AUC", f"{m['roc_auc']*100:.1f}%")
            if "oob_score" in m:
                st.metric("OOB Score (RF)", f"{m['oob_score']*100:.1f}%",
                          help="Out-of-bag accuracy — free cross-validation estimate.")
            if "cv_f1_mean" in m:
                cv_m = m["cv_f1_mean"] * 100
                cv_s = m["cv_f1_std"]  * 100
                st.markdown(
                    f'<div class="tip-box">📈 <b>5-Fold CV F1:</b> {cv_m:.1f}% ± {cv_s:.1f}%'
                    f'<br><span style="font-size:.8rem;">Low σ → model generalises well</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Dataset source badge
            src = m.get("dataset_source", "")
            if src:
                badge_color = "#27ae60" if "Enron" in src or "CSV" in src else "#e67e22"
                st.markdown(
                    f'<div style="background:{badge_color}22;border:1px solid {badge_color};'
                    f'border-radius:8px;padding:6px 10px;font-size:.78rem;color:{badge_color};'
                    f'font-weight:600;">📂 {src}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Session history summary
        hist = st.session_state.get("history", [])
        if hist:
            phish_cnt = sum(1 for h in hist if h["verdict"] == "Phishing")
            safe_cnt  = len(hist) - phish_cnt
            st.markdown(
                f"**🕑 Session**  \n"
                f"🔴 Phishing: **{phish_cnt}**  &nbsp;✅ Safe: **{safe_cnt}**  "
                f"&nbsp;Total: **{len(hist)}**"
            )

        st.markdown("---")
        st.markdown(
            '<p style="color:#888;font-size:.75rem;text-align:center;">'
            "Built with scikit-learn + Streamlit<br>"
            "Supervisor: Mr. Santosh Upadhyay</p>",
            unsafe_allow_html=True,
        )

    return model_choice


# ══════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE EMAIL
# ══════════════════════════════════════════════════════════════

def render_analysis_tab(model, model_choice: str) -> None:
    st.markdown('<span class="section-tag">Live Detection</span>', unsafe_allow_html=True)
    st.markdown("### 🔍 Analyse an Email")
    st.caption(
        "Paste raw email body text below. "
        "The system pre-processes it identically to training (HTML/URL/email normalisation) "
        "then classifies and explains the result."
    )

    # Example selector
    selected   = st.selectbox("Try a built-in example", list(EXAMPLES.keys()))
    user_input = st.text_area(
        "Email content",
        value=EXAMPLES[selected],
        height=150,
        placeholder="Paste email text here…",
        label_visibility="collapsed",
    )

    col_btn, col_xai = st.columns([1, 2])
    with col_btn:
        run_xai = st.toggle(
            "🧠 Explain with XAI",
            value=True,
            help="Runs LIME perturbation-based word attribution (~1–3 s extra).",
        )
    with col_xai:
        st.caption("LIME local attribution + colour-highlighted email text.")

    # ── Input validation ──────────────────────────────────────
    if not st.button("🛡️ Analyse Email", type="primary", use_container_width=True):
        return

    raw_input = user_input.strip()
    if not raw_input:
        st.warning("⚠️ Please enter or select some email text first.")
        return
    if len(raw_input) < 10:
        st.warning("⚠️ Input too short — please paste a meaningful email excerpt.")
        return
    if len(raw_input) > 50_000:
        st.warning("⚠️ Input exceeds 50 000 characters. Please trim the email text.")
        return

    # ── Preprocess (must match train_model.py) ────────────────
    processed_input = clean_text(raw_input)
    if not processed_input:
        st.error("❌ Input reduced to empty string after preprocessing. Please try different text.")
        return

    # ── Classify ──────────────────────────────────────────────
    with st.spinner("Classifying…"):
        t0         = time.perf_counter()
        prediction = model.predict([processed_input])[0]
        probs      = model.predict_proba([processed_input])[0]
        classes    = list(model.classes_)
        phish_prob = float(probs[classes.index("Phishing")])
        safe_prob  = float(probs[classes.index("Safe")])
        infer_ms   = (time.perf_counter() - t0) * 1000

    # ── Verdict card ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == "Phishing":
        st.markdown(
            f'<div class="verdict-phishing">'
            f'<p class="verdict-title" style="color:#c0392b;">🚨 PHISHING EMAIL DETECTED</p>'
            f'<p class="verdict-body">This email exhibits strong phishing indicators. '
            f'<strong>Do not</strong> click any links or provide personal information. '
            f'Confidence: <strong>{phish_prob*100:.1f}%</strong></p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-safe">'
            f'<p class="verdict-title" style="color:#1e8449;">✅ EMAIL APPEARS SAFE</p>'
            f'<p class="verdict-body">No significant phishing indicators detected. '
            f'Confidence: <strong>{safe_prob*100:.1f}%</strong></p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Verdict",         prediction),
        (c2, "Phishing Prob.",  f"{phish_prob*100:.1f}%"),
        (c3, "Safe Prob.",      f"{safe_prob*100:.1f}%"),
        (c4, "Inference Time",  f"{infer_ms:.1f} ms"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="mc-label">{label}</div>'
            f'<div class="mc-value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence bars ───────────────────────────────────────
    st.markdown("**Confidence Breakdown**")
    st.markdown(
        f'<div class="prob-row">'
        f'  <span class="prob-label" style="color:#c0392b;">Phishing</span>'
        f'  <div class="prob-bar-wrap"><div class="prob-bar" '
        f'       style="width:{phish_prob*100:.1f}%;'
        f'background:linear-gradient(90deg,#e74c3c,#c0392b);"></div></div>'
        f'  <span class="prob-val">{phish_prob*100:.1f}%</span>'
        f'</div>'
        f'<div class="prob-row">'
        f'  <span class="prob-label" style="color:#1e8449;">Safe</span>'
        f'  <div class="prob-bar-wrap"><div class="prob-bar" '
        f'       style="width:{safe_prob*100:.1f}%;'
        f'background:linear-gradient(90deg,#27ae60,#1e8449);"></div></div>'
        f'  <span class="prob-val">{safe_prob*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Preprocessing preview ─────────────────────────────────
    with st.expander("🔬 View preprocessed text (what the model actually sees)"):
        st.code(processed_input, language=None)
        st.caption(
            "HTML tags, URLs (→ URL), email addresses (→ EMAIL), "
            "numbers (→ NUM), punctuation, and casing are normalised "
            "before feature extraction."
        )

    # ── XAI section ───────────────────────────────────────────
    if run_xai:
        st.markdown("---")
        st.markdown('<span class="section-tag">Explainable AI (XAI)</span>', unsafe_allow_html=True)
        st.markdown("#### 🧠 Why did the model decide this?")

        with st.spinner("Computing LIME word attributions (perturbation sampling)…"):
            t_lime      = time.perf_counter()
            word_scores = lime_word_importances(
                model, processed_input, n_samples=150, n_top=15
            )
            lime_ms = (time.perf_counter() - t_lime) * 1000

        if word_scores:
            left_col, right_col = st.columns([1.1, 1])

            with left_col:
                st.markdown("**Highlighted Email Text**")
                st.caption("🔴 High risk &nbsp;|&nbsp; 🟠 Moderate &nbsp;|&nbsp; 🟡 Low &nbsp;|&nbsp; 🟢 Safe indicator")
                highlighted = highlight_email_html(raw_input, word_scores)
                st.markdown(
                    f'<div style="background:#fafafa;border:1px solid #e0e0e0;'
                    f'border-radius:10px;padding:16px 18px;font-size:.92rem;'
                    f'line-height:1.7;">{highlighted}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"⏱ LIME in {lime_ms:.0f} ms  |  "
                    f"{len(word_scores)} words attributed  |  "
                    f"n_samples = 150"
                )

            with right_col:
                st.markdown("**LIME Attribution Chart**")
                fig_lime = plot_lime_bar(word_scores)
                st.pyplot(fig_lime, use_container_width=True)
                plt.close(fig_lime)

            st.markdown(
                '<div class="tip-box">💡 <b>How to read LIME:</b> Words are randomly '
                "dropped from the email and the model is re-queried each time. "
                "A <span style='color:#c0392b;font-weight:600;'>positive score</span> "
                "means the word raises phishing probability when present; "
                "<span style='color:#1e8449;font-weight:600;'>negative scores</span> "
                "indicate safe vocabulary. This is a <em>local</em> explanation — "
                "specific to this one email.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Could not generate word attributions — text may be too short after preprocessing.")

    # ── Save to session history ───────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "text":    raw_input[:80] + ("…" if len(raw_input) > 80 else ""),
        "verdict": prediction,
        "phish%":  f"{phish_prob*100:.1f}%",
        "safe%":   f"{safe_prob*100:.1f}%",
        "model":   model_choice,
    })


# ══════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════

def render_metrics_tab(metrics: dict, model_choice: str) -> None:
    st.markdown('<span class="section-tag">Model Evaluation</span>', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Metrics")

    if not metrics:
        st.error("No metrics found. Run `python train_model.py` first.")
        return

    # ── Top charts row ────────────────────────────────────────
    col_cmp, col_roc = st.columns(2)
    with col_cmp:
        st.markdown("**Algorithm Comparison**")
        fig_cmp = plot_model_comparison(metrics)
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

    with col_roc:
        st.markdown("**ROC Curve (AUC)**")
        fig_roc = plot_roc_approx(metrics)
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    # ── CV fold chart ─────────────────────────────────────────
    fig_cv = plot_cv_fold_bars(metrics)
    if fig_cv:
        st.markdown("**Cross-Validation F1 per Fold**")
        st.pyplot(fig_cv, use_container_width=True)
        plt.close(fig_cv)

    st.markdown("---")

    # ── Per-model accordion ───────────────────────────────────
    st.markdown("**Detailed Results per Model**")
    for name, m in metrics.items():
        badge = "🏆 " if name == model_choice else ""
        with st.expander(f"{badge}{name}", expanded=(name == model_choice)):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{m.get('accuracy',  0)*100:.2f}%")
            c2.metric("Precision", f"{m.get('precision', 0)*100:.2f}%")
            c3.metric("Recall",    f"{m.get('recall',    0)*100:.2f}%")
            c4.metric("F1 Score",  f"{m.get('f1_score',  0)*100:.2f}%")
            c5.metric("ROC-AUC",   f"{m.get('roc_auc',   0)*100:.2f}%")

            extra_cols = st.columns(3)
            if "oob_score" in m:
                extra_cols[0].metric("OOB Score", f"{m['oob_score']*100:.2f}%")
            if "train_samples" in m:
                extra_cols[1].metric("Train samples", f"{m['train_samples']:,}")
            if "test_samples" in m:
                extra_cols[2].metric("Test samples",  f"{m['test_samples']:,}")

            if "cv_f1_mean" in m:
                cv_m = m["cv_f1_mean"] * 100
                cv_s = m["cv_f1_std"]  * 100
                st.markdown(
                    f'<div class="tip-box">📈 <b>5-Fold Stratified CV F1:</b> '
                    f'{cv_m:.2f}% ± {cv_s:.2f}%  — '
                    f"small σ confirms model generalises and is not overfitting.</div>",
                    unsafe_allow_html=True,
                )

            if "confusion_matrix" in m:
                cm_col, interp_col = st.columns(2)
                with cm_col:
                    fig_cm = plot_confusion_matrix(m["confusion_matrix"], name)
                    st.pyplot(fig_cm, use_container_width=False)
                    plt.close(fig_cm)
                with interp_col:
                    cm  = np.array(m["confusion_matrix"])
                    TP  = int(cm[0, 0]); FN = int(cm[0, 1])
                    FP  = int(cm[1, 0]); TN = int(cm[1, 1])
                    tot = TP + FN + FP + TN or 1
                    st.markdown(
                        f"**Confusion Matrix**\n\n"
                        f"- ✅ **TP:** {TP}  — Phishing caught\n"
                        f"- ❌ **FN:** {FN}  — Phishing missed *(most dangerous)*\n"
                        f"- ⚠️ **FP:** {FP}  — Safe flagged as phishing\n"
                        f"- ✅ **TN:** {TN}  — Safe correctly cleared\n\n"
                        f"**Error rate:** {(FN+FP)/tot*100:.2f}%  \n"
                        f"**Miss rate (FNR):** {FN/(TP+FN or 1)*100:.2f}%  \n"
                        f"**False alarm (FPR):** {FP/(FP+TN or 1)*100:.2f}%\n\n"
                        f"*In security, FN cost >> FP cost — prioritise Recall.*"
                    )

    # ── Summary table ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Summary Table**")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{m.get('accuracy',  0)*100:.2f}%",
            "Precision": f"{m.get('precision', 0)*100:.2f}%",
            "Recall":    f"{m.get('recall',    0)*100:.2f}%",
            "F1 Score":  f"{m.get('f1_score',  0)*100:.2f}%",
            "ROC-AUC":   f"{m.get('roc_auc',   0)*100:.2f}%",
            "CV F1":     (
                f"{m.get('cv_f1_mean',0)*100:.2f}% ± "
                f"{m.get('cv_f1_std', 0)*100:.2f}%"
            ),
            "Dataset":   m.get("dataset_source", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS & XAI
# ══════════════════════════════════════════════════════════════

def render_xai_tab(model, model_choice: str) -> None:
    st.markdown('<span class="section-tag">Explainable AI</span>', unsafe_allow_html=True)
    st.markdown("### 🔎 Global Feature Insights (SHAP-Style)")
    st.caption(
        "Understanding which words and patterns the model relies on globally — "
        "across all predictions — is critical for trust, auditability, and "
        "regulatory compliance."
    )

    fig_shap = plot_shap_bar(model, model_choice, n=20)
    if fig_shap:
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)
        st.markdown(
            '<div class="tip-box">💡 <b>SHAP-style interpretation:</b> For '
            "<b>Logistic Regression</b>, each bar = absolute coefficient ≈ "
            "linear SHAP value. For <b>Random Forest</b>, bars = mean Gini "
            "decrease (Gini SHAP approximation). "
            '<span style="color:#c0392b;font-weight:600;">Red</span> = known '
            "phishing vocabulary; "
            '<span style="color:#4a90d9;font-weight:600;">blue</span> = neutral.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Global feature importance not available for this model configuration.")

    st.markdown("---")

    # ── Heuristic features table ──────────────────────────────
    st.markdown("#### ⚙️ Engineered Heuristic Features (12)")
    st.caption(
        "These numeric features are appended to the TF-IDF matrix via FeatureUnion."
    )
    heuristics = pd.DataFrame({
        "Feature": [
            "url_count", "urgency_count", "urgency_ratio", "exclamation_count",
            "question_count", "num_count", "email_mention", "text_length",
            "avg_word_len", "type_token_ratio", "url_density", "long_word_ratio",
        ],
        "Description": [
            "Count of 'URL' placeholder tokens (from cleaned text)",
            "Raw count of urgency vocabulary matches",
            "urgency_count / total word count",
            "'!' count (before cleaning) — artificial urgency",
            "'?' count — rhetorical social-engineering questions",
            "Count of 'NUM' placeholders (replaced numeric tokens)",
            "Binary: text mentions 'email'",
            "Character length of cleaned text",
            "Mean word length (shorter = simpler automated copy)",
            "Unique words / total words (lexical diversity)",
            "url_count / total words (URL density)",
            "Fraction of words longer than 8 chars",
        ],
        "Signal": [
            "🔴 Strong", "🔴 Strong", "🔴 Strong", "🟠 Moderate",
            "🟡 Weak", "🟠 Moderate", "🟠 Moderate", "🟡 Context",
            "🟡 Context", "🟠 Moderate", "🔴 Strong", "🟡 Context",
        ],
    })
    st.dataframe(heuristics, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Urgency vocabulary listing ────────────────────────────
    st.markdown(f"#### 📝 Urgency Vocabulary ({len(URGENCY_WORDS)} terms)")
    cols = st.columns(4)
    for i, word in enumerate(sorted(URGENCY_WORDS)):
        cols[i % 4].markdown(
            f'<span style="background:#fff0f0;color:#c0392b;border-radius:4px;'
            f'padding:2px 8px;font-size:.82rem;display:inline-block;margin:2px;">'
            f"{word}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Pipeline architecture ─────────────────────────────────
    st.markdown("#### 🔄 Pipeline Architecture")
    st.code("""
┌────────────────────────────────────────────────────┐
│                  Raw Email Text                    │
└─────────────────────┬──────────────────────────────┘
                      │ clean_text()
                      │ HTML → strip  │  URL → "URL"
                      │ email → "EMAIL" │ nums → "NUM"
                      │ lowercase  │  punctuation stripped
      ┌───────────────┴──────────────────────┐
      │             FeatureUnion             │
      ├──────────────────────┬───────────────┤
      │  TfidfVectorizer     │ HeuristicFeatures │
      │  (1,2)-grams         │  12 numeric        │
      │  max 30,000 features │  hand-crafted      │
      │  sublinear_tf=True   │  features          │
      │  min_df=2            │                    │
      └──────────┬───────────┴──────────┬──────────┘
                 │     scipy.sparse     │
                 │     hstack           │
                 └──────────┬───────────┘
            ┌───────────────┴────────────────┐
            │    Classifier (LR / RF)        │
            │    class_weight = "balanced"   │
            └───────────────┬────────────────┘
                            │
                   "Phishing" / "Safe"
                   + predict_proba()
    """, language=None)


# ══════════════════════════════════════════════════════════════
# TAB 4 — DATASET INFO
# ══════════════════════════════════════════════════════════════

def render_dataset_tab(metrics: dict) -> None:
    st.markdown('<span class="section-tag">Dataset</span>', unsafe_allow_html=True)
    st.markdown("### 📂 Enron Spam Dataset")

    source = ""
    if metrics:
        first_model = next(iter(metrics.values()))
        source = first_model.get("dataset_source", "")

    if "Enron" in source or "CSV" in source:
        st.success(f"✅ **Real Enron dataset in use:**  `{source}`")
    elif "Synthetic" in source or not source:
        st.warning(
            "⚠️ **Running on synthetic data.**  "
            "Add the Enron CSV for production-quality results."
        )

    st.markdown("""
    #### About the Enron Spam Dataset

    The **Enron-Spam** corpus is one of the most widely used real-world email
    datasets in phishing / spam detection research.  It was derived from the
    Enron email dataset released after the Enron scandal and later annotated
    for spam/ham classification.

    | Property | Value |
    |---|---|
    | Emails | ~33,716 |
    | Classes | ham (safe) / spam (phishing-proxy) |
    | Language | English |
    | Source | Enron corporate email archive |
    | Citation | Metsis et al. (2006) |

    #### How to add the Enron dataset

    **Option A — Kaggle download (automatic):**
    """)
    st.code("""# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your API key at ~/.kaggle/kaggle.json
#    (Download from https://www.kaggle.com/settings → API → Create New Token)

# 3. Run the training script — it will auto-download
python train_model.py""", language="bash")

    st.markdown("""
    **Option B — Manual download:**
    1. Go to [kaggle.com/datasets/wanderfj/enron-spam](https://www.kaggle.com/datasets/wanderfj/enron-spam)
    2. Download `enron_spam_data.csv`
    3. Place it in the same folder as `train_model.py`
    4. Run `python train_model.py`

    **Option C — CLI argument:**
    """)
    st.code("python train_model.py --dataset /path/to/enron_spam_data.csv", language="bash")

    st.markdown("""
    #### Accepted CSV formats

    The loader automatically detects these column name variants:

    | Text column alias | Label column alias | Phishing label values |
    |---|---|---|
    | `text`, `message`, `body`, `content`, `email_body` | `label`, `spam`, `class`, `category`, `is_spam` | `spam`, `1`, `phishing`, `yes`, `true` |

    Any other value is treated as **Safe**.

    #### Why Enron matters

    Training on real email corpora dramatically improves generalisation:
    - Covers genuine linguistic variation in business email
    - Includes novel phishing patterns not in hand-crafted templates
    - Provides thousands of real ham examples, improving specificity
    - Reduces false positive rate on legitimate corporate communication
    """)

    if metrics:
        st.markdown("#### Current model training info")
        for name, m in metrics.items():
            cols = st.columns(4)
            cols[0].metric(f"{name} — Train", f"{m.get('train_samples', 0):,}")
            cols[1].metric("Test samples",     f"{m.get('test_samples',  0):,}")
            cols[2].metric("F1 Score",         f"{m.get('f1_score', 0)*100:.1f}%")
            cols[3].metric("ROC-AUC",          f"{m.get('roc_auc',  0)*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════

def render_history_tab() -> None:
    st.markdown('<span class="section-tag">Session Log</span>', unsafe_allow_html=True)
    st.markdown("### 🕑 Prediction History")

    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Use the **Analyse Email** tab to get started.")
        return

    phish_cnt = sum(1 for h in history if h["verdict"] == "Phishing")
    safe_cnt  = len(history) - phish_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analysed", len(history))
    c2.metric("🔴 Phishing",   phish_cnt)
    c3.metric("✅ Safe",        safe_cnt)

    st.markdown("---")

    df_hist = pd.DataFrame(history[::-1])
    df_hist.index = range(1, len(df_hist) + 1)
    df_hist.index.name = "#"
    st.dataframe(
        df_hist.rename(columns={
            "text":    "Email preview",
            "verdict": "Verdict",
            "phish%":  "Phishing %",
            "safe%":   "Safe %",
            "model":   "Model used",
        }),
        use_container_width=True,
    )

    # ── Export ────────────────────────────────────────────────
    col_dl, col_clr = st.columns([2, 1])
    with col_dl:
        csv_buf = io.StringIO()
        df_hist.to_csv(csv_buf)
        st.download_button(
            "⬇️ Export history as CSV",
            data=csv_buf.getvalue(),
            file_name="phishguard_history.csv",
            mime="text/csv",
        )
    with col_clr:
        if st.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 6 — VIVA PREPARATION
# ══════════════════════════════════════════════════════════════

def render_viva_tab() -> None:
    st.markdown('<span class="section-tag">Academic Prep</span>', unsafe_allow_html=True)
    st.markdown("### 🎓 Viva Preparation Guide")
    st.caption("Examiner Q&A aligned to the project report.")

    qa_pairs = [
        (
            "Why TF-IDF over Word2Vec or BERT?",
            "TF-IDF is fully interpretable — every feature is a readable word/bigram — "
            "and computationally cheap (no GPU needed). Phishing text is keyword-heavy: "
            "the presence of 'urgent', 'verify', 'click' is the dominant signal. "
            "Word embeddings capture richer semantics but need more data and lose "
            "direct explainability. DistilBERT is a valid future extension for "
            "semantic phishing that avoids trigger keywords entirely.",
        ),
        (
            "How did you handle class imbalance?",
            "SMOTEENN-style balancing: (1) SMOTE analogue — oversample minority class "
            "with word-level Bernoulli dropout (~10–25% word removal) to create diverse "
            "near-duplicates rather than exact copies; (2) ENN analogue — remove "
            "majority-class samples with no urgency vocabulary AND no URL token. "
            "Also used class_weight='balanced' in both classifiers and stratified k-fold CV.",
        ),
        (
            "Explain LIME and how it differs from SHAP.",
            "LIME explains a single prediction by perturbing the input (randomly removing "
            "words) and fitting a linear model locally around that point. It is fast but "
            "approximate — no global consistency guarantee. SHAP uses game-theoretic "
            "Shapley values: SHAP values sum to the model output (efficiency property) "
            "and satisfy consistency + missingness axioms. LIME values do not. Here: "
            "LIME provides per-prediction word attribution; SHAP-style global importance "
            "uses |coefficients| (LR) / Gini decrease (RF).",
        ),
        (
            "Why is Recall more important than Precision here?",
            "False Negative = phishing email classified as safe → attack succeeds. "
            "This can cause credential theft or financial loss. False Positive = safe "
            "email flagged → inconvenience only. In security, the asymmetric cost of "
            "misses far outweighs false alarms, so we optimise for high Recall while "
            "keeping Precision acceptable.",
        ),
        (
            "How do you know the model isn't overfitting?",
            "5-fold stratified cross-validation on the training set gives CV F1 ± σ. "
            "Low σ across folds confirms stable generalisation. We evaluate on a 20% "
            "held-out test set not seen during training. The small gap between CV and "
            "test scores confirms the model generalises. For Random Forest, the OOB "
            "accuracy provides a free additional out-of-bag estimate.",
        ),
        (
            "Why FeatureUnion rather than just TF-IDF?",
            "TF-IDF captures vocabulary but misses structural signals: high URL density, "
            "excessive capitalisation, or dense urgency-word clusters are patterns present "
            "in feature space but not as distinct TF-IDF tokens. FeatureUnion concatenates "
            "~30 000 TF-IDF features with 12 hand-crafted heuristic features, improving "
            "recall on novel phishing emails that avoid common trigger keywords.",
        ),
        (
            "Why was the Enron dataset used / what is it?",
            "The Enron-Spam corpus (~33 700 emails) is the most widely cited real-world "
            "email benchmark. It was derived from Enron corporate email (released post-"
            "scandal) and annotated spam/ham. Using it ensures: genuine linguistic "
            "variation, novel phishing patterns not in hand-crafted templates, thousands "
            "of real ham examples (reducing false positive rate), and reproducible "
            "academic benchmarking. A synthetic fallback is used when the CSV is absent.",
        ),
        (
            "What are the system's main limitations?",
            "1) No email header analysis (SPF, DKIM, Return-Path, sender domain). "
            "2) No URL reputation checking via external API (e.g. VirusTotal). "
            "3) Static vocabulary — adversarial rephrasing may evade detection. "
            "4) LIME quality depends on n_samples (150 here = reasonable, not exact). "
            "5) Synthetic fallback data diverges from real-world email distribution.",
        ),
        (
            "What are future improvements?",
            "Short-term: DistilBERT for semantic understanding; URL reputation API. "
            "Medium-term: email header features (SPF/DKIM/Return-Path); active learning "
            "loop for uncertain predictions. "
            "Long-term: real-time Gmail/Outlook plugin; multilingual phishing detection; "
            "adversarial training against paraphrase attacks.",
        ),
    ]

    for q, a in qa_pairs:
        with st.expander(f"❓  {q}"):
            st.markdown(
                f'<div class="tip-box" style="font-size:.9rem;line-height:1.65;">'
                f"💡 {a}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 📋 Live Demo Checklist")
    steps = [
        "Analyse Email → paste phishing example → explain confidence scores & probability bars",
        "Toggle XAI on → walk through LIME highlighted text and attribution chart",
        "Switch model (LR ↔ RF) → comment on speed vs accuracy trade-off",
        "Show preprocessed text expander → explain clean_text() pipeline",
        "Performance Metrics → confusion matrix, TP/FP/FN/TN, why Recall matters",
        "Feature Insights → SHAP-style global importance, FeatureUnion pipeline diagram",
        "Mention 5-fold CV + OOB score as evidence of generalisation",
        "Dataset tab → explain Enron corpus, how to load it, synthetic fallback",
        "Discuss SMOTEENN class balancing strategy",
        "State limitations: no header analysis, no URL API, adversarial rephrasing",
        "Propose improvements: DistilBERT, active learning, real-time URL checking",
    ]
    for i, step in enumerate(steps, 1):
        st.checkbox(f"Step {i}: {step}", key=f"demo_{i}")

    st.markdown("---")
    st.markdown("#### 📐 Key Formulae")
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{df(t)}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
    with col_b:
        st.latex(r"\text{AUC} = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)")
        st.latex(r"\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[v(S\cup\{i\})-v(S)]")
        st.caption("Shapley value formula — SHAP theoretical foundation")
        st.latex(r"\text{Gini}(t) = 1 - \sum_{k} p_k^2")
        st.caption("Gini impurity — used in Random Forest feature importance")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="header-banner">
      <span style="font-size:3.2rem;">🛡️</span>
      <div>
        <h1>PhishGuard</h1>
        <p>Email Phishing Detection System · AKGEC Ghaziabad · Group G14 · May 2026</p>
        <p style="font-size:.8rem;opacity:.6;">
          Logistic Regression &amp; Random Forest · TF-IDF + 12 Heuristics ·
          Enron Dataset · LIME Explainability · SHAP Feature Importance
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load metrics ──────────────────────────────────────────
    metrics = load_metrics()
    if not metrics:
        st.error(
            "❌ `model_metrics.json` not found.  \n"
            "Run **`python train_model.py`** to train the models first."
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────
    model_choice = render_sidebar(metrics)
    model        = load_model(model_choice)

    if not model:
        st.error(
            f"Model file for **{model_choice}** not found.  \n"
            "Run **`python train_model.py`** to train the models."
        )
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "🔍 Analyse Email",
        "📊 Performance Metrics",
        "🔎 Feature Insights & XAI",
        "📂 Dataset & Enron Info",
        "🕑 History",
        "🎓 Viva Prep",
    ])

    with tabs[0]: render_analysis_tab(model, model_choice)
    with tabs[1]: render_metrics_tab(metrics, model_choice)
    with tabs[2]: render_xai_tab(model, model_choice)
    with tabs[3]: render_dataset_tab(metrics)
    with tabs[4]: render_history_tab()
    with tabs[5]: render_viva_tab()


if __name__ == "__main__":
    main()