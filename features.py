"""
PhishGuard — Shared Feature Engineering
========================================
Imported by both train_model.py and app.py so that joblib can unpickle
the trained pipelines without a 'HeuristicFeatures' attribute error.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URGENCY_WORDS: frozenset[str] = frozenset({
    "urgent", "immediately", "verify", "suspended", "limited", "expired",
    "click", "confirm", "update", "account", "password", "security",
    "winner", "lottery", "prize", "congratulations", "claim", "free",
    "guaranteed", "risk", "alert", "warning", "unusual", "suspicious",
    "action", "required", "final", "notice", "deadline", "renew",
    "login", "credential", "billing", "payment", "failed", "verify",
    "validate", "unauthorized", "breach", "locked", "suspended", "expire",
    "refund", "compensation", "reward", "selected", "chosen",
})


class HeuristicFeatures(BaseEstimator, TransformerMixin):
    """
    12 hand-crafted numeric features capturing structural phishing patterns
    that bag-of-words / TF-IDF cannot represent directly.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(t) for t in X], dtype=np.float32)

    def _extract(self, text: str) -> list[float]:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        n     = max(len(words), 1)

        url_count         = text.count("url")
        urgency_count     = sum(1 for w in words if w in URGENCY_WORDS)
        urgency_ratio     = urgency_count / n
        exclamation_count = text.count("!")
        question_count    = text.count("?")
        num_count         = text.count("num")
        email_mention     = int("email" in words)
        text_length       = len(text)
        avg_word_len      = sum(len(w) for w in words) / n
        type_token_ratio  = len(set(words)) / n
        url_density       = url_count / n

        return [
            url_count, urgency_count, urgency_ratio,
            exclamation_count, question_count, num_count,
            email_mention, text_length, avg_word_len,
            type_token_ratio, url_density,
            sum(1 for w in words if len(w) > 8) / n,
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "url_count", "urgency_count", "urgency_ratio",
            "exclamation_count", "question_count", "num_count",
            "email_mention", "text_length", "avg_word_len",
            "type_token_ratio", "url_density", "long_word_ratio",
        ])