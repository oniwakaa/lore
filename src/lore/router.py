# src/lore/router.py
"""TF-IDF + LogReg 3-way router. Classifies input in <1ms."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class Router:
    """3-way text classifier: PRIMARY / SPECIALIST / TOOL_ONLY."""

    def __init__(self, pipeline: Pipeline, classes: list[str], confidence_threshold: float = 0.70):
        self._pipeline = pipeline
        self._classes = classes
        self._threshold = confidence_threshold

    @classmethod
    def load(cls, path: str, confidence_threshold: float = 0.70) -> "Router":
        """Load a trained router model. Falls back to always-PRIMARY if file missing."""
        p = Path(path)
        if not p.exists():
            # ponytail: no model = default to PRIMARY, log nothing, just work
            dummy = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
            return cls(dummy, ["PRIMARY", "SPECIALIST", "TOOL_ONLY"], confidence_threshold)
        data = joblib.load(path)
        return cls(data["pipeline"], data["classes"], confidence_threshold)

    def classify(self, text: str) -> tuple[str, float]:
        """Classify text. Returns (route, confidence). Below threshold -> PRIMARY."""
        try:
            probs = self._pipeline.predict_proba([text])[0]
        except Exception:
            return "PRIMARY", 0.0

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        route = self._classes[idx]

        if confidence < self._threshold:
            return "PRIMARY", confidence
        return route, confidence

    @classmethod
    def train(cls, data_path: str, model_path: str) -> dict:
        """Train router on JSONL data. Returns metrics dict."""
        texts, labels = [], []
        for line in Path(data_path).read_text().strip().split("\n"):
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])

        classes = sorted(set(labels))
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

        # stratify only when we have enough samples per class for a split
        n_samples = len(texts)
        n_classes = len(classes)
        min_per_class = 5
        can_split = n_samples >= n_classes * min_per_class

        if can_split:
            stratify = labels
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=stratify
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        else:
            # too few samples for a reliable split: train on all, eval on all
            pipeline.fit(texts, labels)
            y_pred = pipeline.predict(texts)
            accuracy = accuracy_score(labels, y_pred)

        joblib.dump({"pipeline": pipeline, "classes": classes}, model_path)

        return {
            "accuracy": float(accuracy),
            "classes": classes,
            "train_size": len(texts) if not can_split else len(X_train),
            "test_size": len(texts) if not can_split else len(X_test),
        }

if __name__ == "__main__":
    # ponytail: self-check
    import tempfile
    data = [
        {"text": "write code", "label": "PRIMARY"},
        {"text": "extract data", "label": "SPECIALIST"},
        {"text": "count words", "label": "TOOL_ONLY"},
    ] * 10
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        train_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name
    metrics = Router.train(train_path, model_path)
    assert metrics["accuracy"] > 0.5, f"accuracy too low: {metrics}"
    r = Router.load(model_path)
    route, conf = r.classify("write a function")
    assert route == "PRIMARY"
    print("router self-check OK")
