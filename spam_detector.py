"""Train, evaluate, and probe the email spam classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import OUTPUT_PATH, extract_features


ROOT = Path(__file__).resolve().parent
FEATURES = [
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "link_count",
    "urgency_words",
]
LABEL_ORDER = ["legitimate", "spam"]
ATTACKS = [
    (
        "Low-signal spam",
        "Hello, I am following up about the position we discussed. Please review "
        "the attached offer when convenient.",
    ),
    (
        "Urgent legitimate message",
        "Hi team, the meeting is today and the deadline is urgent. Please respond "
        "now so I can finalize the agenda.",
    ),
    (
        "Routine legitimate link",
        "Hi, your requested course schedule is available at "
        "https://college.example/schedule. Let me know if you have questions.",
    ),
]


def load_dataset() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} does not exist. Run prepare_data.py first."
        )
    return pd.read_csv(OUTPUT_PATH)


def main() -> None:
    dataset = load_dataset()
    x = dataset[FEATURES]
    y = dataset["label"]
    row_indices = dataset.index

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000),
    )
    model.fit(x.loc[train_indices], y.loc[train_indices])

    predictions = model.predict(x.loc[test_indices])
    accuracy = accuracy_score(y.loc[test_indices], predictions)
    matrix = confusion_matrix(
        y.loc[test_indices], predictions, labels=LABEL_ORDER
    )
    precision = precision_score(
        y.loc[test_indices], predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y.loc[test_indices], predictions, pos_label="spam", zero_division=0
    )

    train_sources = set(dataset.loc[train_indices, "source_id"])
    test_sources = set(dataset.loc[test_indices, "source_id"])
    shared_sources = sorted(train_sources & test_sources)

    classifier = model.named_steps["logisticregression"]
    weights = pd.Series(classifier.coef_[0], index=FEATURES).sort_values(
        key=lambda values: values.abs(), ascending=False
    )

    print(f"Rows: {len(dataset)} ({(y == 'spam').sum()} spam, {(y == 'legitimate').sum()} legitimate)")
    print(f"Train/test rows: {len(train_indices)}/{len(test_indices)}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted; legitimate, spam):")
    print(matrix)
    print(f"Spam precision: {precision:.3f}")
    print(f"Spam recall: {recall:.3f}")
    print("Feature weights (standardized; positive means more spam-like):")
    print(weights.to_string(float_format=lambda value: f"{value:.3f}"))
    print(
        f"Near-copy source IDs in both splits: {len(shared_sources)}/"
        f"{len(test_sources)} test-source IDs"
    )
    print(", ".join(shared_sources))

    print("Adversarial examples:")
    for name, email_text in ATTACKS:
        features = extract_features(email_text)
        prediction = model.predict(pd.DataFrame([features], columns=FEATURES))[0]
        print(f"- {name}: predicted {prediction}; features={features}")

    results = pd.DataFrame(
        {
            "actual": y.loc[test_indices],
            "predicted": predictions,
            "source_id": dataset.loc[test_indices, "source_id"],
        }
    )
    results.to_csv(ROOT / "data" / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
