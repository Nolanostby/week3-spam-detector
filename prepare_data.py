"""Build a small, reproducible email-feature dataset for the project."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "email_features.csv"

MONEY_WORDS = {
    "bonus",
    "cash",
    "credit",
    "dollar",
    "dollars",
    "income",
    "money",
    "payment",
    "prize",
}
URGENCY_WORDS = {
    "act",
    "deadline",
    "expires",
    "immediately",
    "limited",
    "now",
    "respond",
    "today",
    "urgent",
}
WORD_PATTERN = re.compile(r"\b[\w']+\b")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

BASE_EMAILS = [
    (
        "S01",
        "spam",
        "CONGRATULATIONS! You are a winner. Claim your $5000 cash prize now at "
        "https://prize.example/claim",
    ),
    (
        "S02",
        "spam",
        "URGENT: Verify your payment details immediately to receive your bonus. "
        "Respond now at www.account.example",
    ),
    (
        "S03",
        "spam",
        "Earn guaranteed income from home. This limited offer pays cash today; "
        "act now!",
    ),
    (
        "S04",
        "spam",
        "You have been selected for a free gift and vacation prize. Claim today "
        "before this offer expires!",
    ),
    (
        "S05",
        "spam",
        "Double your money with our investment bonus. Limited time only! "
        "Visit https://invest.example",
    ),
    (
        "S06",
        "spam",
        "Your payment is on hold. Respond immediately to claim your credit "
        "bonus at www.billing.example",
    ),
    (
        "S07",
        "spam",
        "WINNER! Claim your cash reward now. This prize expires today!",
    ),
    (
        "S08",
        "spam",
        "Make extra income and get a $200 bonus. Act now to claim this limited "
        "offer!",
    ),
    (
        "L01",
        "legitimate",
        "Hi Morgan, the project notes are ready. Please review them before our "
        "team meeting on Thursday.",
    ),
    (
        "L02",
        "legitimate",
        "Your library book is due next week. Renew it through the library "
        "website if you need more time.",
    ),
    (
        "L03",
        "legitimate",
        "Thanks for applying. We would like to schedule an interview for the "
        "position. Please reply with a time that works for you.",
    ),
    (
        "L04",
        "legitimate",
        "The monthly utility statement is available in your account. The "
        "amount due is listed on the first page.",
    ),
    (
        "L05",
        "legitimate",
        "Lunch with the team is at noon today. Let me know if you would like "
        "me to save you a seat.",
    ),
    (
        "L06",
        "legitimate",
        "Your order has shipped. Track the delivery using the carrier link in "
        "your order history.",
    ),
    (
        "L07",
        "legitimate",
        "Could you send me the notes from class when you have a chance? I "
        "appreciate your help.",
    ),
    (
        "L08",
        "legitimate",
        "The appointment is confirmed for Tuesday at 10 AM. Call the office "
        "if you need to change the time.",
    ),
    (
        "L08",
        "legitimate",
        "The appointment is confirmed for Tuesday at 10 AM. Call the office "
        "if you need to change the time.",
    ),
]

# These light edits intentionally preserve each original message's label and
# make it possible to measure how a row-wise split can leak near-copies.
VARIATIONS = (
    "",
    " Please see the details below.",
    " Thank you for taking a look.",
)


def extract_features(email_text: str) -> dict[str, int]:
    """Return the six numeric signals used by the classifier."""
    words = WORD_PATTERN.findall(email_text)
    return {
        "word_count": len(words),
        "exclamations": email_text.count("!"),
        "money_words": sum(word.lower() in MONEY_WORDS for word in words),
        "all_caps": sum(
            len(word) > 1 and word.isalpha() and word.isupper() for word in words
        ),
        "link_count": len(URL_PATTERN.findall(email_text)),
        "urgency_words": sum(word.lower() in URGENCY_WORDS for word in words),
    }


def build_dataset() -> pd.DataFrame:
    rows = []
    for source_id, label, base_text in BASE_EMAILS:
        for variant, addition in enumerate(VARIATIONS):
            email_text = f"{base_text}{addition}"
            rows.append(
                {
                    "source_id": source_id,
                    "variant": variant,
                    "email_text": email_text,
                    "label": label,
                    **extract_features(email_text),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(dataset)} emails to {OUTPUT_PATH.relative_to(ROOT)}")
    print(dataset["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
