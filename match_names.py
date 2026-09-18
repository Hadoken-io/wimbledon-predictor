"""
Name matcher: full real-world names -> dataset format ('Lastname F.')

Run: python match_names.py
Then paste full names (one per line, e.g. from an ATP rankings page),
blank line to finish. Reports match results.
"""

import pandas as pd
import re
from difflib import SequenceMatcher

ELO_CURRENT_PATH = "data_processed/elo_current.csv"


def load_dataset_names():
    df = pd.read_csv(ELO_CURRENT_PATH)
    return df["Player"].tolist()


def normalize(name: str) -> str:
    """Lowercase, strip accents-ish chars loosely, collapse whitespace."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z\s\.\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def to_dataset_format(full_name: str) -> str:
    """
    'Carlos Alcaraz' -> 'alcaraz c.'
    Standard assumption: FIRST token is given name, LAST token is surname
    (this is the normal Western full-name convention used by ATP/ESPN/etc).
    Multi-word surnames (e.g. 'Juan Martin del Potro') still take everything
    after the first token as the surname.
    """
    parts = normalize(full_name).split()
    if len(parts) < 2:
        return normalize(full_name)
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last} {first[0]}."


def fuzzy_best_match(target: str, candidates: list, threshold: float = 0.75):
    """Fallback: find closest dataset name by string similarity."""
    best_score = 0
    best_match = None
    target_norm = normalize(target)
    for c in candidates:
        score = SequenceMatcher(None, target_norm, normalize(c)).ratio()
        if score > best_score:
            best_score = score
            best_match = c
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def match_player(full_name: str, dataset_names: list, dataset_names_normalized: dict):
    """
    Try exact heuristic conversion first, then fuzzy fallback.
    Returns (matched_name_or_None, method, confidence)
    """
    heuristic_guess = to_dataset_format(full_name)

    if heuristic_guess in dataset_names_normalized:
        return dataset_names_normalized[heuristic_guess], "exact_heuristic", 1.0

    fuzzy_match, score = fuzzy_best_match(full_name, dataset_names)
    if fuzzy_match:
        return fuzzy_match, "fuzzy", round(score, 3)

    return None, "no_match", 0.0


def main():
    dataset_names = load_dataset_names()
    dataset_names_normalized = {normalize(n): n for n in dataset_names}

    print(f"Loaded {len(dataset_names)} known player names from dataset.\n")
    print("Paste full player names (one per line). Blank line to finish:\n")

    inputs = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        inputs.append(line.strip())

    print("\n" + "=" * 70)
    print(f"{'Input Name':<28} {'Matched To':<22} {'Method':<16} {'Conf'}")
    print("=" * 70)

    unmatched = []
    for name in inputs:
        match, method, conf = match_player(name, dataset_names, dataset_names_normalized)
        display_match = match if match else "*** NO MATCH ***"
        print(f"{name:<28} {display_match:<22} {method:<16} {conf}")
        if not match:
            unmatched.append(name)

    print("=" * 70)
    print(f"\n{len(inputs) - len(unmatched)}/{len(inputs)} matched.")
    if unmatched:
        print("Unmatched:", unmatched)
        print("\n(These either don't exist in your dataset - e.g. brand new players -")
        print(" or need a manual override entry. Check spelling/format first.)")


if __name__ == "__main__":
    main()