"""
Diagnostic: search dataset for name variants of well-known players,
to understand exact format(s) actually present.

Run: python diagnose_names.py
"""

import pandas as pd

ELO_CURRENT_PATH = "data_processed/elo_current.csv"

# last names / partial strings to search for, case-insensitive substring match
SEARCH_TERMS = [
    "alcaraz", "sinner", "djokovic", "novak", "zverev", "medvedev",
    "fritz", "ruud", "draper", "tsitsipas", "rune", "paul",
    "dimitrov", "shelton", "rublev", "federer", "nadal"
]


def main():
    df = pd.read_csv(ELO_CURRENT_PATH)
    names = df["Player"].astype(str)

    print(f"Total players in dataset: {len(names)}\n")
    print("Searching for known players (showing ALL matching variants found):\n")

    for term in SEARCH_TERMS:
        matches = names[names.str.lower().str.contains(term, na=False)]
        if len(matches) > 0:
            print(f"'{term}': {list(matches)}")
        else:
            print(f"'{term}': NOT FOUND in dataset at all")

    print("\n" + "=" * 60)
    print("Sample of 30 random player name strings (raw, as stored):")
    print("=" * 60)
    print(names.sample(30, random_state=1).to_list())


if __name__ == "__main__":
    main()