"""
Diagnostic: print the EXACT feature values app.py's build_match_features()
produces for the known real matches, including raw profile/rank inspection,
to find precisely where things are wrong instead of guessing.

Run from the repo root:
  python diagnose_app_vs_check.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pickle
from app import load_player_profiles, load_h2h, build_match_features, FEATURE_COLS, GENERAL_MODEL_PATH, ELITE_MODEL_PATH
from rank_to_elo import build_rank_elo_curve

TEST_MATCHES = [
    ("Sinner J.", "Dimitrov G.", "Sinner J."),
    ("Cilic M.", "Cobolli F.", "Cobolli F."),
    ("Khachanov K.", "Majchrzak K.", "Khachanov K."),
    ("Sinner J.", "Djokovic N.", "Sinner J."),
    ("Sinner J.", "Alcaraz C.", "Sinner J."),
]


def main():
    print("Loading profiles/H2H/rank-curve...")
    profiles = load_player_profiles()
    h2h, h2h_grass = load_h2h()
    rank_elo_estimate, _ = build_rank_elo_curve()

    with open(GENERAL_MODEL_PATH, "rb") as f:
        model_general = pickle.load(f)
    with open(ELITE_MODEL_PATH, "rb") as f:
        model_elite = pickle.load(f)

    print(f"General model FEATURE_COLS expected (from model itself, if available): {getattr(model_general, 'feature_names_in_', 'N/A')}")
    print(f"Elite model FEATURE_COLS expected: {getattr(model_elite, 'feature_names_in_', 'N/A')}")
    print(f"app.py's FEATURE_COLS list: {FEATURE_COLS}")

    correct_count = 0

    for p1, p2, real_winner in TEST_MATCHES:
        print(f"\n{'='*80}")
        print(f"MATCH: {p1} vs {p2}  |  REAL WINNER: {real_winner}")
        print(f"{'='*80}")

        feat_row, is_elite = build_match_features(p1, p2, profiles, h2h, h2h_grass, rank_elo_estimate, {})
        model = model_elite if is_elite else model_general

        print(f"  is_elite = {is_elite}  (using {'ELITE' if is_elite else 'GENERAL'} model)")
        print(f"  feat_row columns in order: {list(feat_row.columns)}")
        print(f"  feat_row dtypes:\n{feat_row.dtypes.to_string()}")

        proba_p1 = model.predict_proba(feat_row)[0, 1]
        pick = p1 if proba_p1 >= 0.5 else p2
        conf = proba_p1 if proba_p1 >= 0.5 else 1 - proba_p1
        correct = (pick == real_winner)
        correct_count += int(correct)

        print(f"\n  predict_proba raw output: {model.predict_proba(feat_row)}")
        print(f"  P(player_1 wins) = {proba_p1:.4f}")
        print(f"  MODEL PICK: {pick} ({conf:.1%} confidence)  ->  {'CORRECT' if correct else 'WRONG'}")

    print(f"\n{'='*80}")
    print(f"Direct prediction accuracy on these 5: {correct_count}/{len(TEST_MATCHES)}")


if __name__ == "__main__":
    main()