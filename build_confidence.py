"""
Step: Build "Confidence" metric per player at any point in time.
Composite of:
  1. Recency-weighted streak momentum (exponential decay - recent matches matter more)
  2. Upset-adjusted result (beating higher-Elo opponent = bigger confidence boost;
     losing to lower-Elo opponent = bigger confidence hit)
  3. Grass-recency weighting (grass matches weighted higher, since target = Wimbledon)

Output: roughly -1.0 (very low confidence) to +1.0 (very high confidence) scale.

This script computes confidence AT EACH POINT IN HISTORY (chronologically safe,
pre-match only) so it can be merged into features.csv as Confidence_Diff.

Run: python build_confidence.py
In:  data_processed/elo_history.csv
Out: data_processed/confidence_history.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict

IN_PATH = "data_processed/elo_history.csv"
OUT_PATH = "data_processed/confidence_history.csv"

WINDOW = 15              # consider last N matches for confidence calc
DECAY_RATE = 0.85        # each match back in time weighted by DECAY_RATE^age (most recent = age 0)
GRASS_BOOST = 1.6        # multiplier applied to grass matches' weight
UPSET_SENSITIVITY = 400  # same scale as Elo - controls how much opponent strength affects swing


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print("Computing confidence for", len(df), "matches")

    # per-player rolling history: list of dicts {result: +1/-1, opp_elo_diff, is_grass}
    player_history = defaultdict(list)

    p1_confidence = []
    p2_confidence = []

    def compute_confidence(history):
        """history = list of recent match records, OLDEST first, for one player."""
        if not history:
            return 0.0  # neutral - no history yet

        recent = history[-WINDOW:]
        recent = list(reversed(recent))  # most recent first, for decay indexing

        total_weight = 0.0
        weighted_sum = 0.0

        for age, rec in enumerate(recent):
            decay_weight = DECAY_RATE ** age
            surface_weight = GRASS_BOOST if rec["is_grass"] else 1.0
            weight = decay_weight * surface_weight

            # result_value: base +1/-1, adjusted by how surprising it was
            # beating a much stronger opponent (positive elo_diff_vs_opp) -> amplify win
            # losing to a much weaker opponent -> amplify loss
            upset_factor = 1 + abs(rec["opp_elo_diff"]) / UPSET_SENSITIVITY
            upset_factor = min(upset_factor, 2.5)  # cap so one huge upset doesn't dominate everything

            if rec["result"] == 1:
                # win: bigger boost if opponent was higher elo (opp_elo_diff > 0 means opp stronger)
                value = 1.0 * (upset_factor if rec["opp_elo_diff"] > 0 else 1.0)
            else:
                # loss: bigger hit if opponent was lower elo (opp_elo_diff < 0 means opp weaker)
                value = -1.0 * (upset_factor if rec["opp_elo_diff"] < 0 else 1.0)

            weighted_sum += value * weight
            total_weight += weight

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        # normalize roughly into [-1, 1] - raw_score can slightly exceed due to upset multiplier, so clip
        return float(np.clip(raw_score, -1.0, 1.0))

    for idx, row in df.iterrows():
        p1, p2 = row["Player_1"], row["Player_2"]

        conf_p1 = compute_confidence(player_history[p1])
        conf_p2 = compute_confidence(player_history[p2])
        p1_confidence.append(conf_p1)
        p2_confidence.append(conf_p2)

        # update history AFTER computing (pre-match safety)
        won_p1 = row["Player_1_Won"] == 1
        elo_diff_p1_vs_p2 = row["P2_Elo_Overall_Pre"] - row["P1_Elo_Overall_Pre"]  # positive = p2 stronger
        is_grass = row["Surface"] == "Grass"

        player_history[p1].append({"result": 1 if won_p1 else -1, "opp_elo_diff": elo_diff_p1_vs_p2, "is_grass": is_grass})
        player_history[p2].append({"result": -1 if won_p1 else 1, "opp_elo_diff": -elo_diff_p1_vs_p2, "is_grass": is_grass})

        if idx % 10000 == 0:
            print(f"  ...{idx} processed")

    df["P1_Confidence"] = p1_confidence
    df["P2_Confidence"] = p2_confidence
    df["Confidence_Diff"] = df["P1_Confidence"] - df["P2_Confidence"]

    out_cols = ["Date", "Player_1", "Player_2", "P1_Confidence", "P2_Confidence", "Confidence_Diff"]
    df[out_cols].to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # sanity check: show players currently on hot streaks (high confidence) vs cold (low)
    # IMPORTANT: only consider players who (a) have enough match history to be meaningful,
    # and (b) played RECENTLY (active tour players), not one-off entries from decades ago.
    latest_date_per_player = {}
    latest_conf_per_player = {}
    match_count_per_player = defaultdict(int)

    for idx, row in df.iterrows():
        p1, p2, date = row["Player_1"], row["Player_2"], row["Date"]
        latest_date_per_player[p1] = date
        latest_date_per_player[p2] = date
        latest_conf_per_player[p1] = row["P1_Confidence"]
        latest_conf_per_player[p2] = row["P2_Confidence"]
        match_count_per_player[p1] += 1
        match_count_per_player[p2] += 1

    most_recent_date_in_data = df["Date"].max()
    active_cutoff = most_recent_date_in_data - pd.Timedelta(days=180)  # played in last 6 months
    MIN_MATCHES = 10  # need real sample size for confidence to mean anything

    active_players = {
        p: latest_conf_per_player[p]
        for p in latest_conf_per_player
        if latest_date_per_player[p] >= active_cutoff and match_count_per_player[p] >= MIN_MATCHES
    }

    print(f"\n(Filtered to {len(active_players)} players active in last 180 days with 10+ career matches)")

    conf_series = pd.Series(active_players).sort_values(ascending=False)
    print("\n=== TOP 10 HIGHEST CURRENT CONFIDENCE (active players only) ===")
    print(conf_series.head(10).to_string())
    print("\n=== BOTTOM 10 LOWEST CURRENT CONFIDENCE (active players only) ===")
    print(conf_series.tail(10).to_string())


if __name__ == "__main__":
    main()