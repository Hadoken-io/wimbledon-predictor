import pandas as pd, pickle, numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"data_processed\features.csv", parse_dates=["Date"])

# check each Wimbledon year separately
with open(r"models\model_grass_elite.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLS = [
    "Best_of", "Elo_Overall_Diff", "Elo_Grass_Diff", "Elo_Wimbledon_Diff",
    "GrassDNA_Diff", "Rank_Diff", "Form_Diff", "Grass_Form_Diff",
    "Grass_Momentum_Diff", "H2H_P1_Winrate", "H2H_Matches_Played",
    "H2H_Grass_Winrate", "H2H_Grass_Matches", "Wimbledon_History_Diff",
    "Elite_Match", "Is_Wimbledon",
]

wimb = df[
    (df["Is_Wimbledon"] == 1) &
    (df["Rank_1"] <= 50) & (df["Rank_2"] <= 50) &
    (df["Date"] >= "2018-01-01")
].dropna(subset=FEATURE_COLS)

for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
    yw = wimb[wimb["Date"].dt.year == year]
    if len(yw) == 0:
        continue
    acc = accuracy_score(yw["Player_1_Won"], (model.predict_proba(yw[FEATURE_COLS])[:,1] >= 0.5).astype(int))
    print(f"{year}: {len(yw)} matches → {acc:.4f}")