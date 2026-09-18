"""
Rank -> Elo estimator for unseen players (qualifiers, wildcards not in
features.csv's grass history). Without this, unseen players default to
flat Elo 1500 / rank 100 regardless of their REAL rank.

Run standalone to inspect the curve: python rank_to_elo.py
Import build_rank_elo_curve() into app.py to use for real.
"""

import pandas as pd

FEAT_PATH = r"data_processed\features.csv"


def build_rank_elo_curve():
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    cutoff = df["Date"].max() - pd.Timedelta(days=730)
    recent = df[df["Date"] >= cutoff]

    pairs = []
    for _, row in recent.iterrows():
        if pd.notna(row.get("Rank_1")) and row["Rank_1"] > 0:
            pairs.append((row["Rank_1"], row["Elo_Overall_P1"], row.get("Elo_Grass_P1"), row.get("GrassDNA_P1")))
        if pd.notna(row.get("Rank_2")) and row["Rank_2"] > 0:
            pairs.append((row["Rank_2"], row["Elo_Overall_P2"], row.get("Elo_Grass_P2"), row.get("GrassDNA_P2")))

    rank_df = pd.DataFrame(pairs, columns=["Rank", "Elo_Overall", "Elo_Grass", "GrassDNA"]).drop_duplicates()

    bins = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 10000]
    rank_df["bucket"] = pd.cut(rank_df["Rank"], bins=bins)

    medians = rank_df.groupby("bucket")[["Elo_Overall", "Elo_Grass", "GrassDNA"]].median()
    edges = bins

    def estimate(rank):
        if pd.isna(rank) or rank <= 0:
            return {"Elo_Overall": 1500.0, "Elo_Grass": 1500.0, "GrassDNA": 0.5}
        for i in range(len(edges) - 1):
            if edges[i] < rank <= edges[i + 1]:
                row = medians.iloc[i] if i < len(medians) else None
                if row is not None and not row.isna().all():
                    return {
                        "Elo_Overall": float(row["Elo_Overall"]) if pd.notna(row["Elo_Overall"]) else 1500.0,
                        "Elo_Grass": float(row["Elo_Grass"]) if pd.notna(row["Elo_Grass"]) else 1500.0,
                        "GrassDNA": float(row["GrassDNA"]) if pd.notna(row["GrassDNA"]) else 0.5,
                    }
        return {"Elo_Overall": 1500.0, "Elo_Grass": 1500.0, "GrassDNA": 0.5}

    return estimate, medians


def main():
    estimate, medians = build_rank_elo_curve()
    print("Rank bucket medians (last 2 years):")
    print(medians.to_string())
    print("\nSelf-test:")
    for r in [1, 10, 25, 50, 90, 150, 250, 400]:
        print(f"  Rank {r:>4} -> {estimate(r)}")


if __name__ == "__main__":
    main()