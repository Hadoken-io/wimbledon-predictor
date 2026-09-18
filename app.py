"""
FastAPI backend for Wimbledon predictor - REWIRED to the new pipeline.

Uses:
  - data_processed/features.csv    (full feature history for live form/H2H lookups)
  - models/model_grass_general.pkl (used for non-elite matchups)
  - models/model_grass_elite.pkl   (used for top-50-vs-top-50 matchups)

Still reloads model + data fresh from disk on every request, by design.

Run: uvicorn app:app --reload --port 8000
"""

import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from rank_to_elo import build_rank_elo_curve

app = FastAPI(title="Wimbledon Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEAT_PATH = r"data_processed\features.csv"
GENERAL_MODEL_PATH = r"models\model_grass_general.pkl"
ELITE_MODEL_PATH = r"models\model_grass_elite.pkl"

FEATURE_COLS = [
    "Best_of", "Elo_Overall_Diff", "Elo_Grass_Diff", "Elo_Wimbledon_Diff",
    "GrassDNA_Diff", "Rank_Diff", "Form_Diff", "Grass_Form_Diff",
    "Grass_Momentum_Diff", "H2H_P1_Winrate", "H2H_Matches_Played",
    "H2H_Grass_Winrate", "H2H_Grass_Matches", "Wimbledon_History_Diff",
    "Elite_Match", "Is_Wimbledon",
]

DEFAULT_ELO = 1500.0
DEFAULT_RANK = 100.0
ELITE_RANK_THRESHOLD = 50


def load_models():
    with open(GENERAL_MODEL_PATH, "rb") as f:
        general = pickle.load(f)
    with open(ELITE_MODEL_PATH, "rb") as f:
        elite = pickle.load(f)
    return general, elite


def load_player_profiles():
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    df = df[df["Surf_Grass"] == 1].sort_values("Date")
    profiles = {}
    for _, row in df.iterrows():
        for side in ["1", "2"]:
            p = row[f"Player_{side}"]
            profiles[p] = {
                "Elo_Overall": row[f"Elo_Overall_P{side}"],
                "Elo_Grass": row[f"Elo_Grass_P{side}"],
                "Elo_Wimbledon": row[f"Elo_Wimbledon_P{side}"],
                "GrassDNA": row[f"GrassDNA_P{side}"],
                "Rank": row[f"Rank_{side}"],
                "Form": row["Form_Diff"] if side == "1" else -row["Form_Diff"],
                "Grass_Form": row["Grass_Form_Diff"] if side == "1" else -row["Grass_Form_Diff"],
                "Grass_Momentum": row["Grass_Momentum_Diff"] if side == "1" else -row["Grass_Momentum_Diff"],
                "Wimbledon_History": row["Wimbledon_History_Diff"] if side == "1" else -row["Wimbledon_History_Diff"],
            }
    return profiles


def load_h2h():
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    h2h = defaultdict(lambda: {"wins": 0, "total": 0})
    h2h_grass = defaultdict(lambda: {"wins": 0, "total": 0})
    for _, row in df.iterrows():
        p1, p2, won = row["Player_1"], row["Player_2"], row["Player_1_Won"]
        key = tuple(sorted([p1, p2]))
        cp1 = key[0] == p1
        h2h[key]["total"] += 1
        if (cp1 and won) or (not cp1 and not won):
            h2h[key]["wins"] += 1
        if row.get("Surf_Grass", 0) == 1:
            h2h_grass[key]["total"] += 1
            if (cp1 and won) or (not cp1 and not won):
                h2h_grass[key]["wins"] += 1
    return h2h, h2h_grass


def build_match_features(p1, p2, profiles, h2h, h2h_grass, rank_elo_estimate=None, known_ranks=None):
    def get(p, k, default=0):
        val = profiles.get(p, {}).get(k, default)
        return val if (val is not None and pd.notna(val)) else default

    def get_with_fallback(p, k, real_rank, default):
        """For unseen players, estimate from real rank instead of flat default."""
        if p in profiles:
            val = profiles[p].get(k, None)
            if val is not None and pd.notna(val):
                return val
        if rank_elo_estimate is not None and real_rank:
            est = rank_elo_estimate(real_rank)
            est_val = est.get(k, None)
            if est_val is not None and pd.notna(est_val):
                return est_val
        return default

    key = tuple(sorted([p1, p2]))
    p1c = key[0] == p1

    ht = h2h[key]["total"]
    hw = h2h[key]["wins"]
    hwr = (hw / ht) if ht > 0 else 0.5
    if not p1c:
        hwr = 1 - hwr

    ght = h2h_grass[key]["total"]
    ghw = h2h_grass[key]["wins"]
    ghwr = (ghw / ght) if ght > 0 else 0.5
    if not p1c:
        ghwr = 1 - ghwr

    # known_ranks: optional dict of {player_name: real_rank} supplied by the caller
    # for unseen players (since they won't have a Rank in profiles either)
    real_rank_p1 = (known_ranks or {}).get(p1)
    real_rank_p2 = (known_ranks or {}).get(p2)

    raw_rank1 = get(p1, "Rank", None)
    raw_rank2 = get(p2, "Rank", None)

    # Explicit NaN/None check - a player CAN exist in profiles but still have
    # a missing/NaN Rank value in the underlying data. The old `get()` default
    # only fires when the player key itself is absent, so it silently let NaN
    # ranks through uncorrected. Fix: check validity directly, every time.
    def resolve_rank(raw_rank, real_rank_hint):
        if raw_rank is not None and pd.notna(raw_rank):
            return float(raw_rank)
        if real_rank_hint:
            return float(real_rank_hint)
        return DEFAULT_RANK

    rank1 = resolve_rank(raw_rank1, real_rank_p1)
    rank2 = resolve_rank(raw_rank2, real_rank_p2)
    is_elite = 1 if (rank1 <= ELITE_RANK_THRESHOLD and rank2 <= ELITE_RANK_THRESHOLD) else 0

    feats = {
        "Best_of": 5,
        "Elo_Overall_Diff": get_with_fallback(p1, "Elo_Overall", real_rank_p1, DEFAULT_ELO) - get_with_fallback(p2, "Elo_Overall", real_rank_p2, DEFAULT_ELO),
        "Elo_Grass_Diff": get_with_fallback(p1, "Elo_Grass", real_rank_p1, DEFAULT_ELO) - get_with_fallback(p2, "Elo_Grass", real_rank_p2, DEFAULT_ELO),
        "Elo_Wimbledon_Diff": get(p1, "Elo_Wimbledon", DEFAULT_ELO) - get(p2, "Elo_Wimbledon", DEFAULT_ELO),
        "GrassDNA_Diff": get_with_fallback(p1, "GrassDNA", real_rank_p1, 0.5) - get_with_fallback(p2, "GrassDNA", real_rank_p2, 0.5),
        "Rank_Diff": rank1 - rank2,
        "Form_Diff": get(p1, "Form", 0) - get(p2, "Form", 0),
        "Grass_Form_Diff": get(p1, "Grass_Form", 0) - get(p2, "Grass_Form", 0),
        "Grass_Momentum_Diff": get(p1, "Grass_Momentum", 0) - get(p2, "Grass_Momentum", 0),
        "H2H_P1_Winrate": hwr,
        "H2H_Matches_Played": ht,
        "H2H_Grass_Winrate": ghwr,
        "H2H_Grass_Matches": ght,
        "Wimbledon_History_Diff": get(p1, "Wimbledon_History", 0) - get(p2, "Wimbledon_History", 0),
        "Elite_Match": is_elite,
        "Is_Wimbledon": 1,
    }
    return pd.DataFrame([feats])[FEATURE_COLS], is_elite


def predict_win_probability(p1, p2, profiles, h2h, h2h_grass, model_general, model_elite, rank_elo_estimate=None, known_ranks=None):
    feat_row, is_elite = build_match_features(p1, p2, profiles, h2h, h2h_grass, rank_elo_estimate, known_ranks)
    model = model_elite if is_elite else model_general
    proba_p1 = float(model.predict_proba(feat_row)[0, 1])
    return proba_p1, is_elite


class MatchRequest(BaseModel):
    player_1: str
    player_2: str
    player_1_rank: float | None = None
    player_2_rank: float | None = None


class BracketRequest(BaseModel):
    players: List[str]
    n_simulations: int = 10000
    known_ranks: dict | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Wimbledon Predictor API (new pipeline: GrassDNA, decay-fixed, general+elite models). See /docs."}
def build_name_resolver(profiles):
    """
    Maps natural names like 'Novak Djokovic' to the CSV's stored format
    like 'Djokovic N.' so /predict-match accepts normal names.
    """
    resolver = {}
    for csv_name in profiles.keys():
        # csv_name format: "Djokovic N." or "Al-Alawi S.K."
        parts = csv_name.strip().split()
        if len(parts) < 2:
            continue
        last_name = " ".join(parts[:-1])  # handles multi-word surnames
        initial = parts[-1].rstrip(".")
        resolver[csv_name.lower()] = csv_name          # exact match, lowercased
        resolver[f"{last_name.lower()} {initial.lower()}"] = csv_name  # "djokovic n"
    return resolver


def resolve_name(input_name, resolver, profiles):
    """Try exact match first, then last-name + first-initial match."""
    if input_name in profiles:
        return input_name
    
    key = input_name.strip().lower()
    if key in resolver:
        return resolver[key]
    
    # try "Firstname Lastname" -> "Lastname F."
    parts = input_name.strip().split()
    if len(parts) >= 2:
        first, last = parts[0], " ".join(parts[1:])
        candidate_key = f"{last.lower()} {first[0].lower()}"
        if candidate_key in resolver:
            return resolver[candidate_key]
    
    return input_name  # fallback: return as-is, will be "unseen"

@app.get("/player/{name}")
def get_player(name: str):
    profiles = load_player_profiles()
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found in features.csv (grass history).")
    p = profiles[name]
    return {
        "player": name,
        "elo_overall": float(p["Elo_Overall"]),
        "elo_grass": float(p["Elo_Grass"]),
        "elo_wimbledon": float(p["Elo_Wimbledon"]),
        "grass_dna": float(p["GrassDNA"]),
        "rank": float(p["Rank"]) if pd.notna(p["Rank"]) else None,
    }


@app.post("/predict-match")
def predict_match(req: MatchRequest):
    profiles = load_player_profiles()
    h2h, h2h_grass = load_h2h()
    model_general, model_elite = load_models()
    rank_elo_estimate, _ = build_rank_elo_curve()

    resolver = build_name_resolver(profiles)
    resolved_p1 = resolve_name(req.player_1, resolver, profiles)
    resolved_p2 = resolve_name(req.player_2, resolver, profiles)

    known_ranks = {}
    if req.player_1_rank:
        known_ranks[resolved_p1] = req.player_1_rank
    if req.player_2_rank:
        known_ranks[resolved_p2] = req.player_2_rank

    proba_p1, is_elite = predict_win_probability(
        resolved_p1, resolved_p2, profiles, h2h, h2h_grass, model_general, model_elite,
        rank_elo_estimate, known_ranks
    )

    return {
        "player_1": req.player_1,
        "player_2": req.player_2,
        "player_1_win_probability": round(proba_p1, 4),
        "player_2_win_probability": round(1 - proba_p1, 4),
        "predicted_winner": req.player_1 if proba_p1 >= 0.5 else req.player_2,
        "model_used": "elite" if is_elite else "general",
        "unseen_players": [orig for orig, resolved in [(req.player_1, resolved_p1), (req.player_2, resolved_p2)] if resolved not in profiles],
    }

@app.post("/simulate-bracket")
def simulate_bracket(req: BracketRequest):
    n = len(req.players)
    if n & (n - 1) != 0 or n < 2:
        raise HTTPException(status_code=400, detail=f"players list length must be a power of 2 (got {n})")

    profiles = load_player_profiles()
    h2h, h2h_grass = load_h2h()
    model_general, model_elite = load_models()
    rank_elo_estimate, _ = build_rank_elo_curve()
    known_ranks = req.known_ranks or {}
    rng = np.random.default_rng()

    n_rounds = int(np.log2(n))
    round_reach_counts = {p: np.zeros(n_rounds + 1) for p in req.players}
    unseen_players = [p for p in req.players if p not in profiles]

    for _ in range(req.n_simulations):
        current = list(req.players)
        furthest = {p: 0 for p in req.players}
        round_idx = 0

        while len(current) > 1:
            next_round = []
            for i in range(0, len(current), 2):
                p_a, p_b = current[i], current[i + 1]
                prob_a, _ = predict_win_probability(
                    p_a, p_b, profiles, h2h, h2h_grass, model_general, model_elite,
                    rank_elo_estimate, known_ranks
                )
                winner = p_a if rng.random() < prob_a else p_b
                furthest[winner] = round_idx + 1
                next_round.append(winner)
            current = next_round
            round_idx += 1

        for p, f in furthest.items():
            round_reach_counts[p][:f + 1] += 1

    results = []
    for p in req.players:
        counts = round_reach_counts[p]
        results.append({
            "player": p,
            "title_probability": round(counts[-1] / req.n_simulations, 4),
            "round_reach_probabilities": [round(c / req.n_simulations, 4) for c in counts],
        })

    results.sort(key=lambda r: -r["title_probability"])
    return {"n_simulations": req.n_simulations, "results": results, "unseen_players": unseen_players}