"""
Live feature lookup helpers.
Pulls REAL current values (rank, recent form, h2h) from elo_history.csv
so /predict-match and /simulate-bracket aren't using neutral placeholders.

Import this from app.py - not run standalone.
"""

import pandas as pd
from functools import lru_cache

HISTORY_PATH = "data_processed/elo_history.csv"
FORM_WINDOW = 10


@lru_cache(maxsize=1)
def _load_history_cached():
    """
    Cached per-process load. NOTE: app.py reloads model/elo fresh per request
    by design, but history.csv is large (67k rows) and re-parsing it on every
    single request would be slow. This cache is invalidated by restarting the
    server (e.g. after a retrain) - acceptable tradeoff, flagged here on purpose.
    """
    df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def get_latest_rank(player_name: str):
    """Most recent known rank for a player, or None if never seen."""
    df = _load_history_cached()
    p1_rows = df[df["Player_1"] == player_name][["Date", "Rank_1"]].rename(columns={"Rank_1": "Rank"})
    p2_rows = df[df["Player_2"] == player_name][["Date", "Rank_2"]].rename(columns={"Rank_2": "Rank"})
    all_rows = pd.concat([p1_rows, p2_rows]).dropna(subset=["Rank"])
    if all_rows.empty:
        return None
    latest = all_rows.sort_values("Date").iloc[-1]
    return float(latest["Rank"])


def get_recent_form(player_name: str, grass_only: bool = False, window: int = FORM_WINDOW):
    """
    Win% over last `window` matches for this player.
    Returns 0.5 (neutral) if player has no history at all.
    """
    df = _load_history_cached()
    if grass_only:
        df = df[df["Surface"] == "Grass"]

    p1_rows = df[df["Player_1"] == player_name][["Date", "Player_1_Won"]].rename(columns={"Player_1_Won": "Won"})
    p2_rows = df[df["Player_2"] == player_name][["Date", "Player_1_Won"]].copy()
    p2_rows["Won"] = 1 - p2_rows["Player_1_Won"]
    p2_rows = p2_rows[["Date", "Won"]]

    all_rows = pd.concat([p1_rows, p2_rows]).sort_values("Date")
    if all_rows.empty:
        return 0.5

    recent = all_rows.tail(window)
    return float(recent["Won"].mean())


def get_h2h(player_1: str, player_2: str):
    """
    Returns (p1_winrate, matches_played) for all prior matches between these two.
    p1_winrate = 0.5 if they've never played.
    """
    df = _load_history_cached()
    as_p1 = df[(df["Player_1"] == player_1) & (df["Player_2"] == player_2)]
    as_p2 = df[(df["Player_1"] == player_2) & (df["Player_2"] == player_1)]

    p1_wins = as_p1["Player_1_Won"].sum() + (as_p2["Player_1_Won"] == 0).sum()
    total = len(as_p1) + len(as_p2)

    if total == 0:
        return 0.5, 0
    return float(p1_wins / total), int(total)


def get_days_since_last_match(player_name: str, reference_date=None):
    """
    Days since player's last known match in history, relative to reference_date
    (defaults to most recent date in dataset, i.e. 'today' as far as data knows).
    """
    df = _load_history_cached()
    p1_dates = df[df["Player_1"] == player_name]["Date"]
    p2_dates = df[df["Player_2"] == player_name]["Date"]
    all_dates = pd.concat([p1_dates, p2_dates])

    if all_dates.empty:
        return 60  # unseen player - assume neutral/rested

    last_played = all_dates.max()
    ref = reference_date if reference_date is not None else df["Date"].max()
    days = (ref - last_played).days
    return min(max(days, 0), 60)  # clamp same as training-time cap