"""
Wimbledon 2026 — Versus Tracker
--------------------------------
Logs one match at a time into wimbledon_2026
    Match | My Prediction | My % | IBM Prediction | IBM % |
    Real Winner | My Accuracy | IBM Accuracy

- "My Prediction" + "My %"        -> comes from YOUR model (local FastAPI server)
- "IBM Prediction" + "IBM %"      -> comes from bookmaker odds (The Odds API),
                                     vig-removed. Rename later to "IBM" once
                                     you start manually overwriting with real
                                     SlamTracker numbers.
- "Real Winner"                   -> YOU fill this in after the match finishes.
- "My Accuracy" / "IBM Accuracy"  -> auto-calculated every time you run this script.

USAGE:
    python log_match.py "Player A" "Player B"

What happens:
    1. Script calls your model -> gets predicted winner + confidence %
    2. Script calls The Odds API -> finds the match -> converts odds to
       implied probability (vig removed) -> gets market's predicted winner + %
    3. Appends a new row to the Excel file (creates it if missing)
    4. Recalculates the Running Accuracy % columns based on rows where you've
       already filled in "Real Winner"
"""

import sys
import os
import requests
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ============================================================
# CONFIG — edit these
# ============================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    raise RuntimeError(
        "ODDS_API_KEY is not set. Copy .env.example to .env and add your key from the-odds-api.com"
    )
SPORT_KEY = "tennis_atp_wimbledon"     # The Odds API's sport key for Wimbledon (men's).
                                        # Will only return data once tournament is "in season".
                                        # For women's draw, check available sports first
                                        # (see list_sports.py) — likely tennis_wta_wimbledon
REGION = "uk"                          # uk bookmakers tend to have best tennis coverage
EXCEL_PATH = "wimbledon_2026_final.xlsx"
SHEET_NAME = "Versus Results"


# ============================================================
# 1. YOUR MODEL — calls your local FastAPI server
# ============================================================
MY_API_URL = "http://127.0.0.1:8000/predict-match"


def get_my_prediction(player1: str, player2: str):
    """
    Calls your local Wimbledon Predictor API (/predict-match) and returns
    (predicted_winner_name, win_pct).

    Confirmed real response shape from your server:
        {
          "player_1": "...",
          "player_2": "...",
          "player_1_win_probability": 0.5135,
          "player_2_win_probability": 0.4865,
          "predicted_winner": "...",
          "model_used": "general",
          "unseen_players": [...]
        }
    """
    try:
        resp = requests.post(
            MY_API_URL,
            json={"player_1": player1, "player_2": player2},
            timeout=60,
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            "Your local model server took longer than 60 seconds to respond. "
            "Check Terminal 1 (uvicorn) — if it's still processing, just wait and "
            "rerun this script. If uvicorn shows an error, fix that first."
        )
    resp.raise_for_status()
    data = resp.json()

    winner = data["predicted_winner"]
    if winner == player1:
        win_pct = data["player_1_win_probability"] * 100
    else:
        win_pct = data["player_2_win_probability"] * 100

    # heads-up flag, doesn't block anything — just printed so you know
    # the model had no historical data for one/both players
    if data.get("unseen_players"):
        print(f"   ⚠️  Unseen player(s) in this match: {data['unseen_players']} "
              f"(model used rank-estimated ELO — see model_used: '{data.get('model_used')}')")

    return winner, round(win_pct, 1)


# ============================================================
# 2. MARKET ODDS — The Odds API lookup + vig removal
# ============================================================
def get_market_prediction(player1: str, player2: str):
    """
    Fetches current odds for the given match from The Odds API,
    converts decimal odds -> implied probability, removes the
    bookmaker's margin (vig) so both players' % sum to 100.

    Returns: (predicted_winner_name, win_pct) or (None, None) if not found.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    matches = resp.json()

    target = {_norm(player1), _norm(player2)}

    for m in matches:
        names = {_norm(m.get("home_team", "")), _norm(m.get("away_team", ""))}
        # fuzzy-ish: check both target names appear as substrings of the match's names
        if _names_match(target, names):
            if not m.get("bookmakers"):
                continue
            # use the first bookmaker that has an h2h market (or average across all — simple version: first one)
            for bk in m["bookmakers"]:
                h2h = next((mk for mk in bk["markets"] if mk["key"] == "h2h"), None)
                if not h2h:
                    continue
                outcomes = h2h["outcomes"]
                if len(outcomes) != 2:
                    continue

                o1, o2 = outcomes[0], outcomes[1]
                p1_implied = 1 / o1["price"]
                p2_implied = 1 / o2["price"]
                total = p1_implied + p2_implied  # >1 due to vig
                p1_fair = p1_implied / total
                p2_fair = p2_implied / total

                if p1_fair >= p2_fair:
                    return o1["name"], round(p1_fair * 100, 1)
                else:
                    return o2["name"], round(p2_fair * 100, 1)

    return None, None


def _norm(name: str) -> str:
    return name.lower().strip()


def _names_match(target: set, names: set) -> bool:
    # crude but effective: every target name must appear as a substring of
    # at least one name in `names` (handles "C. Alcaraz" vs "Carlos Alcaraz")
    for t in target:
        last_name = t.split()[-1]
        if not any(last_name in n for n in names):
            return False
    return True


# ============================================================
# 3. EXCEL LOGGING
# ============================================================
HEADERS = [
    "Match",
    "My Prediction",
    "My %",
    "IBM Prediction",
    "IBM %",
    "Real Winner",
    "My Accuracy",
    "IBM Accuracy",
]


def _ensure_workbook():
    if os.path.exists(EXCEL_PATH):
        wb = load_workbook(EXCEL_PATH)
        ws = wb[SHEET_NAME]
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = SHEET_NAME
        ws.append(HEADERS)
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        for col_idx, _ in enumerate(HEADERS, start=1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        for col_idx, width in enumerate([28, 16, 8, 22, 8, 16, 14, 14], start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = width
    return wb, ws


def log_match(player1: str, player2: str):
    wb, ws = _ensure_workbook()

    match_label = f"{player1} vs {player2}"
    for row_idx in range(2, ws.max_row + 1):
        existing = ws.cell(row=row_idx, column=1).value
        if existing == match_label:
            print(f"⚠️  '{match_label}' is already logged (row {row_idx}). Skipping duplicate.")
            return

    my_pred, my_pct = get_my_prediction(player1, player2)
    market_pred, market_pct = get_market_prediction(player1, player2)

    if market_pred is None:
        print(f"⚠️  Couldn't find market odds for {player1} vs {player2} yet "
              f"(match may not be listed / too far out / API has no data). "
              f"Logging your prediction with market fields blank.")

    row = [
        f"{player1} vs {player2}",
        my_pred,
        f"{my_pct}%" if my_pct is not None else "",
        market_pred or "",
        f"{market_pct}%" if market_pct is not None else "",
        "",   # Real Winner — fill in after match
        "",   # My Accuracy — recalculated below
        "",   # IBM Accuracy — recalculated below
    ]
    ws.append(row)

    _recalculate_accuracy(ws)

    wb.save(EXCEL_PATH)
    market_pred_display = market_pred if market_pred else "Not available yet"
    market_pct_display = f"{market_pct}%" if market_pct is not None else ""
    print(f"✅ Logged: {player1} vs {player2}")
    print(f"   My pick: {my_pred} ({my_pct}%) | IBM pick: {market_pred_display} {market_pct_display}")
    print(f"   Saved to {EXCEL_PATH}")


def _recalculate_accuracy(ws):
    """Recompute running accuracy for BOTH My Prediction and IBM columns,
    for every row that has a Real Winner filled in."""
    my_correct = 0
    market_correct = 0
    total = 0
    market_total = 0  # only counts rows where IBM actually had a prediction

    for row_idx in range(2, ws.max_row + 1):
        my_pred = ws.cell(row=row_idx, column=2).value
        market_pred = ws.cell(row=row_idx, column=4).value
        real_winner = ws.cell(row=row_idx, column=6).value

        if not real_winner:
            continue
        total += 1

        my_pred_cell = ws.cell(row=row_idx, column=2)
        is_my_correct = bool(my_pred) and real_winner.strip().lower() in my_pred.strip().lower()
        if is_my_correct:
            my_correct += 1
        my_pred_cell.fill = PatternFill(
            start_color="C6EFCE" if is_my_correct else "FFC7CE",
            end_color="C6EFCE" if is_my_correct else "FFC7CE",
            fill_type="solid",
        )

        if market_pred:
            market_total += 1
            market_pred_cell = ws.cell(row=row_idx, column=4)
            is_market_correct = real_winner.strip().lower() in market_pred.strip().lower()
            if is_market_correct:
                market_correct += 1
            market_pred_cell.fill = PatternFill(
                start_color="C6EFCE" if is_market_correct else "FFC7CE",
                end_color="C6EFCE" if is_market_correct else "FFC7CE",
                fill_type="solid",
            )

        my_acc = round((my_correct / total) * 100, 1)
        ws.cell(row=row_idx, column=7).value = f"{my_acc}%"

        if market_total:
            market_acc = round((market_correct / market_total) * 100, 1)
            ws.cell(row=row_idx, column=8).value = f"{market_acc}%"
        else:
            ws.cell(row=row_idx, column=8).value = "N/A"


def refresh_odds():
    """Re-check market odds for every existing row still missing them, update in place."""
    if not os.path.exists(EXCEL_PATH):
        print(f"No file found at {EXCEL_PATH} yet.")
        return

    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]

    checked = 0
    updated = 0
    for row_idx in range(2, ws.max_row + 1):
        match_label = ws.cell(row=row_idx, column=1).value
        market_pred_cell = ws.cell(row=row_idx, column=4)
        market_pct_cell = ws.cell(row=row_idx, column=5)

        if not match_label:
            continue
        if market_pred_cell.value:  # already has a value, skip
            continue

        if " vs " not in match_label:
            continue
        player1, player2 = match_label.split(" vs ", 1)

        checked += 1
        market_pred, market_pct = get_market_prediction(player1, player2)

        if market_pred is not None:
            market_pred_cell.value = market_pred
            market_pct_cell.value = f"{market_pct}%"
            updated += 1
            print(f"   ✅ Found IBM pick for {match_label}: {market_pred} ({market_pct}%)")
        else:
            print(f"   ⏳ Still no odds for {match_label}")

    wb.save(EXCEL_PATH)
    print(f"\nChecked {checked} row(s) missing odds, updated {updated}.")


def fill_results_interactive():
    """Walks through every row missing a Real Winner, lets you type the winner quickly."""
    if not os.path.exists(EXCEL_PATH):
        print(f"No file found at {EXCEL_PATH} yet.")
        return

    wb = load_workbook(EXCEL_PATH)
    ws = wb[SHEET_NAME]

    pending_rows = []
    for row_idx in range(2, ws.max_row + 1):
        match_label = ws.cell(row=row_idx, column=1).value
        real_winner = ws.cell(row=row_idx, column=6).value
        if match_label and not real_winner:
            pending_rows.append((row_idx, match_label))

    if not pending_rows:
        print("No matches are missing a Real Winner. Nothing to fill in.")
        return

    print(f"{len(pending_rows)} match(es) missing a result:\n")
    for row_idx, match_label in pending_rows:
        winner = input(f"  {match_label}\n  Winner (paste exact name, or 's' to skip): ").strip()
        if winner.lower() == "s":
            continue
        ws.cell(row=row_idx, column=6).value = winner

    _recalculate_accuracy(ws)
    wb.save(EXCEL_PATH)
    print(f"\n✅ Saved. Run 'python log_match.py --recalculate' anytime to refresh accuracy again.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--fill-results":
        fill_results_interactive()
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] == "--refresh-odds":
        refresh_odds()
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] == "--recalculate":
        if not os.path.exists(EXCEL_PATH):
            print(f"No file found at {EXCEL_PATH} yet.")
            sys.exit(1)
        wb = load_workbook(EXCEL_PATH)
        ws = wb[SHEET_NAME]
        _recalculate_accuracy(ws)
        wb.save(EXCEL_PATH)

        total = sum(1 for r in range(2, ws.max_row + 1) if ws.cell(row=r, column=6).value)
        if total:
            last_my_acc = ws.cell(row=ws.max_row, column=7).value
            last_market_acc = ws.cell(row=ws.max_row, column=8).value
            print(f"✅ Recalculated. {total} matches have a Real Winner filled in.")
            print(f"   My accuracy:   {last_my_acc}")
            print(f"   IBM accuracy:  {last_market_acc}")
        else:
            print("No matches have a Real Winner filled in yet — nothing to calculate.")
        sys.exit(0)

    if len(sys.argv) != 3:
        print('Usage:')
        print('  python log_match.py "Player A" "Player B"   -> log a new match')
        print('  python log_match.py --recalculate            -> refresh accuracy after filling in winners')
        print('  python log_match.py --refresh-odds            -> re-check market odds for rows still missing them')
        print('  python log_match.py --fill-results            -> interactively type in winners for pending matches')
        sys.exit(1)

    p1, p2 = sys.argv[1], sys.argv[2]
    log_match(p1, p2)