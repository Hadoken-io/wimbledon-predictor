# 🎾 Wimbledon 2026 Predictor

**Data-driven match predictions + Monte Carlo bracket simulation for Wimbledon 2026.**

Built with XGBoost, GrassDNA, Elo decay, and live bookmaker-odds comparison.  
Runs a local FastAPI backend, a bracket simulator, and a versus tracker to log predictions against real results.

---

## 🚀 What it does

- **Predicts match outcomes** using two gradient-boosted models trained on ATP grass-court history
- **Simulates full 128-player brackets** via Monte Carlo (10k+ simulations)
- **Compares model vs market** by pulling live odds from The Odds API and removing vig
- **Logs predictions** in an Excel tracker and recalculates running accuracy as real results come in
- **Handles unseen players** (qualifiers, wildcards) by estimating rank → Elo curves

---

## 🧠 How it works

### 1. Feature engineering (`scripts/`)
| Script | Purpose |
|--------|---------|
| `clean_data.py` | Loads raw ATP data, filters grass/Wimbledon |
| `build_elo.py` | Computes recency-weighted Elo with grass-surface decay |
| `build_feature.py` | Builds `features.csv` with Elo diffs, form, H2H, Wimbledon history |
| `grass_elo.py` | Grass-specific Elo ratings |
| `build_serve_stats.py` | Aggregates Match Charting Project serve stats |
| `merge_serve_stats.py` | Merges serve stats into feature set |
| `calibration.py` | Checks model calibration (reliability) |
| `train.py` | Trains **General** + **Elite** XGBoost models |

### 2. Two-model system
- **General model** — all grass matches post-2010  
- **Elite model** — both players ranked ≤ 50 (used for QF+ rounds)

### 3. FastAPI backend (`app.py`)
- `POST /predict-match` — winner + confidence for any matchup
- `POST /simulate-bracket` — 10k-run Monte Carlo bracket
- `GET /player/{name}` — player profile (Elo, GrassDNA, rank)

### 4. Bracket simulator (`scripts/run_simulation.py`)
- Loads a 128-player draw
- Uses general model R1–R3, elite model R4–Final
- Outputs title probability, SF%, QF% per player

### 5. Versus tracker (`log_match.py`)
- Calls your local API for the model pick
- Calls The Odds API for the market pick
- Logs to Excel with color-coded accuracy tracking

---

## 📊 Model features

- Elo Overall / Grass / Wimbledon differentials
- GrassDNA (surface-specific style vector)
- Recent form & grass-form streaks
- Head-to-head win rate (all surface + grass only)
- Wimbledon historical performance
- Rank differential + elite-match flag

---

## 🛠️ Setup

```bash
git clone https://github.com/Hadoken-io/wimbledon-predictor.git
cd wimbledon-predictor

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Configure environment
```bash
cp .env.example .env
# Edit .env and add your ODDS_API_KEY from https://the-odds-api.com
```

### Train models
```bash
python scripts/train.py
```

### Start API server
```bash
uvicorn app:app --reload --port 8000
```

### Run bracket simulation
```bash
python scripts/run_simulation.py
```

### Log a match prediction
```bash
python log_match.py "Carlos Alcaraz" "Novak Djokovic"
```

---

## 📁 Project structure

```
wimbledon-predictor/
├── app.py                      # FastAPI backend
├── log_match.py                # Excel versus tracker
├── rank_to_elo.py              # Unseen-player rank → Elo estimator
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── scripts/
│   ├── train.py                # Train general + elite XGBoost models
│   ├── run_simulation.py       # Monte Carlo bracket simulator
│   ├── build_elo.py            # Elo with grass decay
│   ├── build_feature.py        # Feature matrix builder
│   ├── build_serve_stats.py    # Serve stat aggregation
│   ├── calibration.py          # Probability calibration check
│   ├── real_matchup_check.py   # Spot-check known matches
│   ├── test_elo_decay.py       # Elo decay validation
│   └── ...
└── visuals/
    ├── bracket.html            # Interactive bracket UI
    └── compare.html            # Model vs odds comparison
```

---

## 🎯 Results & accuracy

The elite model is validated specifically on Wimbledon 2023–2025 matches.  
Accuracy and log-loss are tracked against both bookmaker odds and historical SlamTracker data.

Run `python test_consistency.py` to see per-year Wimbledon accuracy.

---

## ⚠️ Notes

- **Data files** (`data_raw/`, `data_processed/`, `models/`) are gitignored.  
  Run the scripts to regenerate them from ATP + Match Charting Project sources.
- **API key** is stored in `.env` (gitignored). Use `.env.example` as a template.
- **Odds API** only returns Wimbledon data once the tournament is “in season”.

---

## 📜 License

MIT — feel free to fork, remix, and build your own Slam predictor.

---

*Built for Wimbledon 2026. May the best Elo win.*
