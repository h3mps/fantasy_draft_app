# Fantasy Draft Assistant (Dynasty, Keepers, Simulations)

A multi-stage toolkit to:
1) Predict likely keepers by team.
2) Produce post-keeper rankings and tiers.
3) Simulate drafts under uncertainty (ADP + noise, multiple orders).
4) Run a live **Streamlit** draft room with dynamic best-available and suggestions.

## Quickstart (offline, with sample data)
```bash
pip install -r requirements.txt
# Run simulations and keeper predictions
python -m src.keepers --rosters data/sample_rosters.csv --adp data/sample_adp.csv --picks data/sample_picks.csv --out data/predicted_keepers.csv
python -m src.simulate --adp data/sample_adp.csv --keepers data/predicted_keepers.csv --picks data/sample_picks.csv --rounds 12 --sims 200 --out data/sim_summary.csv

# Launch draft room (loads CSVs, lets you tick off picks)
streamlit run app/streamlit_app.py
```

## Files
- `data/sample_rosters.csv` — sample league rosters (Team, Player, Position, LastDraftRound, DraftYear)
- `data/sample_adp.csv` — sample ADP (Player, Position, ADP, ECR, ByeWeek, Team)
- `data/sample_picks.csv` — sample draft capital (Team, Round, PickOverall, HasPick)
- `data/predicted_keepers.csv` — output from keeper model
- `data/sim_summary.csv` — output from simulations

## Notes
- All online fetches are stubbed. Replace with real API calls in `src/data_sources.py` (Sleeper/ESPN/FantasyPros/etc.).
- The keeper algorithm is deliberately transparent and editable; tune the heuristics to your league.
