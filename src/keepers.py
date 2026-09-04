import argparse
import pandas as pd
import numpy as np

# Heuristic keeper model for dynasty with 2-round earlier cost and max 3 keepers.
# If a player was drafted in Round r last year, keeping him costs Round max(1, r-2).
# Value score compares implied cost (pick #) to ADP; lower ADP is better.
# We penalize QB/TE slightly in 1QB leagues and bump elite players.

POS_BASELINE = {"RB": 0.0, "WR": 0.0, "QB": -5.0, "TE": -2.0}

def round_to_pick(round_num: int, picks_per_round: int = 12) -> int:
    return (round_num - 1) * picks_per_round + int(np.ceil(picks_per_round/2))

def compute_value(row, picks_per_round=12):
    # cost round is two rounds earlier, floor at 1
    cost_round = max(1, int(row.get("LastDraftRound", 10)) - 2)
    cost_pick = round_to_pick(cost_round, picks_per_round)
    adp = row.get("ADP", 200.0)
    pos = row.get("Position", "WR")
    # value = how much earlier ADP is vs cost pick (positive is good)
    value = (cost_pick - adp) + POS_BASELINE.get(pos, 0.0)
    # small bump for elite ADP
    if adp <= 12:
        value += 5.0
    return pd.Series({"CostRound": cost_round, "CostPick": cost_pick, "Value": value})

def predict_keepers(rosters_df: pd.DataFrame, adp_df: pd.DataFrame, max_keepers_per_team: int = 3, picks_per_round: int = 12) -> pd.DataFrame:
    merged = rosters_df.merge(adp_df, on=["Player","Position"], how="left")
    vals = merged.apply(lambda r: compute_value(r, picks_per_round), axis=1)
    merged = pd.concat([merged, vals], axis=1)
    # For each team, pick top N value >= threshold
    # Threshold can be tuned; here we accept any positive value (or top by value if none positive)
    keepers = []
    for team, g in merged.groupby("FTeam"):
        g_sorted = g.sort_values(["Value","ADP"], ascending=[False, True]).reset_index(drop=True)
        selected = g_sorted[g_sorted["Value"] > 0].head(max_keepers_per_team)
        if selected.empty:
            selected = g_sorted.head(max_keepers_per_team)
        selected = selected.copy()
        selected["Keep"] = 1
        keepers.append(selected)
    keepers_df = pd.concat(keepers, ignore_index=True)
    cols = ["FTeam","Player","Position","CostRound","CostPick","ADP","Value"]
    return keepers_df[cols].sort_values(["FTeam","Value"], ascending=[True,False])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters", required=True)
    ap.add_argument("--adp", required=True)
    ap.add_argument("--picks", required=True, help="Not used in v1 heuristic; placeholder for validating keeper costs.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppr", type=int, default=1, help="1 for PPR, 0 for standard (adjust heuristics in future)")
    args = ap.parse_args()

    rosters = pd.read_csv(args.rosters)
    adp = pd.read_csv(args.adp)
    picks = pd.read_csv(args.picks)  # placeholder
    keepers = predict_keepers(rosters, adp)
    keepers.to_csv(args.out, index=False)
    print(f"Wrote predicted keepers to {args.out}")

if __name__ == "__main__":
    main()
