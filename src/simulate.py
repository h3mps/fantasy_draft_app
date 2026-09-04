import argparse
import numpy as np
import pandas as pd

def simulate_drafts(board: pd.DataFrame, picks: pd.DataFrame, your_team: str = None, rounds: int = 12, sims: int = 200, pick_noise: float = 8.0, draft_order: list | None = None):
    # Construct order if not provided
    teams = picks["Team"].unique().tolist()
    teams.sort()
    if draft_order is None:
        draft_order = teams.copy()
    n_teams = len(draft_order)
    # Build snake order
    order = []
    for r in range(1, rounds+1):
        seq = draft_order if r % 2 == 1 else list(reversed(draft_order))
        for t in seq:
            order.append((r, t))
    board = board.copy().reset_index(drop=True)
    results = []
    for s in range(sims):
        available = board.copy()
        available["ADP_sim"] = available["ADP"] + np.random.normal(0, pick_noise, size=len(available))
        taken = set()
        roster = {t: [] for t in teams}
        for (rnd, team) in order:
            # Opponents pick by ADP_sim of remaining
            pool = available[~available["Player"].isin(taken)].sort_values("ADP_sim")
            if pool.empty:
                break
            pick = pool.iloc[0]
            taken.add(pick["Player"])
            roster[team].append((rnd, pick["Player"], pick["Position"], float(pick["ADP"])))
        # Evaluate your team
        if your_team is not None:
            yours = roster.get(your_team, [])
            val = sum(200 - p[3] for p in yours)  # crude value metric
            results.append({"sim": s, "your_value": val, "players": "; ".join([p[1] for p in yours])})
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adp", required=True)
    ap.add_argument("--keepers", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--your_team", type=str, default="Alpha")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adp = pd.read_csv(args.adp)
    keepers = pd.read_csv(args.keepers)
    from .rankings import make_post_keeper_board
    board = make_post_keeper_board(adp, keepers)
    picks = pd.read_csv(args.picks)
    simres = simulate_drafts(board, picks, your_team=args.your_team, rounds=args.rounds, sims=args.sims)
    simres.to_csv(args.out, index=False)
    print(f"Wrote {len(simres)} simulation rows to {args.out}")

if __name__ == "__main__":
    main()
