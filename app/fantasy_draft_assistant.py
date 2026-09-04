import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
import re, json, os, sys
from collections import Counter
import numpy as np
import streamlit as st
import statsmodels.api as sm


def _kmeans_2d(X, k, max_iter=50):
    if len(X) == 0 or k <= 0:
        return np.array([], dtype=int)

    n = X.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)

    order = np.argsort(X[:, 1])
    centroids = X[order[np.linspace(0, n - 1, k, dtype=int)]]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
            else:
                centroids[j] = X[np.random.randint(0, n)]
    return labels


def _assign_tiers_by_position(df, position_col="Position", score_col="Fan Pts", rank_col="Pre-Season"):
    tiers = pd.Series(index=df.index, dtype="object")

    def _log_boundaries(size):
        if size <= 8:
            return list(range(0, size + 1))
        targets = [2, 5, 10, 18, 28, 40, 55]
        boundaries = [0]
        for t in targets:
            if t < size:
                boundaries.append(t)
            else:
                break
        boundaries.append(size)
        return boundaries

    for position, group in df.groupby(position_col, sort=False):
        pos_idx = group.index
        fan_pts = pd.to_numeric(group[score_col], errors="coerce")
        rank = pd.to_numeric(group[rank_col], errors="coerce")
        valid = fan_pts.notna() & rank.notna()

        if valid.sum() < 2:
            tiers.loc[pos_idx] = None
            continue

        ordered = group.loc[valid].sort_values(by=[score_col, rank_col], ascending=[False, True]).index
        scores = fan_pts.loc[ordered].values

        boundaries = _log_boundaries(len(scores))
        if boundaries[-1] != len(scores):
            boundaries[-1] = len(scores)

        if len(boundaries) - 1 > 8:
            boundaries = boundaries[:8] + [len(scores)]

        tier_labels = []
        for tier_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            label = f"{str(position).strip().upper()}{tier_index + 1}"
            tier_labels.extend([label] * (end - start))

        if len(tier_labels) != len(ordered):
            tier_labels = [f"{str(position).strip().upper()}{i+1}" for i in range(len(ordered))]

        tiers.loc[ordered] = tier_labels
        tiers.loc[pos_idx[~valid]] = None

    return tiers


sys.path.append(str(Path(__file__).parent.parent / "src"))
from read_yahoo import *

st.set_page_config(page_title="Fantasy Prep", layout="wide")
st.title("🏈 Fantasy Draft Preparation Assistant")
st.caption("Download Fantasy League Data, ADPs and Rankings and upload your keepers and draft order to get a good look at your likely decisions to make on draft day!")

# ----------------------------
# --- Data Pull Section
# ----------------------------

# --- Yahoo League Data Import

# Load cookies once (adjust path)
with open("./data/cookies.json", "r") as f:
    cookies = json.load(f)
cookies_dict = {c['name']: c['value'] for c in cookies}

default_league_id_current = "283250"  # Current year league ID
default_league_id_previous = "117238"  # Previous year league ID

st.sidebar.header("Yahoo League Data Import")
league_id_current = st.sidebar.text_input("Current Year League ID", value=default_league_id_current)
league_id_previous = st.sidebar.text_input("Previous Year League ID", value=default_league_id_previous)

if st.sidebar.button("Fetch Yahoo Data"):
    if not league_id_current or not league_id_previous:
        st.sidebar.error("Please enter both league IDs.")
    else:
        status_sidebar = st.sidebar.empty()

        yahoo_dir = Path(f"data/yahoo/yahoo_{league_id_current}")
        yahoo_dir.mkdir(parents=True, exist_ok=True)

        # Last Year's Draft
        status_sidebar.info("Pulling last year's draft...")
        df_last_draft = pull_last_year_draft(league_id_previous, yahoo_dir, cookies_dict)
        status_sidebar.success(f"Saved last year's draft: {len(df_last_draft)} rows")
        status_sidebar.empty()  # clear message

        # Last Year's Dropped Players
        status_sidebar.info("Pulling dropped players...")
        df_dropped = pull_last_year_dropped(league_id_previous, yahoo_dir, cookies_dict)
        status_sidebar.success(f"Saved dropped players: {len(df_dropped)} rows")
        status_sidebar.empty()

        # Last Season's Rosters
        status_sidebar.info("Pulling last season's rosters...")
        df_rosters = pull_last_season_rosters(league_id_current, yahoo_dir, cookies_dict)
        status_sidebar.success(f"Saved last season's roster: {len(df_rosters)} rows")
        status_sidebar.empty()

        # Last Season's Traded Picks
        status_sidebar.info("Pulling traded picks...")
        df_traded = pull_traded_picks(league_id_current, yahoo_dir, cookies_dict)
        status_sidebar.success(f"Saved traded picks: {len(df_traded)} rows")
        status_sidebar.empty()

        # Last Season's Traded Picks
        status_sidebar.info("Pulling projected points...")
        df_projected = pull_projected_points(league_id_current, yahoo_dir, cookies_dict)
        status_sidebar.success(f"Saved {len(df_projected)} player projections")
        status_sidebar.empty()

        # Final message
        status_sidebar.info("Yahoo data fetch complete!")

st.sidebar.caption("Make sure to periodically update your cookies.json file to ensure access to Yahoo's API. You can find instructions in the README.")

# ----------------------------
# --- Draft Inputs
# ----------------------------

st.text("League to Analyze")
league_id_chosen = st.text_input("Chosen League ID", value=default_league_id_current)

with st.expander("Draft Order Settings", expanded=False):
    # --- Overall Structure ---
    outer1, outer2 = st.columns([2, 3])

    with outer1:
        st.subheader("League Inputs")
        col1, col2, col3 = st.columns([2, 1, 1])  # ratio: radio wider, others narrower
        # Step 1: Initial Selections
        with col1:
            # --- Sel 1: Draft type ---
            draft_type = st.radio("Draft Order Type", ["Known", "Partially Unknown"])

        with col2:
            # --- Sel 2: League size ---
            league_size = st.number_input("League Size", min_value=2, max_value=32, value=14)

        with col3:
            # --- Sel 3: Number of rounds ---
            num_rounds = st.number_input("Rounds", min_value=1, max_value=30, value=15)

        # --- Step 2: Create pick inputs ---
        # --- Load teams from traded picks CSV ---
        traded_picks_path = f"data/yahoo/yahoo_{league_id_chosen}/traded_picks.csv"
        df_traded = pd.read_csv(traded_picks_path)
        teams = sorted(df_traded["Owned By"].unique())

        # --- Take Pick Slot Inputs ---
        pick_inputs = []
        if draft_type == "Partially Unknown":
            st.markdown("### Enter known picks (leave unknown blank)")
            for pick_num in range(1, league_size+1):
                team = st.selectbox(f"Pick {pick_num}", [""] + teams, key=f"pick_{pick_num}")
                pick_inputs.append(team)
        else:
            # Known draft order: just let the user select the order directly
            st.markdown("### Enter full draft order")
            for pick_num in range(1, league_size+1):
                team = st.selectbox(f"Pick {pick_num}", teams, key=f"pick_{pick_num}")
                pick_inputs.append(team)

        # --- Step 3: Simulate (if necessary) & Save ---
        if st.button("Generate Draft Order"):
            # --- Fill only the blank slots in the first round
            remaining_teams = [t for t in teams if t not in pick_inputs]
            first_round = []
            for team in pick_inputs:
                if team:
                    first_round.append(team)
                else:
                    chosen = remaining_teams.pop(0)  # assign remaining teams in order
                    first_round.append(chosen)
            
            # Step 2: Generate full snake draft
            draft_order = []
            for r in range(1, num_rounds + 1):
                if r % 2 == 1:  # odd round = same as first
                    round_order = first_round
                else:           # even round = reversed
                    round_order = list(reversed(first_round))
                for idx, team in enumerate(round_order):
                    pick_number = (r - 1) * league_size + (idx + 1)
                    draft_order.append({
                        "pick_number": pick_number,
                        "round": r,
                        "orig_owner": team   # <-- keep original owner slot
                    })

            # Save to session_state so col2 can access it
            st.session_state.df_draft = pd.DataFrame(draft_order)
        
    with outer2:
        st.subheader("Draft Order")

        # Ensure df_draft always exists with the full expected columns
        if "df_draft" not in st.session_state:
            st.session_state.df_draft = pd.DataFrame({
                "pick_number": [],
                "round": [],
                "orig_owner": []  # needed for merge later
            })

        df_draft = st.session_state.df_draft  # safe to use

        # Step 3: Merge traded picks
        traded_picks_path = f"data/yahoo/yahoo_{league_id_chosen}/traded_picks.csv"
        try:
            traded_picks = pd.read_csv(traded_picks_path)
            traded_picks.columns = traded_picks.columns.str.strip()

            # Merge on (round, orig_owner)
            df_draft = df_draft.merge(
                traded_picks[["Round", "Orig Pick Owner", "Owned By"]],
                left_on=["round", "orig_owner"],
                right_on=["Round", "Orig Pick Owner"],
                how="left"
            )

            # Final team is "Owned By" if traded, otherwise the original owner
            df_draft["team"] = df_draft["Owned By"].fillna(df_draft["orig_owner"])

            df_draft.drop(columns=["Round", "Orig Pick Owner", "Owned By"], inplace=True)

        except FileNotFoundError:
            st.warning("Traded picks file not found — continuing without merging traded picks.")
            df_draft["team"] = df_draft["orig_owner"]

        # Save back to session_state
        st.session_state.df_draft = df_draft

        # Display
        st.dataframe(df_draft.fillna(""), use_container_width=True)

        if not df_draft.empty:
            if draft_type == "Partially Unknown":
                save_path = "data/draft_order_simulated.csv"
                df_draft.to_csv(save_path, index=False)
                st.success(f"Draft order saved to {save_path}")
            else:
                save_path = "data/draft_order_official.csv"
                df_draft.to_csv(save_path, index=False)
                st.success(f"Draft order saved to {save_path}")

        if not df_draft.empty:
            teams_in_draft = df_draft["team"].dropna().unique().tolist()
            selected_team = st.selectbox("Select a team to view their picks", teams_in_draft)

            # Filter draft picks for that team
            team_picks = df_draft[df_draft["team"] == selected_team].sort_values("pick_number")
            st.write(f"Draft picks for {selected_team}:")
            st.dataframe(team_picks[["pick_number", "round"]], use_container_width=True)


# ----------------------------
# --- Keeper Inputs
# ----------------------------

with st.expander("Keepers"):
    # --- Choose how to determine keepers ---
    keeper_option = st.radio(
        "Choose how to determine keepers:",
        ["Predict Keepers", "Input Official Keepers"]
    )

    # --- If predicting keepers, we need to load Projected Rank and rosters ---
    if keeper_option == "Predict Keepers":
        # --- 1. Draft Order selection ---
        draft_order_choice = st.selectbox(
            "Select Draft Order to Use",
            options=["Simulated", "Official"]
        )

        # --- Load the corresponding CSV
        if draft_order_choice == "Simulated":
            df_draft_path = "data/draft_order_simulated.csv"
        else:
            df_draft_path = "data/draft_order_official.csv"

        # Load into df_draft
        try:
            df_draft = pd.read_csv(df_draft_path)
        except FileNotFoundError:
            st.error(f"{df_draft_path} not found. Make sure the file exists.")
            df_draft = pd.DataFrame(columns=["pick_number", "round", "team", "orig_owner"])

        # --- 2. Load Last Season Rosters for Eligible Players ---
        rosters_path = f"data/yahoo/yahoo_{league_id_chosen}/last_season_rosters.csv"
        df_rosters = pd.read_csv(rosters_path)

        # Ensure column names are clean
        df_rosters.columns = df_rosters.columns.str.strip()

        # Filter out dropped or undrafted players
        dropped_path = f"data/yahoo/yahoo_{league_id_chosen}/last_dropped_players.csv"
        try:
            df_dropped = pd.read_csv(dropped_path)
        except FileNotFoundError:
            df_dropped = pd.DataFrame(columns=["Player"])

        # Eligible players: drafted and not dropped
        eligible_players = df_rosters[
            (~df_rosters["Player"].isin(df_dropped["Player"])) &
            (pd.to_numeric(df_rosters["DraftRound"], errors="coerce") > 2)
        ].copy()

        # Adjust draft round (subtract 2)
        eligible_players["DraftPickCost"] = pd.to_numeric(eligible_players["DraftRound"], errors="coerce") - 2
        # Ensure ProjRank is numeric
        eligible_players["ProjRank"] = pd.to_numeric(eligible_players["ProjRank"], errors="coerce")

        # --- 3. Add Pick Required to Keep Player ---
        # Create a lookup of pick_number for each (team, round)
        pick_lookup = df_draft.set_index(["team", "round"])["pick_number"].to_dict()

        # Fill pick_number based on DraftPickCost
        eligible_players["pick_number"] = eligible_players.apply(
            lambda row: pick_lookup.get((row["FTeam"], row["DraftPickCost"]), np.nan),
            axis=1
        )

        # For any remaining missing pick_number, try one round earlier
        mask_missing = eligible_players["pick_number"].isna()
        eligible_players.loc[mask_missing, "pick_number"] = eligible_players.loc[mask_missing].apply(
            lambda row: pick_lookup.get((row["FTeam"], row["DraftPickCost"] - 1), np.nan),
            axis=1
        )

        eligible_players["pick_number"] = eligible_players["pick_number"].fillna(0)

        # --- 4. Calculate Upd_ProjRank and keepers ---
        keepers = pd.DataFrame(columns=eligible_players.columns.tolist() + ["Upd_ProjRank"])
        eligible_players["Upd_ProjRank"] = eligible_players["ProjRank"]

        max_keepers_per_team = 3
        keepers_per_team = {team: 0 for team in eligible_players["FTeam"].unique()}

        new_keepers_added = True

        while new_keepers_added:
            new_keepers_added = False
            
            # Step 1: Identify keeper candidates
            candidates = eligible_players[
                (eligible_players["pick_number"] > 1.25 * eligible_players["Upd_ProjRank"]) &
                (eligible_players["FTeam"].map(keepers_per_team) < 3) &
                (~eligible_players["Player"].isin(keepers["Player"]))
            ].copy()
            
            # Step 2: Limit to top 3 per team
            candidates = candidates.sort_values(["FTeam", "ProjRank"])
            top_candidates = candidates.groupby("FTeam").head(max_keepers_per_team)
            
            # Step 3: Remove already kept players
            top_candidates = top_candidates[~top_candidates["Player"].isin(keepers["Player"])]
            
            if not top_candidates.empty:
                # Step 4: Add to keepers
                keepers = pd.concat([keepers, top_candidates], ignore_index=True)
                new_keepers_added = True

                for team, n in top_candidates["FTeam"].value_counts().items():
                    keepers_per_team[team] += n
            
                # Step 5: Update Upd_ProjRank for remaining players
                remaining = eligible_players[~eligible_players["Player"].isin(keepers["Player"])].copy()
                
                for idx, row in remaining.iterrows():
                    # Count how many already kept players have ProjRank < current player's ProjRank
                    moved_up = sum(keepers["ProjRank"] < row["ProjRank"])
                    
                    # Count how many already kept players have pick_number < current player's pick_number
                    blocked = sum(keepers["pick_number"] < row["pick_number"])
                    
                    # Update
                    eligible_players.loc[idx, "Upd_ProjRank"] = row["ProjRank"] - moved_up + blocked

        
        # --- Streamlit display ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Predicted Keepers")
            st.dataframe(keepers[['Player', 'FTeam', 'pick_number', 'ProjRank', 'Upd_ProjRank']].fillna(""), use_container_width=True)
        
        eligible_players["Kept"] = eligible_players.apply(
            lambda row: ((keepers["Player"] == row["Player"]) & (keepers["FTeam"] == row["FTeam"])).any(),
            axis=1
        )
        keeper_edge_cases = eligible_players[
            (~eligible_players["Kept"]) &
            (eligible_players["pick_number"] > eligible_players["Upd_ProjRank"])
        ].copy()

        with col2:
            st.subheader("Keeper Edge Cases")
            st.dataframe(keeper_edge_cases[['Player', 'FTeam', 'pick_number', 'ProjRank', 'Upd_ProjRank']].fillna(""), use_container_width=True)

        # Save to CSV
        save_path = f"data/keepers_simulated.csv"
        keepers.to_csv(save_path, index=False)
        st.success(f"Predicted keepers saved to {save_path}")

        # Save to CSV
        save_path = f"data/keepers_edge_cases.csv"
        keeper_edge_cases.to_csv(save_path, index=False)
        st.success(f"Keeper Edge Cases saved to {save_path}")

    elif keeper_option == "Input Official Keepers":
        # Load last season's rosters
        try:
            rosters_path = f"data/yahoo/yahoo_{league_id_chosen}/last_season_rosters.csv"
            df_rosters = pd.read_csv(rosters_path)
            teams = df_rosters['FTeam'].unique()
        except FileNotFoundError:
            st.error("Last season rosters file not found.")
            df_rosters = pd.DataFrame()
            teams = []

        # Load dropped players
        try:
            dropped_path = f"data/yahoo/yahoo_{league_id_chosen}/last_dropped_players.csv"
            df_dropped = pd.read_csv(dropped_path)
            dropped_players = df_dropped['Player'].tolist()
        except FileNotFoundError:
            dropped_players = []

        df_rosters = df_rosters[~df_rosters['Player'].isin(df_dropped['Player'])]
        df_rosters = df_rosters[df_rosters['DraftRound'].notna()]

        # Create a multiselect box for each team
        team_keepers = {}
        for team in teams:
            # Filter for this team and rounds after 2
            team_roster = df_rosters[(df_rosters['FTeam'] == team) & (df_rosters['DraftRound'] > 2)].copy()
            
            # Adjust draft round by subtracting 2
            team_roster['DraftPickCost'] = team_roster['DraftRound'] - 2

            # Multiselect for user to pick official keepers
            selected = st.multiselect(
                f"{team} Keepers",
                options=team_roster['Player'].tolist(),
                format_func=lambda x: f"{x} (Round {team_roster.loc[team_roster['Player']==x, 'DraftPickCost'].values[0]})"
            )
            
            # Store selection along with adjusted draft round
            team_keepers[team] = team_roster[team_roster['Player'].isin(selected)][['Player','DraftPickCost']]

        # Button to save all keepers
        if st.button("Submit Official Keepers"):
            # Combine all teams into a single dataframe
            keepers_df = pd.concat(team_keepers.values(), keys=team_keepers.keys(), names=['Team']).reset_index(level=0).rename(columns={'level_0':'Team'})
            
            save_path = "data/keepers_official.csv"
            keepers_df.to_csv(save_path, index=False)
            st.success(f"Official keepers saved to {save_path}")
            st.dataframe(keepers_df)

# ----------------------------
# --- BC Grid Iron Projected Draft Board Rankings
# ----------------------------

# --- Step 1: Inputs side by side ---
col1, col2, col3 = st.columns(3)

with col1:
    draft_type = st.radio("Draft Order Type", ["Simulated", "Official"], index=0)

with col2:
    keeper_type = st.radio("Keepers Type", ["Simulated", "Official"], index=0)

# --- Step 2: Load draft order based on draft_type ---
draft_order_file = f"data/draft_order_{draft_type.lower()}.csv"
draft_order_df = pd.read_csv(draft_order_file)
draft_order_df = draft_order_df.rename(columns={"team": "Pick Owner", "pick_number": "Pick", "round": "Round"})

# Populate team dropdown from draft order
with col3:
    team_options = draft_order_df["Pick Owner"].dropna().unique()
    selected_team = st.selectbox("Select Your Team", options=team_options, index=list(team_options).index("Let Kelce Cook"))

# --- Step 3: Load draft board ---
keepers_file = f"data/keepers_{keeper_type.lower()}.csv"
keepers_df = pd.read_csv(keepers_file)
projected_df = pd.read_csv(f"data/yahoo/yahoo_{league_id_chosen}/projected_points.csv")

# Add tier labels by position based on Fan Pts and Pre-Season ranking
projected_df["Tier"] = _assign_tiers_by_position(projected_df)
projected_df["Adjusted Rank"] = projected_df["Pre-Season"]
projected_df["Pos Rank"] = ""
projected_df["Adjusted Pos Rank"] = np.nan
for pos, group in projected_df.groupby("Position", sort=False):
    ordered_idx = group.sort_values("Pre-Season").index
    projected_df.loc[ordered_idx, "Adjusted Pos Rank"] = np.arange(1, len(ordered_idx) + 1)
    projected_df.loc[ordered_idx, "Pos Rank"] = [f"{str(pos).strip().upper()}{i+1}" for i in range(len(ordered_idx))]

custom_rankings_path = f"data/custom_rankings_{league_id_chosen}.csv"
if os.path.exists(custom_rankings_path):
    saved_custom = pd.read_csv(custom_rankings_path)
    saved_custom.columns = saved_custom.columns.str.strip()
    if {"Player", "Adjusted Pos Rank"}.issubset(saved_custom.columns):
        saved_custom["Adjusted Pos Rank"] = pd.to_numeric(saved_custom["Adjusted Pos Rank"], errors="coerce")
        projected_df = projected_df.merge(saved_custom[["Player", "Adjusted Pos Rank"]], on="Player", how="left", suffixes=("", "_saved"))
        projected_df["Adjusted Pos Rank"] = projected_df["Adjusted Pos Rank_saved"].combine_first(projected_df["Adjusted Pos Rank"])
        projected_df.drop(columns=[c for c in projected_df.columns if c.endswith("_saved")], inplace=True)
projected_df["Adjusted Rank"] = pd.to_numeric(projected_df["Adjusted Rank"], errors="coerce")
projected_df["Adjusted Pos Rank"] = pd.to_numeric(projected_df["Adjusted Pos Rank"], errors="coerce")
projected_df["Adjusted Pos Rank"] = projected_df["Adjusted Pos Rank"].fillna(projected_df.groupby("Position")["Adjusted Pos Rank"].transform(lambda x: x.ffill().bfill()))

# Sort projected players by adjusted ranking
projected_df = projected_df.sort_values("Adjusted Rank").copy()

# Recompute Pos Rank from adjusted position ranking
for pos, group in projected_df.groupby("Position", sort=False):
    ordered_idx = group.sort_values("Adjusted Pos Rank").index
    projected_df.loc[ordered_idx, "Pos Rank"] = [f"{str(pos).strip().upper()}{i+1}" for i in range(len(ordered_idx))]

# Merge keepers
keepers_merged = pd.merge(
    keepers_df[["Player", "pick_number"]],
    projected_df,
    on="Player",
    how="left"
)
keepers_merged["Keeper"] = True

# Non-keepers
non_keepers = projected_df[~projected_df["Player"].isin(keepers_df["Player"])].copy()
non_keepers["Keeper"] = False

# Master pool
player_pool = pd.concat([keepers_merged, non_keepers], ignore_index=True)
available_nonkeepers = non_keepers.copy().reset_index(drop=True)

# Build draft board
draft_board = []
for pick in range(1, 251):
    if pick in keepers_merged["pick_number"].values:
        row = keepers_merged.loc[keepers_merged["pick_number"] == pick].iloc[0].copy()
    else:
        row = available_nonkeepers.iloc[0].copy()
        row["pick_number"] = pick
        available_nonkeepers = available_nonkeepers.iloc[1:].reset_index(drop=True)
    draft_board.append(row)

draft_board_df = pd.DataFrame(draft_board)

# Limit and rename columns
draft_board_df = draft_board_df[[
    "Player", "Team", "Position", "Pos Rank", "Tier", "Bye", "Fan Pts", "Pre-Season", "Keeper", "pick_number"
]]
draft_board_df = draft_board_df.rename(columns={
    "Pre-Season": "xRank",
    "pick_number": "Pick"
})

# Merge draft order (keep only Round and Pick Owner)
draft_board_df = pd.merge(
    draft_board_df,
    draft_order_df[["Pick", "Round", "Pick Owner"]],
    on="Pick",
    how="left"
)

# Reorder columns: Pick, Round, Pick Owner, xRank, Player, Team, Position, Pos Rank, Tier, Bye, Fan Pt, Keeper
draft_board_df = draft_board_df[[
    "Pick", "Round", "Pick Owner", "xRank", "Player", "Team", "Position", "Pos Rank", "Tier", "Bye", "Fan Pts", "Keeper"
]]

# Reset index
draft_board_df = draft_board_df.reset_index(drop=True)

# Convert Pick to integer (already all filled)
draft_board_df["Pick"] = draft_board_df["Pick"].astype(int)

# Convert Round to integer where not null
draft_board_df["Round"] = pd.to_numeric(draft_board_df["Round"], errors="coerce").astype("Int64")

# Round Fan Pt
draft_board_df["Fan Pts"] = pd.to_numeric(draft_board_df["Fan Pts"], errors="coerce").round(2)

# --- Step 4: Highlight picks for selected team ---
def highlight_my_team(row):
    if row["Pick Owner"] == selected_team:
        return ["background-color: lightgreen"] * len(row)
    elif row["Keeper"]:
        return ["background-color: lightgrey"] * len(row)
    else:
        return [""] * len(row)

st.subheader("BC Grid Iron Projected Draft Board Rankings")
st.dataframe(draft_board_df.style.apply(highlight_my_team, axis=1).format({"Fan Pts": "{:.2f}"}))

with st.expander("Custom Draft Ranking Preferences", expanded=False):
    st.write("Adjust rankings for RB, WR, and TE simultaneously. Each column shows the top 75 players by position.")
    pos_cols = st.columns(3)
    editable_positions = ["RB", "WR", "TE"]
    edited_frames = []

    for position, col in zip(editable_positions, pos_cols):
        col.subheader(position)
        rank_editor = projected_df[projected_df["Position"] == position].copy()
        rank_editor = rank_editor.sort_values(["Adjusted Pos Rank", "Player"]).head(75).reset_index(drop=True)
        rank_editor = rank_editor[["Player", "Position", "Team", "Tier", "Pos Rank", "Pre-Season", "Adjusted Pos Rank"]].copy()
        rank_editor = rank_editor.rename(columns={"Pre-Season": "Orig Rank"})

        if rank_editor.empty:
            col.info(f"No players found for {position}.")
            edited_frames.append(rank_editor)
            continue

        try:
            edited_df = col.experimental_data_editor(rank_editor, num_rows="fixed", key=f"rank_editor_{position}")
        except Exception:
            col.warning("Your Streamlit version does not support experimental_data_editor. Adjust Pos Rank manually below.")
            edited_df = rank_editor.copy()
            for i, row in edited_df.iterrows():
                edited_df.at[i, "Adjusted Pos Rank"] = col.number_input(
                    f"{row['Player']} ({row['Team']}, {row['Position']})",
                    min_value=1, max_value=999,
                    value=int(row["Adjusted Pos Rank"]),
                    key=f"rank_adj_{position}_{i}"
                )

        # Recompute Pos Rank in case adjusted position rankings changed
        edited_df = edited_df.sort_values(["Adjusted Pos Rank", "Player"]).reset_index(drop=True)
        edited_df["Pos Rank"] = [f"{position}{i+1}" for i in range(len(edited_df))]
        edited_frames.append(edited_df)

    edited_rankings = pd.concat(edited_frames, ignore_index=True) if edited_frames else pd.DataFrame()

    if st.button("Save Revised Rankings", key="save_custom_rankings"):
        save_df = edited_rankings[["Player", "Adjusted Pos Rank"]].copy()
        save_df.to_csv(custom_rankings_path, index=False)
        st.success("Custom positional rankings saved.")

    if not edited_rankings.empty:
        st.dataframe(edited_rankings, use_container_width=True)

# ----------------------------
# --- Tier Summary by Position ---
position_summaries = []
for pos in ["RB", "WR", "TE"]:
    pos_df = projected_df[projected_df["Position"] == pos].copy()
    if pos_df.empty:
        position_summaries.append((pos, pd.DataFrame(columns=["Player", "Tier"])))
        continue

    players_by_tier = pos_df[["Player", "Tier"]].copy()
    players_by_tier = players_by_tier.sort_values(["Tier", "Player"])  # keep a consistent long-format order
    position_summaries.append((pos, players_by_tier))

with st.expander("Tier Summary by Position", expanded=False):
    st.markdown("### Tier Summary by Position")
    pos_col1, pos_col2, pos_col3 = st.columns(3)

    for col, (pos, summary_df) in zip([pos_col1, pos_col2, pos_col3], position_summaries):
        col.subheader(pos)
        summary_df = summary_df.reset_index(drop=True)

        def _style_tier_rows(df):
            if df.empty or "Tier" not in df.columns:
                return pd.DataFrame("", index=df.index, columns=df.columns)

            colors = ["#f7fbff", "#e7f3fe"]
            style_rows = []
            current_color = 0
            previous_tier = None
            for tier in df["Tier"].astype(str):
                if tier != previous_tier:
                    current_color = 1 - current_color
                    previous_tier = tier
                style_rows.append([f"background-color: {colors[current_color]}" for _ in df.columns])

            return pd.DataFrame(style_rows, index=df.index, columns=df.columns)

        styled = summary_df.style.apply(_style_tier_rows, axis=None)
        col.dataframe(styled, use_container_width=True)

# ----------------------------
# --- Last Year Pick Variance Estimation
# ----------------------------

# Copy projected_points.csv from the folder data/yahoo/yahoo_{last year} and save it to the current year's folder as last_proj_rank.csv
# Pull in last year's projected points
last_year_rank_df = pd.read_csv(f"data/yahoo/yahoo_{league_id_previous}/projected_points.csv")
# Keep only Player, Position, and Pre-Season (Projected Rank) and Team
last_year_rank_df = last_year_rank_df[["Player", "Position", "Pre-Season", "Team"]].copy()
last_year_rank_df = last_year_rank_df.rename(columns={"Pre-Season": "Pick", "Player": "Player Name", "Position": "Pos", "Team": "Team"})
# Save into current year's folder as last_proj_rank.csv
last_year_rank_df.to_csv(f"data/yahoo/yahoo_{league_id_chosen}/last_proj_rank.csv", index=False)

# Load last year's projected ranks
proj_df = pd.read_csv(f"data/yahoo/yahoo_{league_id_chosen}/last_proj_rank.csv")

# Standardize column names (lowercase)
proj_df.columns = proj_df.columns.str.lower()

# Make sure player column is called "player"
if "player name" in proj_df.columns:
    proj_df = proj_df.rename(columns={"player name": "player"})

# Load last year's draft results
draft_df = pd.read_csv(f"data/yahoo/yahoo_{league_id_chosen}/last_draft_results.csv")

# Merge on player
last_year_df = pd.merge(proj_df, draft_df, on="player", how="inner")

# Filter out keepers
last_year_df = last_year_df[last_year_df["is_keeper"] != True].copy()

# Filter out keepers and unwanted positions (DEF, K)
last_year_df = last_year_df[
    (last_year_df["is_keeper"] != True) &
    (~last_year_df["pos"].isin(["DEF", "K"]))
].copy()

# Fix Pick Number
last_year_df["pick_y"] = (last_year_df["round"]-1)*14 + last_year_df["pick_y"]

# Determine variance at each pick
last_year_df["resid"] = last_year_df["pick_y"] - last_year_df["pick_x"]
last_year_df["abs_resid"] = last_year_df["resid"].abs()

# --- Step 2. Model heteroskedasticity (abs residuals ~ pick)
X_var = sm.add_constant(last_year_df["pick_x"])
var_model = sm.OLS(last_year_df["abs_resid"], X_var).fit()
alpha, beta = var_model.params

players = draft_board_df.copy()
players["abs_resid"] = np.minimum(np.maximum(1, alpha + beta * players["Pick"]), 25)

# ----------------------------
# --- Player Pick Probability Model
# ----------------------------

# Inputs
N_SIM = 100
n_picks = last_year_df["pick_y"].max()
rng = np.random.default_rng(42)

st.header("Pick Analysis")

# --- Columns setup
col1, col2, col3, col4 = st.columns([1, 3, 3, 3])

# --- Column 1: controls
with col1:
    st.subheader("Inputs")
    simulate_button = st.button("Simulate Draft Probabilities")
    chosen_pick = st.number_input("Select Pick Number", min_value=1, max_value=n_picks, value=18)

# Paths
prob_file = f"data/prob_matrix_{league_id_chosen}.pkl"

# Check if saved file exists
if os.path.exists(prob_file) and not simulate_button:
    st.success("Loading existing simulation...")
    prob_matrix = pd.read_pickle(prob_file)
else:
    st.info("Running Monte Carlo simulations...")
    prob_matrix = pd.DataFrame(0.0, index=players["Player"], columns=np.arange(1, n_picks+1))
    progress_bar = st.progress(0)

    for sim in range(N_SIM):
        available = players.copy()
        for pick_num in range(1, n_picks+1):
            keepers_this_pick = available[(available["Keeper"]) & (available["Pick"] == pick_num)]
            if not keepers_this_pick.empty:
                player_picked = keepers_this_pick.iloc[0]["Player"]
            else:
                available_non_keepers = available[~available["Keeper"]].copy()
                expected_pick = available_non_keepers["Pick"].values
                stds = available_non_keepers["abs_resid"].values
                draws = rng.normal(loc=expected_pick, scale=stds)
                idx = np.argmin(draws)
                player_picked = available_non_keepers.iloc[idx]["Player"]

            prob_matrix.loc[player_picked, pick_num:] += 1
            available = available[available["Player"] != player_picked].reset_index(drop=True)

        # Update progress
        if sim % max(1, N_SIM // 100) == 0:
            progress_bar.progress((sim+1)/N_SIM)

    # Convert counts to probabilities
    prob_matrix = prob_matrix / N_SIM
    # Save to file
    prob_matrix.to_pickle(prob_file)
    st.success("Simulation complete and saved!")

# Merge probability for chosen pick
prob_taken = prob_matrix[chosen_pick]
players_display = players.merge(
    prob_taken.rename("ProbTakenByPick").reset_index(),
    on="Player",
    how="left"
)
players_display["ProbAv"] = 1 - players_display["ProbTakenByPick"]

players_display = players_display[~players_display["Keeper"]].copy()

# --- Columns 2-4: tables
with col2:
    st.subheader("Potential Fallers (5-40%)")
    fallers = players_display[(players_display["ProbAv"] > 0.05) & 
                              (players_display["ProbAv"] < 0.4)]
    st.dataframe(fallers[["Player", "Pos Rank", "Tier", "Pick", "Fan Pts", "ProbAv"]]
                 .sort_values("ProbAv")
                 .reset_index(drop=True)
                 .style.format({"Fan Pts": "{:.2f}", "ProbAv": "{:.0%}"})
                 )

with col3:
    st.subheader("Likely Picks (40-90%)")
    likely = players_display[(players_display["ProbAv"] >= 0.4) & (players_display["ProbAv"] <= 0.9)]
    st.dataframe(likely[["Player", "Pos Rank", "Tier", "Pick", "Fan Pts", "ProbAv"]]
                 .sort_values("Pick")
                 .reset_index(drop=True)
                 .style.format({"Fan Pts": "{:.2f}", "ProbAv": "{:.0%}"})
                 )

with col4:
    st.subheader("Potential Reaches (>90%)")
    reaches = players_display[players_display["ProbAv"] > 0.9].sort_values("Pick").reset_index(drop=True)
    st.dataframe(reaches.head(10)[["Player", "Position", "Tier", "Pick", "Fan Pts", "ProbAv"]].style.format({"Fan Pts": "{:.2f}", "ProbAv": "{:.0%}"}))