import pandas as pd
import numpy as np

def make_post_keeper_board(adp_df: pd.DataFrame, keepers_df: pd.DataFrame) -> pd.DataFrame:
    kept_players = set(keepers_df["Player"].tolist())
    board = adp_df[~adp_df["Player"].isin(kept_players)].copy()
    # Simple tiers by ECR/ADP gaps (placeholder logic)
    board["Tier"] = pd.qcut(board["ADP"].rank(method="first"), q=6, labels=[1,2,3,4,5,6]).astype(int)
    board["PosRank"] = board.groupby("Position")["ADP"].rank(method="first")
    return board.sort_values("ADP")

# # Spacer / separator
# st.sidebar.markdown("---")

# # --- Fantasy Life ADP Import

# st.sidebar.markdown("## Retrieve Updated Fantasy ADPs")

# # Download button
# st.sidebar.markdown("""
# <a href="https://www.fantasylife.com/tools/nfl-adp" target="_blank">
#     <button style="
#         width:100%;
#         padding:10px;
#         background-color:#4CAF50;
#         color:white;
#         border:none;
#         border-radius:5px;
#         font-size:16px;
#     ">
#         Download ADPs Here
#     </button>
# </a>
# """, unsafe_allow_html=True)

# # Uploader label in larger, centered text
# st.sidebar.markdown("<p style='text-align:center; font-size:16px; font-weight:bold; margin-top:10px;'>Upload ADPs File Below</p>", unsafe_allow_html=True)

# # File uploader with same width
# uploaded_file = st.sidebar.file_uploader("", type="csv")

# if uploaded_file is not None:
#     data_dir = Path("data")
#     data_dir.mkdir(parents=True, exist_ok=True)
#     save_path = data_dir / "fantasy_adp.csv"
    
#     # Save the CSV exactly as uploaded
#     with open(save_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     st.sidebar.success(f"Saved {uploaded_file.name} to {save_path}")

# # Spacer / separator
# st.sidebar.markdown("---")

# # ----------------------------
# # --- Main Draft Tracker App
# # ----------------------------

# data_dir = Path("data")

# adp_path = st.sidebar.text_input("ADP CSV", value=str(data_dir / "sample_adp.csv"))
# keepers_path = st.sidebar.text_input("Predicted/Actual Keepers CSV", value=str(data_dir / "predicted_keepers.csv"))
# picks_path = st.sidebar.text_input("Draft Picks CSV", value=str(data_dir / "sample_picks.csv"))

# col1, col2, col3 = st.columns([2,2,1])

# @st.cache_data
# def load_board(adp_csv, keepers_csv):
#     adp = pd.read_csv(adp_csv)
#     try:
#         keepers = pd.read_csv(keepers_csv)
#         kept = set(keepers["Player"].tolist())
#         board = adp[~adp["Player"].isin(kept)].copy()
#     except Exception:
#         board = adp.copy()
#     board["Tier"] = pd.qcut(board["ADP"].rank(method="first"), q=6, labels=[1,2,3,4,5,6]).astype(int)
#     board["PosRank"] = board.groupby("Position")["ADP"].rank(method="first")
#     return board.sort_values("ADP")

# @st.cache_data
# def load_picks(csv):
#     return pd.read_csv(csv)

# board = load_board(adp_path, keepers_path)
# picks = load_picks(picks_path)

# with col1:
#     st.subheader("Best Available")
#     search = st.text_input("Search player/pos/team")    
#     filt = board.copy()
#     if search:
#         s = search.lower()
#         filt = filt[filt.apply(lambda r: s in str(r["Player"]).lower() or s in str(r["Position"]).lower() or s in str(r["Team"]).lower(), axis=1)]
#     pos_filter = st.multiselect("Positions", sorted(board["Position"].unique().tolist()), default=[])
#     if pos_filter:
#         filt = filt[filt["Position"].isin(pos_filter)]
#     st.dataframe(filt.reset_index(drop=True), use_container_width=True, height=500)

# with col2:
#     st.subheader("Draft Tracker")
#     st.caption("Mark selected players. This updates your board on the left.")
#     if "taken" not in st.session_state:
#         st.session_state.taken = []
#     options = board[~board["Player"].isin(st.session_state.taken)]["Player"].tolist()
#     pick_player = st.selectbox("Select drafted player", options=options, index=0 if options else None)
#     if st.button("Add Pick") and pick_player:
#         st.session_state.taken.append(pick_player)
#         st.toast(f"Picked: {pick_player}")
#         st.cache_data.clear()
#         board = load_board(adp_path, keepers_path)
#     st.write("Drafted so far:")
#     st.write(st.session_state.taken)

# with col3:
#     st.subheader("Your Roster & Suggestions")
#     if "your_roster" not in st.session_state:
#         st.session_state.your_roster = []
#     add_me = st.selectbox("Add to my roster", options=board["Player"].tolist())
#     if st.button("Add to My Roster") and add_me:
#         st.session_state.your_roster.append(add_me)
#         st.toast(f"Added to your roster: {add_me}")
#     st.write("Your roster:")
#     st.write(st.session_state.your_roster)
#     need = {}
#     for p in ["QB","RB","WR","TE"]:
#         have = sum(1 for x in st.session_state.your_roster if p in board[board["Player"]==x]["Position"].values)
#         need[p] = have
#     least = sorted(need.items(), key=lambda x: x[1])[0][0] if need else None
#     st.markdown("**Suggested targets (by need + ADP):**")
#     sugg = board[board["Position"]==least].head(10) if least else board.head(10)
#     st.dataframe(sugg[["Player","Position","ADP","Tier","Team"]], use_container_width=True)

# st.divider()
# st.markdown("Tip: Replace CSV paths with your league exports and re-run. For actual keepers, load your finalized `predicted_keepers.csv` or a hand-made file with columns Team, Player, Position.")