import pandas as pd

def load_adp_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_rosters_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_picks_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# --- Stubs for live web sources (replace with real fetch logic) ---
def fetch_adp_from_fantasypros() -> pd.DataFrame:
    """TODO: Implement actual FantasyPros fetch (respect ToS).

    Return columns: Player, Position, ADP, ECR, ByeWeek, Team
"""
    raise NotImplementedError

def fetch_rosters_from_sleeper(league_id: str) -> pd.DataFrame:
    """TODO: Implement Sleeper API roster fetch.
Return columns: Team, Player, Position, LastDraftRound, DraftYear
"""
    raise NotImplementedError

def fetch_draft_picks_from_sleeper(league_id: str) -> pd.DataFrame:
    """TODO: Implement Sleeper API draft capital fetch.
Return columns: Team, Round, PickOverall, HasPick
"""
    raise NotImplementedError
