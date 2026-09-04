import pandas as pd
import requests
from bs4 import BeautifulSoup
import json, os, re
from datetime import datetime
from collections import Counter

# Load cookies
with open("./data/cookies.json", "r") as f:
    cookies = json.load(f)
cookies_dict = {c['name']: c['value'] for c in cookies}


def pull_last_year_draft(league_id_previous, save_dir):
    previous_year = datetime.now().year - 1
    url = f"https://football.fantasysports.yahoo.com/{previous_year}/f1/{league_id_previous}/draftresults"
    response = requests.get(url, cookies=cookies_dict)
    soup = BeautifulSoup(response.text, "html.parser")

    draft_tables_div = soup.find("div", id="drafttables")
    tables = draft_tables_div.find_all("table")

    rows = []
    for round_num, table in enumerate(tables, start=1):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 3:
                pick_raw = tds[0].get_text(strip=True)
                pick_clean = pick_raw.replace(".", "")

                player_raw = tds[1].get_text(strip=True)
                is_keeper = any(ord(ch) > 127 for ch in player_raw)
                player_clean = re.sub(r'[^\x00-\x7F]+', '', player_raw).strip()

                fteam_raw = tds[2].get_text(strip=True)
                fteam_clean = fteam_raw.rstrip('.')

                rows.append({
                    "round": round_num,
                    "pick": pick_clean,
                    "player": player_clean,
                    "FTeam": fteam_clean,
                    "is_keeper": is_keeper
                })

    df = pd.DataFrame(rows)
    
    out_path = os.path.join(save_dir, "last_draft_results.csv")  # change filename as needed
    df.to_csv(out_path, index=False, encoding="utf-8")
    
    return df


def pull_last_year_dropped(league_id_previous, save_dir):
    previous_year = datetime.now().year - 1
    base_url = f"https://football.fantasysports.yahoo.com/{previous_year}/f1/{league_id_previous}/transactions?transactionsfilter=drop&count={{}}"
    all_players = set()
    rows = []

    count = 0
    while True:
        url = base_url.format(count)
        res = requests.get(url, cookies=cookies_dict)
        soup = BeautifulSoup(res.text, "html.parser")

        transactions_div = soup.find("div", id="transactions")
        if not transactions_div:
            break

        table = transactions_div.find("table")
        if not table:
            break

        new_found = False
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            td = tds[1]
            a_tag = td.find("a")
            if a_tag:
                player_name = a_tag.get_text(strip=True)
                if player_name not in all_players:
                    all_players.add(player_name)
                    rows.append({"Player": player_name})
                    new_found = True

        if not new_found:
            break
        count += 25

    df = pd.DataFrame(rows)

    out_path = os.path.join(save_dir, "last_dropped_players.csv")  # change filename as needed
    df.to_csv(out_path, index=False, encoding="utf-8")

    return df


def pull_last_season_rosters(league_id_current, save_dir):
    last_season_rosters_url = f"https://football.fantasysports.yahoo.com/f1/{league_id_current}/lastseason"

    res = requests.get(last_season_rosters_url, cookies=cookies_dict)
    soup = BeautifulSoup(res.text, "html.parser")

    rows = []

    # Loop through each roster section
    for section in soup.find_all("section", class_="Mod"):
        header = section.find("header").find("h3")
        if not header:
            continue
        fteam = header.get_text(strip=True)

        table_div = section.find("div", class_="Bd No-p")
        if not table_div:
            continue
        table = table_div.find("table")
        if not table:
            continue

        for tr in table.find_all("tr")[1:]:  # skip header row
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            # Player name
            first_td = tds[0]
            a_tags = first_td.find_all("a")
            if not a_tags:
                continue
            player_name = a_tags[0].get_text(strip=True)

            # Team abbreviation and position
            span = first_td.find("span", class_="F-position")
            if span:
                team_pos = span.get_text(strip=True).split("-")
                team_abbr = team_pos[0].strip() if len(team_pos) > 0 else ""
                position = team_pos[1].strip() if len(team_pos) > 1 else ""
            else:
                team_abbr = ""
                position = ""

            # Draft position
            draft_text = tds[1].get_text(strip=True)
            round_num, pick_num = None, None
            if draft_text and draft_text != "-":
                match = re.search(r"Round (\d+), Pick (\d+)", draft_text)
                if match:
                    round_num = int(match.group(1))
                    pick_num = int(match.group(2))

            # Last year and projected ranks
            last_year_rank = tds[2].get_text(strip=True)
            proj_rank = tds[3].get_text(strip=True)

            rows.append({
                "FTeam": fteam,
                "Player": player_name,
                "TeamAbbr": team_abbr,
                "Position": position,
                "DraftRound": round_num,
                "DraftPick": pick_num,
                "LastYearRank": last_year_rank,
                "ProjRank": proj_rank
            })

    df = pd.DataFrame(rows)
    
    out_path = os.path.join(save_dir, "last_season_rosters.csv")  # change filename as needed
    df.to_csv(out_path, index=False, encoding="utf-8")

    return df

def pull_traded_picks(league_id, save_dir):
    url = f"https://football.fantasysports.yahoo.com/f1/{league_id}/showtradedpicks?order=byteam"
    response = requests.get(url, cookies=cookies_dict)
    soup = BeautifulSoup(response.text, "html.parser")

    traded_picks_div = soup.find(id="ysf-showtradedpicks-tables")

    tables = traded_picks_div.find_all("table")

    all_rows = []

    for table in tables:
        # Get all original pick owners in the table (to guess Owned By)
        owners_found = []
        for td in table.find_all("td", class_="owned"):
            owners_found.extend([li.get_text(strip=True) for li in td.find_all("li")])

        # Infer Owned By from the most common owner name in this table
        if owners_found:
            owned_by = Counter(owners_found).most_common(1)[0][0]
        else:
            owned_by = None  # No picks in table

        # Parse table body rows
        for tr in table.find_all("tr"):
            # Round cell
            round_td = tr.find("td", class_="round")
            if not round_td:
                continue
            round_num_text = round_td.get_text(strip=True)
            try:
                round_num = int(round_num_text)
            except ValueError:
                continue  # skip rows where round isn’t a number

            # Owned picks cell
            owned_cell = tr.find("td", class_="owned")
            if owned_cell:
                pick_owners = [li.get_text(strip=True) for li in owned_cell.find_all("li")]
                for orig_owner in pick_owners:
                    if orig_owner:  # skip blanks
                        all_rows.append({
                            "Round": round_num,
                            "Orig Pick Owner": orig_owner,
                            "Owned By": owned_by
                        })

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Clean: drop blanks in Orig Pick Owner
    df = df[df["Orig Pick Owner"].str.strip() != ""]

    # Sort
    df.sort_values(by=["Round", "Owned By"], inplace=True)

    # Save
    out_path = os.path.join(save_dir, "traded_picks.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    return df

def _make_unique(names):
    counts = {}
    out = []
    for n in names:
        base = n if n else "Col"
        if base not in counts:
            counts[base] = 1
            out.append(base)
        else:
            counts[base] += 1
            out.append(f"{base}_{counts[base]}")
    return out

def pull_projected_points(league_id, save_dir=".", max_players=300):
    base_url = f"https://football.fantasysports.yahoo.com/f1/{league_id}/players?status=ALL&eteam=ALL&fteam=NONE&pos=O&cut_type=9&stat1=S_PS_2025&myteam=0&sort=OR&sdir=1&count={{}}"
    rows = []
    all_players = set()

    count = 0
    headers = None

    while True:
        url = base_url.format(count)
        res = requests.get(url, cookies=cookies_dict)
        soup = BeautifulSoup(res.text, "html.parser")

        players_div = soup.find("div", id="players-table")
        if not players_div:
            break

        table = players_div.find("table")
        if not table:
            break

        # --- Extract headers once (from the second header row) ---
        if headers is None:
            thead = table.find("thead")
            if thead:
                header_rows = thead.find_all("tr")
                if len(header_rows) > 1:
                    header_cells = header_rows[1].find_all("th")
                    # Get text even if wrapped in <div>, then: skip first 2, skip last 1
                    raw_headers = [
                        " ".join(list(cell.stripped_strings))
                        for cell in header_cells[3:-1]
                    ]
                    # Optionally strip weird unicode (stars, etc.)
                    raw_headers = [re.sub(r"[^\x20-\x7E]", "", h).strip() for h in raw_headers]
                    # Make duplicates unique by suffix
                    headers = _make_unique(raw_headers)

        # --- Extract player rows ---
        tbody = table.find("tbody")
        if not tbody:
            break  # gracefully stop

        new_found = False
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            # Third cell: player info
            player_div = tds[2].find("div", class_="ysf-player-name")
            if not player_div:
                continue

            a_tags = player_div.find_all("a")
            if not a_tags:
                continue
            player_name = a_tags[0].get_text(strip=True)

            # Team & position: e.g., "KC - QB" → Team="KC", Position="QB"
            span_tags = player_div.find_all("span")
            team_pos = span_tags[-1].get_text(strip=True) if span_tags else ""
            team, pos = None, None
            if " " in team_pos:
                parts = team_pos.split(" ", 1)
                team = parts[0].strip()
                pos = parts[1].replace("-", "").strip() if len(parts) > 1 else None

            # Remaining cells (skip last one)
            stats = [td.get_text(strip=True) for td in tds[3:-1]]

            if player_name not in all_players:
                all_players.add(player_name)
                row = {"Player": player_name, "Team": team, "Position": pos}
                if headers and len(stats) == len(headers):
                    row.update(dict(zip(headers, stats)))
                else:
                    # Fallback if mismatch: generic numbered columns
                    row.update({f"Stat{i}": v for i, v in enumerate(stats, start=1)})
                rows.append(row)
                new_found = True

            if len(rows) >= max_players:
                break

        if not new_found or len(rows) >= max_players:
            break

        count += 25

    df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, "projected_points.csv")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} players to {out_path}")

    return df


def run_yahoo_data_pipeline(league_id_current, league_id_previous):
    save_dir = os.path.join("data", "yahoo", f"yahoo_{league_id_current}")
    os.makedirs(save_dir, exist_ok=True)  # create folders if they don't exist

    pull_last_year_draft(league_id_previous, save_dir)
    pull_last_year_dropped(league_id_previous, save_dir)
    pull_last_season_rosters(league_id_current, save_dir)
    pull_traded_picks(league_id_current, save_dir)
    pull_projected_points(league_id_current, save_dir)

    return save_dir

# Run just one part for testing
league_id_current = 117238  # Example current league ID
# league_id_previous = 37744  # Example previous league ID

save_dir = os.path.join("data", "yahoo", f"yahoo_{league_id_current}")
os.makedirs(save_dir, exist_ok=True)  # create folders if they don't exist
pull_projected_points(league_id_current, save_dir)

# Run whole thing
# run_yahoo_data_pipeline(league_id_current, league_id_previous)  # Example league IDs, replace as needed