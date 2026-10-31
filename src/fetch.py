"""Load NFL schedules, results and team stats via nflreadpy.

Run directly (python src\fetch.py) to print a check table for schedules.
"""

import nflreadpy as nfl
import pandas as pd
import polars as pl

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Columns the notebook and the Step 3 check rely on.
COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
]

# String columns holding lists of kick distances. They are noisy and not useful for modeling.
TEAM_STATS_DROP = ["fg_made_list", "fg_missed_list", "fg_blocked_list"]


def load_schedules_pl(seasons: list[int] = SEASONS) -> pl.DataFrame:
    """All schedule columns as a Polars frame. The database schema is built from this."""
    return nfl.load_schedules(seasons=seasons)


def load_team_stats_pl(seasons: list[int] = SEASONS) -> pl.DataFrame:
    """Team stats as a Polars frame, minus the three fg_*_list string columns."""
    df = nfl.load_team_stats(seasons=seasons)
    return df.drop([c for c in TEAM_STATS_DROP if c in df.columns])

def load_teams_pl() -> pl.DataFrame:
    """Team reference data: abbreviation, full name, nickname, conference and division."""
    df = nfl.load_teams()
    keep = ["team_abbr", "team_name", "team_nick", "team_conf", "team_division"]
    return df.select([c for c in keep if c in df.columns])

def fetch_schedules(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Return schedules and results for the given seasons as a pandas DataFrame (10 columns)."""
    df = load_schedules_pl(seasons).to_pandas()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing from nflverse schedule data: {missing}")
    return df[COLUMNS].copy()


def main() -> None:
    df = fetch_schedules()
    played = df["home_score"].notna() & df["away_score"].notna()

    print(f"Rows: {len(df)} | played: {int(played.sum())} | unplayed: {int((~played).sum())}")
    print()
    print(df.groupby(["season", "game_type"]).agg(
        games=("game_id", "count"),
        played=("home_score", lambda s: int(s.notna().sum())),
    ))
    print()
    pd.set_option("display.width", 200)
    print(df.sort_values(["season", "week", "gameday"]).head(10).to_string(index=False))
    print("...")
    print(df[played].sort_values(["season", "week", "gameday"]).tail(10).to_string(index=False))
    print()
    print(f"Duplicate game_ids: {int(df['game_id'].duplicated().sum())}")


if __name__ == "__main__":
    main()



# Per-game quarterback stats: the columns the QB projections use.
QB_STATS_COLUMNS = ["player_id", "player_name", "team", "opponent_team", "season", "week", "season_type",
                    "game_id", "completions", "attempts", "passing_yards", "passing_tds",
                    "passing_interceptions", "sacks_suffered", "passing_epa"]


def load_qb_stats_pl(seasons: list[int] = SEASONS) -> pl.DataFrame:
    """One row per quarterback per game, for every QB who threw at least one pass."""
    df = nfl.load_player_stats(seasons=seasons)
    df = df.filter((pl.col("position") == "QB") & (pl.col("attempts") > 0))
    return df.select([c for c in QB_STATS_COLUMNS if c in df.columns])