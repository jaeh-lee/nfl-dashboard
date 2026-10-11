"""Load NFL schedules, results and team stats via nflreadpy.

Run directly (python src\fetch.py) to print a check table for schedules.
"""

import nflreadpy as nfl
import pandas as pd
import polars as pl

SEASONS = [2025, 2026]

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