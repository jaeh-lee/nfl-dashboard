"""Inspect nflverse team stats so the team_game_stats schema is based on real columns."""

import nflreadpy as nfl
import pandas as pd

SEASONS = [2025, 2026]

# Candidate key columns. The script reports which of these actually exist.
KEY_CANDIDATES = ["season", "week", "season_type", "team", "opponent_team"]


def main() -> None:
    df = nfl.load_team_stats(seasons=SEASONS).to_pandas()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print(f"Shape: {df.shape}")
    print()

    print("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    print()

    present = [c for c in KEY_CANDIDATES if c in df.columns]
    absent = [c for c in KEY_CANDIDATES if c not in df.columns]
    print(f"Key candidates present: {present}")
    print(f"Key candidates absent: {absent}")
    print()

    if "season" in df.columns and "week" in df.columns:
        summary = df.groupby("season").agg(
            rows=("week", "size"),
            min_week=("week", "min"),
            max_week=("week", "max"),
        )
        print("Rows and week range per season:")
        print(summary)
        print()

    key = [c for c in ["season", "week", "team"] if c in df.columns]
    if len(key) == 3:
        dups = int(df.duplicated(subset=key).sum())
        print(f"Duplicate rows on {key}: {dups}")
        if "season_type" in df.columns:
            dups_st = int(df.duplicated(subset=key + ["season_type"]).sum())
            print(f"Duplicate rows on {key + ['season_type']}: {dups_st}")
        print()

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    print("Columns with nulls (count):")
    print(nulls.head(15).to_string() if len(nulls) else "  none")
    print()

    print("Sample rows:")
    print(df.head(5).T.to_string())


if __name__ == "__main__":
    main()