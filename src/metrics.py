"""Step 6: trend metrics per team per game, stored in the team_game_metrics table.

Run from the repo root:  python -m src.metrics            (recompute and store everything)
                         python -m src.metrics --team KC --season 2025   (choose the spot-check)

Every metric is computed per season, in game order, and resets when a new season starts.
Each metric exists in two forms:
  - plain name (e.g. roll_win_pct): the value AFTER the game, for the dashboard.
  - pre_ name (e.g. pre_roll_win_pct): the value BEFORE the game, using only earlier games.
    Phase 2 models use the pre_ columns so no feature ever includes the game being predicted.
"""

import argparse

import numpy as np
import pandas as pd
import polars as pl

from src.db import ensure_table, get_connection, upsert_df

WINDOW = 4          # games in each rolling window
PYTH_EXP = 2.37     # Pythagorean exponent commonly cited for the NFL
TABLE = "team_game_metrics"
PRIMARY_KEY = ["game_id", "team"]


def pythag(pf, pa, exponent: float = PYTH_EXP):
    """Pythagorean win %: points_for^e / (points_for^e + points_against^e)."""
    pf_e = pf ** exponent
    return pf_e / (pf_e + pa ** exponent)


def load_team_games(conn) -> pd.DataFrame:
    """One row per team per played game, built from the games table (like Cell 11 in the notebook)."""
    games = pd.read_sql_query(
        """
        SELECT game_id, season, game_type, week, gameday, gametime,
               home_team, away_team, home_score, away_score
        FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """,
        conn,
    )

    # Each game becomes two rows: the home team's view and the away team's view.
    shared = ["game_id", "season", "game_type", "week", "gameday", "gametime"]
    home = games[shared].assign(
        team=games["home_team"], opponent=games["away_team"], location="home",
        points_for=games["home_score"], points_against=games["away_score"],
    )
    away = games[shared].assign(
        team=games["away_team"], opponent=games["home_team"], location="away",
        points_for=games["away_score"], points_against=games["home_score"],
    )
    tg = pd.concat([home, away], ignore_index=True)

    tg["is_playoff"] = (tg["game_type"] != "REG").astype(int)   # WC, DIV, CON and SB all count as playoffs
    tg["margin"] = tg["points_for"] - tg["points_against"]
    tg["win"] = (tg["margin"] > 0).astype(float) + 0.5 * (tg["margin"] == 0)  # win 1, tie 0.5, loss 0

    # Game order within each team and season. gameday is an ISO date string, so it sorts correctly.
    return tg.sort_values(["season", "team", "gameday", "gametime", "game_id"]).reset_index(drop=True)


def compute_metrics(tg: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative, rolling and pre-game metrics to the team-game table."""
    tg = tg.copy()
    by_team = tg.groupby(["season", "team"], sort=False)

    # Season-to-date totals after each game.
    tg["game_number"] = by_team.cumcount() + 1
    tg["games_before"] = tg["game_number"] - 1
    tg["cum_pf"] = by_team["points_for"].cumsum()
    tg["cum_pa"] = by_team["points_against"].cumsum()
    tg["cum_wins"] = by_team["win"].cumsum()
    tg["cum_margin"] = by_team["margin"].cumsum()
    tg["season_win_pct"] = tg["cum_wins"] / tg["game_number"]
    tg["pythag_win_pct"] = pythag(tg["cum_pf"], tg["cum_pa"])

    # Opponent strength: the opponent's average margin over games played BEFORE this game.
    # Using only earlier games keeps the adjustment leakage-free. A team with no prior
    # games has no average yet (NaN) and is treated as average (0) below.
    prior_games = tg["games_before"].replace(0, np.nan)
    tg["prior_avg_margin"] = (tg["cum_margin"] - tg["margin"]) / prior_games

    # Look up the opponent's prior_avg_margin for the same game (their row in the same game_id).
    opp = tg[["game_id", "team", "prior_avg_margin"]].rename(
        columns={"team": "opponent", "prior_avg_margin": "opp_prior_avg_margin"}
    )
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left", validate="m:1")

    # Opponent-adjusted margin: how much better or worse than the opponent's usual margin.
    tg["adj_margin"] = tg["margin"] + tg["opp_prior_avg_margin"].fillna(0)

    # Rolling metrics over the last WINDOW games (fewer at the start of a season).
    by_team = tg.groupby(["season", "team"], sort=False)

    def rolling_mean(col: str) -> pd.Series:
        return by_team[col].transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())

    tg["roll_games"] = by_team["win"].transform(lambda s: s.rolling(WINDOW, min_periods=1).count())
    tg["roll_win_pct"] = rolling_mean("win")
    tg["roll_margin"] = rolling_mean("margin")
    tg["roll_adj_margin"] = rolling_mean("adj_margin")

    # Pre-game versions: the same metrics as of the previous game (NaN before a team's first game).
    by_team = tg.groupby(["season", "team"], sort=False)
    for col in ["season_win_pct", "pythag_win_pct", "roll_win_pct", "roll_margin", "roll_adj_margin"]:
        tg["pre_" + col] = by_team[col].shift(1)

    keep = [
        "game_id", "team", "season", "week", "game_type", "is_playoff", "gameday", "location",
        "opponent", "points_for", "points_against", "margin", "win", "game_number", "games_before",
        "season_win_pct", "pythag_win_pct", "roll_games", "roll_win_pct", "roll_margin",
        "opp_prior_avg_margin", "adj_margin", "roll_adj_margin",
        "pre_season_win_pct", "pre_pythag_win_pct", "pre_roll_win_pct", "pre_roll_margin",
        "pre_roll_adj_margin",
    ]
    return tg[keep]


def write_metrics(conn, metrics: pd.DataFrame) -> None:
    """Create the table if needed, then upsert every row (safe to rerun)."""
    frame = pl.from_pandas(metrics)  # NaN becomes NULL
    ensure_table(conn, TABLE, frame, PRIMARY_KEY)
    upsert_df(conn, TABLE, frame, PRIMARY_KEY)


def read_metrics(conn) -> pd.DataFrame:
    """Read the stored metrics table (this is what the Streamlit app will call)."""
    return pd.read_sql_query(f'SELECT * FROM "{TABLE}"', conn)


def season_summary(m: pd.DataFrame, season: int, scope: str = "both", exponent: float = PYTH_EXP) -> pd.DataFrame:
    """One row per team for a season, computed only from the games in the chosen scope.

    scope: "regular", "playoffs" or "both". Totals, home/away splits and the Pythagorean
    win % are all recomputed from the filtered games.
    """
    if scope not in ("regular", "playoffs", "both"):
        raise ValueError("scope must be 'regular', 'playoffs' or 'both'")

    d = m[m["season"] == season]
    if scope == "regular":
        d = d[d["is_playoff"] == 0]
    elif scope == "playoffs":
        d = d[d["is_playoff"] == 1]

    out = d.groupby("team").agg(
        games=("win", "size"),
        wins=("win", lambda s: int((s == 1).sum())),
        losses=("win", lambda s: int((s == 0).sum())),
        ties=("win", lambda s: int((s == 0.5).sum())),
        win_pct=("win", "mean"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
        margin_per_game=("margin", "mean"),
    )
    out["pythag_win_pct"] = pythag(out["points_for"], out["points_against"], exponent)

    # Home and away splits, from the same filtered games.
    for loc in ("home", "away"):
        part = d[d["location"] == loc].groupby("team").agg(**{
            f"{loc}_games": ("win", "size"),
            f"{loc}_win_pct": ("win", "mean"),
            f"{loc}_margin_per_game": ("margin", "mean"),
        })
        out = out.join(part)

    return out.sort_values(["win_pct", "margin_per_game"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=None, help="team code for the spot-check, e.g. KC")
    parser.add_argument("--season", type=int, default=None, help="season for the spot-check")
    args = parser.parse_args()

    conn = get_connection()
    tg = load_team_games(conn)
    metrics = compute_metrics(tg)
    write_metrics(conn, metrics)

    stored = read_metrics(conn)
    print(f"Team-game rows computed: {len(metrics)} | rows in {TABLE}: {len(stored)}")
    print(f"Rows computed equal twice the played games: {len(metrics) == 2 * tg['game_id'].nunique()}")
    print(f"Window: {WINDOW} games | Pythagorean exponent: {PYTH_EXP}\n")

    # Spot-check: one team's season, game by game, to verify against the raw scores by hand.
    season = args.season or int(stored["season"].min())
    team = args.team or sorted(stored["team"].unique())[0]
    cols = ["week", "game_type", "location", "opponent", "points_for", "points_against", "margin",
            "win", "roll_win_pct", "roll_margin", "opp_prior_avg_margin", "adj_margin",
            "roll_adj_margin", "pythag_win_pct"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 40)
    print(f"Spot-check: {team}, season {season}")
    print(stored[(stored["team"] == team) & (stored["season"] == season)]
          .sort_values("game_number")[cols]
          .round(3).to_string(index=False))
    print()

    print(f"Top 5 teams, season {season}, regular season only:")
    print(season_summary(stored, season, "regular").head(5).round(3).to_string())
    conn.close()


if __name__ == "__main__":
    main()