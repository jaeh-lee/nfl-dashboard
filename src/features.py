"""Phase 2 training table: one row per game with pre-game features and targets.

Run from the repo root:  python -m src.features                    (build, store and verify)
                         python -m src.features --game 2025_03_MIA_BUF   (print one game for a hand check)

Rules that keep it leakage-free:
  - Team features for a played game come from the team_game_metrics pre_ columns, which use only
    games played before it.
  - Team features for an upcoming game are each team's latest values after its last played game.
  - Betting lines are stored with a market_ prefix as benchmark columns. They are not features.
"""

import argparse

import numpy as np
import pandas as pd
import polars as pl

from src.db import ensure_table, get_connection, upsert_df
from src.metrics import WINDOW, pythag, read_metrics

TABLE = "game_features"
PRIMARY_KEY = ["game_id"]
MIN_PRIOR_GAMES = 3   # a game is usable for modeling when both teams have played at least this many

# Pre-game team metrics to use, as named in team_game_metrics, and the short names used here.
FEATURES = {
    "pre_season_win_pct": "season_win_pct",
    "pre_pythag_win_pct": "pythag_win_pct",
    "pre_roll_win_pct": "roll_win_pct",
    "pre_roll_margin": "roll_margin",
    "pre_roll_adj_margin": "roll_adj_margin",
    "pre_roll_pf": "roll_pf",
    "pre_roll_pa": "roll_pa",
    "pre_season_ppg_for": "season_ppg_for",
    "pre_season_ppg_against": "season_ppg_against",
}
# The same metrics measured after a game, used for upcoming games (latest value per team).
AFTER = {"season_win_pct": "pre_season_win_pct", "pythag_win_pct": "pre_pythag_win_pct",
         "roll_win_pct": "pre_roll_win_pct", "roll_margin": "pre_roll_margin",
         "roll_adj_margin": "pre_roll_adj_margin", "roll_pf": "pre_roll_pf", "roll_pa": "pre_roll_pa",
         "season_ppg_for": "pre_season_ppg_for", "season_ppg_against": "pre_season_ppg_against"}


def read_games_for_features(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT game_id, season, week, game_type, gameday, gametime, home_team, away_team,
               home_score, away_score, home_rest, away_rest, div_game,
               spread_line, total_line, home_moneyline, away_moneyline
        FROM games
        """,
        conn,
    )


def build_features(games: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the game-level table from the games and team_game_metrics tables."""
    g = games.copy()
    g["played"] = g["home_score"].notna() & g["away_score"].notna()

    # Pre-game values for games already played, straight from the metrics table.
    played_vals = metrics[["game_id", "team", "games_before"] + list(FEATURES)]

    # For upcoming games: each team's latest values after its last played game this season.
    latest = (metrics.sort_values("game_number").groupby(["season", "team"]).tail(1)
              [["season", "team", "game_number"] + list(AFTER)]   # only the after-game columns
              .rename(columns=AFTER)                               # now named like the pre_ columns
              .rename(columns={"game_number": "games_before"}))    # games played so far

    def side_frame(side: str) -> pd.DataFrame:
        d = g[["game_id", "season", "played"]].assign(team=g[f"{side}_team"])
        d = d.merge(played_vals, on=["game_id", "team"], how="left")
        d = d.merge(latest, on=["season", "team"], how="left", suffixes=("", "_latest"))
        for col in ["games_before"] + list(FEATURES):
            d[col] = np.where(d["played"], d[col], d[col + "_latest"])
        d["games_before"] = d["games_before"].fillna(0).astype(int)  # no games yet means zero
        out = d[["game_id", "games_before"] + list(FEATURES)]
        names = {"games_before": f"{side}_games_before"}
        names.update({k: f"{side}_{v}" for k, v in FEATURES.items()})
        return out.rename(columns=names)

    out = g[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"]].copy()
    out["is_playoff"] = (g["game_type"] != "REG").astype(int)
    out = out.merge(side_frame("home"), on="game_id").merge(side_frame("away"), on="game_id")

    # Home minus away difference for each team metric.
    for short in FEATURES.values():
        out[f"diff_{short}"] = out[f"home_{short}"] - out[f"away_{short}"]

    # Scoring features for the total: both teams' offense and defense together.
    out["sum_roll_pf"] = out["home_roll_pf"] + out["away_roll_pf"]
    out["sum_roll_pa"] = out["home_roll_pa"] + out["away_roll_pa"]
    out["sum_season_ppg_for"] = out["home_season_ppg_for"] + out["away_season_ppg_for"]
    out["sum_season_ppg_against"] = out["home_season_ppg_against"] + out["away_season_ppg_against"]
    # Expected total: each team's scoring averaged with what its opponent usually allows.
    out["exp_total_roll"] = (out["sum_roll_pf"] + out["sum_roll_pa"]) / 2
    out["exp_total_season"] = (out["sum_season_ppg_for"] + out["sum_season_ppg_against"]) / 2

    # Schedule features.
    out["home_rest"] = g["home_rest"].values
    out["away_rest"] = g["away_rest"].values
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["div_game"] = g["div_game"].values

    # Targets, empty for games not yet played. A tie has no home win.
    margin = (g["home_score"] - g["away_score"]).where(g["played"])
    out["margin"] = margin.values
    out["total"] = (g["home_score"] + g["away_score"]).where(g["played"]).values
    out["home_win"] = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, np.nan))

    # Betting lines, kept as benchmark columns only. A positive spread means the home team is favored.
    out["market_spread"] = g["spread_line"].values
    out["market_total"] = g["total_line"].values
    out["market_home_moneyline"] = g["home_moneyline"].values
    out["market_away_moneyline"] = g["away_moneyline"].values

    # Usable for modeling: both teams have enough history.
    out["usable"] = ((out["home_games_before"] >= MIN_PRIOR_GAMES)
                     & (out["away_games_before"] >= MIN_PRIOR_GAMES)).astype(int)
    return out.sort_values(["season", "gameday", "game_id"]).reset_index(drop=True)


def verify(games: pd.DataFrame, feats: pd.DataFrame, n: int = 400, seed: int = 0) -> tuple[int, int]:
    """Recompute some features from raw scores with plain loops. Returns (values checked, mismatches).

    For each sampled played game and side, only that team's games earlier in the same season are used.
    Covers season win %, rolling win %, rolling margin, Pythagorean win % and the points scored and
    allowed averages. Opponent-adjusted margin was checked when the metrics were built.
    """
    played = games[games["home_score"].notna() & games["away_score"].notna()]
    rows = []  # one raw row per team per game
    for r in played.itertuples():
        rows.append((r.season, r.home_team, r.gameday, r.home_score, r.away_score))
        rows.append((r.season, r.away_team, r.gameday, r.away_score, r.home_score))
    raw = pd.DataFrame(rows, columns=["season", "team", "gameday", "pf", "pa"]).sort_values("gameday")
    history = {k: v for k, v in raw.groupby(["season", "team"])}

    pool = feats[(feats["usable"] == 1) & feats["margin"].notna()]
    sample = pool.sample(min(n, len(pool)), random_state=seed)   # fewer rows than n is fine
    checked = bad = 0
    for r in sample.itertuples():
        for side, team in (("home", r.home_team), ("away", r.away_team)):
            h = history[(r.season, team)]
            prior = h[h["gameday"] < r.gameday]
            win = (prior["pf"] > prior["pa"]) + 0.5 * (prior["pf"] == prior["pa"])
            last = prior.tail(WINDOW)
            last_win = (last["pf"] > last["pa"]) + 0.5 * (last["pf"] == last["pa"])
            expected = {
                "season_win_pct": win.mean(),
                "roll_win_pct": last_win.mean(),
                "roll_margin": (last["pf"] - last["pa"]).mean(),
                "pythag_win_pct": float(pythag(prior["pf"].sum(), prior["pa"].sum())),
                "roll_pf": last["pf"].mean(),
                "roll_pa": last["pa"].mean(),
                "season_ppg_for": prior["pf"].mean(),
                "season_ppg_against": prior["pa"].mean(),
            }
            for name, value in expected.items():
                checked += 1
                bad += abs(getattr(r, f"{side}_{name}") - value) > 1e-9
    return checked, int(bad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None, help="game_id to print for a hand check")
    args = parser.parse_args()

    conn = get_connection()
    games = read_games_for_features(conn)
    metrics = read_metrics(conn)
    feats = build_features(games, metrics)

    frame = pl.from_pandas(feats)  # NaN becomes NULL
    ensure_table(conn, TABLE, frame, PRIMARY_KEY)
    upsert_df(conn, TABLE, frame, PRIMARY_KEY)

    stored = conn.execute(f'SELECT COUNT(*) FROM "{TABLE}"').fetchone()[0]
    print(f"Games in source: {len(games)} | rows in {TABLE}: {stored} | match: {stored == len(games)}")
    print(f"Duplicate game_ids: {int(feats['game_id'].duplicated().sum())}\n")

    played = feats["margin"].notna()
    summary = feats.groupby("season").agg(
        games=("game_id", "size"),
        played=("margin", "count"),
        usable=("usable", "sum"),
    )
    summary["training_rows"] = feats[played & (feats["usable"] == 1)].groupby("season").size()
    summary["upcoming"] = feats[~played].groupby("season").size()
    print(summary.fillna(0).astype(int).to_string())
    train = feats[played & (feats["usable"] == 1)]
    print(f"\nTraining rows (played and usable): {len(train)} "
          f"| winner model rows (ties excluded): {int(train['home_win'].notna().sum())}")
    print(f"Minimum prior games per team: {MIN_PRIOR_GAMES}")

    checked, bad = verify(games, feats)
    print(f"\nLeakage check: recomputed {checked} feature values from raw scores of earlier games, "
          f"{bad} mismatches")

    if args.game:
        row = feats[feats["game_id"] == args.game]
        if row.empty:
            print(f"\nNo game {args.game}")
        else:
            pd.set_option("display.max_rows", 100)
            print(f"\nFeatures for {args.game}:")
            print(row.iloc[0].to_string())
    conn.close()


if __name__ == "__main__":
    main()