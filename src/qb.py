"""Quarterback passing yards: features, a walk-forward backtest, and projections for the next week.

Run from the repo root:  python -m src.qb backtest   (walk-forward test of the models, prints scores)
                         python -m src.qb predict    (project the next unplayed week and store tables)

Everything about a game uses only information from before it:
  - The QB's recent form is his last WINDOW_STARTS starts, pulled toward the league average when he
    has few starts (shrinkage), so a QB with 3 starts is not trusted like one with 8.
  - The opponent's pass defense is what it has allowed to opposing offenses earlier in the same
    season, shrunk toward the league average the same way.
  - Context: home or away, the team's implied total and spread (betting lines), rest days, dome.

A "start" is a regular season game where the QB threw at least MIN_ATTEMPTS passes. The QB projected
for an upcoming game is the one who threw the most passes in his team's most recent game.
"""

import argparse

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.db import ensure_table, get_connection, upsert_df

MIN_ATTEMPTS = 15        # pass attempts that make a game a start
WINDOW_STARTS = 8        # how many recent starts describe a QB
MIN_PRIOR_STARTS = 3     # a QB needs this many earlier starts to be modeled or projected
K_QB = 4                 # shrinkage strength: a QB's n starts count as n / (n + K_QB) of his own average
K_DEF = 4                # same for a defense's games
START_SEASON = 2023      # backtest evaluation starts here
MIN_TRAIN_ROWS = 300

FEATURES = ["qb_att", "qb_ypa", "def_att", "def_ypa", "is_home", "implied_total", "fav_by",
            "rest_days", "dome"]

# The model whose projections are stored and shown in the app.
PRIMARY = "ridge_direct"


# ---------------------------------------------------------------- building the feature frame

def date_prior_means(df: pd.DataFrame, sums: dict[str, str]) -> pd.DataFrame:
    """For each row, totals over all rows with a strictly earlier gameday.

    sums maps an output name to a column. Rows whose value is empty (upcoming games) add nothing.
    Returns a frame indexed like df with one column per output name plus 'n' (rows counted).
    """
    d = pd.DataFrame({"gameday": df["gameday"].values})
    for name, col in sums.items():
        d[name] = df[col].fillna(0).values
    d["n"] = df[list(sums.values())[0]].notna().astype(int).values
    daily = d.groupby("gameday").sum().sort_index().cumsum().shift(1)   # totals before each date
    out = daily.reindex(d["gameday"]).reset_index(drop=True)
    out.index = df.index
    return out


def build_frame(qbg: pd.DataFrame, games: pd.DataFrame, team: pd.DataFrame):
    """Feature rows for every QB start, plus one row per expected starter of the next unplayed week.

    Returns (frame, defense_state). The frame's 'upcoming' column is 1 for the next-week rows.
    """
    g = games.copy()
    g["played"] = g["home_score"].notna() & g["away_score"].notna()

    # ---- QB starts and the expected starters of the next unplayed regular season week
    qbg = qbg.merge(g[["game_id", "gameday", "gametime"]], on="game_id", how="left")
    starts = qbg[(qbg["season_type"] == "REG") & (qbg["attempts"] >= MIN_ATTEMPTS)].copy()
    starts["upcoming"] = 0

    unplayed = g[(g["game_type"] == "REG") & (~g["played"])]
    rows = []
    if len(unplayed):
        key = unplayed["season"] * 100 + unplayed["week"]
        nxt = unplayed[key == key.min()]
        last = (qbg.sort_values(["gameday", "gametime", "attempts"])
                .groupby("team").tail(50))                       # recent rows, ordered by time
        for team_code, rows_t in last.groupby("team"):
            latest_game = rows_t.sort_values(["gameday", "gametime"]).iloc[-1]["game_id"]
            gm = rows_t[rows_t["game_id"] == latest_game].sort_values("attempts").iloc[-1]
            for _, u in nxt[(nxt["home_team"] == team_code) | (nxt["away_team"] == team_code)].iterrows():
                if gm["season"] != u["season"]:
                    continue    # the QB's latest game was last season, so the starter is unknown
                rows.append({
                    "player_id": gm["player_id"], "player_name": gm["player_name"], "team": team_code,
                    "opponent_team": u["away_team"] if u["home_team"] == team_code else u["home_team"],
                    "season": u["season"], "week": u["week"], "game_id": u["game_id"],
                    "gameday": u["gameday"], "gametime": u["gametime"], "attempts": np.nan,
                    "passing_yards": np.nan, "upcoming": 1,
                })
    frame = pd.concat([starts, pd.DataFrame(rows)], ignore_index=True)
    frame = frame.sort_values(["player_id", "gameday", "gametime"]).reset_index(drop=True)

    # ---- QB form: last WINDOW_STARTS starts, shrunk toward the league average
    by_qb = frame.groupby("player_id", sort=False)
    prior_att = by_qb["attempts"].transform(lambda s: s.shift(1).rolling(WINDOW_STARTS, min_periods=1).sum())
    prior_yds = by_qb["passing_yards"].transform(lambda s: s.shift(1).rolling(WINDOW_STARTS, min_periods=1).sum())
    n = by_qb["attempts"].transform(lambda s: s.shift(1).rolling(WINDOW_STARTS, min_periods=1).count()).fillna(0)
    frame["qb_n_prior"] = n.astype(int)
    frame["qb_last_yards"] = prior_yds / n.replace(0, np.nan)        # raw average of the last starts

    lg = date_prior_means(frame, {"att": "attempts", "yds": "passing_yards"})
    lg_att, lg_ypa = lg["att"] / lg["n"], lg["yds"] / lg["att"]       # league per start, league yards per attempt
    w = n / (n + K_QB)
    frame["qb_att"] = w * (prior_att / n.replace(0, np.nan)).fillna(0) + (1 - w) * lg_att
    frame["qb_ypa"] = w * (prior_yds / prior_att.replace(0, np.nan)).fillna(0) + (1 - w) * lg_ypa

    # ---- Opponent pass defense: what each defense allowed earlier this season
    t = team.merge(g[["game_id", "season", "game_type", "gameday", "gametime", "home_team", "away_team", "played"]],
                   on="game_id")
    t["defense"] = np.where(t["team"] == t["home_team"], t["away_team"], t["home_team"])
    allowed = t[["game_id", "season", "game_type", "gameday", "gametime", "defense", "attempts", "passing_yards"]].copy()
    ph = g[(g["game_type"] == "REG") & (~g["played"])]
    ph_rows = pd.concat([
        pd.DataFrame({"game_id": ph["game_id"], "season": ph["season"], "game_type": "REG", "gameday": ph["gameday"],
                      "gametime": ph["gametime"], "defense": ph[side], "attempts": np.nan, "passing_yards": np.nan})
        for side in ("home_team", "away_team")
    ])
    allowed = pd.concat([allowed, ph_rows], ignore_index=True).sort_values(["season", "defense", "gameday", "gametime"])
    gd = allowed.groupby(["season", "defense"], sort=False)
    valid = allowed["attempts"].notna().astype(int)
    for col in ("attempts", "passing_yards"):
        allowed["cum_" + col] = gd[col].transform(lambda s: s.fillna(0).cumsum()) - allowed[col].fillna(0)
    allowed["n_before"] = gd["attempts"].transform(lambda s: s.notna().astype(int).cumsum()) - valid

    lt = date_prior_means(allowed, {"att": "attempts", "yds": "passing_yards"})
    lt_att, lt_ypa = lt["att"] / lt["n"], lt["yds"] / lt["att"]        # league per team-game
    nb = allowed["n_before"]
    wd = nb / (nb + K_DEF)
    allowed["def_att"] = wd * (allowed["cum_attempts"] / nb.replace(0, np.nan)).fillna(0) + (1 - wd) * lt_att
    allowed["def_ypa"] = wd * (allowed["cum_passing_yards"] / allowed["cum_attempts"].replace(0, np.nan)).fillna(0) + (1 - wd) * lt_ypa
    allowed["def_yards_pg"] = allowed["def_att"] * allowed["def_ypa"]
    allowed["league_yards_pg"] = lt_att * lt_ypa

    # Season totals per defense, to compare a QB's game with what the defense allowed in its other games.
    reg_played = allowed[allowed["attempts"].notna() & (allowed["game_type"] == "REG")]   # regular season only
    sums = reg_played.groupby(["season", "defense"])["passing_yards"].agg(["sum", "count"])
    allowed = allowed.merge(sums.rename(columns={"sum": "season_yds_allowed", "count": "season_games"}),
                            left_on=["season", "defense"], right_index=True, how="left")
    allowed["opp_avg_allowed_other"] = ((allowed["season_yds_allowed"] - allowed["passing_yards"])
                                        / (allowed["season_games"] - 1).replace(0, np.nan))

    frame = frame.merge(
        allowed[["game_id", "defense", "def_att", "def_ypa", "def_yards_pg", "league_yards_pg",
                 "opp_avg_allowed_other"]].rename(columns={"defense": "opponent_team"}),
        on=["game_id", "opponent_team"], how="left")

    # ---- Context from the schedule
    ctx = g[["game_id", "home_team", "spread_line", "total_line", "roof", "home_rest", "away_rest"]]
    frame = frame.merge(ctx, on="game_id", how="left")
    frame["is_home"] = (frame["team"] == frame["home_team"]).astype(int)
    frame["implied_total"] = np.where(frame["is_home"] == 1, (frame["total_line"] + frame["spread_line"]) / 2,
                                      (frame["total_line"] - frame["spread_line"]) / 2)
    frame["fav_by"] = np.where(frame["is_home"] == 1, frame["spread_line"], -frame["spread_line"])
    frame["rest_days"] = np.where(frame["is_home"] == 1, frame["home_rest"], frame["away_rest"])
    frame["dome"] = frame["roof"].isin(["dome", "closed"]).astype(int)
    frame["location"] = np.where(frame["is_home"] == 1, "home", "away")
    frame["vs_opp_avg"] = frame["passing_yards"] - frame["opp_avg_allowed_other"]
    frame["ypa"] = frame["passing_yards"] / frame["attempts"]

    frame["eligible"] = ((frame["qb_n_prior"] >= MIN_PRIOR_STARTS) & frame[FEATURES].notna().all(axis=1)).astype(int)
    frame = frame.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    final_def = allowed[allowed["attempts"].isna()]
    return frame, final_def


# ---------------------------------------------------------------- models

def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit every model on train and return each model's yard predictions for test."""
    out = {
        "baseline_league": np.full(len(test), train["passing_yards"].mean()),
        "baseline_qb_recent": test["qb_last_yards"].values,
        "baseline_qb_shrunk": (test["qb_att"] * test["qb_ypa"]).values,
    }
    direct = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])).fit(train[FEATURES], train["passing_yards"])
    out["ridge_direct"] = direct.predict(test[FEATURES])

    att = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])).fit(train[FEATURES], train["attempts"])
    ypa = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])).fit(train[FEATURES], train["ypa"])
    out["ridge_structural"] = att.predict(test[FEATURES]) * ypa.predict(test[FEATURES])

    gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30,
                                       l2_regularization=1.0, random_state=0).fit(train[FEATURES], train["passing_yards"])
    out["gradient_boosting"] = gb.predict(test[FEATURES])

    resid = train["passing_yards"] - direct.predict(train[FEATURES])
    out["_q10"], out["_q90"] = np.quantile(resid, 0.10), np.quantile(resid, 0.90)
    out["_attempts"] = att.predict(test[FEATURES])
    return out


def walk_forward(frame: pd.DataFrame, start_season: int = START_SEASON) -> pd.DataFrame:
    """Train on earlier weeks and predict each week from start_season on. Returns one row per QB start and model."""
    d = frame[(frame["upcoming"] == 0) & (frame["eligible"] == 1)].copy()
    d["key"] = d["season"] * 100 + d["week"]
    rows = []
    for k in sorted(x for x in d["key"].unique() if x // 100 >= start_season):
        train, test = d[d["key"] < k], d[d["key"] == k]
        assert train["key"].max() < k
        if len(train) < MIN_TRAIN_ROWS:
            continue
        preds = fit_predict(train, test)
        for name, p in preds.items():
            if name.startswith("_"):
                continue
            rows.append(pd.DataFrame({"game_id": test["game_id"].values, "player_id": test["player_id"].values,
                                      "season": test["season"].values, "model": name, "prediction": p,
                                      "actual": test["passing_yards"].values,
                                      "low": p + preds["_q10"], "high": p + preds["_q90"]}))
    return pd.concat(rows, ignore_index=True)


def summarize(preds: pd.DataFrame, baseline: str = "baseline_qb_recent") -> pd.DataFrame:
    """MAE and RMSE per model on the same starts, with the difference from a baseline and its 95% range."""
    preds = preds.assign(err=(preds["prediction"] - preds["actual"]).abs())
    base = preds[preds["model"] == baseline].set_index(["game_id", "player_id"])["err"]
    rows = []
    for model, g in preds.groupby("model"):
        diff = g.set_index(["game_id", "player_id"])["err"] - base.reindex(g.set_index(["game_id", "player_id"]).index).values
        rows.append({"model": model, "starts": len(g), "mae": g["err"].mean(),
                     "rmse": np.sqrt(((g["prediction"] - g["actual"]) ** 2).mean()),
                     f"mae_vs_{baseline}": diff.mean(), "plus_minus_95": 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))})
    return pd.DataFrame(rows).set_index("model").sort_values("mae")


def read_inputs(conn):
    qbg = pd.read_sql_query("SELECT * FROM qb_game_stats", conn)
    games = pd.read_sql_query(
        """SELECT game_id, season, week, game_type, gameday, gametime, home_team, away_team,
                  home_score, away_score, spread_line, total_line, roof, home_rest, away_rest FROM games""", conn)
    team = pd.read_sql_query("SELECT game_id, team, attempts, passing_yards FROM team_game_stats", conn)
    return qbg, games, team


# ---------------------------------------------------------------- commands

def cmd_backtest() -> None:
    conn = get_connection()
    frame, _ = build_frame(*read_inputs(conn))
    d = frame[(frame["upcoming"] == 0) & (frame["eligible"] == 1)]
    print(f"QB starts available to model: {len(d)} ({d['season'].min()} to {d['season'].max()}), "
          f"{d['player_id'].nunique()} quarterbacks")
    preds = walk_forward(frame)
    n_starts = preds.drop_duplicates(["game_id", "player_id"]).shape[0]
    print(f"Walk-forward from season {START_SEASON}: {n_starts} QB starts scored\n")
    pd.set_option("display.width", 200)
    print("PASSING YARDS (lower is better; the last two columns compare each model with last-8-starts average)")
    print(summarize(preds).round(2).to_string())
    print("\nMAE by season:")
    err = preds.assign(err=(preds["prediction"] - preds["actual"]).abs())
    print(err.pivot_table(index="model", columns="season", values="err", aggfunc="mean").round(2).to_string())
    pm = preds[preds["model"] == PRIMARY]
    cover = ((pm["actual"] >= pm["low"]) & (pm["actual"] <= pm["high"])).mean()
    print(f"\n80% range from {PRIMARY}: contained the actual yards {cover:.1%} of the time "
          f"(average width {(pm['high'] - pm['low']).mean():.0f} yards)")
    conn.close()


def cmd_predict() -> None:
    conn = get_connection()
    frame, final_def = build_frame(*read_inputs(conn))
    train = frame[(frame["upcoming"] == 0) & (frame["eligible"] == 1)]
    todo = frame[(frame["upcoming"] == 1) & (frame["eligible"] == 1)].copy()

    # Descriptive table: every QB start with how the opponent's defense usually does.
    games_tbl = frame[frame["upcoming"] == 0][[
        "game_id", "player_id", "player_name", "team", "opponent_team", "season", "week", "gameday",
        "location", "attempts", "passing_yards", "opp_avg_allowed_other", "vs_opp_avg"]]
    gframe = pl.from_pandas(games_tbl)
    ensure_table(conn, "qb_games", gframe, ["game_id", "player_id"])
    upsert_df(conn, "qb_games", gframe, ["game_id", "player_id"])

    if todo.empty:
        print("No QB projections: no upcoming regular season game has an expected starter with enough history.")
        conn.close()
        return

    p = fit_predict(train, todo)
    todo["projected_yards"] = p[PRIMARY]
    todo["low_yards"] = p[PRIMARY] + p["_q10"]
    todo["high_yards"] = p[PRIMARY] + p["_q90"]
    todo["projected_attempts"] = p["_attempts"]

    # Where the opponent ranks among defenses this season (1 = allows the most passing yards per game).
    cur = final_def[final_def["season"] == todo["season"].iloc[0]].sort_values("gameday").groupby("defense").tail(1)
    rank = cur["def_yards_pg"].rank(ascending=False, method="min").astype(int)
    rank.index = cur["defense"].values
    todo["opp_rank"] = todo["opponent_team"].map(rank)

    out = todo[["game_id", "player_id", "player_name", "team", "opponent_team", "season", "week", "gameday",
                "gametime", "location", "projected_yards", "low_yards", "high_yards", "projected_attempts",
                "qb_last_yards", "qb_n_prior", "def_yards_pg", "league_yards_pg", "opp_rank",
                "implied_total", "fav_by"]].rename(columns={"def_yards_pg": "opp_allowed_yards_pg"})
    out["model"] = PRIMARY
    out["train_starts"] = len(train)
    out["trained_through"] = train["gameday"].max()
    pframe = pl.from_pandas(out)
    ensure_table(conn, "qb_predictions", pframe, ["game_id", "player_id"])
    upsert_df(conn, "qb_predictions", pframe, ["game_id", "player_id"])

    print(f"Projected season {int(out['season'].iloc[0])} week {int(out['week'].iloc[0])}: {len(out)} quarterbacks "
          f"(model {PRIMARY}, trained on {len(train)} starts through {out['trained_through'].iloc[0]})")
    show = out.assign(opp=out["opponent_team"], proj=out["projected_yards"].round(0), low=out["low_yards"].round(0),
                      high=out["high_yards"].round(0), last8=out["qb_last_yards"].round(0),
                      opp_allows=out["opp_allowed_yards_pg"].round(0), league=out["league_yards_pg"].round(0))
    pd.set_option("display.width", 200)
    print(show[["player_name", "team", "opp", "location", "proj", "low", "high", "last8", "opp_allows", "league", "opp_rank"]]
          .sort_values("proj", ascending=False).to_string(index=False))
    skipped = int(((frame["upcoming"] == 1) & (frame["eligible"] == 0)).sum())
    print(f"\nExpected starters skipped (fewer than {MIN_PRIOR_STARTS} earlier starts or no betting line yet): {skipped}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["backtest", "predict"])
    args = parser.parse_args()
    {"backtest": cmd_backtest, "predict": cmd_predict}[args.command]()


if __name__ == "__main__":
    main()