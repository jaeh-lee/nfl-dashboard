"""Phase 2: walk-forward backtest for home win probability and game margin (the spread).

Run from the repo root:  python -m src.backtest
                         python -m src.backtest --start-season 2024

For every (season, week) from the start season on, each model is trained only on regular season
games from EARLIER weeks, then predicts that week. Nothing from the predicted week or later is used.
Results are scored against baselines and the betting market on the same games, and every
prediction is stored in the backtest_predictions table.
"""

import argparse
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.db import ensure_table, get_connection, upsert_df

TABLE = "backtest_predictions"
PRIMARY_KEY = ["game_id", "target", "model"]
MIN_TRAIN_GAMES = 200    # skip a week if fewer earlier games than this are available

# Features for the full models: home minus away team metrics plus schedule features.
FEATURES = ["diff_season_win_pct", "diff_pythag_win_pct", "diff_roll_win_pct", "diff_roll_margin",
            "diff_roll_adj_margin", "rest_diff", "div_game"]
ONE_FEATURE = ["diff_pythag_win_pct"]   # the one-feature baseline model


# ---------------------------------------------------------------- data

def load_frame(conn) -> pd.DataFrame:
    """Played, usable, regular season games with features, targets and market columns."""
    df = pd.read_sql_query(
        """
        SELECT * FROM game_features
        WHERE usable = 1 AND is_playoff = 0 AND margin IS NOT NULL
        """,
        conn,
    )
    df["key"] = df["season"] * 100 + df["week"]   # orders games by season, then week
    return df.sort_values(["key", "game_id"]).reset_index(drop=True)


def implied_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """Market win probability for the home team from moneylines, with the bookmaker's margin removed."""
    def raw(ml):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    h, a = raw(home_ml), raw(away_ml)
    return pd.Series(h / (h + a), index=home_ml.index)


# ---------------------------------------------------------------- models

def winner_models() -> dict:
    """Home win probability models. Each value is (feature columns, estimator factory)."""
    return {
        "logistic_pythag_only": (ONE_FEATURE, lambda: make_pipeline(StandardScaler(), LogisticRegression())),
        "logistic": (FEATURES, lambda: make_pipeline(StandardScaler(), LogisticRegression())),
        "gradient_boosting": (FEATURES, lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0)),
    }


def margin_models() -> dict:
    """Margin (home minus away points) models."""
    alphas = [1.0, 10.0, 100.0, 1000.0]
    return {
        "ridge_pythag_only": (ONE_FEATURE, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))),
        "ridge": (FEATURES, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))),
        "gradient_boosting": (FEATURES, lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0)),
    }


# ---------------------------------------------------------------- walk-forward

def walk_forward(df: pd.DataFrame, start_season: int) -> pd.DataFrame:
    """Train on earlier weeks and predict each week from start_season on. Returns long predictions."""
    out = []
    w_models, m_models = winner_models(), margin_models()
    keys = sorted(k for k in df["key"].unique() if k // 100 >= start_season)

    for k in keys:
        test = df[df["key"] == k]
        train = df[df["key"] < k]
        assert train["key"].max() < k, "training data must come strictly before the predicted week"
        if len(train) < MIN_TRAIN_GAMES:
            continue
        base = {"game_id": test["game_id"].values, "season": test["season"].values, "week": test["week"].values}

        # Winner: ties have no home win, so they are left out of training and scoring.
        tw = train[train["home_win"].notna()]
        te = test[test["home_win"].notna()]
        if len(te):
            b = {"game_id": te["game_id"].values, "season": te["season"].values, "week": te["week"].values}
            actual = te["home_win"].values
            preds = {"baseline_home_rate": np.full(len(te), tw["home_win"].mean()),
                     "market": implied_home_prob(te["market_home_moneyline"], te["market_away_moneyline"]).values}
            for name, (cols, make) in w_models.items():
                preds[name] = make().fit(tw[cols], tw["home_win"]).predict_proba(te[cols])[:, 1]
            for name, p in preds.items():
                out.append(pd.DataFrame({**b, "target": "home_win", "model": name, "prediction": p, "actual": actual}))

        # Margin.
        actual = test["margin"].values
        preds = {"baseline_mean_margin": np.full(len(test), train["margin"].mean()),
                 "market": test["market_spread"].values}   # positive spread means the home team is favored
        for name, (cols, make) in m_models.items():
            preds[name] = make().fit(train[cols], train["margin"]).predict(test[cols])
        for name, p in preds.items():
            out.append(pd.DataFrame({**base, "target": "margin", "model": name, "prediction": p, "actual": actual}))

    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- scoring

def per_game_loss(preds: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per game and model with the per-game loss: log loss for winner, absolute error for margin."""
    d = preds[preds["target"] == target].copy()
    if target == "home_win":
        p = d["prediction"].clip(1e-6, 1 - 1e-6)
        d["loss"] = -(d["actual"] * np.log(p) + (1 - d["actual"]) * np.log(1 - p))
        d["brier"] = (d["prediction"] - d["actual"]) ** 2
        d["correct"] = ((d["prediction"] > 0.5) == (d["actual"] == 1)).astype(float)
    else:
        d["loss"] = (d["prediction"] - d["actual"]).abs()
        d["sq"] = (d["prediction"] - d["actual"]) ** 2
    return d


def summarize(preds: pd.DataFrame, target: str) -> pd.DataFrame:
    """Scores per model on the games where every model has a prediction, with a comparison to the market."""
    d = per_game_loss(preds, target)
    counts = d.groupby("game_id")["model"].nunique()
    d = d[d["game_id"].isin(counts[counts == counts.max()].index)]   # same games for every model

    market = d[d["model"] == "market"].set_index("game_id")["loss"]
    rows = []
    for model, g in d.groupby("model"):
        diff = g.set_index("game_id")["loss"] - market.reindex(g["game_id"]).values
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        row = {"model": model, "games": len(g)}
        if target == "home_win":
            row.update({"accuracy": g["correct"].mean(), "log_loss": g["loss"].mean(), "brier": g["brier"].mean()})
            main = "log_loss"
        else:
            row.update({"mae": g["loss"].mean(), "rmse": np.sqrt(g["sq"].mean())})
            main = "mae"
        row[f"{main}_vs_market"] = diff.mean()
        row["plus_minus_95"] = 1.96 * se
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values(main)


def by_season(preds: pd.DataFrame, target: str) -> pd.DataFrame:
    """The main score (log loss or MAE) for each model, by season."""
    d = per_game_loss(preds, target)
    return d.pivot_table(index="model", columns="season", values="loss", aggfunc="mean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2023)
    args = parser.parse_args()

    conn = get_connection()
    df = load_frame(conn)
    print(f"Games available (regular season, usable, played): {len(df)} "
          f"| seasons {df['season'].min()} to {df['season'].max()}")

    t0 = time.time()
    preds = walk_forward(df, args.start_season)
    print(f"Walk-forward from season {args.start_season}: {preds['game_id'].nunique()} games predicted "
          f"in {time.time() - t0:.0f} seconds\n")

    pd.set_option("display.width", 200)
    print("HOME WIN PROBABILITY (lower log loss and Brier are better; vs_market below 0 beats the market)")
    print(summarize(preds, "home_win").round(4).to_string())
    print("\nLog loss by season:")
    print(by_season(preds, "home_win").round(4).to_string())

    print("\nMARGIN / SPREAD (lower MAE is better; vs_market below 0 beats the market)")
    print(summarize(preds, "margin").round(3).to_string())
    print("\nMAE by season:")
    print(by_season(preds, "margin").round(3).to_string())

    frame = pl.from_pandas(preds)
    ensure_table(conn, TABLE, frame, PRIMARY_KEY)
    upsert_df(conn, TABLE, frame, PRIMARY_KEY)
    stored = conn.execute(f'SELECT COUNT(*) FROM "{TABLE}"').fetchone()[0]
    print(f"\nPredictions stored in {TABLE}: {stored} rows (this run produced {len(preds)})")
    conn.close()


if __name__ == "__main__":
    main()