"""Predict the next unplayed regular season week and store the results in the predictions table.

Run from the repo root:  python -m src.predict

Trains one primary model per target (see PRIMARY) on every completed, usable regular season game,
then predicts the games of the earliest week that still has unplayed games. The refresh workflow
runs this after the features step. Predictions for games that are already played are kept as a
record. Nothing time-dependent is stored, so rerunning on unchanged data leaves the database as is.

Not predicted yet: playoff games, and early-season games where a team has fewer than
MIN_PRIOR_GAMES prior games (the usable flag in game_features).
"""

import numpy as np
import pandas as pd
import polars as pl

from src.backtest import implied_home_prob, margin_models, total_models, winner_models
from src.db import ensure_table, get_connection, upsert_df

TABLE = "predictions"
PRIMARY_KEY = ["game_id", "target", "model"]

# The model shown in the app for each target, with the model set it comes from.
PRIMARY = {"home_win": "logistic", "margin": "ridge", "total": "ridge_expected_total_only"}
MODEL_SETS = {"home_win": winner_models, "margin": margin_models, "total": total_models}


def make_predictions(feats: pd.DataFrame) -> pd.DataFrame:
    """Fit the primary models on completed games and predict the next unplayed week."""
    reg = feats[feats["is_playoff"] == 0]
    train = reg[(reg["margin"].notna()) & (reg["usable"] == 1)]
    unplayed = reg[reg["margin"].isna()]
    if unplayed.empty:
        return pd.DataFrame()

    # The earliest week with games still to play.
    key = unplayed["season"] * 100 + unplayed["week"]
    todo = unplayed[(key == key.min()) & (unplayed["usable"] == 1)]
    if todo.empty:
        return pd.DataFrame()

    out = []
    for target, model_name in PRIMARY.items():
        cols, make = MODEL_SETS[target]()[model_name]
        fit_rows = train[train["home_win"].notna()] if target == "home_win" else train
        est = make().fit(fit_rows[cols], fit_rows[target])
        pred = est.predict_proba(todo[cols])[:, 1] if target == "home_win" else est.predict(todo[cols])
        out.append(pd.DataFrame({
            "game_id": todo["game_id"].values, "target": target, "model": model_name,
            "prediction": pred, "season": todo["season"].values, "week": todo["week"].values,
            "train_games": len(fit_rows), "trained_through": fit_rows["gameday"].max(),
        }))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    conn = get_connection()
    feats = pd.read_sql_query("SELECT * FROM game_features", conn)
    preds = make_predictions(feats)

    if preds.empty:
        print("No predictions made: there are no upcoming regular season games with enough history.")
        conn.close()
        return

    frame = pl.from_pandas(preds)
    ensure_table(conn, TABLE, frame, PRIMARY_KEY)
    upsert_df(conn, TABLE, frame, PRIMARY_KEY)

    week = preds.iloc[0]
    print(f"Predicted season {week['season']} week {week['week']}: {preds['game_id'].nunique()} games")
    for target in PRIMARY:
        p = preds[preds["target"] == target].iloc[0]
        print(f"  {target}: {p['model']}, trained on {p['train_games']} games through {p['trained_through']}")

    # A readable comparison with the market.
    wide = preds.pivot(index="game_id", columns="target", values="prediction")
    game = feats.set_index("game_id").loc[wide.index]
    table = pd.DataFrame({
        "away": game["away_team"], "home": game["home_team"],
        "model_home_win%": (wide["home_win"] * 100).round(1),
        "market_home_win%": (implied_home_prob(game["market_home_moneyline"], game["market_away_moneyline"]) * 100).round(1),
        "model_margin": wide["margin"].round(1), "market_spread": game["market_spread"],
        "model_total": wide["total"].round(1), "market_total": game["market_total"],
    })
    pd.set_option("display.width", 200)
    print()
    print(table.to_string())

    stored = conn.execute(f'SELECT COUNT(*) FROM "{TABLE}"').fetchone()[0]
    print(f"\nRows in {TABLE}: {stored} (this run wrote {len(preds)})")
    conn.close()


if __name__ == "__main__":
    main()