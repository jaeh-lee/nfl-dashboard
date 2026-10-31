"""Read-only queries used by the Streamlit app."""

import pandas as pd


def read_games(conn) -> pd.DataFrame:
    """Every game in the database, played or not, with the columns the weekly view needs."""
    return pd.read_sql_query(
        """
        SELECT game_id, season, week, game_type, gameday, weekday, gametime,
               away_team, away_score, home_team, home_score
        FROM games
        """,
        conn,
    )


def read_team_names(conn) -> dict[str, str]:
    """Map team abbreviation to full name, for example KC to Kansas City Chiefs.

    Returns an empty dict if the teams table does not exist yet, so the app falls
    back to abbreviations instead of failing.
    """
    try:
        teams = pd.read_sql_query("SELECT team_abbr, team_name FROM teams", conn)
    except Exception:
        return {}
    return dict(zip(teams["team_abbr"], teams["team_name"]))


def read_predictions(conn) -> pd.DataFrame:
    """Stored model predictions with the betting lines for the same games.

    One row per game, target and model. Returns an empty frame if the predictions table
    does not exist yet, so the app shows a message instead of failing.
    """
    try:
        return pd.read_sql_query(
            """
            SELECT p.game_id, p.target, p.model, p.prediction, p.season, p.week,
                   p.train_games, p.trained_through,
                   f.market_spread, f.market_total, f.market_home_moneyline, f.market_away_moneyline
            FROM predictions p
            JOIN game_features f ON f.game_id = p.game_id
            """,
            conn,
        )
    except Exception:
        return pd.DataFrame()

def read_qb_predictions(conn) -> pd.DataFrame:
    """Stored quarterback passing yard projections. Empty if the table does not exist yet."""
    try:
        return pd.read_sql_query("SELECT * FROM qb_predictions", conn)
    except Exception:
        return pd.DataFrame()


def read_qb_games(conn) -> pd.DataFrame:
    """Every quarterback start with the opponent's usual passing yards allowed. Empty if missing."""
    try:
        return pd.read_sql_query("SELECT * FROM qb_games", conn)
    except Exception:
        return pd.DataFrame()