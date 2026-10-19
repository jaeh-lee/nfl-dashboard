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