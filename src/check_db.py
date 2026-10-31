"""Check data/nfl.db for duplicate, orphaned or inconsistent rows.

Run from the repo root:  python -m src.check_db
                         python -m src.check_db --upstream   (also compares against current nflverse data)

Prints OK or PROBLEM for each check and exits with a non-zero status if any check finds a problem.
The tables have primary keys, so exact duplicates cannot exist. The checks below look for the things a
primary key cannot catch: the same matchup under two game_ids, rows that point at a missing game, row
counts that do not line up between tables, and rows that no longer exist upstream.
"""

import argparse
import sys

from src.db import DB_PATH, get_connection, quote

# table name, primary key columns
TABLES = {
    "games": ["game_id"],
    "team_game_stats": ["game_id", "team"],
    "teams": ["team_abbr"],
    "team_game_metrics": ["game_id", "team"],
    "game_features": ["game_id"],
    "backtest_predictions": ["game_id", "target", "model"],
    "predictions": ["game_id", "target", "model"],
    "qb_game_stats": ["game_id", "player_id"], "qb_games": ["game_id", "player_id"], "qb_predictions": ["game_id", "player_id"],
}

problems = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    """Print one check result and count failures."""
    global problems
    problems += 0 if ok else 1
    print(f"  {'OK     ' if ok else 'PROBLEM'} {label}" + (f": {detail}" if detail else ""))


def scalar(conn, sql: str):
    return conn.execute(sql).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="store_true", help="compare games with current nflverse data")
    args = parser.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"Database not found at {DB_PATH}")
    conn = get_connection()
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    print("1. File integrity")
    report(scalar(conn, "PRAGMA integrity_check") == "ok", "SQLite integrity_check")

    print("\n2. Primary keys and duplicate keys")
    for table, key in TABLES.items():
        if table not in existing:
            print(f"  --      {table}: table not present, skipped")
            continue
        info = list(conn.execute(f"PRAGMA table_info({quote(table)})"))
        pk_cols = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
        rows = scalar(conn, f"SELECT COUNT(*) FROM {quote(table)}")
        cols = ", ".join(quote(c) for c in key)
        distinct = scalar(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT {cols} FROM {quote(table)})")
        report(pk_cols == key and rows == distinct, f"{table}: {rows} rows",
               f"key {pk_cols}, {rows - distinct} duplicate keys" if (pk_cols != key or rows != distinct) else "")

    print("\n3. Same game under two game_ids")
    dup_matchups = scalar(conn, """
        SELECT COUNT(*) FROM (SELECT 1 FROM games GROUP BY season, game_type, week, home_team, away_team HAVING COUNT(*) > 1)""")
    report(dup_matchups == 0, "same season, week, home and away team more than once", f"{dup_matchups} found" if dup_matchups else "")
    twice_in_week = scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT season, week, team FROM (
                SELECT season, week, home_team AS team FROM games WHERE game_type = 'REG'
                UNION ALL
                SELECT season, week, away_team FROM games WHERE game_type = 'REG')
            GROUP BY season, week, team HAVING COUNT(*) > 1)""")
    report(twice_in_week == 0, "a team playing twice in the same regular season week", f"{twice_in_week} found" if twice_in_week else "")
    same_day = scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT gameday, team FROM (
                SELECT gameday, home_team AS team FROM games UNION ALL SELECT gameday, away_team FROM games)
            GROUP BY gameday, team HAVING COUNT(*) > 1)""")
    report(same_day == 0, "a team playing twice on the same day", f"{same_day} found" if same_day else "")

    print("\n4. Rows that point at a missing game")
    for table in ("team_game_stats", "team_game_metrics", "game_features", "backtest_predictions", "predictions"):
        if table in existing:
            n = scalar(conn, f"""SELECT COUNT(*) FROM {quote(table)} t
                                 LEFT JOIN games g ON g.game_id = t.game_id WHERE g.game_id IS NULL""")
            report(n == 0, f"{table} rows with no matching game", f"{n} found" if n else "")

    print("\n5. Row counts line up between tables")
    played = scalar(conn, "SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL AND away_score IS NOT NULL")
    total = scalar(conn, "SELECT COUNT(*) FROM games")
    if "team_game_metrics" in existing:
        n = scalar(conn, "SELECT COUNT(*) FROM team_game_metrics")
        report(n == 2 * played, f"team_game_metrics {n} rows = 2 x {played} played games")
    if "game_features" in existing:
        n = scalar(conn, "SELECT COUNT(*) FROM game_features")
        report(n == total, f"game_features {n} rows = {total} games")
    if "team_game_stats" in existing:
        n = scalar(conn, "SELECT COUNT(*) FROM team_game_stats")
        report(n <= 2 * played, f"team_game_stats {n} rows, at most 2 x {played} played games",
               "" if n == 2 * played else f"{2 * played - n} played team-games have no stats yet (normal right after a game)")

    if args.upstream:
        print("\n6. Compared with current nflverse data")
        from src.fetch import load_schedules_pl
        up = load_schedules_pl()
        up_ids = set(up["game_id"].to_list())
        db_ids = {r[0] for r in conn.execute("SELECT game_id FROM games")}
        stale = sorted(db_ids - up_ids)
        missing = sorted(up_ids - db_ids)
        report(not stale, "games in the database that nflverse no longer has", f"{stale[:5]}{' ...' if len(stale) > 5 else ''}" if stale else "")
        report(not missing, "games in nflverse that the database does not have (run python -m src.load)",
               f"{len(missing)} missing" if missing else "")

    print(f"\n{'All checks passed' if problems == 0 else str(problems) + ' check(s) found a problem'}")
    conn.close()
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()