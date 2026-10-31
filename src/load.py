"""Step 5: backfill games and team stats into data/nfl.db (idempotent).

Run from the repo root:  python -m src.load
Safe to run any number of times: rows are upserted on their primary key, so reruns
update existing rows instead of duplicating them.
"""

from src.db import DB_PATH, ensure_table, get_connection, quote, upsert_df
from src.fetch import (SEASONS, load_qb_stats_pl, load_schedules_pl, load_team_stats_pl, load_teams_pl)

# (table name, loader, primary key columns)
TABLES = [
    ("games", load_schedules_pl, ["game_id"]),
    ("team_game_stats", load_team_stats_pl, ["game_id", "team"]),
    ("teams", load_teams_pl, ["team_abbr"]),
    ("qb_game_stats", load_qb_stats_pl, ["game_id", "player_id"]),
]

def count_rows(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0]


def main() -> None:
    conn = get_connection()
    print(f"Database file: {DB_PATH}")
    print(f"Seasons: {SEASONS}\n")

    for table, loader, pk in TABLES:
        df = loader()                                   # download current data
        added = ensure_table(conn, table, df, pk)       # create the table or add new columns
        before = count_rows(conn, table)
        upsert_df(conn, table, df, pk)                  # insert new rows, update existing ones
        after = count_rows(conn, table)

        # Rows that were new this run vs rows that already existed and were refreshed.
        inserted = after - before
        print(f"Table: {table}")
        print(f"  columns added: {added if added else 'none'}")
        print(f"  source rows: {df.height} | database rows before: {before} | after: {after}")
        print(f"  inserted: {inserted} | updated in place: {df.height - inserted}")
        print(f"  database rows match source: {after == df.height}\n")

    # Integrity checks across the two tables.
    orphans = conn.execute(
        "SELECT COUNT(*) FROM team_game_stats t "
        "LEFT JOIN games g ON g.game_id = t.game_id WHERE g.game_id IS NULL"
    ).fetchone()[0]
    scored_no_stats = conn.execute(
        "SELECT COUNT(*) FROM games g "
        "WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL "
        "AND (SELECT COUNT(*) FROM team_game_stats t WHERE t.game_id = g.game_id) = 0"
    ).fetchone()[0]
    print("Integrity checks:")
    print(f"  team_game_stats rows with no matching game: {orphans}")
    print(f"  scored games with no team stats yet: {scored_no_stats}")

    conn.close()


if __name__ == "__main__":
    main()