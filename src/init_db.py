"""Step 4: create the SQLite schema from the real nflverse frames and print a check report.

Run from the repo root:  python -m src.init_db
Safe to run more than once: existing tables are kept and only new columns are added.
"""

from src.db import DB_PATH, ensure_table, get_connection, quote
from src.fetch import SEASONS, load_schedules_pl, load_team_stats_pl

# (table name, loader, primary key columns)
TABLES = [
    ("games", load_schedules_pl, ["game_id"]),
    ("team_game_stats", load_team_stats_pl, ["game_id", "team"]),
]


def main() -> None:
    conn = get_connection()
    print(f"Database file: {DB_PATH}")
    print(f"Seasons: {SEASONS}\n")

    for table, loader, pk in TABLES:
        df = loader()  # download the current frame; we only use its schema here
        added = ensure_table(conn, table, df, pk)

        info = list(conn.execute(f"PRAGMA table_info({quote(table)})"))
        pk_cols = [row[1] for row in sorted(info, key=lambda r: r[5]) if row[5] > 0]
        rows = conn.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0]

        print(f"Table: {table}")
        print(f"  columns in database: {len(info)} | columns in source frame: {df.width}")
        print(f"  primary key: {pk_cols}")
        print(f"  columns added this run: {added if added else 'none'}")
        print(f"  rows in database: {rows} (loading happens in Step 5)")
        print(f"  rows in source frame: {df.height}\n")

    conn.close()


if __name__ == "__main__":
    main()