"""SQLite helpers: connect, create or extend tables from a Polars schema, and upsert rows.

The database file is data/nfl.db at the repo root. Tables are created from the incoming
frame's column types, so new upstream columns are added instead of breaking the load.
"""

import sqlite3
from pathlib import Path

import polars as pl

# Repo root is the parent of the src folder, so the path works from any working directory.
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "nfl.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def sqlite_type(dtype: pl.DataType) -> str:
    """Map a Polars dtype to a SQLite column type."""
    if dtype.is_integer() or dtype == pl.Boolean:
        return "INTEGER"
    if dtype.is_float():
        return "REAL"
    return "TEXT"  # strings, dates and anything else are stored as text


def quote(name: str) -> str:
    """Quote an identifier so names like 'temp' or 'result' are always safe in SQL."""
    return '"' + name.replace('"', '""') + '"'


def ensure_table(
    conn: sqlite3.Connection, table: str, df: pl.DataFrame, primary_key: list[str]
) -> list[str]:
    """Create `table` from df's schema if it is missing; otherwise add any new columns.

    Returns the list of columns added to an existing table (empty on a first create
    or when nothing changed). Columns that disappear upstream are left in place.
    """
    missing_pk = [c for c in primary_key if c not in df.columns]
    if missing_pk:
        raise KeyError(f"Primary key columns not in frame: {missing_pk}")

    col_types = {name: sqlite_type(dtype) for name, dtype in df.schema.items()}

    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()

    if not exists:
        col_defs = ", ".join(f"{quote(c)} {t}" for c, t in col_types.items())
        pk = ", ".join(quote(c) for c in primary_key)
        conn.execute(f"CREATE TABLE {quote(table)} ({col_defs}, PRIMARY KEY ({pk}))")
        conn.commit()
        return []

    # Table exists: compare its columns with the frame and add what is new.
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({quote(table)})")}
    added = [c for c in col_types if c not in existing]
    for c in added:
        conn.execute(f"ALTER TABLE {quote(table)} ADD COLUMN {quote(c)} {col_types[c]}")
    conn.commit()
    return added


def upsert_df(
    conn: sqlite3.Connection, table: str, df: pl.DataFrame, primary_key: list[str]
) -> int:
    """Insert every row of df, or update the row if its primary key already exists.

    This is what makes reruns safe: the same game or team-week never appears twice,
    and a corrected value upstream overwrites the old one. Returns the rows written.
    """
    # SQLite's Python driver handles ints, floats and strings. Cast anything else
    # (dates, datetimes) to text so no adapter is needed.
    plain = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32,
             pl.UInt64, pl.Float32, pl.Float64, pl.String, pl.Boolean)
    df = df.with_columns(
        [pl.col(c).cast(pl.String) for c, t in df.schema.items() if t not in plain]
    )

    cols = df.columns
    col_list = ", ".join(quote(c) for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    pk = ", ".join(quote(c) for c in primary_key)

    # On a key conflict, overwrite every non-key column with the incoming value.
    updates = ", ".join(f"{quote(c)} = excluded.{quote(c)}" for c in cols if c not in primary_key)

    sql = (
        f"INSERT INTO {quote(table)} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk}) DO UPDATE SET {updates}"
    )

    # df.rows() yields plain Python tuples, with None for nulls.
    conn.executemany(sql, df.rows())
    conn.commit()
    return df.height