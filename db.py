"""SQLite helpers used by the loader, trainer, SQL insights, and dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import DATABASE_PATH


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating the parent directory when needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def database_exists() -> bool:
    return DATABASE_PATH.exists()


def read_query(query: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    """Run a read-only SQL query and return its result as a DataFrame."""
    with get_connection() as connection:
        return pd.read_sql_query(query, connection, params=params)
