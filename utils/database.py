# ============================================================
# FILE: utils/database.py
# PURPOSE: All SQLite database operations
# EXPLANATION:
#   SQLite = a file-based database, no server needed.
#   We store every detection with: plate text, confidence,
#   timestamp, image path, and source (webcam/image/video).
#   Duplicate detection prevents saving the same plate twice
#   within a short time window (e.g. 5 seconds).
# ============================================================

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
DB_CONFIG = {
    "db_path"         : "database/anpr.db",
    "duplicate_window": 5,    # seconds — ignore same plate within this window
}


# ── Context Manager for DB connections ────────────────────
@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically commits on success, rolls back on error,
    and always closes the connection — even if an exception occurs.

    Usage:
        with get_db_connection() as conn:
            conn.execute(...)
    """
    Path(DB_CONFIG["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_CONFIG["db_path"])
    conn.row_factory = sqlite3.Row   # access columns by name like a dict
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


# ── Database Initialization ────────────────────────────────
def init_database():
    """
    Create the database and tables if they don't exist.
    Called once at startup.

    Table: detections
      id           : auto-increment primary key
      plate_text   : the cleaned OCR result e.g. "LHR-1234"
      confidence   : OCR confidence score 0-1
      timestamp    : when it was detected (ISO format string)
      image_path   : path to saved screenshot
      source       : 'webcam', 'image', or 'video'
      vehicle_conf : YOLOv8 bounding box confidence
      is_duplicate : 1 if this was a duplicate detection
    """
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text    TEXT    NOT NULL,
                confidence    REAL    DEFAULT 0.0,
                timestamp     TEXT    NOT NULL,
                image_path    TEXT    DEFAULT '',
                source        TEXT    DEFAULT 'unknown',
                vehicle_conf  REAL    DEFAULT 0.0,
                is_duplicate  INTEGER DEFAULT 0
            )
        """)

        # Index on plate_text for fast duplicate lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_text
            ON detections (plate_text)
        """)

        # Index on timestamp for time-range queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON detections (timestamp)
        """)

    logger.info(f"Database initialized at: {DB_CONFIG['db_path']}")


# ── CRUD Operations ────────────────────────────────────────

def insert_detection(
    plate_text   : str,
    confidence   : float = 0.0,
    image_path   : str   = "",
    source       : str   = "unknown",
    vehicle_conf : float = 0.0,
) -> tuple[int, bool]:
    """
    Insert a new detection record.
    Checks for duplicates first.

    Args:
        plate_text   : cleaned plate text
        confidence   : OCR confidence
        image_path   : path to saved image
        source       : 'webcam', 'image', or 'video'
        vehicle_conf : YOLO confidence

    Returns:
        (record_id, is_duplicate)
        is_duplicate=True means same plate was seen recently
    """
    if not plate_text:
        logger.warning("Attempted to insert empty plate text.")
        return -1, False

    timestamp    = datetime.now().isoformat()
    is_duplicate = _is_duplicate(plate_text)

    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO detections
                (plate_text, confidence, timestamp, image_path,
                 source, vehicle_conf, is_duplicate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            plate_text.upper(),
            round(confidence, 4),
            timestamp,
            image_path,
            source,
            round(vehicle_conf, 4),
            int(is_duplicate),
        ))
        record_id = cursor.lastrowid

    logger.info(
        f"Saved: {plate_text} | conf={confidence:.2f} | "
        f"source={source} | duplicate={is_duplicate}"
    )
    return record_id, is_duplicate


def _is_duplicate(plate_text: str) -> bool:
    """
    Check if this plate was detected within the last N seconds.
    Prevents flooding the database with the same plate in a video.

    duplicate_window: defined in DB_CONFIG (default 5 seconds)
    """
    window_start = (
        datetime.now() - timedelta(seconds=DB_CONFIG["duplicate_window"])
    ).isoformat()

    with get_db_connection() as conn:
        result = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM detections
            WHERE plate_text = ?
              AND timestamp   > ?
              AND is_duplicate = 0
        """, (plate_text.upper(), window_start)).fetchone()

    return result["cnt"] > 0


def get_all_detections(limit: int = 500) -> pd.DataFrame:
    """
    Fetch all detections as a Pandas DataFrame.
    Used by the Streamlit dashboard.

    Args:
        limit: max rows to return (default 500, newest first)

    Returns:
        DataFrame with columns matching the detections table
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(limit,)
        )
    return df


def get_unique_plates() -> pd.DataFrame:
    """
    Get one record per unique plate (most recent per plate).
    Useful for 'fleet tracking' view in dashboard.
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT plate_text,
                   COUNT(*)           AS total_detections,
                   MAX(timestamp)     AS last_seen,
                   AVG(confidence)    AS avg_confidence,
                   MAX(source)        AS source
            FROM detections
            WHERE is_duplicate = 0
            GROUP BY plate_text
            ORDER BY last_seen DESC
            """,
            conn
        )
    return df


def get_stats() -> dict:
    """
    Get summary statistics for the dashboard.
    Returns dict with total detections, unique plates, etc.
    """
    with get_db_connection() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        unique    = conn.execute(
            "SELECT COUNT(DISTINCT plate_text) FROM detections"
        ).fetchone()[0]
        today_str = datetime.now().strftime("%Y-%m-%d")
        today     = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?",
            (f"{today_str}%",)
        ).fetchone()[0]

    return {
        "total_detections" : total,
        "unique_plates"    : unique,
        "today_detections" : today,
    }


def export_to_csv(filepath: str = "logs/detections_export.csv") -> str:
    """
    Export all detections to a CSV file.
    Returns the filepath of the saved CSV.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df = get_all_detections(limit=100000)
    df.to_csv(filepath, index=False)
    logger.info(f"Exported {len(df)} records to {filepath}")
    return filepath


def delete_all_records():
    """Delete all records. Use with caution! (for testing/reset)"""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM detections")
    logger.warning("All detection records deleted!")