import sqlite3
from datetime import datetime

DB_NAME = "history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT,
            location TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_entry(object_name, location):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO detections (object_name, location, timestamp) VALUES (?, ?, ?)",
        (object_name, location, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT object_name, location, timestamp FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows