import sqlite3
import hashlib
from typing import List, Tuple, Optional

from config import DB_PATH


def init_db() -> None:
    """Initialize SQLite database with users, patients, and scans tables.

    The schema matches the integrated CURASCAN admin app so both admin
    and modular pages (e.g. `scan_upload_page.py`) share the same data.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute(
        """CREATE TABLE IF NOT EXISTS users
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )

    # Patients table
    c.execute(
        """CREATE TABLE IF NOT EXISTS patients
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            dob DATE,
            gender TEXT,
            clinical_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT)"""
    )

    # Scans table
    c.execute(
        """CREATE TABLE IF NOT EXISTS scans
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            scan_type TEXT,
            scan_path TEXT,
            result TEXT,
            confidence REAL,
            overlay_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id))"""
    )

    # Lightweight schema migrations for older databases
    c.execute("PRAGMA table_info(scans)")
    scan_columns = [row[1] for row in c.fetchall()]

    if "result" not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN result TEXT")
    if "confidence" not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN confidence REAL DEFAULT 0.0")
    if "overlay_path" not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN overlay_path TEXT")
    if "created_at" not in scan_columns:
        c.execute(
            "ALTER TABLE scans ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
    if "created_by" not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN created_by TEXT")

    c.execute("PRAGMA table_info(patients)")
    patient_columns = [row[1] for row in c.fetchall()]
    if "created_at" not in patient_columns:
        c.execute(
            "ALTER TABLE patients ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
    if "created_by" not in patient_columns:
        c.execute("ALTER TABLE patients ADD COLUMN created_by TEXT")

    c.execute("PRAGMA table_info(users)")
    user_columns = [row[1] for row in c.fetchall()]
    if "role" not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    if "created_at" not in user_columns:
        c.execute(
            "ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )

    # Seed default admin user if missing
    try:
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", password_hash, "admin"),
        )
    except sqlite3.IntegrityError:
        # Admin already exists
        pass

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Return SHA256 hash of a plaintext password."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username: str, password: str) -> Optional[Tuple]:
    """Return full user row if credentials are valid, else None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hash_password(password)
    c.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password_hash),
    )
    user = c.fetchone()
    conn.close()
    return user


def get_patients() -> List[Tuple]:
    """Fetch all patients ordered by creation time (newest first)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY created_at DESC")
    patients = c.fetchall()
    conn.close()
    return patients


def add_patient(
    name: str, dob: str, gender: str, clinical_notes: str, created_by: str
) -> str:
    """Create a new patient and return its generated patient_id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM patients")
    count = c.fetchone()[0]

    patient_id = chr(65 + (count % 26)) + "-" + str(count + 1)

    c.execute(
        "INSERT INTO patients (patient_id, name, dob, gender, clinical_notes, created_by) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (patient_id, name, dob, gender, clinical_notes, created_by),
    )
    conn.commit()
    conn.close()
    return patient_id


def save_scan(
    patient_id: str,
    scan_type: str,
    scan_path: str,
    result: str,
    confidence: float,
    overlay_path: str | None,
    created_by: str,
) -> None:
    """Insert a scan record linked to a patient."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scans (patient_id, scan_type, scan_path, result, confidence, overlay_path, created_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (patient_id, scan_type, scan_path, result, confidence, overlay_path, created_by),
    )
    conn.commit()
    conn.close()


def get_patient_scans(patient_id: str) -> List[Tuple]:
    """Return all scans for the given patient ordered by newest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM scans WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    )
    scans = c.fetchall()
    conn.close()
    return scans

