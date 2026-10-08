"""Local storage for VetAI — uses Python's built-in sqlite3.
Handles users, animals, predictions, vaccinations, and deworming records.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vetai.db")
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS animals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                tag_number TEXT,
                animal_type TEXT NOT NULL,
                breed TEXT,
                age_months INTEGER,
                weight_kg INTEGER,
                gender TEXT,
                created TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                animal_id INTEGER,
                disease_id TEXT,
                disease_name TEXT,
                confidence REAL,
                risk_level TEXT,
                mode TEXT,
                created TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (animal_id) REFERENCES animals(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS vaccinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                animal_id INTEGER NOT NULL,
                vaccine_name TEXT NOT NULL,
                dose_number TEXT,
                administered_date TEXT NOT NULL,
                next_due_date TEXT,
                batch_number TEXT,
                veterinarian TEXT,
                notes TEXT,
                created TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (animal_id) REFERENCES animals(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS deworming (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                animal_id INTEGER NOT NULL,
                medicine_name TEXT NOT NULL,
                dose_amount TEXT,
                administered_date TEXT NOT NULL,
                next_due_date TEXT,
                veterinarian TEXT,
                notes TEXT,
                created TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (animal_id) REFERENCES animals(id) ON DELETE CASCADE
            );
            """
        )
        # Migration: ensure tag_number exists on existing databases
        try:
            c.execute("ALTER TABLE animals ADD COLUMN tag_number TEXT")
        except sqlite3.OperationalError:
            pass  # Already exists


# ---------- users ----------
def create_user(name, phone, pw_hash, salt):
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO users (name, phone, pw_hash, salt, created) VALUES (?,?,?,?,?)",
            (name, phone, pw_hash, salt, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_user_by_phone(phone):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        return dict(row) if row else None


def get_user(user_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ---------- animals ----------
def add_animal(user_id, data):
    tag = data.get("tag_number")
    tag_cleaned = tag.strip() if tag and tag.strip() else None
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO animals (user_id, name, tag_number, animal_type, breed, age_months,
                weight_kg, gender, created) VALUES (?,?,?,?,?,?,?,?,?)""",
            (user_id, data.get("name"), tag_cleaned, data.get("animal_type"), data.get("breed"),
             data.get("age_months"), data.get("weight_kg"), data.get("gender"),
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_animals(user_id):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM animals WHERE user_id = ? ORDER BY created DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_animal(user_id, animal_id):
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM animals WHERE id = ? AND user_id = ?", (animal_id, user_id)
        ).fetchone()
        return dict(row) if row else None


def delete_animal(user_id, animal_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM animals WHERE id = ? AND user_id = ?", (animal_id, user_id))


# ---------- predictions ----------
def save_prediction(user_id, animal_id, result):
    top = result.get("top", {})
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO predictions (user_id, animal_id, disease_id, disease_name,
                confidence, risk_level, mode, created) VALUES (?,?,?,?,?,?,?,?)""",
            (user_id, animal_id, top.get("disease_id"), top.get("name"),
             top.get("confidence"), top.get("risk_level"), result.get("mode"),
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def animal_history(user_id, animal_id):
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM predictions WHERE user_id = ? AND animal_id = ?
               ORDER BY created DESC""", (user_id, animal_id)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- vaccinations (Phase 3) ----------
def add_vaccination(user_id, animal_id, data):
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO vaccinations (user_id, animal_id, vaccine_name, dose_number,
                administered_date, next_due_date, batch_number, veterinarian, notes, created)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                animal_id,
                data.get("vaccine_name"),
                data.get("dose_number"),
                data.get("administered_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                data.get("next_due_date"),
                data.get("batch_number"),
                data.get("veterinarian"),
                data.get("notes"),
                datetime.now(timezone.utc).isoformat()
            )
        )
        return cur.lastrowid


def list_vaccinations(user_id, animal_id=None):
    with _conn() as c:
        if animal_id:
            rows = c.execute(
                """SELECT v.*, a.name as animal_name, a.tag_number, a.animal_type
                   FROM vaccinations v
                   JOIN animals a ON v.animal_id = a.id
                   WHERE v.user_id = ? AND v.animal_id = ?
                   ORDER BY v.administered_date DESC""",
                (user_id, animal_id)
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT v.*, a.name as animal_name, a.tag_number, a.animal_type
                   FROM vaccinations v
                   JOIN animals a ON v.animal_id = a.id
                   WHERE v.user_id = ?
                   ORDER BY v.administered_date DESC""",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_vaccination(user_id, vac_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM vaccinations WHERE id = ? AND user_id = ?", (vac_id, user_id)).fetchone()
        return dict(row) if row else None


def delete_vaccination(user_id, vac_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM vaccinations WHERE id = ? AND user_id = ?", (vac_id, user_id))


# ---------- deworming (Phase 3) ----------
def add_deworming(user_id, animal_id, data):
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO deworming (user_id, animal_id, medicine_name, dose_amount,
                administered_date, next_due_date, veterinarian, notes, created)
                VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                animal_id,
                data.get("medicine_name"),
                data.get("dose_amount"),
                data.get("administered_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                data.get("next_due_date"),
                data.get("veterinarian"),
                data.get("notes"),
                datetime.now(timezone.utc).isoformat()
            )
        )
        return cur.lastrowid


def list_deworming(user_id, animal_id=None):
    with _conn() as c:
        if animal_id:
            rows = c.execute(
                """SELECT d.*, a.name as animal_name, a.tag_number, a.animal_type
                   FROM deworming d
                   JOIN animals a ON d.animal_id = a.id
                   WHERE d.user_id = ? AND d.animal_id = ?
                   ORDER BY d.administered_date DESC""",
                (user_id, animal_id)
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT d.*, a.name as animal_name, a.tag_number, a.animal_type
                   FROM deworming d
                   JOIN animals a ON d.animal_id = a.id
                   WHERE d.user_id = ?
                   ORDER BY d.administered_date DESC""",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def delete_deworming(user_id, dew_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM deworming WHERE id = ? AND user_id = ?", (dew_id, user_id))


# ---------- dashboard stats & health summary ----------
def dashboard_stats(user_id):
    with _conn() as c:
        animals = c.execute(
            "SELECT * FROM animals WHERE user_id = ?", (user_id,)
        ).fetchall()
        preds = c.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created DESC", (user_id,)
        ).fetchall()
        vacs = c.execute(
            "SELECT * FROM vaccinations WHERE user_id = ?", (user_id,)
        ).fetchall()
        dews = c.execute(
            "SELECT * FROM deworming WHERE user_id = ?", (user_id,)
        ).fetchall()

    animals = [dict(a) for a in animals]
    preds = [dict(p) for p in preds]
    vacs = [dict(v) for v in vacs]
    dews = [dict(d) for d in dews]

    by_species, by_disease, by_risk = {}, {}, {}
    for a in animals:
        by_species[a["animal_type"]] = by_species.get(a["animal_type"], 0) + 1
    for p in preds:
        if p["disease_name"]:
            by_disease[p["disease_name"]] = by_disease.get(p["disease_name"], 0) + 1
        if p["risk_level"]:
            by_risk[p["risk_level"]] = by_risk.get(p["risk_level"], 0) + 1

    # Compute upcoming / overdue vaccination counts
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    overdue_count = 0
    upcoming_30_count = 0
    for v in vacs:
        due = v.get("next_due_date")
        if due:
            if due < today:
                overdue_count += 1
            else:
                try:
                    d_due = datetime.strptime(due, "%Y-%m-%d")
                    d_today = datetime.strptime(today, "%Y-%m-%d")
                    diff_days = (d_due - d_today).days
                    if 0 <= diff_days <= 30:
                        upcoming_30_count += 1
                except ValueError:
                    pass

    return {
        "total_animals": len(animals),
        "total_diagnoses": len(preds),
        "total_vaccines": len(vacs),
        "total_deworming": len(dews),
        "vaccines_overdue": overdue_count,
        "vaccines_due_30": upcoming_30_count,
        "by_species": by_species,
        "by_disease": by_disease,
        "by_risk": by_risk,
        "recent": preds[:8],
    }
