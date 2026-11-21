
"""Authentication utilities for NEWSINTEL AI."""

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from src import database

# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_connection(db_path):
    database.init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# --------------------------------------------------
# Password security
# --------------------------------------------------

def hash_password(password):
    salt = secrets.token_bytes(16)

    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
    )

    return (
        f"scrypt$16384$8$1$"
        f"{salt.hex()}${password_hash.hex()}"
    )


def verify_password(password, stored_hash):
    try:
        algorithm, n, r, p, salt_hex, hash_hex = stored_hash.split("$")

        if algorithm != "scrypt":
            return False

        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
        )

        return hmac.compare_digest(actual_hash, expected_hash)

    except (ValueError, TypeError):
        return False


# --------------------------------------------------
# User registration
# --------------------------------------------------

def register_user(db_path, email, password):
    email = email.strip().lower()

    if not email or "@" not in email:
        return False, "Please enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash = hash_password(password)

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (?, ?)
                """,
                (email, password_hash),
            )

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


# --------------------------------------------------
# User login
# --------------------------------------------------

def authenticate_user(db_path, email, password):
    email = email.strip().lower()

    with get_connection(db_path) as conn:
        user = conn.execute(
            """
            SELECT id, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

        if user is None:
            return None

        if not verify_password(password, user["password_hash"]):
            return None

        conn.execute(
            """
            UPDATE users
            SET last_login_at = ?
            WHERE id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                user["id"],
            ),
        )

        return {
            "id": user["id"],
            "email": user["email"],
        }


# --------------------------------------------------
# Session management
# --------------------------------------------------

def create_session(db_path, user_id, days=7):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=days)).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions
            (user_id, token_hash, created_at, expires_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                token_hash,
                now.isoformat(),
                expires_at,
                now.isoformat(),
            ),
        )

    return token


def get_session_user(db_path, token):
    if not token:
        return None

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with get_connection(db_path) as conn:
        session = conn.execute(
            """
            SELECT users.id, users.email, sessions.expires_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

        if session is None:
            return None

        expires_at = datetime.fromisoformat(session["expires_at"])

        if expires_at <= datetime.now(timezone.utc):
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (token_hash,),
            )
            return None

        conn.execute(
            """
            UPDATE sessions
            SET last_seen_at = ?
            WHERE token_hash = ?
            """,
            (datetime.now(timezone.utc).isoformat(), token_hash),
        )

        return {
            "id": session["id"],
            "email": session["email"],
        }


def logout_user(db_path, token):
    if not token:
        return

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (token_hash,),
        )
