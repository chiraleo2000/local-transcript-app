"""SQLite user accounts with PBKDF2 password hashes and signed session tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.paths import app_root

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")
_PBKDF2_ITERS = 200_000
_TOKEN_TTL_S = 7 * 24 * 3600


@dataclass(frozen=True)
class UserRecord:
    id: int
    username: str
    is_active: bool


def users_db_path() -> Path:
    raw = os.getenv("APP_USERS_DB", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = app_root() / path
        return path
    return app_root() / "storage" / "users.db"


def _auth_secret() -> bytes:
    configured = os.getenv("APP_AUTH_SECRET", "").strip()
    if configured:
        return configured.encode("utf-8")
    # Dev fallback derived from install path (set APP_AUTH_SECRET in production).
    material = f"lta|{app_root().resolve()}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _connect() -> sqlite3.Connection:
    path = users_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        _PBKDF2_ITERS,
    )
    return (
        f"pbkdf2_sha256${_PBKDF2_ITERS}$"
        f"{base64.b64encode(salt_bytes).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters_s, salt_b64, digest_b64 = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iters,
    )
    return hmac.compare_digest(actual, expected)


def init_user_db() -> None:
    """Create schema and seed the bootstrap admin user when missing."""
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()
            _seed_user_unlocked(conn)
        finally:
            conn.close()


def _seed_user_unlocked(conn: sqlite3.Connection) -> None:
    seed_user = os.getenv("APP_SEED_USER", "chira").strip() or "chira"
    # Empty APP_SEED_PASSWORD= in dotenv must not create an empty-password user.
    seed_password = (os.getenv("APP_SEED_PASSWORD") or "").strip() or "leo2569"
    if not _USERNAME_RE.match(seed_user):
        logger.warning("Invalid APP_SEED_USER=%r; skipping seed.", seed_user)
        return
    row = conn.execute(
        "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
        (seed_user,),
    ).fetchone()
    if row is not None:
        return
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at, is_active) VALUES (?, ?, ?, 1)",
        (seed_user, _hash_password(seed_password), time.time()),
    )
    conn.commit()
    logger.info("Seeded user account username=%s", seed_user)


def validate_username(username: str) -> str:
    name = (username or "").strip()
    if not _USERNAME_RE.match(name):
        raise ValueError(
            "Username must be 3–64 chars: letters, digits, underscore, dot, or hyphen."
        )
    return name


def register_user(username: str, password: str) -> UserRecord:
    name = validate_username(username)
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    init_user_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            if existing is not None:
                raise ValueError("Username already taken.")
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, created_at, is_active) "
                "VALUES (?, ?, ?, 1)",
                (name, _hash_password(password), time.time()),
            )
            conn.commit()
            return UserRecord(id=int(cur.lastrowid), username=name, is_active=True)
        finally:
            conn.close()


def authenticate_user(username: str, password: str) -> UserRecord | None:
    init_user_db()
    name = (username or "").strip()
    if not name or not password:
        return None
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, password_hash, is_active FROM users "
                "WHERE username = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
        finally:
            conn.close()
    if row is None or int(row["is_active"]) != 1:
        return None
    if not _verify_password(password, str(row["password_hash"])):
        return None
    return UserRecord(
        id=int(row["id"]),
        username=str(row["username"]),
        is_active=True,
    )


def get_user_by_id(user_id: int) -> UserRecord | None:
    init_user_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, is_active FROM users WHERE id = ?",
                (int(user_id),),
            ).fetchone()
        finally:
            conn.close()
    if row is None or int(row["is_active"]) != 1:
        return None
    return UserRecord(
        id=int(row["id"]),
        username=str(row["username"]),
        is_active=True,
    )


def get_user_by_username(username: str) -> UserRecord | None:
    init_user_db()
    name = (username or "").strip()
    if not name:
        return None
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, username, is_active FROM users "
                "WHERE username = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
        finally:
            conn.close()
    if row is None or int(row["is_active"]) != 1:
        return None
    return UserRecord(
        id=int(row["id"]),
        username=str(row["username"]),
        is_active=True,
    )


def issue_session_token(user: UserRecord, *, ttl_s: int = _TOKEN_TTL_S) -> str:
    payload = {
        "uid": user.id,
        "u": user.username,
        "exp": int(time.time()) + max(60, ttl_s),
        "n": secrets.token_hex(8),
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    sig = hmac.new(_auth_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_session_token(token: str | None) -> UserRecord | None:
    if not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    expected = hmac.new(_auth_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    pad = "=" * (-len(body) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + pad).decode("utf-8"))
    except ValueError:
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    user = get_user_by_id(int(payload.get("uid") or 0))
    if user is None:
        return None
    if user.username.lower() != str(payload.get("u") or "").lower():
        return None
    return user


def gradio_auth_credentials(username: str, password: str) -> bool:
    """Gradio ``auth=`` callable — True when credentials match an active user."""
    return authenticate_user(username, password) is not None


def user_public_dict(user: UserRecord) -> dict[str, Any]:
    return {"id": user.id, "username": user.username}
