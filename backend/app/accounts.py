"""Lightweight auth using only Python's standard library — no external packages.

- Passwords: PBKDF2-HMAC-SHA256 with a per-user random salt.
- Tokens: a signed value "user_id.expiry.signature" (HMAC-SHA256). Stateless,
  no session storage needed. The signing key is generated once per install.
"""

import hashlib
import hmac
import os
import secrets
import time

from fastapi import Header, HTTPException
from typing import Optional

from app import db

# A signing key persisted next to the database so tokens survive restarts.
_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".vetai_key")


def _signing_key():
    if os.path.exists(_KEY_PATH):
        with open(_KEY_PATH, "rb") as f:
            return f.read()
    key = secrets.token_bytes(32)
    with open(_KEY_PATH, "wb") as f:
        f.write(key)
    return key


_KEY = _signing_key()
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 days


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return dk.hex(), salt


def verify_password(password, pw_hash, salt):
    calc, _ = hash_password(password, salt)
    return hmac.compare_digest(calc, pw_hash)


def make_token(user_id):
    exp = int(time.time()) + TOKEN_TTL
    msg = f"{user_id}.{exp}"
    sig = hmac.new(_KEY, msg.encode(), hashlib.sha256).hexdigest()
    return f"{msg}.{sig}"


def verify_token(token):
    try:
        user_id, exp, sig = token.split(".")
        msg = f"{user_id}.{exp}"
        expected = hmac.new(_KEY, msg.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        if int(exp) < time.time():
            return None
        return int(user_id)
    except Exception:
        return None


def current_user(authorization: Optional[str] = Header(None)):
    """FastAPI dependency: REQUIRES a valid token, else 401."""
    user = optional_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in.")
    return user


def optional_user(authorization: Optional[str] = Header(None)):
    """FastAPI dependency: returns the user dict if logged in, else None."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    uid = verify_token(authorization.split(" ", 1)[1])
    if not uid:
        return None
    return db.get_user(uid)
