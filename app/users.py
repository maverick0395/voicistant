"""Manage invited users (admin CLI, run on the VPS).

uv run -m app.users add <name>      # prompts for the password
uv run -m app.users remove <name>   # also deletes their history
uv run -m app.users list
"""

import argparse
import getpass
import sqlite3
import sys

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from app.db import connect

ph = PasswordHasher()


def add_user(conn: sqlite3.Connection, name: str, password: str) -> int:
    cur = conn.execute(
        "INSERT INTO users (name, password_hash) VALUES (?, ?)", (name, ph.hash(password))
    )
    conn.commit()
    return cur.lastrowid


def authenticate(conn: sqlite3.Connection, name: str, password: str) -> int | None:
    """Return the user id if the password matches, else None."""
    row = conn.execute("SELECT id, password_hash FROM users WHERE name = ?", (name,)).fetchone()
    if row is None:
        ph.hash(password)  # same work as a real check, so timing doesn't reveal which names exist
        return None
    try:
        ph.verify(row["password_hash"], password)
    except VerificationError:
        return None
    return row["id"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("action", choices=["add", "remove", "list"])
    p.add_argument("name", nargs="?")
    a = p.parse_args(argv)
    conn = connect()
    if a.action == "list":
        for r in conn.execute("SELECT name, created_at FROM users ORDER BY name"):
            print(f"{r['name']}\t{r['created_at']}")
        return
    if not a.name:
        p.error("name is required")
    if a.action == "add":
        password = getpass.getpass(f"Password for {a.name}: ")
        if len(password) < 10:
            sys.exit("password must be at least 10 characters")
        if password != getpass.getpass("Repeat: "):
            sys.exit("passwords don't match")
        try:
            add_user(conn, a.name, password)
        except sqlite3.IntegrityError:
            sys.exit(f"user {a.name!r} already exists")
        print(f"added {a.name}")
    else:
        n = conn.execute("DELETE FROM users WHERE name = ?", (a.name,)).rowcount
        conn.commit()
        print(f"removed {a.name}" if n else f"no user {a.name!r}")


if __name__ == "__main__":
    main()
