"""Conversation history: persisted per turn, recent turns fed back as LLM context (SPEC §3.3)."""

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

MAX_MESSAGES = 30
MAX_TOKENS = 3000


def save_turn(
    conn: sqlite3.Connection,
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    interrupted: bool = False,
) -> None:
    """Store one final turn. For an interrupted reply, `content` is only the part the user heard."""
    if not content.strip():
        return
    conn.execute(
        "INSERT INTO messages (user_id, session_id, role, content, interrupted) VALUES (?, ?, ?, ?, ?)",
        (user_id, session_id, role, content.strip(), int(interrupted)),
    )
    conn.commit()


def clear(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()


def estimate_tokens(text: str) -> int:
    # ponytail: ~4 chars per token for English; use the model's tokenizer if budgets get tight
    return len(text) // 4 + 1


def load_context(
    conn: sqlite3.Connection,
    user_id: int,
    tz: ZoneInfo,
    max_messages: int = MAX_MESSAGES,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    """Most recent earlier turns as chat messages, oldest first, within both budgets.

    The first user message of each earlier session gets a date prefix, so the model knows these
    are past conversations. (A system message per session would be cleaner, but Qwen chat
    templates only accept a system message in first position.)
    """
    rows = conn.execute(
        "SELECT session_id, role, content, interrupted, created_at FROM messages"
        " WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, max_messages),
    ).fetchall()
    picked, used = [], 0
    for r in rows:  # newest first
        used += estimate_tokens(r["content"])
        if used > max_tokens:
            break
        picked.append(r)
    picked.reverse()
    # never open on a reply to a question that fell outside the budget
    while picked and picked[0]["role"] == "assistant":
        picked.pop(0)

    messages, session, undated = [], None, False
    for r in picked:
        content = r["content"]
        if r["interrupted"]:
            content += " [interrupted]"
        if r["session_id"] != session:
            session, undated = r["session_id"], True
        if undated and r["role"] == "user":
            when = datetime.fromisoformat(r["created_at"]).astimezone(tz)
            content = f"(earlier conversation, {when:%d %b %Y}) {content}"
            undated = False
        messages.append({"role": r["role"], "content": content})
    return messages
