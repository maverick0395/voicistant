from zoneinfo import ZoneInfo

import pytest

from app import history
from app.db import connect
from app.users import add_user, authenticate

KYIV = ZoneInfo("Europe/Kyiv")


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


def turns(conn, uid, session, *pairs, when="2026-09-18T20:30:00Z"):
    for role, text in pairs:
        history.save_turn(conn, uid, session, role, text)
    conn.execute("UPDATE messages SET created_at = ? WHERE session_id = ?", (when, session))


def test_auth(conn):
    uid = add_user(conn, "anna", "correct horse battery")
    assert authenticate(conn, "anna", "correct horse battery") == uid
    assert authenticate(conn, "anna", "wrong") is None
    assert authenticate(conn, "nobody", "whatever") is None


def test_context_order_dates_and_interrupts(conn):
    uid = add_user(conn, "anna", "x" * 10)
    turns(conn, uid, "s1", ("user", "Hi"), ("assistant", "Hello!"), when="2026-09-17T22:30:00Z")
    turns(conn, uid, "s2", ("assistant", "Welcome back."), ("user", "Tell me a story"))
    history.save_turn(conn, uid, "s2", "assistant", "Once upon a time", interrupted=True)

    ctx = history.load_context(conn, uid, KYIV)
    assert [m["role"] for m in ctx] == ["user", "assistant", "assistant", "user", "assistant"]
    # 17 Sep 22:30 UTC is already 18 Sep in Kyiv
    assert ctx[0]["content"] == "(earlier conversation, 18 Sep 2026) Hi"
    # a session opened by the bot's greeting gets the date on its first user turn
    assert ctx[2]["content"] == "Welcome back."
    assert ctx[3]["content"].startswith("(earlier conversation, 18 Sep 2026) Tell")
    assert ctx[4]["content"] == "Once upon a time [interrupted]"


def test_budgets_keep_newest_and_never_open_on_assistant(conn):
    uid = add_user(conn, "anna", "x" * 10)
    for i in range(20):
        turns(conn, uid, "s1", ("user", f"q{i}"), ("assistant", f"a{i}"))

    ctx = history.load_context(conn, uid, KYIV, max_messages=5)
    assert ctx[-1]["content"] == "a19"
    assert ctx[0]["role"] == "user" and len(ctx) == 4  # newest 5 = a17..a19; leading a17 dropped

    long = "word " * 400  # ~500 tokens each
    turns(conn, uid, "s2", ("user", long), ("assistant", long))
    ctx = history.load_context(conn, uid, KYIV, max_tokens=800)
    assert ctx == []  # only the newest (assistant) fits, and it can't open the context


def test_clear_and_user_isolation(conn):
    a, b = add_user(conn, "anna", "x" * 10), add_user(conn, "bob", "y" * 10)
    turns(conn, a, "s1", ("user", "mine"))
    turns(conn, b, "s2", ("user", "his"))
    history.clear(conn, a)
    assert history.load_context(conn, a, KYIV) == []
    assert history.load_context(conn, b, KYIV)[0]["content"].endswith("his")


def test_blank_turns_are_not_stored(conn):
    uid = add_user(conn, "anna", "x" * 10)
    history.save_turn(conn, uid, "s1", "assistant", "   ")
    assert history.load_context(conn, uid, KYIV) == []
