# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""Phase 0 spike: measure the model servers against the SPEC latency budget and gate C.

Run on the GPU box (defaults) or on the VPS through the tunnel:
    MODEL_API_KEY=... uv run gpu/bench.py
    uv run gpu/bench.py --selftest
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import time

import httpx

SYSTEM = (
    "You are a friendly voice assistant. Answer in one to three short spoken sentences. "
    "Start with a short first sentence. No markdown, no lists."
)
CHAT_PROMPTS = [
    "Hi! How are you today?",
    "What's a good way to fall asleep faster?",
    "Explain what a black hole is.",
    "Give me an idea for a quick dinner.",
    "Why is the sky blue?",
    "Tell me a fun fact about octopuses.",
    "How do I stay focused while working from home?",
    "What should I pack for a weekend hike?",
    "Can you recommend a classic science fiction book?",
    "What's the difference between weather and climate?",
]
SPEECH_SENTENCES = [
    "Remind me to call my mother tomorrow at six in the evening.",
    "What is the weather going to be like this weekend in Kyiv?",
    "Add milk, eggs and bread to my shopping list.",
    "How many kilometers are there in twenty six miles?",
    "Tell me something interesting about the history of Rome.",
]


def tool(name, desc, **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": t} for k, t in props.items()},
                "required": list(props),
            },
        },
    }


TOOLS = [
    tool("get_time", "Current local date and time for the user."),
    tool("calculate", "Evaluate an arithmetic expression.", expression="string"),
    tool("set_timer", "Start a countdown timer.", seconds="integer", label="string"),
    tool("add_note", "Save a free-text note.", text="string"),
    tool("add_todo", "Add an item to the todo list.", text="string"),
    tool("list_todos", "Read the user's todo list."),
    tool("web_search", "Search the web for current information.", query="string"),
]
# (prompt, expected tool or None for "answer directly")
TOOL_CASES = [
    ("What time is it?", "get_time"),
    ("What's today's date?", "get_time"),
    ("Is it already evening?", "get_time"),
    ("What is 17 times 23?", "calculate"),
    ("How much is 15 percent of 240?", "calculate"),
    ("Divide 1000 by 7 for me.", "calculate"),
    ("Set a timer for 10 minutes.", "set_timer"),
    ("Remind me in 30 seconds to check the oven.", "set_timer"),
    ("Start a five minute timer for tea.", "set_timer"),
    ("Note that the wifi password is sunflower42.", "add_note"),
    ("Write down: parking spot is level 3, row B.", "add_note"),
    ("Save a note: Anna's birthday is on May 4th.", "add_note"),
    ("Add buy milk to my todo list.", "add_todo"),
    ("Put 'renew passport' on my todos.", "add_todo"),
    ("I need to remember to pay rent, add it to my list.", "add_todo"),
    ("What's on my todo list?", "list_todos"),
    ("Read me my todos.", "list_todos"),
    ("Do I have anything left to do?", "list_todos"),
    ("Who won the last football world cup final?", "web_search"),
    ("Search for the latest news about SpaceX.", "web_search"),
    ("What's the current price of bitcoin?", "web_search"),
    ("Look up opening hours of the Louvre.", "web_search"),
    ("Find reviews of the newest iPhone.", "web_search"),
    ("Hi there!", None),
    ("Tell me a joke.", None),
    ("What's the capital of France?", None),
    ("Thanks, that's all.", None),
    ("How do you say hello in Spanish?", None),
    ("Explain photosynthesis in one sentence.", None),
    ("Can you speak slower?", None),
]
SENTENCE_END = re.compile(r"[.!?](\s|$)")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def ms(s):
    return f"{s * 1000:.0f} ms"


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / reference length (case and punctuation ignored)."""
    r, h = (re.findall(r"[a-z0-9']+", s.lower()) for s in (ref, hyp))
    row = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hw in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
    return row[-1] / max(len(r), 1)


async def llm_stream(c: httpx.AsyncClient, prompt: str) -> tuple[float, float]:
    """Return (time to first token, time to end of first sentence)."""
    body = {
        "model": "llm",
        "stream": True,
        "max_tokens": 150,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0, ttft, text = time.perf_counter(), None, ""
    async with c.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and ttft is None:
                ttft = time.perf_counter() - t0
            text += delta
            if SENTENCE_END.search(text):
                return ttft, time.perf_counter() - t0
    return ttft or 0.0, time.perf_counter() - t0  # no sentence end: whole reply


async def bench_llm(c: httpx.AsyncClient) -> dict:
    await llm_stream(c, "Say ok.")  # warm-up
    seq = [await llm_stream(c, p) for p in CHAT_PROMPTS]
    par = await asyncio.gather(*(llm_stream(c, p) for p in CHAT_PROMPTS[:3]))
    ttft, ttfs = [s[0] for s in seq], [s[1] for s in seq]
    print(f"LLM  time to first token     p50 {ms(pct(ttft, 50))}  p90 {ms(pct(ttft, 90))}")
    print(f"LLM  time to first sentence  p50 {ms(pct(ttfs, 50))}  p90 {ms(pct(ttfs, 90))}")
    print(f"LLM  3 parallel sessions     first sentence max {ms(max(s[1] for s in par))}")
    return {"ttfs_p50": pct(ttfs, 50)}


async def tool_case(c: httpx.AsyncClient, prompt: str, expected: str | None) -> str | None:
    body = {
        "model": "llm",
        "max_tokens": 200,
        "tools": TOOLS,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = await c.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    calls = r.json()["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return None if expected is None else "no tool call"
    fn = calls[0]["function"]
    if fn["name"] != expected:
        return f"called {fn['name']}"
    try:
        args = json.loads(fn["arguments"] or "{}")
    except json.JSONDecodeError:
        return "arguments not JSON"
    required = next(t for t in TOOLS if t["function"]["name"] == expected)["function"]
    missing = set(required["parameters"]["required"]) - set(args)
    return f"missing {sorted(missing)}" if missing else None


async def bench_tools(c: httpx.AsyncClient) -> dict:
    results = await asyncio.gather(*(tool_case(c, p, e) for p, e in TOOL_CASES))
    fails = [(p, e, err) for (p, e), err in zip(TOOL_CASES, results) if err]
    rate = 1 - len(fails) / len(TOOL_CASES)
    print(f"LLM  tool calls              {rate:.0%} correct ({len(TOOL_CASES)} cases)")
    for p, e, err in fails:
        print(f"       FAIL {p!r}: expected {e}, {err}")
    return {"tool_rate": rate}


async def bench_speech(c: httpx.AsyncClient, stt_model: str, tts_model: str) -> dict:
    first, stt, errs = [], [], []
    for s in SPEECH_SENTENCES:
        body = {"model": tts_model, "voice": "af_heart", "input": s, "response_format": "wav"}
        t0, t_first, audio = time.perf_counter(), None, b""
        async with c.stream("POST", "/v1/audio/speech", json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                t_first = t_first or time.perf_counter() - t0
                audio += chunk
        first.append(t_first)
        t0 = time.perf_counter()
        r = await c.post(
            "/v1/audio/transcriptions",
            files={"file": ("s.wav", audio, "audio/wav")},
            data={"model": stt_model},
        )
        r.raise_for_status()
        stt.append(time.perf_counter() - t0)
        errs.append(wer(s, r.json()["text"]))
    print(f"TTS  first audio chunk       p50 {ms(pct(first, 50))}  max {ms(max(first))}")
    print(f"STT  ~4 s utterance          p50 {ms(pct(stt, 50))}  max {ms(max(stt))}")
    print(f"STT  word error rate         {statistics.mean(errs):.1%} (on TTS audio: optimistic)")
    return {"tts_first_p50": pct(first, 50), "stt_p50": pct(stt, 50)}


def vram():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print(f"GPU  {out}")
    except (OSError, subprocess.CalledProcessError):
        print("GPU  nvidia-smi not available here (run on the GPU box for VRAM)")


async def main(a):
    headers = {"Authorization": f"Bearer {a.api_key}"}
    timeout = httpx.Timeout(120)
    async with (
        httpx.AsyncClient(base_url=a.llm_url, headers=headers, timeout=timeout) as llm,
        httpx.AsyncClient(base_url=a.speaches_url, headers=headers, timeout=timeout) as sp,
    ):
        vram()
        res = await bench_llm(llm) | await bench_tools(llm)
        res |= await bench_speech(sp, a.stt_model, a.tts_model)
    gates = [
        ("LLM first sentence p50 <= 400 ms", res["ttfs_p50"] <= 0.4),
        ("tool calls >= 95%", res["tool_rate"] >= 0.95),
        ("STT p50 <= 250 ms", res["stt_p50"] <= 0.25),
        ("TTS first chunk p50 <= 150 ms", res["tts_first_p50"] <= 0.15),
    ]
    print("\nGate C (cold start: see /var/log/voicistant/onstart.log '[+Ns] ready')")
    for name, ok in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def selftest():
    assert wer("Hello, world!", "hello world") == 0
    assert wer("a b c d", "a x c") == 0.5  # one substitution + one deletion
    assert wer("", "anything") == 1.0
    assert pct([3, 1, 2], 50) == 2 and pct([1, 2, 3, 4], 90) == 4
    assert SENTENCE_END.search("Sure. Here") and not SENTENCE_END.search("3.5 kg")
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--llm-url", default="http://127.0.0.1:8000")
    p.add_argument("--speaches-url", default="http://127.0.0.1:8001")
    p.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", ""))
    p.add_argument("--stt-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    p.add_argument("--tts-model", default="speaches-ai/Kokoro-82M-v1.0-ONNX")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    selftest() if args.selftest else asyncio.run(main(args))
