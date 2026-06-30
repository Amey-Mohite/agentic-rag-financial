"""
memory.py — conversational memory so the agent handles follow-up questions.

THE PROBLEM IT SOLVES
---------------------
Without memory every question is treated in isolation. Ask "What was Apple's FY2025 revenue?"
then "and the year before?" and the second question is meaningless on its own — there's no
"Apple" and no "revenue" in it. Conversational memory keeps a short rolling history of prior
(user, assistant) turns keyed by a SESSION id and replays it to the model, so the follow-up
resolves against earlier context.

WHAT WE STORE (and what we DON'T)
---------------------------------
We persist only the clean conversational turns — the user's question and the agent's final
answer text. We deliberately do NOT persist the intermediate tool-call/tool-result blocks from
the agent loop: they're large, transient, and would bloat the context. This "clean transcript"
pattern is what production assistants do.

PLUGGABLE BACKEND
-----------------
`SessionStore` is a Protocol (structural interface). `InMemorySessionStore` keeps everything in a
process-local dict — perfect for a single API worker or tests. For real multi-worker deployments
you'd implement the same interface over Redis/Postgres (sketched in the docstring) and swap it in
via `make_session_store`. Nothing else in the codebase changes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

from .config import MemoryConfig


# A single conversational turn as the Anthropic Messages API expects it:
#   {"role": "user"|"assistant", "content": "<text>"}
Turn = dict


class SessionStore(Protocol):
    """Structural interface for anything that can persist per-session conversation turns."""

    def history(self, session_id: str) -> list[Turn]:
        """Return the stored turns for a session, oldest-first (empty list if none)."""
        ...

    def append(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Append one completed (user, assistant) exchange to a session's history."""
        ...

    def reset(self, session_id: str) -> None:
        """Forget a session's history (e.g. when the user starts a new conversation)."""
        ...


class InMemorySessionStore:
    """Process-local SessionStore backed by a dict of bounded deques.

    Each session maps to a `deque(maxlen=2*max_turns)` so the history self-trims to the most
    recent `max_turns` exchanges (each exchange = 2 messages: user + assistant). Bounding the
    history is essential: unbounded context grows token cost and latency without bound.

    NOTE: in-process state lives only as long as the worker. For horizontally-scaled deployments,
    implement this same interface over Redis:
        def history(self, sid):  return json.loads(redis.get(sid) or "[]")
        def append(self, sid, u, a):  push two messages, LTRIM to 2*max_turns
    and select it in `make_session_store`.
    """

    def __init__(self, cfg: MemoryConfig):
        self._cfg = cfg
        # maxlen bounds each session to the last `max_turns` exchanges (2 messages per exchange).
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=2 * cfg.max_turns))

    def history(self, session_id: str) -> list[Turn]:
        """Return a plain list copy of the session's turns (oldest-first)."""
        return list(self._sessions[session_id])

    def append(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Record one full exchange. The deque's maxlen auto-evicts the oldest turns."""
        dq = self._sessions[session_id]
        dq.append({"role": "user", "content": user_text})
        dq.append({"role": "assistant", "content": assistant_text})

    def reset(self, session_id: str) -> None:
        """Drop all stored turns for the session."""
        self._sessions.pop(session_id, None)


def make_session_store(cfg: MemoryConfig) -> SessionStore:
    """Factory: build the configured SessionStore implementation.

    Parameters
    ----------
    cfg : MemoryConfig
        Provides the backend name and history bound.

    Returns
    -------
    SessionStore
        Currently always an InMemorySessionStore; extend here for Redis/Postgres backends.
    """
    if cfg.backend == "memory":
        return InMemorySessionStore(cfg)
    raise ValueError(f"unknown memory backend: {cfg.backend}")
