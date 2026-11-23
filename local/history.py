"""Wadia's memory of past conversations (from the laptop and the phone).

Every conversation is saved as a JSON file in local/history. When a conversation ends, the
brain adds a one or two sentence summary; the latest summaries go into Wadia's instructions,
and the full transcripts can be searched ("What did we talk about yesterday?").
"""

import json
import re
import secrets
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "history"
_lock = threading.Lock()
STOPWORDS = set("""a an the and or but of to in on at for with about from by is are was were be been it its
this that these those i you he she we they me my your what which who when where how did do does
el la los las un una y o de del en con por para que qué es son fue era lo le se mi tu""".split())


class ConversationLog:
    """One conversation, saved to disk after every turn."""

    def __init__(self, device: str) -> None:
        now = datetime.now()
        self.id = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        self.data = {"id": self.id, "device": device, "started": now.isoformat(timespec="seconds"),
                     "ended": None, "summary": "", "messages": []}

    @property
    def messages(self) -> list[dict]:
        return self.data["messages"]

    def add(self, role: str, text: str) -> None:
        if text.strip():
            self.messages.append({"role": role, "text": text.strip(), "time": datetime.now().isoformat(timespec="seconds")})

    def save(self) -> None:
        if not self.messages:
            return
        self.data["ended"] = datetime.now().isoformat(timespec="seconds")
        _write(self.data)


def _path(conversation_id: str) -> Path:
    return HISTORY_DIR / f"{re.sub(r'[^0-9a-f-]', '', conversation_id)}.json"


def _write(data: dict) -> None:
    with _lock:
        HISTORY_DIR.mkdir(exist_ok=True)
        _path(data["id"]).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load(conversation_id: str) -> dict | None:
    try:
        return json.loads(_path(conversation_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def all_conversations() -> list[dict]:
    """Every saved conversation, newest first."""
    conversations = []
    for file in HISTORY_DIR.glob("*.json") if HISTORY_DIR.exists() else []:
        try:
            conversations.append(json.loads(file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(conversations, key=lambda c: c["started"], reverse=True)


def set_summary(conversation_id: str, summary: str) -> None:
    if data := load(conversation_id):
        data["summary"] = summary.strip()
        _write(data)


def delete(conversation_id: str) -> None:
    with _lock:
        _path(conversation_id).unlink(missing_ok=True)


def delete_all() -> None:
    with _lock:
        for file in HISTORY_DIR.glob("*.json") if HISTORY_DIR.exists() else []:
            file.unlink(missing_ok=True)


def when(timestamp: str) -> str:
    """'today at 10:42', 'yesterday at 18:05', 'Monday at 09:10', or '3 March at 12:00'."""
    moment = datetime.fromisoformat(timestamp)
    days = (date.today() - moment.date()).days
    day = "today" if days == 0 else "yesterday" if days == 1 else moment.strftime("%A") if days < 7 else \
        f"{moment.day} {moment:%B}"
    return f"{day} at {moment:%H:%M}"


def recent_summaries(limit: int = 8, exclude: str | None = None) -> list[str]:
    """Lines like "yesterday at 18:05 (phone): The user asked for a pasta recipe..." for the prompt."""
    lines = []
    for conversation in all_conversations():
        if conversation["id"] == exclude or not conversation.get("summary"):
            continue
        lines.append(f"{when(conversation['started'])} ({conversation['device']}): {conversation['summary']}")
        if len(lines) == limit:
            break
    return lines


def search(query: str = "", days_ago: int = -1) -> str:
    """Finds past conversations by words they contain and/or by day, for the recall tool."""
    conversations = all_conversations()
    if days_ago >= 0:
        day = date.today() - timedelta(days=days_ago)
        conversations = [c for c in conversations if datetime.fromisoformat(c["started"]).date() == day]
    words = {w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) > 1}

    scored = []
    for conversation in conversations:
        if not words:
            scored.append((1, conversation, conversation["messages"][:4]))
            continue
        hits = [m for m in conversation["messages"] if words & set(re.findall(r"\w+", m["text"].lower()))]
        summary_hits = len(words & set(re.findall(r"\w+", conversation.get("summary", "").lower())))
        if hits or summary_hits:
            scored.append((len(hits) + 2 * summary_hits, conversation, hits[:4]))
    if not scored:
        return "No past conversation matches that."

    results = []
    for _, conversation, messages in sorted(scored, key=lambda s: s[0], reverse=True)[:3]:
        lines = [f"Conversation {when(conversation['started'])} ({conversation['device']})"]
        if conversation.get("summary"):
            lines.append(f"Summary: {conversation['summary']}")
        lines += [f"{'User' if m['role'] == 'user' else 'You'}: {m['text'][:300]}" for m in messages]
        results.append("\n".join(lines))
    return "\n\n".join(results)


def unsummarized(idle_minutes: int) -> list[dict]:
    """Conversations that ended a while ago but have no summary yet (e.g. the app was closed)."""
    cutoff = datetime.now() - timedelta(minutes=idle_minutes)
    return [c for c in all_conversations()
            if not c.get("summary") and c.get("ended") and datetime.fromisoformat(c["ended"]) < cutoff]
