"""Tools the assistant can call on its own. All are free and need no API key.

Each function's docstring is what the LLM reads to decide when to use it, so keep
them clear. Every tool returns a plain string.
"""

import ast
import json
import math
import operator
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from ddgs import DDGS

MEMORY_FILE = Path(__file__).parent / "memory.json"
SETTINGS_FILE = Path(__file__).parent / "settings.json"


def get_current_datetime() -> str:
    """Get the current local date, time and day of the week.

    Returns:
        The current date and time.
    """
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


def web_search(query: str) -> str:
    """Search the internet live for current or factual information: news, recent events,
    sports results, prices, opening hours, people, places, or anything you are not sure about.
    It also reads the top web pages, so the answer is up to date.

    Args:
        query: What to search for, written like a search engine query.

    Returns:
        The top search results, with the relevant parts of the best pages.
    """
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=4)
    news = pool.submit(_search, "news", query, 4)
    pages = pool.submit(_search, "text", query, 6)
    news = sorted(news.result(), key=lambda r: r.get("date", ""), reverse=True)  # newest first
    pages = pages.result()
    # Read the newest article and the best pages at the same time, but never keep the user waiting for slow sites.
    urls = list(dict.fromkeys([r.get("url") for r in news[:1]] + [r.get("href") for r in pages]))
    readers = [pool.submit(_read_page, url, query) for url in [u for u in urls if u][:READ_PAGES]]
    done, _ = wait(readers, timeout=max(1.5, SEARCH_BUDGET - (time.monotonic() - started)))
    read = [f.result() for f in readers if f in done and f.result()]
    pool.shutdown(wait=False)  # pages still loading finish in the background and are ignored
    lines = [f"Today is {datetime.now():%B %d, %Y}. For questions about the latest or current anything, trust the newest dated results."]
    lines += [f"[{r.get('date', '')[:10]}] {r['title']} ({r.get('source', '')}): {r['body']}" for r in news]
    lines += [f"{r['title']} ({_site(r.get('href', ''))}): {r['body']}" for r in pages]
    lines += [f"From the page {_site(url)}:\n{text}" for url, text in read]
    return "\n\n".join(lines) if lines else "No results found."


def read_webpage(url: str, question: str = "") -> str:
    """Open a web page and read it, for when the user names a website or asks about a
    specific page (for example "what's on the BBC News front page").

    Args:
        url: The page address, for example "https://www.bbc.com/news" or "elpais.com".
        question: What the user wants to know from the page, to find the relevant parts.

    Returns:
        The relevant text of the page.
    """
    if "://" not in url:
        url = "https://" + url.strip()
    result = _read_page(url, question, limit=4000)
    return f"From {_site(url)}:\n{result[1]}" if result else f"Couldn't open {url}."


READ_PAGES = 2  # how many of the top results web_search reads in full
SEARCH_BUDGET = 5.0  # seconds web_search takes at most, page reading included


def _search(kind: str, query: str, count: int) -> list[dict]:
    try:
        with DDGS(timeout=5) as search:  # "auto" combines several search engines (Google, Bing, Brave, ...)
            return getattr(search, kind)(query, max_results=count, backend="auto")
    except Exception:
        return []


def _site(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _read_page(url: str, query: str, limit: int = 1500) -> tuple[str, str] | None:
    """The parts of a page that best match the query (menus and other clutter left out)."""
    try:
        with DDGS(timeout=4) as reader:
            text = reader.extract(url, fmt="text_plain")["content"]
    except Exception:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    # Glue the page's short lines into passages of a few hundred characters.
    passages, current = [], ""
    for line in (l.strip() for l in text.splitlines()):
        if line:
            current = f"{current} | {line}" if current else line
        if len(current) >= 350 or (not line and len(current) >= 120):
            passages.append(current)
            current = ""
    if current:
        passages.append(current)
    words = {w for w in re.findall(r"\w{3,}", query.lower())}
    this_year = str(datetime.now().year)

    def score(i: int) -> float:
        passage = passages[i].lower()
        hits = sum(passage.count(w) for w in words)
        # Numbers often matter, recent passages beat old ones, and earlier in the page is better.
        return hits + 0.5 * bool(re.search(r"\d", passage)) + (this_year in passage) - 0.002 * i
    best = sorted(range(len(passages)), key=score, reverse=True) if words else range(len(passages))
    chosen, size = [], 0
    for i in best:
        if size + len(passages[i]) > limit:
            continue
        chosen.append(i)
        size += len(passages[i])
    excerpt = "\n".join(passages[i] for i in sorted(chosen))
    return (url, excerpt) if excerpt else None


def get_weather(city: str = "") -> str:
    """Get the current weather and the forecast for the next 3 days. Leave city empty for
    the weather where the user is.

    Args:
        city: The city name, for example "Madrid" or "New York". Empty means the user's location.

    Returns:
        Current conditions and a short daily forecast.
    """
    import location

    here = location.current()
    if not city.strip() and here:
        place = {"name": here.get("city") or location.describe(here), "country": here.get("country", ""),
                 "latitude": here["latitude"], "longitude": here["longitude"]}
    elif not city.strip():
        return "The user's location is unknown. Ask them which city they mean."
    else:
        places = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=10
        ).json()
        if not places.get("results"):
            return f"Couldn't find a city called {city}."
        place = places["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=10,
    ).json()

    now = weather["current"]
    lines = [
        f"{place['name']}, {place.get('country', '')}. Now: {WEATHER_CODES.get(now['weather_code'], 'unknown')}, "
        f"{now['temperature_2m']}°C (feels like {now['apparent_temperature']}°C), wind {now['wind_speed_10m']} km/h."
    ]
    daily = weather["daily"]
    for i, day in enumerate(daily["time"]):
        lines.append(
            f"{day}: {WEATHER_CODES.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]} to {daily['temperature_2m_max'][i]}°C, "
            f"{daily['precipitation_probability_max'][i]}% chance of rain."
        )
    return "\n".join(lines)


def calculate(expression: str) -> str:
    """Evaluate a math expression exactly. Use this for any arithmetic instead of
    calculating in your head. Supports + - * / ** % parentheses and functions like
    sqrt, sin, cos, tan, log, log10, exp, plus the constants pi and e.

    Args:
        expression: The math expression, for example "(17.5 * 3) / 4" or "sqrt(2) * pi".

    Returns:
        The result.
    """
    try:
        result = _eval_math(ast.parse(expression, mode="eval").body)
    except Exception as e:
        return f"Could not calculate {expression!r}: {e}"
    return str(round(result, 10)) if isinstance(result, float) else str(result)


def recall_conversations(query: str = "", days_ago: int = -1) -> str:
    """Search your past conversations with the user, from the laptop and the phone. Use it
    when the user refers to something you talked about before ("what was that movie you
    recommended?", "what did we talk about yesterday?").

    Args:
        query: Words to look for, e.g. "movie recommendation". Empty to list conversations from a day.
        days_ago: Only conversations from that day: 0 = today, 1 = yesterday. -1 = any day.

    Returns:
        The best matching past conversations, with their summaries and messages.
    """
    import history

    return history.search(query, int(days_ago))


def change_voice(gender: str) -> str:
    """Switch your own speaking voice between a male and a female voice. Use it only
    when the user asks you to change your voice.

    Args:
        gender: "male" or "female".

    Returns:
        Confirmation of the change.
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female"):
        return 'The voice must be "male" or "female".'
    set_voice_gender(gender)
    return f"Done: from now on you speak with a {gender} voice. Confirm it briefly to the user."


def set_voice_gender(gender: str) -> None:
    """Switches to the default male or female voice in every language (replacing any
    specific voice picked in the app)."""
    settings = load_settings()
    settings["voice"] = gender
    for key in [k for k in settings if k.startswith("voice_")]:
        del settings[key]
    _write_settings(settings)


def get_voice_gender() -> str:
    return load_settings().get("voice", "female")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_setting(key: str, value) -> None:
    settings = load_settings()
    if value is None:
        settings.pop(key, None)
    else:
        settings[key] = value
    _write_settings(settings)


def _write_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# Long-term memory. These aren't offered to the LLM as tools: local_assistant.update_memory
# decides what to add or remove after each turn, which is far more reliable.


def add_memories(facts: list[str]) -> None:
    memories = load_memories()
    memories += [f for f in facts if f not in memories]
    _save_memories(memories)


def remove_memories(facts: list[str]) -> None:
    _save_memories([f for f in load_memories() if f not in facts])


def load_memories() -> list[str]:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_memories(memories: list[str]) -> None:
    MEMORY_FILE.write_text(json.dumps(memories, indent=2, ensure_ascii=False), encoding="utf-8")


TOOLS = [get_current_datetime, web_search, read_webpage, get_weather, calculate, recall_conversations, change_voice]
TOOLS_BY_NAME = {tool.__name__: tool for tool in TOOLS}


def run_tool(name: str, arguments: dict) -> str:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"Unknown tool {name}."
    try:
        return tool(**arguments)
    except Exception as e:
        return f"The tool {name} failed: {e}"


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp", "floor", "ceil", "factorial", "radians", "degrees")
} | {"abs": abs, "round": round}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval_math(node):
    """Safely evaluates a parsed math expression (no arbitrary code execution)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _eval_math(node.left), _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_math(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        return _FUNCTIONS[node.func.id](*[_eval_math(arg) for arg in node.args])
    raise ValueError("unsupported expression")


WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}
