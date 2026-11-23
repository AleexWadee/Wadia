"""Where the user is, so Wadia can answer "Will it rain today?" or "What's open near me?".

The location comes from one of three places (all free, no API key):
- automatic: estimated from the internet connection (city level, refreshed every few hours)
- manual: a place the user typed in the app
- phone: the phone's GPS, sent from the phone app
"""

import threading
from collections import Counter
from datetime import datetime, timedelta

import requests

import tools

REFRESH_AFTER = timedelta(hours=6)  # how often the automatic location is re-detected
HEADERS = {"User-Agent": "Wadia voice assistant"}
_refreshing = threading.Lock()


def current() -> dict | None:
    """The saved location. An automatic one is refreshed in the background when it's old."""
    location = tools.load_settings().get("location")
    if location is None or (location.get("source") == "auto" and _is_stale(location)):
        threading.Thread(target=refresh_automatic, daemon=True).start()
    return location


def describe(location: dict | None = None) -> str:
    location = location or current()
    if not location:
        return ""
    parts = [location.get("city"), location.get("region"), location.get("country")]
    return ", ".join(dict.fromkeys(p for p in parts if p))  # dict.fromkeys drops repeats like "Madrid, Madrid"


def for_prompt() -> str:
    """A line for Wadia's instructions telling it where the user is."""
    location = current()
    if not location:
        return ""
    how = {"auto": "approximate, estimated from the internet connection", "manual": "set by the user",
           "phone": "from the user's phone"}.get(location.get("source"), "")
    return (f"The user's location: {describe(location)} ({how}). Use it for weather, local time, nearby places "
            "and local news, unless the user mentions another place.")


def refresh_automatic() -> dict | None:
    """Detects the location from the internet connection and saves it (unless the user set one)."""
    if not _refreshing.acquire(blocking=False):
        return None
    try:
        saved = tools.load_settings().get("location")
        if saved and saved.get("source") != "auto":
            return saved
        location = detect_from_ip()
        return _save(location, "auto") if location else None
    finally:
        _refreshing.release()


def use_automatic() -> dict | None:
    """Switches back to the automatic location."""
    tools.save_setting("location", None)
    return refresh_automatic()


def set_from_place(place: str) -> dict | None:
    """Saves a place the user typed, e.g. "Telde" or "Madrid, Spain". Returns None if not found."""
    name, _, country = place.partition(",")
    results = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name.strip(), "count": 10, "language": "en"}, timeout=10,
    ).json().get("results", [])
    if country.strip():  # "Madrid, Spain": prefer results in that country or region
        wanted = country.strip().lower()
        results = [r for r in results if wanted in (r.get("country", "") + " " + r.get("admin1", "")).lower()] or results
    if not results:
        return None
    best = results[0]
    location = {"city": best["name"], "region": best.get("admin1", ""), "country": best.get("country", ""),
                "latitude": best["latitude"], "longitude": best["longitude"], "timezone": best.get("timezone", "")}
    return _save(location, "manual")


def set_from_coordinates(latitude: float, longitude: float) -> dict:
    """Saves the phone's GPS position, with the place name looked up from the coordinates."""
    place = {}
    try:
        place = requests.get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={"latitude": latitude, "longitude": longitude, "localityLanguage": "en"}, timeout=10,
        ).json()
    except Exception:
        pass
    location = {"city": place.get("city") or place.get("locality") or "", "region": place.get("principalSubdivision", ""),
                "country": place.get("countryName", ""), "latitude": round(latitude, 4), "longitude": round(longitude, 4)}
    return _save(location, "phone")


def detect_from_ip() -> dict | None:
    """Asks several free IP-location services and keeps the city most of them agree on
    (they often disagree by a town or two)."""
    answers = []
    for url, read in _SERVICES:
        try:
            data = requests.get(url, timeout=6, headers=HEADERS).json()
            if (answer := read(data)) and answer.get("city"):
                answers.append(answer)
        except Exception:
            continue
    if not answers:
        return None
    city, _ = Counter(a["city"] for a in answers).most_common(1)[0]
    return next(a for a in answers if a["city"] == city)


_SERVICES = [
    ("https://get.geojs.io/v1/ip/geo.json", lambda d: {
        "city": d.get("city"), "region": d.get("region"), "country": d.get("country"),
        "latitude": float(d["latitude"]), "longitude": float(d["longitude"]), "timezone": d.get("timezone")}),
    ("https://ipwho.is/", lambda d: d.get("success", True) and {
        "city": d.get("city"), "region": d.get("region"), "country": d.get("country"),
        "latitude": d["latitude"], "longitude": d["longitude"], "timezone": (d.get("timezone") or {}).get("id")}),
    ("https://ipapi.co/json/", lambda d: {
        "city": d.get("city"), "region": d.get("region"), "country": d.get("country_name"),
        "latitude": d["latitude"], "longitude": d["longitude"], "timezone": d.get("timezone")}),
]


def _save(location: dict, source: str) -> dict:
    location = {**location, "source": source, "updated": datetime.now().isoformat(timespec="seconds")}
    tools.save_setting("location", location)
    return location


def _is_stale(location: dict) -> bool:
    try:
        return datetime.now() - datetime.fromisoformat(location["updated"]) > REFRESH_AFTER
    except (KeyError, ValueError):
        return True
