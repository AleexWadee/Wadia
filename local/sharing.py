"""A public link to Wadia, so people can try it from anywhere (4G, any Wi-Fi, any country).

Two free ways, tried in this order:

1. Tailscale Funnel (if Tailscale is installed, signed in, and Funnel is enabled for the account):
   a permanent address like https://my-laptop.tail1234.ts.net that never changes.
2. Cloudflare quick tunnel (cloudflared, no account needed): a random
   https://….trycloudflare.com address that changes every time the link is started.

Either way, visitors need nothing installed, and the laptop must stay on.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
CLOUDFLARED = HERE / "bin" / "cloudflared.exe"
PID_FILE = HERE / "bin" / "cloudflared.pid"
DOWNLOAD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class PublicLink:
    """Publishes Wadia's local public port and reports the address it gets."""

    def __init__(self, local_port: int, on_change) -> None:
        self.local_port = local_port
        self.on_change = on_change  # called with this object whenever the status or address changes
        self.url = ""
        self.provider = ""  # "tailscale" or "cloudflare"
        self.status = "off"  # off / starting / on / error
        self.error = ""
        self.note = ""  # e.g. why Tailscale wasn't used
        self._process: subprocess.Popen | None = None
        self._wanted = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._wanted:
                return
            self._wanted = True
        threading.Thread(target=self._run, daemon=True, name="public-link").start()

    def stop(self) -> None:
        with self._lock:
            self._wanted = False
            process, self._process = self._process, None
        if process and process.poll() is None:
            process.terminate()
        PID_FILE.unlink(missing_ok=True)
        if self.provider == "tailscale":
            tailscale_funnel_off()
        self._set("off", "")

    def _run(self) -> None:
        self._set("starting", "")
        url, self.note = tailscale_funnel_on(self.local_port)
        if url:
            self.provider = "tailscale"
            self._set("on", url)
            return
        self.provider = "cloudflare"
        self._run_cloudflare()

    def _run_cloudflare(self) -> None:
        """Keeps a Cloudflare quick tunnel up while it's wanted, restarting it if it drops."""
        _stop_leftover()
        while self._wanted:
            if not CLOUDFLARED.exists():
                try:
                    _download()
                except Exception as e:
                    self._set("error", "", f"Couldn't download cloudflared: {e}")
                    return
            self._set("starting", "")
            process = subprocess.Popen(
                [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{self.local_port}", "--no-autoupdate"],
                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW,
            )
            with self._lock:
                self._process = process
            PID_FILE.write_text(str(process.pid))
            for line in process.stderr:  # cloudflared prints the address it got in its log
                if match := re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line):
                    self._set("on", match.group(0))
            if self._wanted:  # it stopped by itself (e.g. lost internet): try again shortly
                self._set("starting", "")
                time.sleep(5)

    def _set(self, status: str, url: str, error: str = "") -> None:
        changed = (status, url) != (self.status, self.url)
        self.status, self.url, self.error = status, url, error
        if changed or error:
            self.on_change(self)


# --- Tailscale Funnel ---

def _tailscale() -> str | None:
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    return exe if Path(exe).exists() else None


def _run_tailscale(*args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run([_tailscale(), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=_NO_WINDOW)


def tailscale_funnel_on(local_port: int) -> tuple[str, str]:
    """Publishes the port with Tailscale Funnel. Returns (address, "") or ("", reason it wasn't used)."""
    if not _tailscale():
        return "", ""
    try:
        status = json.loads(_run_tailscale("status", "--json", timeout=10).stdout)
        if status.get("BackendState") != "Running":
            return "", "Tailscale is installed but not signed in, so a temporary link is used instead."
        name = status["Self"]["DNSName"].rstrip(".")
        # When Funnel isn't enabled yet, this prints a link to enable it and waits, so give up quickly.
        result = _run_tailscale("funnel", "--bg", str(local_port), timeout=15)
        output, failed = result.stdout + result.stderr, result.returncode != 0
    except subprocess.TimeoutExpired as e:
        output, failed = f"{e.stdout or ''}{e.stderr or ''}", True
    except Exception:
        return "", ""
    if failed or "not enabled" in output.lower():
        enable = re.search(r"https://login\.tailscale\.com/\S+", output)
        return "", ("Using a temporary link: Tailscale Funnel isn't enabled for your account yet"
                    + (f". Enable it here, then turn the public link off and on: {enable.group(0)}" if enable else "."))
    return f"https://{name}", ""


def tailscale_funnel_off() -> None:
    if _tailscale():
        try:
            _run_tailscale("funnel", "--https=443", "off", timeout=15)
        except Exception:
            pass


# --- Cloudflare quick tunnel ---

def _download() -> None:
    import requests

    CLOUDFLARED.parent.mkdir(exist_ok=True)
    partial = CLOUDFLARED.with_suffix(".part")
    with requests.get(DOWNLOAD_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(partial, "wb") as file:
            for chunk in response.iter_content(1 << 20):
                file.write(chunk)
    partial.replace(CLOUDFLARED)


def _stop_leftover() -> None:
    """Stops a tunnel left running by a previous Wadia that didn't close cleanly."""
    try:
        pid = int(PID_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return
    # Only if that process is still cloudflared (Windows reuses process numbers).
    subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe", "/FI", f"PID eq {pid}"],
                   capture_output=True, creationflags=_NO_WINDOW)
    PID_FILE.unlink(missing_ok=True)
