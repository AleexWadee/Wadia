"""The Wadia app for the laptop. Open it from the Wadia shortcut (Desktop or Start menu).

It opens the Wadia window, lives in the system tray (next to the clock) and keeps listening
for "Wadia" when the window is closed. The tray menu has the microphone switch, "Start with
Windows" and Quit. It also serves the phone app on your Wi-Fi.

    pythonw local/app.py               # what the shortcut runs (no console window)
    python local/app.py                # same, but showing a log in the console
    python local/app.py --background   # start in the tray without opening the window
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent
LOG_FILE = HERE / "wadia.log"
sys.path.insert(0, str(HERE))


def setup_output() -> None:
    """Without a console (pythonw), print() has nowhere to go, so log to a file instead."""
    if sys.stdout is None or sys.stderr is None:
        log = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def full_speed() -> None:
    """Windows puts windowless background programs (like Wadia in the tray) in "efficiency mode",
    which makes the voice up to 15 times slower. Opt out of it, and run a bit above normal priority."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()
    # Control execution speed (1) and timer resolution (4), with both throttles off (state 0).
    setting = PowerThrottling(1, 1 | 4, 0)
    kernel32.SetProcessInformation(process, 4, ctypes.byref(setting), ctypes.sizeof(setting))  # ProcessPowerThrottling
    kernel32.SetPriorityClass(process, 0x8000)  # ABOVE_NORMAL_PRIORITY_CLASS


setup_output()
full_speed()

import server  # noqa: E402  (after the output is set up, since imports can print)
import tools  # noqa: E402
from assistant import Hub  # noqa: E402

APP_URL = f"http://127.0.0.1:{server.DESKTOP_PORT}/"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def already_running() -> bool:
    try:
        return urllib.request.urlopen(APP_URL + "ping", timeout=1).read() == b"wadia"
    except OSError:
        return False


def open_window() -> None:
    """Opens Wadia in its own app window (Edge app mode: no tabs or address bar)."""
    edge = next(
        (p for p in [
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(shutil.which("msedge") or ""),
        ] if p.name == "msedge.exe" and p.exists()),
        None,
    )
    if edge:
        subprocess.Popen([str(edge), f"--app={APP_URL}", "--window-size=440,780"], creationflags=_NO_WINDOW)
    else:
        webbrowser.open(APP_URL)


class Controller:
    """Passes commands from the window and tray to the assistant, once it has loaded."""

    def __init__(self) -> None:
        self.assistant = None

    def command(self, command: dict) -> None:
        if command.get("type") == "mute":
            tools.save_setting("mic_muted", bool(command.get("on")))
        if self.assistant:
            self.assistant.command(command)


def ensure_ollama(hub: Hub) -> None:
    """Starts the Ollama app if it isn't running, and waits for it."""
    import ollama

    for attempt in range(60):
        try:
            ollama.list()
            return
        except Exception:
            if attempt == 0:
                hub.emit({"type": "state", "state": "loading", "detail": "Starting Ollama…"})
                app = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama app.exe"
                if app.exists():
                    subprocess.Popen([str(app)], creationflags=_NO_WINDOW)
            time.sleep(1)
    raise RuntimeError("Ollama isn't running. Install it from ollama.com, or start the Ollama app.")


def boot(models: server.Models, hub: Hub, controller: Controller) -> None:
    """Loads the models (about a minute), then runs the laptop assistant."""
    import brain
    import speech
    from assistant import LaptopAssistant

    try:
        ensure_ollama(hub)
        hub.emit({"type": "state", "state": "loading", "detail": "Waking up the brain…"})
        brain.warm_up()  # load the LLM first so it gets the GPU memory it needs
        hub.emit({"type": "state", "state": "loading", "detail": "Loading hearing…"})
        models.vad = speech.load_vad()
        models.transcriber = speech.Transcriber()
        hub.emit({"type": "state", "state": "loading", "detail": "Loading voice…"})
        models.voice = speech.Voice()
        models.ready.set()
        print(f"Ready. Ears: {models.transcriber.description}", flush=True)
        import location

        location.current()  # detects the location in the background if it isn't known yet
        threading.Thread(target=brain.summarize_pending, daemon=True).start()
    except Exception as e:
        message = str(e)
        if "not found" in message.lower():
            message = "The wadia model is missing. Run: python local\\create_model.py"
        hub.emit({"type": "state", "state": "error", "detail": message})
        return

    controller.assistant = LaptopAssistant(models.vad, models.transcriber, models.voice, hub)
    controller.assistant.run()


PUBLIC_LINK = None  # the shareable link (sharing.PublicLink), created in main()


def on_setting(key: str, value) -> None:
    if key == "autostart":
        import shortcuts

        shortcuts.set_autostart(value)
        tools.save_setting("autostart", value)
    elif key == "sharing":
        tools.save_setting("sharing", value)
        PUBLIC_LINK.start() if value else PUBLIC_LINK.stop()


def run_tray(hub: Hub, controller: Controller) -> None:
    """The icon next to the clock. Blocks until Quit."""
    import pystray
    from PIL import Image

    def muted() -> bool:
        return bool(tools.load_settings().get("mic_muted", False))

    def toggle_mic(icon, item) -> None:
        controller.command({"type": "mute", "on": not muted()})

    def toggle_autostart(icon, item) -> None:
        on_setting("autostart", not tools.load_settings().get("autostart", False))

    def quit_app(icon, item) -> None:
        icon.stop()

    icon = pystray.Icon(
        "Wadia",
        Image.open(HERE / "web" / "icon-512.png").resize((64, 64)),
        "Wadia",
        menu=pystray.Menu(
            pystray.MenuItem("Open Wadia", lambda icon, item: open_window(), default=True),
            pystray.MenuItem("Microphone on", toggle_mic, checked=lambda item: not muted()),
            pystray.MenuItem("Start with Windows", toggle_autostart,
                             checked=lambda item: bool(tools.load_settings().get("autostart", False))),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Wadia", quit_app),
        ),
    )
    labels = {"loading": "starting…", "sleeping": "say “Wadia”", "listening": "listening", "thinking": "thinking",
              "speaking": "speaking", "muted": "microphone off", "error": "needs attention"}

    def on_event(event: dict) -> None:
        if event["type"] == "state":
            icon.title = f"Wadia – {labels.get(event['state'], event['state'])}"

    hub.listeners.append(on_event)
    icon.run()


def main() -> None:
    if already_running():
        open_window()
        return

    import sharing

    global PUBLIC_LINK
    hub, models, controller = Hub(), server.Models(), Controller()

    def link_changed(link) -> None:
        if link.status == "on":
            print(f"Public link: {link.url}/?guest={tools.load_settings().get('guest_key', '')}", flush=True)
        hub.emit({"type": "sharing", "status": link.status, "error": link.error})

    PUBLIC_LINK = sharing.PublicLink(server.PUBLIC_PORT, link_changed)
    phone = os.getenv("PHONE", "on") != "off"
    phone_url = server.start(models, hub, controller, on_setting, phone=phone, link=PUBLIC_LINK)
    if phone_url:
        print(f"Phone app (same Wi-Fi): {phone_url}", flush=True)
    if tools.load_settings().get("sharing"):
        PUBLIC_LINK.start()
    threading.Thread(target=boot, args=(models, hub, controller), daemon=True, name="wadia-boot").start()
    if "--background" not in sys.argv:
        open_window()
    run_tray(hub, controller)
    PUBLIC_LINK.stop()  # close the public link before quitting
    os._exit(0)  # Quit: stop every background thread at once


if __name__ == "__main__":
    main()
