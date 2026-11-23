"""Windows shortcuts for the Wadia app: Desktop, Start menu, and "Start with Windows".

    python local/shortcuts.py      # creates the Desktop and Start menu shortcuts
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ICON = HERE / "web" / "wadia.ico"
APPDATA = Path(os.environ.get("APPDATA", ""))
START_MENU = APPDATA / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Wadia.lnk"
STARTUP = APPDATA / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Wadia.lnk"


def desktop_folder() -> Path:
    """The real Desktop folder (it may be redirected, e.g. to OneDrive)."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    return Path(result.stdout.strip() or Path.home() / "Desktop")


def pythonw() -> Path:
    """The windowless Python of this project's virtual environment."""
    return Path(sys.prefix) / "Scripts" / "pythonw.exe"


def create(link: Path, background: bool = False) -> None:
    ensure_icon()
    link.parent.mkdir(parents=True, exist_ok=True)
    arguments = f'"{HERE / "app.py"}"' + (" --background" if background else "")
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:WADIA_LINK);"
        "$s.TargetPath = $env:WADIA_TARGET; $s.Arguments = $env:WADIA_ARGS;"
        "$s.WorkingDirectory = $env:WADIA_DIR; $s.IconLocation = $env:WADIA_ICON;"
        "$s.Description = 'Wadia voice assistant'; $s.Save()"
    )
    env = {**os.environ, "WADIA_LINK": str(link), "WADIA_TARGET": str(pythonw()), "WADIA_ARGS": arguments,
           "WADIA_DIR": str(HERE.parent), "WADIA_ICON": str(ICON)}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env, check=True, creationflags=_NO_WINDOW)


def set_autostart(on: bool) -> None:
    if on:
        create(STARTUP, background=True)
    else:
        STARTUP.unlink(missing_ok=True)


def ensure_icon() -> None:
    """Makes the .ico Windows uses for shortcuts, from the app's PNG icon."""
    if ICON.exists():
        return
    from PIL import Image

    Image.open(HERE / "web" / "icon-512.png").save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


if __name__ == "__main__":
    desktop = desktop_folder() / "Wadia.lnk"
    create(desktop)
    create(START_MENU)
    print(f"Created shortcuts:\n  {desktop}\n  {START_MENU}")
