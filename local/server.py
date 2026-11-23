"""The web backend for both Wadia apps.

- Laptop app: http://127.0.0.1:8765, only reachable from the laptop itself. It shows what the
  laptop assistant is doing and controls it (no microphone in the page; the laptop listens itself).
- Public link (optional): http://127.0.0.1:8766, published as https://….trycloudflare.com by sharing.py,
  so the owner can use Wadia from anywhere and testers can try it in guest mode.
- Phone app: https://<laptop>:8443 on your Wi-Fi, protected by a private key in the link. The phone
  streams its microphone here; the laptop detects speech, transcribes it, thinks, and streams the
  spoken reply back. The phone's echo cancellation lets you interrupt Wadia just by talking.

Phones only allow the microphone on HTTPS pages. On the Wi-Fi, a self-signed certificate is created
in local/certs and the phone warns about it once. The public link has a proper certificate.
"""

import asyncio
import datetime
import io
import ipaddress
import json
import re
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

import brain
import history
import location
import speech
import tools

HERE = Path(__file__).parent
WEB_DIR = HERE / "web"
CERT_DIR = HERE / "certs"
DESKTOP_PORT = int(os.getenv("DESKTOP_PORT", "8765"))
PHONE_PORT = int(os.getenv("PHONE_PORT", "8443"))
PUBLIC_PORT = int(os.getenv("PUBLIC_PORT", "8766"))  # what the public link forwards to
LOCAL_ADDRESSES = {"127.0.0.1", "::1"}


class Models:
    """The speech models, filled in once they've loaded (the server starts before that)."""

    def __init__(self) -> None:
        self.vad = self.transcriber = self.voice = None
        self.ready = threading.Event()


class PhoneSession:
    """One connected phone: its own conversation, speech detector and awake state.

    A guest (a tester using the shared link) gets a conversation that isn't saved or remembered,
    and voice choices that only apply to their own session."""

    def __init__(self, ws: web.WebSocketResponse, loop, models: Models, guest: bool = False) -> None:
        self.ws, self.loop, self.models, self.guest = ws, loop, models, guest
        self.detector = speech.SpeechDetector(models.vad)
        self.conversation = brain.Conversation(
            device="phone", guest=guest, on_tool=lambda name, args: self.send({"type": "tool", "name": name})
        )
        self.voice_prefs: dict = {}  # a guest's own voice choices
        self.pending = np.zeros(0, np.float32)
        self.awake = False  # tapped the orb, or said "Wadia" in hands-free mode
        self.hands_free = False  # listening for "Wadia" all the time
        self.stop = threading.Event()  # set to cut off the current reply
        self.turn_lock = threading.Lock()  # one reply at a time
        self.busy = False

    # --- messages to the phone (safe to call from worker threads) ---

    def send(self, message: dict) -> None:
        self._run(self.ws.send_str(json.dumps(message)))

    def send_audio(self, samples: np.ndarray) -> None:
        self._run(self.ws.send_bytes(to_pcm16(samples)))

    def _run(self, coroutine) -> None:
        try:
            in_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:
            in_loop = False
        if in_loop:
            asyncio.ensure_future(coroutine)
            return
        try:
            asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=10)
        except Exception:
            self.stop.set()  # the phone disconnected

    # --- messages from the phone ---

    def on_audio(self, data: bytes) -> None:
        self.pending = np.concatenate([self.pending, np.frombuffer(data, dtype="<f4")])
        while len(self.pending) >= speech.BLOCK:
            block, self.pending = self.pending[: speech.BLOCK], self.pending[speech.BLOCK :]
            event, audio = self.detector.feed(block)
            if event == "started" and (self.awake or self.busy):
                self.stop.set()  # the user talked over Wadia: stop replying
                self.send({"type": "heard"})
            elif event == "ended":
                self.loop.run_in_executor(None, self.run_turn, audio)

    def on_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "wake":  # orb tapped
            self.awake = True
        elif kind == "sleep":  # orb tapped again, or no follow-up in time
            self.awake = False
            self.stop.set()
        elif kind == "hands_free":
            self.hands_free = bool(message.get("on"))
        elif kind == "voice_prefs" and self.guest:
            self.voice_prefs = {
                "voice_en": str(message.get("voice_en", ""))[:80], "voice_es": str(message.get("voice_es", ""))[:80],
                "speed": min(1.5, max(0.7, float(message.get("speed") or 1.0))),
            }
        elif kind == "text" and message.get("text", "").strip():  # typed message
            self.awake = True
            self.loop.run_in_executor(None, self.run_turn, None, message["text"].strip()[:1000])

    def synthesize(self, text: str, lang: str) -> np.ndarray:
        """The owner's voice settings, or a guest's own choices (never changing the owner's)."""
        if not self.guest:
            return self.models.voice.synthesize(text, lang)
        defaults = speech.DEFAULT_VOICES.get(lang, speech.DEFAULT_VOICES[speech.DEFAULT_LANGUAGE])
        voice = (defaults[self.conversation.voice_gender] if self.conversation.voice_gender
                 else self.voice_prefs.get(f"voice_{lang}") or defaults["female"])
        return self.models.voice.synthesize(text, lang, voice=voice, speed=self.voice_prefs.get("speed", 1.0))

    # --- one turn of conversation (runs in a worker thread) ---

    def run_turn(self, audio: np.ndarray | None, typed: str | None = None) -> None:
        if audio is not None and not (self.awake or self.hands_free):
            return
        with self.turn_lock:
            self.busy, self.stop = True, threading.Event()
            try:
                self._turn(audio, typed)
            finally:
                self.busy = False
                self.send({"type": "done"})

    def _turn(self, audio, typed) -> None:
        conversation = self.conversation
        if typed:
            text, lang = typed, brain.typed_language(typed, conversation.lang)
        else:
            if self.awake:
                self.send({"type": "state", "state": "thinking"})
            text, lang = self.models.transcriber.transcribe(audio, conversation.lang)
            if not text:
                return
            if not self.awake:  # hands-free and asleep: only react to "Wadia"
                found, text = brain.find_wake_word(text)
                if not found:
                    return
                self.awake = True
                self.send({"type": "wake"})
                if not text:
                    return  # just "Wadia": the phone now listens for the question
                self.send({"type": "state", "state": "thinking"})
        conversation.lang = lang
        self.send({"type": "user", "text": text})

        if brain.is_sleep_phrase(text):
            bye = brain.phrase(lang, "bye")
            self.send({"type": "assistant", "text": bye})
            self.send_audio(self.synthesize(bye, lang))
            self.awake = False
            self.send({"type": "sleep"})
            return

        replies = conversation.reply(text)
        interrupted = False
        try:
            for sentence in replies:
                if self.stop.is_set():
                    interrupted = True
                    break
                samples = self.synthesize(sentence, lang)
                if self.stop.is_set():
                    interrupted = True
                    break
                self.send({"type": "assistant", "text": sentence})
                self.send_audio(samples)
        finally:
            replies.close()
            conversation.finish(interrupted=interrupted)


def as_fact(text: str) -> str:
    """Turns "I'm vegetarian" typed in the app into "The user is vegetarian.", like the facts
    Wadia learns by itself."""
    fact = text.strip().rstrip(".")
    for mine, theirs in [(r"^I'?m\b", "The user is"), (r"^I am\b", "The user is"), (r"^I have\b", "The user has"),
                         (r"^I was\b", "The user was"), (r"^My\b", "The user's"), (r"^I\b", "The user")]:
        if re.match(mine, fact, re.I):
            fact = re.sub(mine, theirs, fact, count=1, flags=re.I)
            break
    fact = re.sub(r"\bmy\b", "their", fact, flags=re.I) if fact.startswith("The user") else fact
    return fact[:1].upper() + fact[1:] + "."


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def to_wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(speech.VOICE_RATE)
        wav.writeframes(to_pcm16(samples))
    return buffer.getvalue()


def settings_view() -> dict:
    """The settings the apps can show and change."""
    settings = tools.load_settings()
    return {
        "voice_en": speech.voice_for("en"),
        "voice_es": speech.voice_for("es"),
        "speed": speech.voice_speed(),
        "gender": tools.get_voice_gender(),
        "microphone": settings.get("microphone", ""),
        "mic_muted": bool(settings.get("mic_muted", False)),
        "autostart": bool(settings.get("autostart", False)),
    }


def guest_view() -> dict:
    """What a guest's voice picker starts with (the default voices, not the owner's)."""
    return {"voice_en": speech.DEFAULT_VOICES["en"]["female"], "voice_es": speech.DEFAULT_VOICES["es"]["female"], "speed": 1.0}


def make_app(models: Models, hub, assistant, key: str, phone_url: str, on_setting, link=None) -> web.Application:
    def is_local(request: web.Request) -> bool:
        """The laptop app window: only reachable from this laptop, on its own port."""
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        return request.remote in LOCAL_ADDRESSES and bool(sockname) and sockname[1] == DESKTOP_PORT

    def role(request: web.Request) -> str | None:
        if is_local(request) or secrets.compare_digest(request.query.get("key", ""), key):
            return "owner"
        # Guests: the shareable link's code, or the Try-out Android app's own code (which "New link"
        # doesn't change, so installed apps keep working). Both only work while sharing is on.
        given = request.query.get("guest", "")
        codes = [tools.load_settings().get(name, "") for name in ("guest_key", "app_guest_key")]
        if given and link and link.status == "on" and any(c and secrets.compare_digest(given, c) for c in codes):
            return "guest"
        return None

    def require(request: web.Request, local_only: bool = False, guests: bool = False) -> str:
        who = role(request)
        allowed = is_local(request) if local_only else who == "owner" or (guests and who == "guest")
        if not allowed:
            raise web.HTTPForbidden(text="Wadia: this link isn't valid any more. Ask for a new one.")
        return who

    def sharing_view() -> dict:
        public = link.url if link and link.status == "on" else ""
        return {
            "sharing": bool(tools.load_settings().get("sharing", False)),
            "sharing_status": link.status if link else "off", "sharing_error": link.error if link else "",
            "sharing_provider": link.provider if link else "", "sharing_note": link.note if link else "",
            "owner_url": f"{public}/?key={key}" if public else phone_url,
            "guest_url": f"{public}/?guest={tools.load_settings().get('guest_key', '')}" if public else "",
        }

    async def index(request: web.Request) -> web.StreamResponse:
        require(request, guests=True)
        return web.FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    async def manifest(request: web.Request) -> web.Response:
        if is_local(request):
            start = "/"
        elif request.query.get("guest"):
            start = f"/?guest={request.query.get('guest', '')}"
        else:
            start = f"/?key={request.query.get('key', '')}"
        return web.json_response({
            "name": "Wadia", "short_name": "Wadia", "id": "/", "start_url": start, "scope": "/",
            "display": "standalone", "orientation": "portrait",
            "background_color": "#0b0b14", "theme_color": "#0b0b14",
            "icons": [
                {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
                {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
                {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
            ],
        })

    async def service_worker(request: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "sw.js", headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-store"})

    async def ping(request: web.Request) -> web.Response:
        return web.Response(text="wadia")

    async def voices(request: web.Request) -> web.Response:
        who = require(request, guests=True)
        return web.json_response({"voices": speech.voice_catalog(), "settings": settings_view() if who == "owner" else guest_view()})

    async def preview(request: web.Request) -> web.Response:
        require(request, guests=True)
        if not models.ready.is_set():
            raise web.HTTPServiceUnavailable(text="Still starting")
        voice_id = request.query.get("voice", "af_heart")
        lang = speech.ACCENTS.get(voice_id[:1], speech.ACCENTS["a"])[0]
        text = {"en": "Hi, I'm Wadia. This is how I sound.", "es": "Hola, soy Wadia. Así sueno yo.",
                "fr": "Bonjour, je suis Wadia.", "it": "Ciao, sono Wadia.", "pt": "Olá, eu sou a Wadia."}[lang]
        speed = float(request.query.get("speed", speech.voice_speed()))
        samples = await asyncio.get_running_loop().run_in_executor(
            None, models.voice.synthesize, text, lang, voice_id, speed
        )
        return web.Response(body=to_wav(samples), content_type="audio/wav")

    async def update_settings(request: web.Request) -> web.Response:
        require(request)
        changes = await request.json()
        if changes.get("gender") in ("male", "female"):
            tools.set_voice_gender(changes["gender"])
        for lang in ("en", "es"):
            if f"voice_{lang}" in changes:
                tools.save_setting(f"voice_{lang}", str(changes[f"voice_{lang}"])[:80] or None)
        if "speed" in changes:
            tools.save_setting("speed", round(min(1.5, max(0.7, float(changes["speed"]))), 2))
        if is_local(request):  # laptop-only settings
            if "microphone" in changes:
                tools.save_setting("microphone", str(changes["microphone"]))
            if "mic_muted" in changes and assistant:
                assistant.command({"type": "mute", "on": bool(changes["mic_muted"])})
            if "autostart" in changes:
                on_setting("autostart", bool(changes["autostart"]))
        return web.json_response(settings_view())

    async def microphones(request: web.Request) -> web.Response:
        require(request, local_only=True)
        import sounddevice as sd

        apis = sd.query_hostapis()
        names = [d["name"] for d in sd.query_devices()
                 if d["max_input_channels"] > 0 and apis[d["hostapi"]]["name"] == "MME" and "Asignador" not in d["name"]
                 and "Mapper" not in d["name"]]
        return web.json_response({"microphones": names})

    async def phone_qr(request: web.Request) -> web.Response:
        """QR code for the owner's phone link (?for=owner) or the testers' link (?for=guest)."""
        require(request, local_only=True)
        import qrcode
        import qrcode.image.svg

        view = sharing_view()
        url = view["guest_url"] if request.query.get("for") == "guest" else view["owner_url"]
        if not url:
            raise web.HTTPNotFound()
        buffer = io.BytesIO()
        qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=1).save(buffer)
        return web.Response(body=buffer.getvalue(), content_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    async def sharing(request: web.Request) -> web.Response:
        """Turns the public link on or off, or makes a new testers' link (cancelling the old one)."""
        require(request, local_only=True)
        change = await request.json()
        if "on" in change:
            on_setting("sharing", bool(change["on"]))
        if change.get("new_guest_link"):
            guest_key(new=True)
        return web.json_response(sharing_view())

    async def memory(request: web.Request) -> web.Response:
        """Everything Wadia knows: location, facts about the user, and past conversations."""
        require(request)
        conversations = [
            {"id": c["id"], "device": c["device"], "when": history.when(c["started"]), "summary": c.get("summary", ""),
             "first": next((m["text"] for m in c["messages"] if m["role"] == "user"), ""), "turns": len(c["messages"]) // 2}
            for c in history.all_conversations()
        ]
        here = location.current()
        return web.json_response({
            "location": {"place": location.describe(here), "source": (here or {}).get("source", "")} if here else None,
            "facts": tools.load_memories(), "conversations": conversations,
        })

    async def conversation(request: web.Request) -> web.Response:
        require(request)
        data = history.load(request.query.get("id", ""))
        if not data:
            raise web.HTTPNotFound()
        return web.json_response(data)

    async def change_memory(request: web.Request) -> web.Response:
        require(request)
        change = await request.json()
        action, value = change.get("action"), str(change.get("value", "")).strip()[:300]
        if action == "add_fact" and value:
            tools.add_memories([as_fact(value)])
        elif action == "forget_fact":
            tools.remove_memories([value])
        elif action == "delete_conversation":
            history.delete(value)
        elif action == "delete_all_conversations":
            history.delete_all()
        return await memory(request)

    async def change_location(request: web.Request) -> web.Response:
        """Sets the location from a typed place, the phone's GPS, or back to automatic."""
        require(request)
        change = await request.json()
        run = asyncio.get_running_loop().run_in_executor
        if change.get("auto"):
            found = await run(None, location.use_automatic)
        elif "latitude" in change and "longitude" in change:
            found = await run(None, location.set_from_coordinates, float(change["latitude"]), float(change["longitude"]))
        else:
            found = await run(None, location.set_from_place, str(change.get("place", ""))[:100])
        if not found:
            return web.json_response({"error": "Couldn't find that place. Try adding the country, e.g. \"Telde, Spain\"."}, status=404)
        return web.json_response({"place": location.describe(found), "source": found["source"]})

    async def desktop_socket(request: web.Request) -> web.WebSocketResponse:
        """The laptop app window: receives what Wadia is doing, sends commands."""
        require(request, local_only=True)
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()

        def forward(event: dict) -> None:
            asyncio.run_coroutine_threadsafe(ws.send_str(json.dumps(event)), loop)

        for event in hub.snapshot():
            await ws.send_str(json.dumps(event))
        await ws.send_str(json.dumps({"type": "settings", **settings_view(), **sharing_view()}))
        with hub.lock:
            hub.listeners.append(forward)
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT and assistant:
                    assistant.command(json.loads(message.data))
        finally:
            with hub.lock:
                hub.listeners.remove(forward)
        return ws

    async def phone_socket(request: web.Request) -> web.WebSocketResponse:
        who = require(request, guests=True)
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        if not models.ready.is_set():
            await ws.send_str(json.dumps({"type": "state", "state": "loading"}))
            await asyncio.get_running_loop().run_in_executor(None, models.ready.wait)
        session = PhoneSession(ws, asyncio.get_running_loop(), models, guest=who == "guest")
        await ws.send_str(json.dumps({"type": "ready", "guest": session.guest}))
        hub.emit({"type": "phone", "connected": True, "guest": session.guest})
        try:
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    session.on_audio(message.data)
                elif message.type == WSMsgType.TEXT:
                    session.on_message(json.loads(message.data))
        finally:
            session.stop.set()
            session.conversation.end()  # save and summarize what was said on the phone
            hub.emit({"type": "phone", "connected": False, "guest": session.guest})
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/manifest.json", manifest)
    app.router.add_get("/sw.js", service_worker)
    app.router.add_get("/ping", ping)
    app.router.add_get("/api/voices", voices)
    app.router.add_get("/api/preview", preview)
    app.router.add_post("/api/settings", update_settings)
    app.router.add_get("/api/microphones", microphones)
    app.router.add_get("/api/phone-qr.svg", phone_qr)
    app.router.add_post("/api/sharing", sharing)
    app.router.add_get("/api/memory", memory)
    app.router.add_post("/api/memory", change_memory)
    app.router.add_get("/api/conversation", conversation)
    app.router.add_post("/api/location", change_location)
    app.router.add_get("/desktop", desktop_socket)
    app.router.add_get("/ws", phone_socket)
    app.router.add_static("/static", WEB_DIR)
    return app


def local_ip() -> str:
    """The laptop's address on the Wi-Fi (the one other devices can reach)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.0.1", 9))  # no packet is sent; this just picks the right network card
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def self_signed_certificate(ip: str) -> ssl.SSLContext:
    """Creates (once) a self-signed HTTPS certificate for this laptop's Wi-Fi address."""
    cert_file, key_file, ip_file = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "ip.txt"
    if not cert_file.exists() or not ip_file.exists() or ip_file.read_text() != ip:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        CERT_DIR.mkdir(exist_ok=True)
        private_key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Wadia")])
        now = datetime.datetime.now(datetime.timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.IPAddress(ipaddress.ip_address(ip)), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.DNSName("localhost"), x509.DNSName(socket.gethostname()),
                ]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )
        key_file.write_bytes(private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        ))
        cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        ip_file.write_text(ip)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_file, key_file)
    return context


def guest_key(new: bool = False) -> str:
    """The code in the testers' link. A new one cancels the old link."""
    code = tools.load_settings().get("guest_key")
    if new or not code:
        code = secrets.token_urlsafe(6)
        tools.save_setting("guest_key", code)
    return code


def app_guest_key() -> str:
    """The guest code built into the Try-out Android app."""
    code = tools.load_settings().get("app_guest_key")
    if not code:
        code = secrets.token_urlsafe(9)
        tools.save_setting("app_guest_key", code)
    return code


def access_key() -> str:
    key = tools.load_settings().get("phone_key")
    if not key:
        key = secrets.token_urlsafe(9)
        tools.save_setting("phone_key", key)
    return key


def start(models: Models, hub, assistant, on_setting=lambda key, value: None, phone: bool = True, link=None) -> str:
    """Starts the servers in a background thread. Returns the phone link (on the Wi-Fi)."""
    key = access_key()
    guest_key()
    # The Wi-Fi link always uses this laptop's own certificate; the public link (sharing.py)
    # is what works from anywhere.
    host = local_ip()
    ssl_context = self_signed_certificate(host) if phone else None
    phone_url = f"https://{host}:{PHONE_PORT}/?key={key}" if phone else ""
    app = make_app(models, hub, assistant, key, phone_url, on_setting, link)
    started = threading.Event()

    def serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app, access_log=None)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", DESKTOP_PORT).start())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", PUBLIC_PORT).start())
        if phone:
            loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", PHONE_PORT, ssl_context=ssl_context).start())
        started.set()
        loop.run_forever()

    threading.Thread(target=serve, daemon=True, name="wadia-server").start()
    started.wait(10)
    return phone_url
