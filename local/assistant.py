"""Wadia on the laptop: sleeps until it hears its name, then listens, thinks and speaks.

It reports what it's doing to the app window through a Hub, and takes commands from
it (wake up, stop talking, typed messages, mute the microphone).
"""

import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

import brain
import speech
import tools

FOLLOW_UP_SECONDS = 8.0  # after answering, how long Wadia waits for a follow-up without its name
COMMAND = object()  # returned by Ears.listen when the app sent a command


class Hub:
    """Collects what Wadia is doing and passes it on to any open app windows."""

    def __init__(self) -> None:
        self.state = "loading"
        self.detail = "Starting…"
        self.recent: deque[dict] = deque(maxlen=12)  # last captions, for a window opened mid-conversation
        self.listeners: list = []
        self.lock = threading.Lock()

    def emit(self, event: dict) -> None:
        with self.lock:
            if event["type"] == "state":
                self.state, self.detail = event["state"], event.get("detail", "")
            elif event["type"] in ("user", "assistant", "tool"):
                self.recent.append(event)
            listeners = list(self.listeners)
        _log(event)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [{"type": "state", "state": self.state, "detail": self.detail}, *self.recent]


def _log(event: dict) -> None:
    kind = event["type"]
    if kind == "user":
        print(f"\nYou:   {event['text']}", flush=True)
    elif kind == "assistant":
        print(f"Wadia: {event['text']}", flush=True)
    elif kind == "tool":
        print(f"       [{event['name']}]", flush=True)
    elif kind == "memory":
        for fact in event["add"]:
            print(f"       [remembered: {fact}]", flush=True)
        for fact in event["remove"]:
            print(f"       [forgot: {fact}]", flush=True)
    elif kind == "state" and event["state"] in ("sleeping", "muted", "error"):
        print(f"({event['state']}{': ' + event['detail'] if event.get('detail') else ''})", flush=True)


def find_microphone(setting: str) -> int | None:
    """Turns a microphone setting (part of its name, or its number) into a device number.
    A name can match the same mic once per Windows audio system, so the first match is used."""
    if not setting:
        return None
    if str(setting).isdigit():
        return int(setting)
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0 and setting.lower() in device["name"].lower():
            return index
    return None


class Ears:
    """Listens on the laptop microphone for one utterance at a time."""

    def __init__(self, vad) -> None:
        self.detector = speech.SpeechDetector(vad)

    def listen(self, device, start_timeout: float | None, signal: threading.Event):
        """Records until the user stops talking. Returns the audio, None if nobody starts
        talking within `start_timeout` seconds, or COMMAND if the app sent a command."""
        blocks: "queue.Queue[np.ndarray]" = queue.Queue()
        self.detector.reset()
        deadline = time.monotonic() + start_timeout if start_timeout else None

        def on_audio(indata, frames, time_info, status) -> None:
            blocks.put(indata[:, 0].copy())

        # The mic is only open while listening, so the assistant never hears itself.
        with sd.InputStream(
            device=device, samplerate=speech.MIC_RATE, channels=1, dtype="float32",
            blocksize=speech.BLOCK, callback=on_audio,
        ):
            while True:
                if signal.is_set():
                    return COMMAND
                try:
                    block = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                event, audio = self.detector.feed(block)
                if event == "ended":
                    return audio
                if deadline and not self.detector.speaking and time.monotonic() > deadline:
                    return None


class Mouth:
    """Speaks on the laptop speakers. Synthesis of the next sentence overlaps playback of
    the current one, and stop() cuts it off mid-sentence."""

    def __init__(self, voice: speech.Voice) -> None:
        self.voice = voice
        self.current_stop = threading.Event()

    def speak(self, sentences, lang: str, on_sentence=None) -> bool:
        """Speaks each sentence as soon as it arrives. Returns False if it was stopped."""
        audio_queue: "queue.Queue[np.ndarray | None]" = queue.Queue()
        stop = self.current_stop = threading.Event()
        done = threading.Event()

        player = threading.Thread(target=self._play, args=(audio_queue, stop), daemon=True)
        player.start()
        threading.Thread(target=_watch_keyboard, args=(stop, done), daemon=True).start()
        try:
            for sentence in sentences:
                if stop.is_set():
                    break
                if on_sentence:
                    on_sentence(sentence)
                audio_queue.put(self.voice.synthesize(sentence, lang))
        finally:
            audio_queue.put(None)
            player.join()
            done.set()
        return not stop.is_set()

    def stop(self) -> None:
        self.current_stop.set()

    @staticmethod
    def _play(audio_queue: "queue.Queue[np.ndarray | None]", stop: threading.Event) -> None:
        chunk = speech.VOICE_RATE // 10  # write 100 ms at a time so stopping is near-instant
        with sd.OutputStream(samplerate=speech.VOICE_RATE, channels=1, dtype="float32") as out:
            while (samples := audio_queue.get()) is not None:
                for i in range(0, len(samples), chunk):
                    if stop.is_set():
                        out.abort()
                        return
                    out.write(samples[i : i + chunk])
        # Leaving the `with` block waits for the last audio to finish playing.

    @staticmethod
    def chime(rising: bool) -> None:
        sd.play(speech.chime(rising), speech.VOICE_RATE)
        sd.wait()


def _watch_keyboard(stop: threading.Event, done: threading.Event) -> None:
    """Sets `stop` if Enter is pressed while Wadia talks (only when run from a console)."""
    if not (sys.stdin and sys.stdin.isatty()):
        return
    try:
        import msvcrt
    except ImportError:
        return
    while msvcrt.kbhit():  # ignore keys pressed before Wadia started talking
        msvcrt.getwch()
    while not done.wait(0.05):
        if msvcrt.kbhit():
            while msvcrt.kbhit():
                msvcrt.getwch()
            stop.set()
            return


class LaptopAssistant:
    """The always-listening assistant on the laptop."""

    def __init__(self, vad, transcriber: speech.Transcriber, voice: speech.Voice, hub: Hub) -> None:
        self.transcriber, self.hub = transcriber, hub
        self.ears, self.mouth = Ears(vad), Mouth(voice)
        self.conversation = brain.Conversation(
            on_tool=lambda name, args: hub.emit({"type": "tool", "name": name}),
            on_memory=lambda changes: hub.emit({"type": "memory", **changes}),
        )
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self.signal = threading.Event()
        self.muted = bool(tools.load_settings().get("mic_muted", False))
        self.awake = False

    # --- commands from the app window or tray (any thread) ---

    def command(self, command: dict) -> None:
        if command["type"] in ("wake", "stop", "sleep", "text", "mute"):
            self.mouth.stop()  # any of these cuts Wadia off if it's talking
        self.commands.put(command)
        self.signal.set()

    # --- main loop (its own thread) ---

    def run(self) -> None:
        self.mouth.chime(rising=True)
        while True:
            try:
                self._step()
            except Exception as e:  # never let one bad turn kill the assistant
                self.hub.emit({"type": "state", "state": "error", "detail": str(e)})
                time.sleep(2)

    def _step(self) -> None:
        if not self.commands.empty():
            self.signal.clear()
            self._handle(self.commands.get())
            return
        if self.muted:
            self._set_state("muted")
            self.signal.wait()
            return

        self._set_state("listening" if self.awake else "sleeping")
        device = find_microphone(tools.load_settings().get("microphone") or os.getenv("MIC_DEVICE", ""))
        result = self.ears.listen(device, FOLLOW_UP_SECONDS if self.awake else None, self.signal)
        if result is COMMAND:
            return
        if result is None:  # no follow-up: back to sleep
            self._sleep()
            return

        if self.awake:
            self._set_state("thinking")
        text, lang = self.transcriber.transcribe(result, self.conversation.lang)
        if not text:
            return
        if not self.awake:
            found, text = brain.find_wake_word(text)
            if not found:
                return  # not talking to Wadia
            self._wake()
            if not text:  # just "Wadia": wait for the actual question
                return
        self._turn(text, lang)

    def _handle(self, command: dict) -> None:
        kind = command["type"]
        if kind == "wake":
            self._sleep() if self.awake and command.get("toggle") else self._wake()
        elif kind == "sleep":
            self._sleep()
        elif kind == "text" and command.get("text", "").strip():
            self.awake = True
            text = command["text"].strip()
            self._turn(text, brain.typed_language(text, self.conversation.lang))
        elif kind == "mute":
            self.muted = bool(command.get("on"))
            tools.save_setting("mic_muted", self.muted)
            if self.muted:
                self.awake = False

    def _wake(self) -> None:
        if not self.awake:
            self.awake = True
            self._set_state("listening")
            self.mouth.chime(rising=True)

    def _sleep(self) -> None:
        if self.awake:
            self.awake = False
            self.mouth.chime(rising=False)
        self._set_state("muted" if self.muted else "sleeping")

    def _turn(self, text: str, lang: str) -> None:
        self.conversation.lang = lang
        self.hub.emit({"type": "user", "text": text})

        if brain.is_sleep_phrase(text):
            bye = brain.phrase(lang, "bye")
            self._set_state("speaking")
            self.mouth.speak([bye], lang, on_sentence=self._said)
            self.awake = False
            self._set_state("sleeping")
            return

        self._set_state("thinking")
        replies = self.conversation.reply(text)
        first = True

        def said(sentence: str) -> None:
            nonlocal first
            if first:
                self._set_state("speaking")
                first = False
            self._said(sentence)

        finished = self.mouth.speak(replies, lang, on_sentence=said)
        replies.close()  # stops the LLM if Wadia was cut off
        self.conversation.finish(interrupted=not finished)
        self.awake = True

    def _said(self, sentence: str) -> None:
        self.hub.emit({"type": "assistant", "text": sentence})

    def _set_state(self, state: str) -> None:
        if self.hub.state != state:
            self.hub.emit({"type": "state", "state": state})
