"""Microphone/speaker bridge between the local machine and an ElevenLabs agent.

ElevenLabs streams 16-bit PCM, mono, 16 kHz audio in both directions. This
implementation uses `sounddevice` (ships prebuilt wheels on Windows/macOS/Linux)
instead of PyAudio.
"""

import queue
import threading
from typing import Callable

import sounddevice as sd
from elevenlabs.conversational_ai.conversation import AudioInterface

SAMPLE_RATE = 16_000
INPUT_CHUNK = 4_000  # 250 ms per chunk, as recommended by the SDK


class SoundDeviceAudioInterface(AudioInterface):
    def __init__(self) -> None:
        self._output_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._in_stream: sd.RawInputStream | None = None
        self._out_stream: sd.RawOutputStream | None = None
        self._out_thread: threading.Thread | None = None

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        def on_mic_audio(indata, frames, time, status) -> None:
            input_callback(bytes(indata))

        self._in_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=INPUT_CHUNK,
            callback=on_mic_audio,
        )
        self._out_stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")

        self._stop.clear()
        self._in_stream.start()
        self._out_stream.start()
        self._out_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._out_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._out_thread:
            self._out_thread.join(timeout=1)
        for stream in (self._in_stream, self._out_stream):
            if stream:
                stream.stop()
                stream.close()
        self._in_stream = self._out_stream = self._out_thread = None

    def output(self, audio: bytes) -> None:
        self._output_queue.put(audio)

    def interrupt(self) -> None:
        # The user started talking over the agent: drop any audio not yet played.
        try:
            while True:
                self._output_queue.get_nowait()
        except queue.Empty:
            pass

    def _play_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self._output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._out_stream.write(audio)
