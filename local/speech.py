"""Wadia's ears and voice, shared by the PC assistant and the phone app.

- SpeechDetector: finds where speech starts and ends with Silero VAD (ignores fans, typing...)
- Transcriber:    speech -> text with Whisper on the GPU, detecting the language
- Voice:          text -> speech with Kokoro, in a male or female voice per language
"""

import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

import tools

HERE = Path(__file__).parent

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_FALLBACK = "small"  # used if the main model doesn't fit in GPU memory
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto" = GPU if it works, else CPU
KOKORO_MODEL = HERE / "voices" / "kokoro-v1.0.onnx"
KOKORO_VOICES = HERE / "voices" / "voices-v1.0.bin"

MIC_RATE = 16_000  # Whisper and the VAD expect 16 kHz mono
VOICE_RATE = 24_000  # Kokoro's output rate
BLOCK = 512  # 32 ms per block, the size Silero VAD works with
VAD_WINDOW = 8  # blocks of context the VAD sees (~256 ms)
SPEECH_START = 0.5  # VAD probability that counts as speech
SPEECH_CONTINUE = 0.25  # lower bar to keep counting as speech once the user is talking
START_BLOCKS = 3  # ~100 ms of speech before we decide the user started talking
SILENCE_TO_END = 1.0  # seconds of silence that end the user's turn
PRE_ROLL = 0.3  # seconds kept from before speech started, so the first word isn't clipped
MAX_UTTERANCE = 30.0

# Languages Wadia speaks, and the default female/male voice for each. Voices can be changed in
# the app; see voice_catalog() and https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
DEFAULT_VOICES = {
    "en": {"female": "af_heart", "male": "am_michael"},
    "es": {"female": "ef_dora", "male": "em_alex"},
    "fr": {"female": "ff_siwis", "male": "ff_siwis"},
    "it": {"female": "if_sara", "male": "im_nicola"},
    "pt": {"female": "pf_dora", "male": "pm_alex"},
}
VOICES = DEFAULT_VOICES  # the languages Wadia can detect and answer in
DEFAULT_LANGUAGE = "en"
# The first letter of a Kokoro voice name is its accent; the second, f/m, its gender.
ACCENTS = {
    "a": ("en", "en-us", "American English"), "b": ("en", "en-gb", "British English"),
    "e": ("es", "es", "Spanish"), "f": ("fr", "fr-fr", "French"), "i": ("it", "it", "Italian"),
    "p": ("pt", "pt-br", "Brazilian Portuguese"),
}
RECOMMENDED = {"af_heart", "af_bella", "bf_emma", "am_michael", "am_fenrir", "bm_george", "ef_dora", "em_alex"}
NAME_HINTS = {"Wadia", "Alejandro"}  # names Whisper should always expect


class SpeechDetector:
    """Push-based turn detection: feed it 512-sample blocks, it tells you when speech
    starts and hands back the whole utterance when the speaker stops."""

    def __init__(self, vad) -> None:
        self.vad = vad
        self.reset()

    def reset(self) -> None:
        self.recorded: list[np.ndarray] = []
        self.speaking = False
        self.speech_streak = self.silent_streak = 0

    def feed(self, block: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """Returns ("started", None) when speech begins, ("ended", audio) when it's over,
        otherwise (None, None)."""
        self.recorded.append(block)
        probability = self._speech_probability()

        if not self.speaking:
            self.speech_streak = self.speech_streak + 1 if probability > SPEECH_START else 0
            if self.speech_streak >= START_BLOCKS:
                self.speaking = True
                return "started", None
            self.recorded = self.recorded[-int(PRE_ROLL * MIC_RATE / BLOCK) :]
            return None, None

        self.silent_streak = 0 if probability > SPEECH_CONTINUE else self.silent_streak + 1
        if self.silent_streak >= SILENCE_TO_END * MIC_RATE / BLOCK or len(self.recorded) >= MAX_UTTERANCE * MIC_RATE / BLOCK:
            audio = np.concatenate(self.recorded)
            self.reset()
            return "ended", audio
        return None, None

    def _speech_probability(self) -> float:
        window = np.concatenate(self.recorded[-VAD_WINDOW:])
        if len(window) < VAD_WINDOW * BLOCK:
            window = np.concatenate([np.zeros(VAD_WINDOW * BLOCK - len(window), np.float32), window])
        return float(np.asarray(self.vad(window))[-1])


def load_vad():
    from faster_whisper.vad import get_vad_model

    return get_vad_model()


class Transcriber:
    """Whisper speech recognition. Thread-safe, so the PC and the phone can share it."""

    def __init__(self) -> None:
        self.model, self.description = self._load()
        self.lock = threading.Lock()

    @staticmethod
    def _load():
        """Loads Whisper on the GPU if it fits and CUDA works, falling back to a smaller model or the CPU."""
        from faster_whisper import WhisperModel

        if WHISPER_DEVICE in ("auto", "cuda"):
            _add_cuda_dlls()
            for name in dict.fromkeys([WHISPER_MODEL, WHISPER_FALLBACK]):
                try:
                    model = WhisperModel(name, device="cuda", compute_type="int8_float16")
                    list(model.transcribe(np.zeros(MIC_RATE, dtype=np.float32))[0])  # fails here if CUDA is broken
                    return model, f"Whisper {name} on GPU"
                except Exception as e:
                    print(f"(couldn't run Whisper {name} on the GPU: {e})")
            if WHISPER_DEVICE == "cuda":
                sys.exit("Whisper failed on the GPU. Set WHISPER_DEVICE=cpu to use the CPU.")
        return WhisperModel(WHISPER_FALLBACK, device="cpu", compute_type="int8"), f"Whisper {WHISPER_FALLBACK} on CPU"

    def transcribe(self, audio: np.ndarray, current_lang: str) -> tuple[str, str]:
        """Returns (text, language). Short or ambiguous clips keep the current language,
        since Whisper's language guess is unreliable on a word or two."""
        with self.lock:
            segments, info = self.model.transcribe(audio, beam_size=1, vad_filter=True, hotwords=name_hints())
            segments = list(segments)
            if info.language not in VOICES or info.language_probability < 0.6:
                if info.language != current_lang:
                    segments, _ = self.model.transcribe(
                        audio, beam_size=1, vad_filter=True, hotwords=name_hints(), language=current_lang
                    )
                    segments = list(segments)
                lang = current_lang
            else:
                lang = info.language
        return " ".join(s.text for s in segments).strip(), lang


class Voice:
    """Kokoro text-to-speech on the CPU. The voice for each language and the speed come
    from the saved settings, read on every sentence so changes apply immediately.

    A voice is a Kokoro name ("af_heart") or a blend of two ("af_heart:60,af_bella:40")."""

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        self.tts = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        self.names = set(self.tts.get_voices())
        self.lock = threading.Lock()

    def synthesize(self, text: str, lang: str, voice: str | None = None, speed: float | None = None) -> np.ndarray:
        voice = voice or voice_for(lang)
        style, kokoro_lang = self._style(voice)
        with self.lock:
            samples, _ = self.tts.create(text, voice=style, lang=kokoro_lang, speed=speed or voice_speed())
        return samples.astype(np.float32)

    def _style(self, voice: str):
        """Returns (Kokoro voice style, Kokoro language) for a voice name or blend."""
        parts = []
        for part in voice.split(","):
            name, _, weight = part.strip().partition(":")
            if name in self.names:
                parts.append((name, float(weight or 100)))
        if not parts:
            parts = [(DEFAULT_VOICES[DEFAULT_LANGUAGE]["female"], 100.0)]
        kokoro_lang = ACCENTS.get(parts[0][0][0], ACCENTS["a"])[1]
        if len(parts) == 1:
            return parts[0][0], kokoro_lang
        total = sum(weight for _, weight in parts)
        style = sum(self.tts.get_voice_style(name) * (weight / total) for name, weight in parts)
        return style.astype(np.float32), kokoro_lang


def voice_for(lang: str) -> str:
    """The voice chosen for a language, or the default female/male voice for it."""
    settings = tools.load_settings()
    if chosen := settings.get(f"voice_{lang}"):
        return chosen
    return DEFAULT_VOICES.get(lang, DEFAULT_VOICES[DEFAULT_LANGUAGE])[tools.get_voice_gender()]


def voice_speed() -> float:
    return float(tools.load_settings().get("speed", 1.0))


def voice_catalog() -> list[dict]:
    """Every voice Wadia can use, for the voice picker in the app."""
    names = sorted(np.load(KOKORO_VOICES).files) if KOKORO_VOICES.exists() else []
    catalog = []
    for name in names:
        if name[0] not in ACCENTS or name[1] not in "fm":
            continue
        lang, _, accent = ACCENTS[name[0]]
        catalog.append({
            "id": name, "name": name[3:].capitalize(), "language": lang, "accent": accent,
            "gender": "female" if name[1] == "f" else "male", "recommended": name in RECOMMENDED,
        })
    return catalog


def chime(rising: bool = True) -> np.ndarray:
    """A soft two-note chime: rising when Wadia wakes up, falling when it goes to sleep."""
    notes = (660.0, 880.0) if rising else (880.0, 587.3)
    parts = []
    for frequency in notes:
        t = np.arange(int(0.09 * VOICE_RATE)) / VOICE_RATE
        tone = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 18)
        parts.append(tone * np.minimum(1, t * 400))  # tiny fade-in avoids a click
    return (np.concatenate(parts) * 0.25).astype(np.float32)


def name_hints() -> str:
    """Names Whisper should expect (NAME_HINTS plus any saved in memory), so words
    like "Wadia" or the user's name aren't misheard."""
    names = set(NAME_HINTS)
    for fact in tools.load_memories():
        names.update(re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-zà-ÿ]+", fact))
    return " ".join(sorted(names))


def _add_cuda_dlls() -> None:
    """Makes the pip-installed CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable on Windows."""
    try:
        import nvidia
    except ImportError:
        return
    for package in ("cublas", "cudnn", "cuda_nvrtc"):
        dll_dir = Path(nvidia.__path__[0]) / package / "bin"
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
