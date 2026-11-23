"""Wadia's brain, shared by the PC assistant and the phone app: the LLM conversation
with tools, long-term memory, and wake-word / sleep-phrase detection."""

import json
import os
import random
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path

import ollama

import history
import location
import tools
from speech import DEFAULT_LANGUAGE

HERE = Path(__file__).parent

LLM_MODEL = os.getenv("LLM_MODEL", "wadia")
PERSONA = (HERE / "persona.txt").read_text(encoding="utf-8")
MAX_HISTORY = 30  # messages kept in memory (plus the system prompt)
MAX_TOOL_ROUNDS = 4  # tool calls the LLM may chain before it must answer
OLD_TOOL_RESULT = 600  # characters of a tool result kept after the turn it was used in
SLOW_TOOLS = {"web_search", "read_webpage", "get_weather"}  # say "one moment" before these
CONVERSATION_IDLE_MINUTES = 10  # after this long without talking, a conversation ends and gets summarized
RECENT_CONVERSATIONS = 8  # summaries of past conversations included in Wadia's instructions

PHRASES = {
    "en": {"ready": "I'm ready. Just say Wadia when you need me.", "bye": "Okay, I'll be here if you need me.",
           "wait": ["One moment.", "Let me look.", "Give me a second.", "Checking."],
           "stuck": "Sorry, I got stuck trying to work that out."},
    "es": {"ready": "Ya estoy. Di Wadia cuando me necesites.", "bye": "Vale, aquí estaré si me necesitas.",
           "wait": ["Un momento.", "Déjame ver.", "Dame un segundo.", "Lo miro."],
           "stuck": "Perdona, me he atascado intentando resolver eso."},
}

WAKE_WORD = "wadia"
WAKE_SIMILARITY = 0.7  # how close a heard word must be to "wadia" ("guadia", "badia"... count)
GREETINGS = {"hey", "hi", "hello", "ok", "okay", "oye", "hola", "eh", "so", "oh", "ey"}
SLEEP_PHRASES = re.compile(
    r"\b(goodbye|good bye|bye|that'?s all|that is all|never ?mind|stop|go to sleep|"
    r"adi[oó]s|eso es todo|nada m[aá]s|olv[ií]dalo|duerme|hasta luego|chao)\b",
    re.I,
)


SPANISH_HINTS = re.compile(
    r"[¿¡ñ]|\b(qu[eé]|c[oó]mo|cu[aá]ndo|d[oó]nde|por qu[eé]|hola|hoy|tiempo|est[aá]s?|eres|tienes|puedes|"
    r"el|la|los|las|una?|de|del|y|pero|para|con|es|son|hay|me|mi|tu|gracias)\b", re.I)
ENGLISH_HINTS = re.compile(
    r"\b(what|how|when|where|why|who|hello|hi|today|weather|are|is|you|your|can|the|a|an|of|and|but|for|"
    r"with|there|me|my|thanks|please)\b", re.I)


def typed_language(text: str, current: str) -> str:
    """The language of a typed message (Whisper only detects it for speech): Spanish or English."""
    spanish, english = len(SPANISH_HINTS.findall(text)), len(ENGLISH_HINTS.findall(text))
    return "es" if spanish > english else "en" if english > spanish else current


def phrase(lang: str, key: str) -> str:
    """A fixed phrase in the user's language (picked at random when there are several, to sound less robotic)."""
    choice = PHRASES.get(lang, PHRASES["en"])[key]
    return random.choice(choice) if isinstance(choice, list) else choice


def find_wake_word(text: str) -> tuple[bool, str]:
    """Looks for "Wadia" (or a near miss like "Guadia") at the start of what was said,
    optionally after a greeting ("Hey Wadia", "Oye Wadia"). Returns (found, what was said after it)."""
    words = list(re.finditer(r"[\wÀ-ÿ']+", text))
    start = 0
    while start < len(words) - 1 and start < 2 and words[start].group().lower() in GREETINGS:
        start += 1
    words = words[start : start + 2]
    candidates = [(w.group(), w.end()) for w in words]
    candidates += [(a.group() + b.group(), b.end()) for a, b in zip(words, words[1:])]  # "wa dia" -> "wadia"
    for word, end in candidates:
        if len(word) >= 4 and SequenceMatcher(None, word.lower(), WAKE_WORD).ratio() >= WAKE_SIMILARITY:
            return True, text[end:].lstrip(" ,.!?¡¿:;-—")
    return False, ""


MALE_VOICE = re.compile(r"\b(male|man'?s|masculine|guy'?s|boy'?s)\s+voice\b|\bvoz\s+(de\s+)?(hombre|chico|masculina|var[oó]n)\b", re.I)
FEMALE_VOICE = re.compile(r"\b(female|woman'?s|feminine|girl'?s|lady'?s)\s+voice\b|\bvoz\s+(de\s+)?(mujer|chica|femenina)\b", re.I)


def voice_request(text: str) -> str | None:
    """Spots "use a male voice" / "pon una voz de mujer" so the switch always happens,
    instead of trusting the LLM to call its tool."""
    if MALE_VOICE.search(text) and not FEMALE_VOICE.search(text):
        return "male"
    if FEMALE_VOICE.search(text) and not MALE_VOICE.search(text):
        return "female"
    return None


def is_sleep_phrase(text: str) -> bool:
    """Short utterances like "that's all" or "goodbye" end the conversation."""
    return bool(SLEEP_PHRASES.search(text)) and len(text.split()) <= 5


EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def clean_for_speech(text: str) -> str:
    return EMOJI.sub("", re.sub(r"[*#_`>\[\]]", "", text)).strip()


def sentences_from_stream(token_stream):
    """Groups streamed LLM tokens into sentences to speak. The first piece is cut early,
    at a comma once there are a few words, so Wadia starts talking sooner."""
    buffer, first = "", True
    for token in token_stream:
        buffer += token
        while True:
            match = re.search(r"(?<=[.!?])\s+|\n+", buffer)
            if first and not match:
                match = next(
                    (c for c in re.finditer(r"(?<=[,;:])\s+", buffer) if len(buffer[: c.start()].split()) >= 5), None
                )
            if not match:
                break
            piece, buffer = buffer[: match.start()], buffer[match.end() :]
            if (piece := clean_for_speech(piece)) and (first or not is_filler_question(piece)):
                first = False
                yield piece
    if (piece := clean_for_speech(buffer)) and (first or not is_filler_question(piece)):
        yield piece


FILLER_QUESTION = re.compile(
    r"anything else|something else|what else|want me to (check|look|keep)|quieres que (mire|busque)|anything more|know more|learn more|hear more|more (details|info)|another one|"
    r"can i help|may i help|help you (with )?(today|anything)|is there (anything|something)|"
    r"algo m[aá]s|saber m[aá]s|m[aá]s (detalles|informaci[oó]n)|te puedo ayudar|puedo ayudarte|necesitas algo|"
    r"hay algo (m[aá]s|espec[ií]fico)", re.I)


def is_filler_question(sentence: str) -> bool:
    """A generic offer like "Anything else?" or "Want to know more?", which a good voice
    assistant doesn't add: the user asks if they want more. Real questions get through."""
    return sentence.endswith("?") and len(sentence.split()) <= 16 and bool(FILLER_QUESTION.search(sentence))


GUEST_NOTE = """You're talking with a guest who is trying Wadia out through a shared link. You don't know
anything about them and won't remember this conversation. Never share anything about the person
who owns this Wadia (their name, location, memories or past conversations). You don't know where
the guest is: if they ask about the weather or anything local, ask which city."""
GUEST_TOOLS = ["get_current_datetime", "web_search", "read_webpage", "get_weather", "calculate"]


def system_prompt(current_conversation: str | None = None, guest: bool = False) -> str:
    """Wadia's instructions: persona, date and time, where the user is, what it knows about
    them, and what you talked about recently. Guests get none of the personal parts."""
    prompt = PERSONA + f"\nCurrent date and time: {tools.get_current_datetime()}."
    if guest:
        return prompt + "\n\n" + GUEST_NOTE
    if where := location.for_prompt():
        prompt += "\n" + where
    if memories := tools.load_memories():
        prompt += "\n\nWhat you know about the user:\n" + "\n".join(f"- {fact}" for fact in memories)
    if recent := history.recent_summaries(RECENT_CONVERSATIONS, exclude=current_conversation):
        prompt += ("\n\nYour recent conversations with the user (newest first), so you can pick up where you left off:\n"
                   + "\n".join(f"- {line}" for line in recent)
                   + "\nFor details of these or older conversations, use recall_conversations.")
    else:
        prompt += ("\n\nYou have no summaries of earlier conversations. If the user asks about a past conversation, "
                   "use recall_conversations, and if it finds nothing, say you don't have a record of it.")
    return prompt


def respond(messages: list[dict], lang: str = DEFAULT_LANGUAGE, on_tool=None, guest: bool = False):
    """Runs the LLM, executing any tools it asks for, and streams the reply text.

    Yields the text to speak (tokens of the answer, plus a short filler while slow
    tools run). Appends the assistant/tool messages to `messages` as it goes.
    """
    for _ in range(MAX_TOOL_ROUNDS + 1):
        content, tool_calls = "", []
        stream = ollama.chat(
            model=LLM_MODEL, messages=messages, stream=True, think=False, keep_alive="60m",
            tools=[tools.TOOLS_BY_NAME[name] for name in GUEST_TOOLS] if guest else tools.TOOLS,
        )
        for part in stream:
            if part.message.tool_calls:
                tool_calls.extend(part.message.tool_calls)
            if token := part.message.content:
                content += token
                yield token

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls or None})
        if not tool_calls:
            return

        if content:
            yield "\n"  # end the text spoken before the tool call as its own sentence
        elif any(call.function.name in SLOW_TOOLS for call in tool_calls):
            yield phrase(lang, "wait") + "\n"
        for call in tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            if on_tool:
                on_tool(name, args)
            messages.append({"role": "tool", "tool_name": name, "content": _run_tool(name, args, guest)})

    yield phrase(lang, "stuck")


def _run_tool(name: str, args: dict, guest: bool) -> str:
    if guest:  # guests only get the general tools, and never the owner's location
        if name not in GUEST_TOOLS:
            return "That isn't available in guest mode."
        if name == "get_weather" and not str(args.get("city", "")).strip():
            return "The guest's location is unknown. Ask them which city they mean."
    return tools.run_tool(name, args)


class Conversation:
    """One ongoing conversation (the laptop and each phone have their own).

    Every turn is saved to history. After CONVERSATION_IDLE_MINUTES without talking, the
    conversation ends: it gets summarized in the background, and the next one starts fresh
    (the summary keeps Wadia aware of what was said)."""

    def __init__(self, device: str = "laptop", on_tool=None, on_memory=None, guest: bool = False) -> None:
        self.device = device
        self.guest = guest  # a tester using a shared link: nothing is saved or remembered
        self.voice_gender: str | None = None  # a guest's "use a male voice" (theirs only)
        self.history: list[dict] = []
        self.lang = DEFAULT_LANGUAGE
        self.on_tool, self.on_memory = on_tool, on_memory
        self.log: history.ConversationLog | None = None
        self._idle_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def reply(self, user_text: str):
        """Streams the reply as speakable sentences. Call finish() afterwards."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
            if self.log is None and not self.guest:
                self.log = history.ConversationLog(self.device)
            log_id = self.log.id if self.log else None
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-MAX_HISTORY:]
        while self.history and self.history[0]["role"] == "tool":
            self.history.pop(0)  # never start the history with an orphaned tool result
        self._user_text = user_text
        prompt = system_prompt(current_conversation=log_id, guest=self.guest)
        if gender := voice_request(user_text):
            if self.guest:
                self.voice_gender = gender
            else:
                tools.set_voice_gender(gender)
            prompt += f"\n\nYour voice was just switched to a {gender} voice as the user asked; confirm it in a few words."
        self._messages = [{"role": "system", "content": prompt}] + self.history
        self._spoken: list[str] = []
        for sentence in sentences_from_stream(respond(self._messages, self.lang, self.on_tool, self.guest)):
            self._spoken.append(sentence)
            yield sentence

    def finish(self, interrupted: bool = False) -> None:
        if interrupted and self._messages[-1]["role"] != "assistant":
            self._messages.append({"role": "assistant", "content": " ".join(self._spoken) + " [interrupted by the user]"})
        # Web pages are long: once answered, keep only the start of each tool result, so the
        # conversation stays fast and the instructions never fall out of the model's memory.
        self.history = [
            {**m, "content": m["content"][:OLD_TOOL_RESULT] + " …"}
            if m["role"] == "tool" and len(m.get("content") or "") > OLD_TOOL_RESULT else m
            for m in self._messages[1:]
        ]
        with self._lock:
            if self.log is not None:
                self.log.add("user", self._user_text)
                self.log.add("assistant", " ".join(self._spoken) + (" …" if interrupted else ""))
                self.log.save()
            self._idle_timer = threading.Timer(CONVERSATION_IDLE_MINUTES * 60, self.end)
            self._idle_timer.daemon = True
            self._idle_timer.start()
        # Runs in the background, so it never delays the next reply. Guests aren't remembered.
        if not self.guest:
            threading.Thread(target=self._update_memory_quietly, args=(self._user_text,), daemon=True).start()

    def end(self) -> None:
        """Ends the conversation: it's summarized in the background, and the next reply starts fresh."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
            log, self.log = self.log, None
            self.history = []
        if log is not None and log.messages:
            threading.Thread(target=summarize_quietly, args=(log.id,), daemon=True).start()

    def _update_memory_quietly(self, user_text: str) -> None:
        try:
            changes = update_memory(user_text)
            if self.on_memory:
                self.on_memory(changes)
        except Exception:
            pass  # memory is best-effort; never break the conversation over it


SUMMARY_PROMPT = """Summarize this conversation between the user and Wadia (their voice assistant)
in one or two short sentences (at most 40 words), in English and in the past tense. Mention what
the user wanted, the key answers, recommendations or decisions, and anything to follow up on.
Be specific (names, places, numbers). Always refer to the user as "the user": never by name and
never as he or she. Write only the summary."""


def summarize(conversation_id: str) -> str:
    """Writes the short summary of a finished conversation and saves it."""
    data = history.load(conversation_id)
    if not data or not data["messages"]:
        return ""
    transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Wadia'}: {m['text']}" for m in data["messages"])
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": transcript[-12000:]}],
        think=False, options={"temperature": 0.2}, keep_alive="60m",
    )
    summary = clean_for_speech(response.message.content).replace("\n", " ")
    history.set_summary(conversation_id, summary)
    return summary


def summarize_quietly(conversation_id: str) -> None:
    try:
        summarize(conversation_id)
    except Exception:
        pass  # it's retried at the next start


def summarize_pending() -> None:
    """Summarizes conversations left without a summary (e.g. the app was closed mid-conversation)."""
    for conversation in history.unsummarized(CONVERSATION_IDLE_MINUTES):
        summarize_quietly(conversation["id"])


def warm_up() -> None:
    """Loads the LLM into GPU memory (raises if Ollama or the model is missing)."""
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "hi"}], think=False, keep_alive="60m")


MEMORY_PROMPT = """You maintain a personal assistant's long-term memory about its user.
You get the facts already saved and what the user just said. Decide:
- "add": NEW lasting facts about the user worth remembering in future conversations:
  name, age, location, job, studies, family, pets, likes and dislikes, goals, plans
  with dates, or anything they explicitly ask you to remember. Write each as a short
  third-person sentence in English, e.g. "The user's name is Ana."
- "remove": saved facts (copied exactly) that the user asks to forget, or that what
  they just said shows are no longer true.
When a fact changes (new job, moved city...), remove the old fact AND add the new one.
Ignore questions, small talk, one-off requests, and facts already saved.
The assistant is called Wadia: "Wadia" is never the user's name. If the message looks garbled
or cut off, don't guess facts from it.
Most of the time both lists are empty."""

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "add": {"type": "array", "items": {"type": "string"}},
        "remove": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["add", "remove"],
}


def update_memory(user_text: str) -> dict:
    """Saves new facts from what the user said and removes facts they want forgotten."""
    saved = tools.load_memories()
    known = "\n".join(f"- {fact}" for fact in saved) or "(nothing yet)"
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MEMORY_PROMPT},
            {"role": "user", "content": f"Saved facts:\n{known}\n\nThe user just said: {user_text}"},
        ],
        format=MEMORY_SCHEMA,
        think=False,
        options={"temperature": 0},
        keep_alive="60m",
    )
    changes = json.loads(response.message.content)
    changes["remove"] = [fact for fact in changes.get("remove", []) if fact in saved]
    tools.remove_memories(changes["remove"])
    tools.add_memories(changes.get("add", []))
    return changes
