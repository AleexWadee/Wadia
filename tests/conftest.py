"""Test setup: makes local/ importable and replaces the heavy runtime dependencies
(the Ollama client, web search and the speech models) with small stand-ins, so the
pure logic can be tested on any machine without a GPU or the models."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local"))

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

ddgs = types.ModuleType("ddgs")
ddgs.DDGS = object
sys.modules.setdefault("ddgs", ddgs)

speech = types.ModuleType("speech")
speech.DEFAULT_LANGUAGE = "en"
sys.modules.setdefault("speech", speech)
