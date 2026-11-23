"""Builds our own "wadia" model in Ollama from a base model plus Wadia's persona.

Run again after editing persona.txt, or to switch the base model:
    python local/create_model.py                 # default base model
    python local/create_model.py qwen3.5:4b      # lighter / faster base

Afterwards `ollama run wadia` also works straight from the terminal.
"""

import sys
from pathlib import Path

import ollama

BASE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:9b"
persona = (Path(__file__).parent / "persona.txt").read_text(encoding="utf-8")

installed = {m.model for m in ollama.list().models}
if BASE_MODEL not in installed and f"{BASE_MODEL}:latest" not in installed:
    print(f"Downloading {BASE_MODEL}...")
    ollama.pull(BASE_MODEL)

ollama.create(
    model="wadia",
    from_=BASE_MODEL,
    system=persona,
    parameters={"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
)
print(f"Created model 'wadia' from {BASE_MODEL}. Try it:  ollama run wadia")
