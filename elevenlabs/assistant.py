"""Voice virtual assistant powered by ElevenLabs Agents.

The ElevenLabs agent runs the full conversation loop in real time:
  1. Speech recognition  - your microphone audio is streamed and transcribed
  2. Response generation - the LLM configured on the agent writes a reply
  3. Audio creation      - the reply is synthesized to speech and streamed back

Run it, talk, and press Ctrl+C to end the conversation.
"""

import os
import sys
import threading

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation

from audio_interface import SoundDeviceAudioInterface


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    if not api_key or not agent_id:
        sys.exit("Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in .env (run create_agent.py first).")

    client = ElevenLabs(api_key=api_key)
    session_ended = threading.Event()

    conversation = Conversation(
        client,
        agent_id,
        requires_auth=True,  # private agent: authenticated with the API key
        audio_interface=SoundDeviceAudioInterface(),
        callback_user_transcript=lambda text: print(f"You:       {text}"),
        callback_agent_response=lambda text: print(f"Assistant: {text}"),
        callback_agent_response_correction=lambda original, corrected: print(
            f"Assistant (interrupted): {corrected}"
        ),
        callback_latency_measurement=lambda ms: print(f"           [latency {ms} ms]"),
        callback_end_session=session_ended.set,
    )

    print("Connecting... speak when the assistant greets you. Press Ctrl+C to quit.\n")
    conversation.start_session()
    try:
        # Short waits keep Ctrl+C responsive (a plain join can block it on Windows).
        while not session_ended.wait(0.5):
            pass
    except KeyboardInterrupt:
        conversation.end_session()
    conversation_id = conversation.wait_for_session_end()
    print(f"\nConversation ended. ID: {conversation_id}")


if __name__ == "__main__":
    main()
