"""One-time setup: create the voice assistant agent on ElevenLabs.

Prints the new agent ID; put it in `.env` as ELEVENLABS_AGENT_ID.

The free ElevenLabs plan includes about 15 agent minutes per month, so the agent is
set up not to waste them: it hangs up when you say goodbye, after 20 seconds of
silence, and after 5 minutes at most.
"""

import os

from dotenv import load_dotenv
from elevenlabs import ConversationalConfig
from elevenlabs.client import ElevenLabs

SYSTEM_PROMPT = """You are Wadia, a friendly and helpful voice assistant.
You are speaking out loud, so:
- Keep answers short and conversational (1-3 sentences unless asked for more).
- Never use markdown, lists, emojis, or URLs - only plain spoken language.
- If you didn't understand the user, ask them to repeat.
- When the user says goodbye or that they're done, say a short goodbye and then use
  the end_call tool to end the conversation.
"""

MAX_CONVERSATION_SECONDS = 300
END_AFTER_SILENCE_SECONDS = 20

load_dotenv()
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

agent = client.conversational_ai.agents.create(
    name="Wadia Voice Assistant",
    conversation_config=ConversationalConfig(
        agent={
            "first_message": "Hi, I'm Wadia, your voice assistant. How can I help you today?",
            "language": "en",
            "prompt": {
                "prompt": SYSTEM_PROMPT,
                "llm": "gemini-2.5-flash",
                "temperature": 0.5,
                "built_in_tools": {
                    "end_call": {"name": "end_call", "description": "", "params": {"system_tool_type": "end_call"}}
                },
            },
        },
        # "Rachel" - swap for any voice ID from your ElevenLabs Voice Library.
        tts={"voice_id": os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")},
        turn={"silence_end_call_timeout": END_AFTER_SILENCE_SECONDS},
        conversation={"max_duration_seconds": MAX_CONVERSATION_SECONDS},
    ),
)

print(f"Agent created.\nAdd this line to your .env file:\n\nELEVENLABS_AGENT_ID={agent.agent_id}")
