"""Smoke test: confirms the API key and model work."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
client = OpenAI()
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
)
print(model, "->", reply.choices[0].message.content)
