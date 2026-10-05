"""Smoke test: confirms the API key and model work with the Responses API."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
response = OpenAI().responses.create(model=model, input="Reply with the single word: ready")
print(model, "->", response.output_text)
