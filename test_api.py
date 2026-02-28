"""
Smoke test: verify LiteLLM can reach Claude via OpenRouter.
Run with: python3.12 test_api.py
"""
import os
from dotenv import load_dotenv
import litellm

# Load OPENROUTER_API_KEY from .env file
load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")
if not key or key == "sk-or-paste-your-key-here":
    raise SystemExit("ERROR: Set your OPENROUTER_API_KEY in .env first")

print("Calling Claude via OpenRouter...")

response = litellm.completion(
    model="openrouter/anthropic/claude-opus-4-6",
    api_key=key,
    api_base="https://openrouter.ai/api/v1",
    messages=[{"role": "user", "content": "Reply with exactly: API_OK"}],
)

reply = response.choices[0].message.content
print(f"Response: {reply}")
print(f"Tokens used: {response.usage.total_tokens}")
