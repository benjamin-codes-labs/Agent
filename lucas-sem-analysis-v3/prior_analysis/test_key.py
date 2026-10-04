"""Minimal check that the API key in .env works with Claude Opus 5.5 at max effort."""
import os

import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
client = anthropic.Anthropic()

try:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        output_config={"effort": "max"},
        messages=[{"role": "user", "content": "Reply with exactly: key works"}],
    )
except anthropic.AuthenticationError as e:
    print(f"Key rejected (401): {e.message}")
except anthropic.PermissionDeniedError as e:
    print(f"Key lacks access (403): {e.message}")
except anthropic.APIStatusError as e:
    print(f"API error {e.status_code}: {e.message}")
except anthropic.APIConnectionError as e:
    print(f"Could not reach the API: {e}")
else:
    if response.stop_reason == "refusal":
        print("Request was declined (refusal).")
    else:
        text = "".join(b.text for b in response.content if b.type == "text")
        print(f"OK - model={response.model} reply={text!r}")
        print(f"usage: in={response.usage.input_tokens} out={response.usage.output_tokens}")
