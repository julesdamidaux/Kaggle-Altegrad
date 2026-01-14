# gemini_client.py
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import os

load_dotenv()


def extract_non_thought_text(response):
    """Extract only non-thought text parts from Gemini 3 response."""
    text = ""
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                # Skip thought parts - check for thought attribute
                if hasattr(part, 'thought') and part.thought:
                    continue
                # Only include text parts that aren't thoughts
                if hasattr(part, 'text') and part.text:
                    text += part.text
    return text.strip()


class GeminiChatCallable:
    """
    Gemini wrapper that conforms to GEPA's ChatCompletionCallable protocol.
    Takes messages list: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    Returns: str (assistant response)
    """

    def __init__(self, model="gemini-3-flash-preview", max_retries=5):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.max_retries = max_retries

    def __call__(self, messages):
        """GEPA calls this with list of message dicts OR a string for reflection."""
        # Handle case where messages is a string (reflection LM call)
        if isinstance(messages, str):
            full_prompt = messages
        else:
            # Extract system prompt and user message from list of dicts
            system_content = ""
            user_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]

            # Combine into single prompt for Gemini
            full_prompt = f"{system_content}\n\n{user_content}" if system_content else user_content

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,  # Lower for consistency
                        max_output_tokens=49152,
                    )
                )
                # Extract only non-thought text parts
                text = extract_non_thought_text(response)
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Gemini API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Gemini API error after {self.max_retries} retries: {e}")
                    return ""


class GeminiClient:
    """Simple client for direct generation (used in inference)."""

    def __init__(self, model="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model

    def generate(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=16384,  # Increased for detailed descriptions
                    )
                )
                # Extract only non-thought text parts
                text = extract_non_thought_text(response)
                return text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Gemini API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Gemini API error after {max_retries} retries: {e}")
                    return ""
