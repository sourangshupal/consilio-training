"""Unified multi-provider ask(), ported from notebook 06's PROVIDER pattern.

Supports OpenAI, Gemini, and Groq behind one interface so every prompting
demo in the app can call `ask()` without caring which provider is active.
"""

DEFAULT_MODELS = {
    "OpenAI": "gpt-5.6-luna",
    "Gemini": "gemini-3.6-flash",
    "Groq": "llama-3.3-70b-versatile",
}


class LLMRouterError(RuntimeError):
    pass


def ask(provider: str, api_key: str, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    if not api_key:
        raise LLMRouterError(f"No API key set for {provider}. Enter one in the sidebar.")

    try:
        if provider == "OpenAI":
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=DEFAULT_MODELS["OpenAI"],
                messages=messages,
            )
            return response.choices[0].message.content

        if provider == "Gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=DEFAULT_MODELS["Gemini"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            return response.text

        if provider == "Groq":
            from groq import Groq

            client = Groq(api_key=api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=DEFAULT_MODELS["Groq"],
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content

        raise LLMRouterError(f"Unknown provider: {provider}")

    except LLMRouterError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any provider SDK error to the UI
        raise LLMRouterError(f"{provider} call failed: {exc}") from exc
