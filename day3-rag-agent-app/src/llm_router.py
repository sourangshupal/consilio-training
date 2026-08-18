"""Unified multi-provider LLM access: OpenAI, Gemini, and Groq behind one interface.

Every page calls `ask()` for simple prompt/response calls, or
`get_langchain_chat_model()` when a LangChain-wrapped chat model is needed
(RAGAS judging, LangGraph agents) — so provider branching lives in exactly
one place.
"""

DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Gemini": "gemini-2.5-flash",
    "Groq": "llama-3.3-70b-versatile",
}


class LLMRouterError(RuntimeError):
    pass


def ask(provider: str, api_key: str, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
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


def get_langchain_chat_model(provider: str, api_key: str, temperature: float = 0.0):
    """Returns a LangChain-wrapped chat model for the given provider, for use
    with RAGAS (LangchainLLMWrapper) and LangGraph agent graphs."""
    if not api_key:
        raise LLMRouterError(f"No API key set for {provider}. Enter one in the sidebar.")

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=DEFAULT_MODELS["OpenAI"], api_key=api_key)

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=DEFAULT_MODELS["Gemini"], temperature=temperature, google_api_key=api_key)

    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=DEFAULT_MODELS["Groq"], temperature=temperature, api_key=api_key)

    raise LLMRouterError(f"Unknown provider: {provider}")


def normalize_label(text: str, allowed: tuple[str, ...]) -> str:
    """Maps a one-word LLM verdict onto `allowed`, tolerating the punctuation,
    markdown, and preamble models add ("**Correct.**", "Answer: complex").
    Returns "" when no allowed label appears. Used by the CRAG grader and the
    Adaptive RAG classifier, which both prompt for a single bare word.
    """
    lowered = text.lower()
    for label in allowed:
        if label.lower() in lowered:
            return label
    return ""
