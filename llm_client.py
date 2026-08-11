"""
llm_client.py

Thin wrapper around the model backend so agent/orchestration code
never calls ollama.chat() (or any provider SDK) directly.

Why this matters:
- Swapping phi3 -> qwen2.5 -> Gemini Flash -> Claude later is a
  one-file change, not a rewrite of every agent.
- Gives you one place to add retries, logging, and token/cost
  tracking as your agents get more complex.
"""

import ollama


DEFAULT_MODEL = "phi3:mini"


def chat(messages, model: str = DEFAULT_MODEL, tools: list | None = None, **kwargs):
    """
    Send a chat request to the local Ollama model.

    Args:
        messages: list of {"role": "user"/"assistant"/"system", "content": str}
        model: which local model to use (default phi3:mini)
        tools: optional list of tool schemas, for tool-calling agents
        **kwargs: passed through to ollama.chat (e.g. options={"temperature": 0.2})

    Returns:
        str: the model's text response
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        **kwargs,
    )
    return response["message"]["content"]


def chat_raw(messages, model: str = DEFAULT_MODEL, tools: list | None = None, **kwargs):
    """
    Same as chat(), but returns the full response object instead of
    just the text. Use this when you need tool_calls, not just content.
    """
    return ollama.chat(model=model, messages=messages, tools=tools, **kwargs)


if __name__ == "__main__":
    # quick manual test: python3 llm_client.py
    reply = chat([{"role": "user", "content": "Say hello in exactly 5 words."}])
    print(reply)
