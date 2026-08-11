"""
tools.py

Three deliberately simple mock tools. The point isn't that these tools
do anything useful — it's to stress-test whether phi3:mini can:
  1. decide which tool fits a query
  2. format the tool call correctly (right name, right args)
  3. use the tool's result in its final answer

Each tool has an OpenAI-style schema (Ollama uses the same format)
plus a real Python function that executes it.
"""


def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression. e.g. '12 * 7'"""
    try:
        # eval is fine here: local sandbox, throwaway learning project only
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"error: {e}"


def search(query: str) -> str:
    """Fake search tool — returns a canned result so you can test
    whether the model calls it appropriately, without needing real
    internet access or an API key."""
    fake_index = {
        "capital of france": "Paris is the capital of France.",
        "sap bdc": "SAP BDC (Business Data Cloud) is SAP's unified data platform combining Datasphere, Databricks, and AI Core.",
    }
    key = query.lower().strip()
    for k, v in fake_index.items():
        if k in key:
            return v
    return f"No results found for: {query}"


def lookup_user(user_id: str) -> str:
    """Fake database lookup — returns canned user records so you can
    test structured-argument tool calls (not just free text)."""
    fake_db = {
        "1": "Uday Charak — Role: SAP Technical Architect",
        "2": "Test User — Role: QA",
    }
    return fake_db.get(str(user_id), f"No user found with id: {user_id}")


# Tool schemas in the format Ollama's tool-calling API expects
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '12 * 7'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for general knowledge facts",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Look up a user record by their user ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's ID",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
]

# Map tool name -> real Python function, so the harness can execute
# whatever the model decides to call
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "search": search,
    "lookup_user": lookup_user,
}
