"""
harness.py

Runs a set of test queries against a local model with tool access,
and logs exactly what happened at each step:
  - did the model call a tool at all?
  - did it pick the RIGHT tool?
  - did it format the arguments correctly?
  - did it use the tool result to produce a sane final answer?

This is the core learning exercise: watching where a small model's
tool-calling behavior breaks down, not just whether the final answer
looks right.

Usage:
    python3 harness.py
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ollama
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL = "phi3:mini"

# Each test case: the query, and which tool we EXPECT it to use (for grading)
TEST_CASES = [
    {"query": "What is 47 times 12?", "expected_tool": "calculator"},
    {"query": "What's the capital of France?", "expected_tool": "search"},
    {"query": "Look up user with id 1", "expected_tool": "lookup_user"},
    {"query": "Tell me about SAP BDC", "expected_tool": "search"},
    {"query": "What's 100 divided by 4, then add 10?", "expected_tool": "calculator"},
    # No tool should be needed for this one — tests over-triggering
    {"query": "Say a short greeting.", "expected_tool": None},
]


def run_single_case(query: str, expected_tool: str | None):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"EXPECTED TOOL: {expected_tool or '(none)'}")
    print("-" * 60)

    messages = [{"role": "user", "content": query}]

    # Step 1: model decides whether/which tool to call
    response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    msg = response["message"]

    tool_calls = msg.get("tool_calls")

    result = {
        "query": query,
        "expected_tool": expected_tool,
        "called_tool": None,
        "tool_args": None,
        "tool_result": None,
        "final_answer": None,
        "correct_tool_choice": None,
        "notes": [],
    }

    if not tool_calls:
        # Model chose not to call any tool
        result["final_answer"] = msg.get("content", "")
        result["correct_tool_choice"] = expected_tool is None
        print(f"NO TOOL CALLED. Direct answer: {result['final_answer']}")
        return result

    # Model called (at least) one tool — grade the first one
    call = tool_calls[0]
    tool_name = call["function"]["name"]
    raw_args = call["function"]["arguments"]

    result["called_tool"] = tool_name
    result["correct_tool_choice"] = tool_name == expected_tool

    print(f"MODEL CALLED: {tool_name}")
    print(f"RAW ARGS: {raw_args}")

    # Parse args defensively — this is exactly where small models
    # sometimes produce malformed output
    try:
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
        result["tool_args"] = args
    except Exception as e:
        result["notes"].append(f"FAILED TO PARSE ARGS: {e}")
        print(f"⚠️  ARG PARSE FAILURE: {e}")
        return result

    # Execute the real tool function
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        result["notes"].append(f"MODEL CALLED UNKNOWN TOOL: {tool_name}")
        print(f"⚠️  UNKNOWN TOOL: {tool_name}")
        return result

    try:
        tool_result = func(**args)
        result["tool_result"] = tool_result
        print(f"TOOL RESULT: {tool_result}")
    except Exception as e:
        result["notes"].append(f"TOOL EXECUTION FAILED: {e}")
        print(f"⚠️  TOOL EXECUTION FAILED: {e}")
        return result

    # Step 2: feed tool result back to model for a final answer
    messages.append(msg)
    messages.append({
        "role": "tool",
        "content": str(tool_result),
    })
    final_response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    result["final_answer"] = final_response["message"].get("content", "")
    print(f"FINAL ANSWER: {result['final_answer']}")

    return result


def main():
    results = [run_single_case(tc["query"], tc["expected_tool"]) for tc in TEST_CASES]

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    correct = sum(1 for r in results if r["correct_tool_choice"])
    total = len(results)

    for r in results:
        status = "✅" if r["correct_tool_choice"] else "❌"
        print(f"{status} '{r['query'][:40]}...' -> called: {r['called_tool']}, expected: {r['expected_tool']}")
        if r["notes"]:
            for n in r["notes"]:
                print(f"    ⚠️  {n}")

    print(f"\nTool-choice accuracy: {correct}/{total} ({100*correct//total}%)")

    # Save full results to a log file for later review
    log_path = os.path.join(os.path.dirname(__file__), "results_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {log_path}")


if __name__ == "__main__":
    main()
