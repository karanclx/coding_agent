import os
import sys
import json
from openai import OpenAI

# System Prompt
SYSTEM_PROMPT = """You are a helpful coding assistant that can read, write, and manage files.

You have access to the following tools:
- read_file: Read the contents of a file
- write_file: Write content to a file (creates or overwrites)
- list_files: List files in a directory

When given a task:
1. Think about what you need to do
2. Use tools to gather information or make changes
3. Continue until the task is complete
4. Explain what you did

Always be careful when writing files - make sure you understand the existing content first."""

# Tool Schemas (OpenAI format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory (defaults to current).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": []
            }
        }
    }
]

# Tool Functions (same as before)
def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, 'w') as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        return f"Error: {e}"

def list_files(path: str = ".") -> str:
    try:
        return "\n".join(os.listdir(path)) if os.listdir(path) else "No files"
    except Exception as e:
        return f"Error: {e}"

def execute_tool(name: str, args: dict) -> str:
    if name == "read_file":
        return read_file(args["path"])
    elif name == "write_file":
        return write_file(args["path"], args["content"])
    elif name == "list_files":
        return list_files(args.get("path", "."))
    return "Unknown tool"

# ReAct Loop - Fixed for Mistral/OpenAI API
def run_agent(user_message: str, history: list = None) -> None:
    client = OpenAI(
    api_key="8mbEOddOU3XaBQVA9oSz1revWB7IbxJH",   # ← Put your real Mistral key here inside the quotes
    base_url="https://api.mistral.ai/v1"
)

    if history is None:
        history = []

    history.append({"role": "user", "content": user_message})

    while True:
        # Call Mistral with streaming
        stream = client.chat.completions.create(
            model="mistral-large-latest",  # or "open-mistral-7b" if rate limited
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
            stream=True
        )

        # Streaming response
        print("\nAgent: ", end="", flush=True)
        full_text = ""
        tool_calls = []

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_text += text

            # Accumulate tool calls
            if chunk.choices[0].delta.tool_calls:
                for tc_delta in chunk.choices[0].delta.tool_calls:
                    idx = tc_delta.index
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                    if tc_delta.function.name:
                        tool_calls[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

        print()  # New line

        # Add assistant message to history
        assistant_msg = {"role": "assistant", "content": full_text}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        history.append(assistant_msg)

        # If tool calls, execute them
        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except:
                    args = {}
                result = execute_tool(name, args)

                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result
                })
            # Loop again to let model respond to tool results
        else:
            # Done
            break

# Main loop (unchanged)
def main():
    print("=" * 60)
    print("Baby Code Phase 1: Minimum Viable Coding Agent (Mistral)")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'clear' to reset")
    print("=" * 60)

    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except:
            break

        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            history = []
            print("Cleared.")
            continue
        if not user_input:
            continue

        run_agent(user_input, history)

if __name__ == "__main__":
    main()