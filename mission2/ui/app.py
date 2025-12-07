import base64
import io
import json
import os
import subprocess
import threading
from typing import Optional, Tuple
from flask import Flask, abort, jsonify, render_template, request, send_file, Response
import requests

app = Flask(__name__)

CURRENT_PLAN = {"steps": []}
EXECUTION_LOG = []
EXECUTION_STATUS = {"running": False, "current_action": None}

# OpenAI-compatible API endpoint
LLM_API_BASE = os.environ.get("LLM_API_BASE", "http://localhost:8080")

# Path to test_policy.py
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code")

# Available tools for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_rake",
            "description": "Execute the rake skill to draw lines in the zen garden sand",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_place_rock",
            "description": "Execute the place rock skill to place a decorative rock in the zen garden",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_complete",
            "description": "Report that the zen garden is complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A zen-like message about the completed garden"
                    }
                },
                "required": ["message"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are ZenBot, a robotic arm that creates beautiful zen gardens.
You have access to the following skills:
- execute_rake: Draw peaceful lines in the sand
- execute_place_rock: Place a decorative rock in the garden
- report_complete: Report when the garden is finished

When asked to create a zen garden, plan and execute the actions step by step.
Always rake the sand first to create a peaceful base, then optionally place a rock.
After completing all actions, call report_complete with a zen-like message.

Be thoughtful and deliberate in your actions, like a zen master."""

ROCK_IMAGE_BASE64 = os.environ.get("ROCK_IMAGE_BASE64")
ROCK_BASE64_FILE = os.path.join(app.root_path, "static", "assets", "rock_base64.txt")
KARESANSUI_IMAGE_BASE64 = os.environ.get("KARESANSUI_IMAGE_BASE64")
KARESANSUI_BASE64_FILE = os.path.join(
    app.root_path, "static", "assets", "karesansui_base64.txt"
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/plan", methods=["GET"])
def get_plan():
    return jsonify(CURRENT_PLAN)


@app.route("/api/plan", methods=["POST"])
def update_plan():
    global CURRENT_PLAN
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid payload"}), 400
    if "steps" not in data or not isinstance(data["steps"], list):
        return jsonify({"status": "error", "message": "Plan must include steps list"}), 400
    CURRENT_PLAN = {"steps": data.get("steps", [])}
    return jsonify({"status": "ok", "plan": CURRENT_PLAN})


def _load_base64_from_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _decode_base64_payload(payload: Optional[str]) -> Tuple[Optional[io.BytesIO], Optional[str]]:
    """Return (BytesIO, mimetype) for a given base64 payload."""

    if not payload:
        return None, None

    mimetype = "image/png"
    cleaned = payload

    if payload.startswith("data:image"):
        try:
            header, encoded = payload.split(",", 1)
            cleaned = encoded
            if ";" in header:
                mimetype = header.split(":", 1)[1].split(";", 1)[0]
            else:
                mimetype = header.split(":", 1)[1]
        except ValueError:
            return None, None

    try:
        data = base64.b64decode(cleaned)
    except (ValueError, TypeError):
        return None, None

    buffer = io.BytesIO(data)
    buffer.seek(0)
    return buffer, mimetype


@app.route("/rock-image")
def rock_image():
    """Serve a rock image from the committed/base64 payload."""

    payload = ROCK_IMAGE_BASE64 or _load_base64_from_file(ROCK_BASE64_FILE)
    buffer, mimetype = _decode_base64_payload(payload)
    if buffer and mimetype:
        return send_file(buffer, mimetype=mimetype)

    abort(404)


@app.route("/karesansui-image")
def karesansui_image():
    """Serve the karesansui logo from a base64 payload.

    The payload is loaded from the `static/assets/karesansui_base64.txt` file by
    default. You may also override it via the KARESANSUI_IMAGE_BASE64 environment
    variable (with or without a `data:` prefix). The logic mirrors the rock
    image handler so both assets can be updated via committed base64 strings
    instead of local binary files.
    """

    payload = KARESANSUI_IMAGE_BASE64 or _load_base64_from_file(
        KARESANSUI_BASE64_FILE
    )
    buffer, mimetype = _decode_base64_payload(payload)
    if buffer and mimetype:
        return send_file(buffer, mimetype=mimetype)

    abort(404)


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    global EXECUTION_STATUS
    
    EXECUTION_STATUS["current_action"] = tool_name
    EXECUTION_LOG.append({"action": tool_name, "status": "started", "arguments": arguments})
    
    try:
        if tool_name == "execute_rake":
            # Run test_policy.py with rake checkpoint
            result = run_policy("rake")
            EXECUTION_LOG.append({"action": tool_name, "status": "completed", "result": result})
            return f"Rake action completed: {result}"
            
        elif tool_name == "execute_place_rock":
            # Run test_policy.py with place_rock checkpoint
            result = run_policy("place_rock")
            EXECUTION_LOG.append({"action": tool_name, "status": "completed", "result": result})
            return f"Place rock action completed: {result}"
            
        elif tool_name == "report_complete":
            message = arguments.get("message", "The garden is complete.")
            EXECUTION_LOG.append({"action": tool_name, "status": "completed", "message": message})
            return f"Garden complete: {message}"
            
        else:
            return f"Unknown tool: {tool_name}"
            
    except Exception as e:
        EXECUTION_LOG.append({"action": tool_name, "status": "error", "error": str(e)})
        return f"Error executing {tool_name}: {str(e)}"
    finally:
        EXECUTION_STATUS["current_action"] = None


def run_policy(skill: str) -> str:
    """Run test_policy.py with the specified skill checkpoint."""
    # Map skill to checkpoint path
    checkpoints = {
        "rake": "outputs/smolvla_rake8_from_base/checkpoints/last/pretrained_model",
        "place_rock": "outputs/smolvla_place_rock3_from_base/checkpoints/last/pretrained_model",
    }
    
    checkpoint = checkpoints.get(skill)
    if not checkpoint:
        return f"Unknown skill: {skill}"
    
    checkpoint_path = os.path.join(CODE_DIR, checkpoint)
    
    # Check if checkpoint exists, if not try HuggingFace
    if not os.path.exists(checkpoint_path):
        hf_checkpoints = {
            "rake": "wmeddie/smolvla_rake8",
            "place_rock": "wmeddie/smolvla_place_rock3",
        }
        checkpoint_path = hf_checkpoints.get(skill, checkpoint_path)
    
    cmd = [
        "python", os.path.join(CODE_DIR, "test_policy.py"),
        "--checkpoint", checkpoint_path,
        "--max-speed", "10",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return "Success"
        else:
            return f"Policy execution failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Policy execution timed out"
    except Exception as e:
        return f"Error running policy: {str(e)}"


def call_llm_with_tools(messages: list) -> dict:
    """Call the LLM API with tools support."""
    url = f"{LLM_API_BASE}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1024,
    }
    
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def run_agent(user_message: str):
    """Run the agent loop with tool calling."""
    global EXECUTION_STATUS
    
    EXECUTION_STATUS["running"] = True
    EXECUTION_LOG.clear()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    try:
        max_iterations = 10
        for _ in range(max_iterations):
            response = call_llm_with_tools(messages)
            choice = response["choices"][0]
            message = choice["message"]
            
            # Add assistant message to history
            messages.append(message)
            
            # Check if we need to call tools
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    tool_name = function["name"]
                    arguments = json.loads(function.get("arguments", "{}"))
                    
                    # Execute the tool
                    result = execute_tool(tool_name, arguments)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                    
                    # If this was report_complete, we're done
                    if tool_name == "report_complete":
                        EXECUTION_STATUS["running"] = False
                        return {"status": "complete", "message": result, "log": EXECUTION_LOG}
            else:
                # No more tool calls, we're done
                final_message = message.get("content", "Garden creation complete.")
                EXECUTION_STATUS["running"] = False
                return {"status": "complete", "message": final_message, "log": EXECUTION_LOG}
        
        EXECUTION_STATUS["running"] = False
        return {"status": "max_iterations", "message": "Reached maximum iterations", "log": EXECUTION_LOG}
        
    except Exception as e:
        EXECUTION_STATUS["running"] = False
        return {"status": "error", "message": str(e), "log": EXECUTION_LOG}


@app.route("/api/agent/run", methods=["POST"])
def api_agent_run():
    """Start the agent to create a zen garden."""
    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "Please create a beautiful zen garden with raked lines and a rock.")
    
    # Run in background thread
    def run_in_background():
        run_agent(user_message)
    
    thread = threading.Thread(target=run_in_background)
    thread.start()
    
    return jsonify({"status": "started", "message": "Agent is creating your zen garden..."})


@app.route("/api/agent/status", methods=["GET"])
def api_agent_status():
    """Get the current agent execution status."""
    return jsonify({
        "running": EXECUTION_STATUS["running"],
        "current_action": EXECUTION_STATUS["current_action"],
        "log": EXECUTION_LOG
    })


@app.route("/api/execute/<skill>", methods=["POST"])
def api_execute_skill(skill: str):
    """Directly execute a skill without the agent."""
    valid_skills = ["rake", "place_rock"]
    if skill not in valid_skills:
        return jsonify({"status": "error", "message": f"Invalid skill. Choose from: {valid_skills}"}), 400
    
    result = run_policy(skill)
    return jsonify({"status": "ok", "result": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
