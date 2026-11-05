import json
import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("API_KEY"),
)

MODEL = "openai/gpt-oss-120b"
MAX_MESSAGES = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OWNER_FILE = os.path.join(BASE_DIR, "Remember_owner.json")

BASE_PROMPT = "You are Clustra AI. Answer in 1-3 short sentences."


def load_owner_info():
    if os.path.exists(OWNER_FILE):
        try:
            with open(OWNER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def build_system_message():
    content = BASE_PROMPT
    owner = load_owner_info()
    if owner:
        content += (
            "\n\nThese are facts about your owner. Always remember them and use "
            "them whenever relevant:\n"
            + json.dumps(owner, ensure_ascii=False, indent=2)
        )
    return {"role": "system", "content": content}


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clustra AI</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 16px; }
  #chat { border: 1px solid #ccc; border-radius: 8px; padding: 12px; height: 60vh; overflow-y: auto; }
  .you { color: #0a58ca; margin: 8px 0; }
  .ai { color: #222; margin: 8px 0; }
  #row { display: flex; gap: 8px; margin-top: 12px; }
  #msg { flex: 1; padding: 10px; }
  button { padding: 10px 14px; }
</style>
</head>
<body>
<h2>Clustra AI</h2>
<div id="chat"></div>
<div id="row">
  <input id="msg" placeholder="Type a message..." autocomplete="off">
  <button onclick="send()">Send</button>
  <button onclick="clearChat()">Clear</button>
</div>
<script>
let history = [];
try { history = JSON.parse(localStorage.getItem("clustra_history")) || []; } catch (e) {}

const chat = document.getElementById("chat");
const input = document.getElementById("msg");

function add(cls, who, text) {
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = who + ": " + text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function save() {
  try { localStorage.setItem("clustra_history", JSON.stringify(history)); } catch (e) {}
}

history.forEach(m => add(m.role === "user" ? "you" : "ai", m.role === "user" ? "You" : "Clustra AI", m.content));

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  add("you", "You", text);
  history.push({ role: "user", content: text });
  const waiting = add("ai", "Clustra AI", "...");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: history })
    });
    const data = await res.json();
    if (data.reply) {
      waiting.textContent = "Clustra AI: " + data.reply;
      history.push({ role: "assistant", content: data.reply });
      save();
    } else {
      waiting.textContent = "Error: " + (data.error || "unknown");
      history.pop();
    }
  } catch (e) {
    waiting.textContent = "Error: could not reach server";
    history.pop();
  }
}

function clearChat() {
  history = [];
  save();
  chat.innerHTML = "";
}

input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
</script>
</body>
</html>"""


@app.route("/")
def home():
    return PAGE


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])

    # only allow user/assistant messages from the browser
    clean = [
        {"role": m["role"], "content": str(m["content"])}
        for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m
    ]

    if not clean:
        return jsonify({"error": "No message"}), 400

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[build_system_message()] + clean[-MAX_MESSAGES:],
            max_tokens=2000,
            reasoning_effort="low",
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
