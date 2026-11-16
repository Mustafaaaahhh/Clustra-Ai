import json
import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY or "missing",
)

# ---------- Supabase logging ----------
_url = (os.environ.get("SUPABASE_URL") or "").strip().strip('"').rstrip("/")
if _url.endswith("/rest/v1"):
    _url = _url[:-len("/rest/v1")]
_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = None
try:
    from supabase import create_client
    if _url and _key:
        supabase = create_client(_url, _key)
except Exception as e:
    print("Supabase setup error:", e, flush=True)


def log_messages(rows):
    if supabase is None:
        return
    try:
        supabase.table("messages").insert(rows).execute()
    except Exception as e:
        print("Log error:", e, flush=True)
# --------------------------------------

MODEL = "openai/gpt-oss-120b"
MAX_MESSAGES = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OWNER_FILE = os.path.join(BASE_DIR, "Remember_owner.json")

# Website folders
HTML_DIR = os.path.join(BASE_DIR, "html")          # new website
OLD_HTML_DIR = os.path.join(BASE_DIR, "html_old")  # old website (kept for fun)

BASE_PROMPT = "You are Clustra AI. Answer in 1-2 short sentences."


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


@app.route("/")
def home():
    return send_from_directory(HTML_DIR, "index.html")


@app.route("/old")
def old_home():
    return send_from_directory(OLD_HTML_DIR, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])

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
            max_tokens=7000,
            reasoning_effort="high",
        )
        reply = response.choices[0].message.content
        log_messages([
            {"role": "user", "content": clean[-1]["content"]},
            {"role": "ai", "content": reply},
        ])
        return jsonify({"reply": reply})
    except Exception as e:
        log_messages([{"role": "user", "content": clean[-1]["content"]}])
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
