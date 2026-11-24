import json
import os
import re
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

BASE_PROMPT = (
    "You are Clustra AI. Read the Question thoroughly and answer it after visiting the web."
    "(longer only when the user asks for code or detail).\n\n"
    "DOCUMENTS: when the user asks you to create a standalone document "
    "(a plan, report, essay, study notes, guide, letter, email draft, resume, "
    "proposal, outline, checklist, README, article, and similar), put the "
    "complete document inside exactly one block like this:\n"
    '<doc title="Short Professional Title">\n'
    "...the document in Markdown (# title, ## sections, - bullets, 1. steps)...\n"
    "</doc>\n"
    "Outside the block write only one short sentence saying it is ready. "
    "Never use a <doc> block for ordinary questions, quick answers or code "
    "snippets. If the user asks to change an earlier document, output the full "
    "updated document in a <doc> block with the same title."
)

TITLE_PROMPT = (
    "Write a short, professional title (2 to 4 words, Title Case, no quotes, "
    "no ending punctuation) for a chat that starts with the user's message. "
    "Greetings such as 'hi' or 'hello' become 'Greetings'. Examples: "
    "'hi' -> Greetings; 'help me plan a website' -> Website Project Plan; "
    "'explain this NameError' -> Resolving Python NameError; "
    "'what is 12*7' -> Quick Calculation. Reply with the title only."
)

DOC_RE = re.compile(
    r"<doc\s+title\s*=\s*(?:\"([^\"]*)\"|'([^']*)')\s*>(.*?)(?:</doc>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def split_doc(text):
    """Split a model reply into (chat_text, doc). doc is None or {title, markdown}."""
    m = DOC_RE.search(text)
    if not m:
        return text.strip(), None
    title = (m.group(1) or m.group(2) or "Untitled document").strip()
    body = m.group(3).strip()
    chat_text = (text[:m.start()] + text[m.end():]).strip()
    if not body:
        return chat_text, None
    if not chat_text:
        chat_text = 'I created "' + title + '" in Docs.'
    return chat_text, {"title": title, "markdown": body}


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
        reply = response.choices[0].message.content or ""
        log_messages([
            {"role": "user", "content": clean[-1]["content"]},
            {"role": "ai", "content": reply},
        ])
        chat_text, doc = split_doc(reply)
        if not chat_text:
            return jsonify({"error": "Empty reply from the model"}), 502
        return jsonify({"reply": chat_text, "doc": doc})
    except Exception as e:
        log_messages([{"role": "user", "content": clean[-1]["content"]}])
        return jsonify({"error": str(e)}), 500


@app.route("/title", methods=["POST"])
def title():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:500]
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=8000,
            reasoning_effort="high",
        )
        raw = (response.choices[0].message.content or "").strip()
        line = raw.splitlines()[0] if raw else ""
        return jsonify({"title": line.strip(" \"'`.")[:60]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
