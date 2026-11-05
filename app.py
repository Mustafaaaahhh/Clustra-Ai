import json
import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY or "missing",
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


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clustra AI</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: #000;
    color: #ededed;
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .app {
    width: 100%;
    max-width: 820px;
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 0 16px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0;
    border-bottom: 1px solid #1c1c1c;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, #6d5dfc, #00c2ff);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 18px; color: #fff;
  }
  .logo.small { width: 28px; height: 28px; border-radius: 8px; font-size: 14px; }
  .title { font-size: 17px; font-weight: 600; }
  .status { font-size: 12px; color: #8a8a8a; display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #2ecc71; box-shadow: 0 0 8px #2ecc71; }
  .right { display: flex; align-items: center; gap: 12px; }
  .badge {
    display: flex; align-items: center; gap: 8px;
    background: #0e0e0e; border: 1px solid #262626;
    padding: 6px 12px 6px 6px; border-radius: 12px;
    font-size: 14px; font-weight: 600;
  }
  .clear-btn {
    background: transparent; color: #9a9a9a; border: 1px solid #262626;
    padding: 8px 14px; border-radius: 10px; cursor: pointer; font-size: 13px;
  }
  .clear-btn:hover { color: #fff; border-color: #444; }

  #chat { flex: 1; overflow-y: auto; padding: 20px 0; scroll-behavior: smooth; }
  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-thumb { background: #222; border-radius: 3px; }

  .welcome { text-align: center; margin: 8vh auto 0; max-width: 520px; }
  .welcome .logo { width: 64px; height: 64px; font-size: 28px; border-radius: 18px; margin: 0 auto 18px; }
  .welcome h1 { font-size: 26px; margin: 0 0 8px; }
  .welcome p { color: #8a8a8a; margin: 0 0 24px; font-size: 15px; line-height: 1.5; }
  .chips { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
  .chip {
    background: #0e0e0e; border: 1px solid #262626; color: #cfcfcf;
    padding: 10px 14px; border-radius: 12px; cursor: pointer; font-size: 14px;
  }
  .chip:hover { border-color: #6d5dfc; color: #fff; }

  .msg { display: flex; gap: 12px; margin: 16px 0; align-items: flex-start; }
  .msg.user { flex-direction: row-reverse; }
  .avatar {
    width: 32px; height: 32px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
  }
  .msg.ai .avatar { background: linear-gradient(135deg, #6d5dfc, #00c2ff); color: #fff; }
  .msg.user .avatar { background: #1d1d1d; color: #bbb; }
  .bubble {
    max-width: 78%; padding: 12px 16px; border-radius: 16px;
    line-height: 1.55; font-size: 15px; white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.ai .bubble { background: #0f0f0f; border: 1px solid #1f1f1f; border-top-left-radius: 4px; }
  .msg.user .bubble { background: #2a2370; border-top-right-radius: 4px; }
  .msg.error .bubble { border-color: #5a1f1f; color: #ff8a8a; }

  .typing { display: inline-flex; gap: 5px; padding: 4px 0; }
  .typing span {
    width: 7px; height: 7px; border-radius: 50%; background: #666;
    animation: blink 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }

  .composer { padding: 12px 0 8px; }
  .box {
    display: flex; align-items: flex-end; gap: 10px;
    background: #0c0c0c; border: 1px solid #262626; border-radius: 16px; padding: 10px 10px 10px 16px;
  }
  .box:focus-within { border-color: #6d5dfc; }
  textarea {
    flex: 1; background: transparent; border: none; outline: none; resize: none;
    color: #fff; font-size: 15px; font-family: inherit; max-height: 140px; line-height: 1.5; padding: 6px 0;
  }
  textarea::placeholder { color: #555; }
  .send {
    width: 40px; height: 40px; border: none; border-radius: 12px; cursor: pointer;
    background: linear-gradient(135deg, #6d5dfc, #00c2ff); color: #fff; font-size: 18px;
  }
  .send:disabled { opacity: 0.4; cursor: not-allowed; }
  footer { text-align: center; color: #555; font-size: 12px; padding: 6px 0 14px; }

  @media (max-width: 600px) {
    .bubble { max-width: 86%; }
    .welcome h1 { font-size: 22px; }
    .badge span { display: none; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="brand">
      <div class="logo">C</div>
      <div>
        <div class="title">Clustra AI</div>
        <div class="status"><span class="dot"></span>Online</div>
      </div>
    </div>
    <div class="right">
      <button class="clear-btn" onclick="clearChat()">Clear chat</button>
      <div class="badge">
        <div class="logo small">C</div>
        <span>Clustra AI</span>
      </div>
    </div>
  </header>

  <div id="chat"></div>

  <div class="composer">
    <div class="box">
      <textarea id="msg" rows="1" placeholder="Message Clustra AI..." autocomplete="off"></textarea>
      <button class="send" id="sendBtn" onclick="send()" aria-label="Send">&#10148;</button>
    </div>
  </div>
  <footer>Clustra AI can make mistakes. Press Enter to send, Shift+Enter for a new line.</footer>
</div>

<script>
let history = [];
try { history = JSON.parse(localStorage.getItem("clustra_his
