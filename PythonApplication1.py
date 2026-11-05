import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
api_key = os.environ.get("API_KEY")
)

MODEL = "openai/gpt-oss-120b"
HISTORY_FILE = "chat_history.json"
OWNER_FILE = "Remember_owner.json"
MAX_MESSAGES = 20  # how many recent messages to remember

BASE_PROMPT = "You are Clustra AI. Answer in 1-3 short sentences."


def load_owner_info():
    """Read the owner file. Returns the data, or None if missing/broken."""
    if os.path.exists(OWNER_FILE):
        try:
            with open(OWNER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("Warning: could not read", OWNER_FILE, "- check it is valid JSON.")
    return None


def build_system_message():
    """Combine the base prompt with the owner facts, fresh every time."""
    content = BASE_PROMPT
    owner = load_owner_info()
    if owner:
        content += (
            "\n\nThese are facts about your owner. Always remember them and use "
            "them whenever relevant:\n"
            + json.dumps(owner, ensure_ascii=False, indent=2)
        )
    return {"role": "system", "content": content}


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


history = load_history()  # saved messages, without the system message

if history:
 #   print(f"Clustra AI remembers your last chat ({len(history)} messages)." 
    print("Hello User! ")



while True:
    user_message = input("You: ").strip()

    if user_message.lower() == "bye":
        print("Clustra AI: Goodbye!")
        break

    if user_message.lower() == "clear":
        history = []
        save_history(history)
        print("Clustra AI: Memory cleared.")
        continue
    if not user_message:
        continue

    history.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[build_system_message()] + history[-MAX_MESSAGES:],
            stream=True,
            max_tokens=7000,
            reasoning_effort="high",
        )

        print("Clustra AI: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                reply += text
        print()

        history.append({"role": "assistant", "content": reply})
        save_history(history)

    except Exception as e:
        print("\nError:", e)
        history.pop()  # remove the failed message

        
        