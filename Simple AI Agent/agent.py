import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not set!")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
messages = [{"role": "system", "content": "You are a helpful assistant."}]


def query(prompt):
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 250,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    assistan_reply = data["choices"][0]["message"]["content"]

    messages.append({"role": "assistant", "content": assistan_reply})

    return assistan_reply


def agent_loop():
    print("AI Agent Ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        try:
            answer = query(user_input)
            print("Agent:", answer, "\n")
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    agent_loop()
