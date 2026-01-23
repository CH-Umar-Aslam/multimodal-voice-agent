import os
import requests
from dotenv import load_dotenv

# ========================
# Load environment variables
# ========================
load_dotenv()  # loads variables from .env file

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ========================
# OpenRouter endpoint
# ========================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ========================
# Models to test
# ========================
MODELS = {
    "deepseek": "deepseek/deepseek-v3.2",
    "moonshot": "moonshotai/kimi-k2-0905",
    "gemini": "google/gemini-2.5-flash"
}

# ========================
# Function to query a model
# ========================
def query_openrouter(model_name, prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODELS[model_name],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # OpenRouter responses usually in: choices[0].message.content
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying {model_name}: {e}"

# ========================
# Main
# ========================
if __name__ == "__main__":
    prompt = input("Enter your text: ")

    print("\n--- Model Responses ---\n")
    for model in MODELS:
        output = query_openrouter(model, prompt)
        print(f"[{model.upper()}]: {output}\n")
