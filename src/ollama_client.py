import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


def ask_ollama(prompt, model_name, max_tokens=2000):
    url = f"{OLLAMA_HOST}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "completion" in data:
            return data["completion"]
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["text"]
        else:
            return ""
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""
