"""
27_list_openai_compatible_models.py

Lists available models from an OpenAI-compatible endpoint.

LM Studio:
    python src/27_list_openai_compatible_models.py --base-url http://localhost:1234/v1

Ollama local:
    python src/27_list_openai_compatible_models.py --base-url http://localhost:11434/v1
"""

import argparse
import os
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key-env", default="")
    args = parser.parse_args()

    headers = {}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

    url = args.base_url.rstrip("/") + "/models"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    print(f"Endpoint: {url}")
    print("Available model IDs:")
    for item in data.get("data", []):
        print("-", item.get("id"))


if __name__ == "__main__":
    main()
