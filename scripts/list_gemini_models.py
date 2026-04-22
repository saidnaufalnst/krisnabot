"""Print Gemini models available to your API key."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from google import genai


def get_api_key() -> str:
    load_dotenv()

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_CHAT_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_GEMINI_CHAT_API_KEY":
        raise SystemExit(
            "API key belum ditemukan. Isi GEMINI_API_KEY di file .env terlebih dahulu."
        )

    return api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tampilkan model Gemini yang tersedia untuk API key Anda."
    )
    parser.add_argument(
        "--action",
        default="generateContent",
        help=(
            "Filter berdasarkan supported action, misalnya generateContent, "
            "embedContent, countTokens. Default: generateContent."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tampilkan semua model tanpa filter action.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Tampilkan supported actions dan limit token.",
    )
    return parser.parse_args()


def usable_model_name(model: object) -> str:
    return str(getattr(model, "name", "") or "").removeprefix("models/")


def print_model(model: object, show_details: bool) -> None:
    name = usable_model_name(model)
    if not show_details:
        print(name)
        return

    actions = ", ".join(getattr(model, "supported_actions", None) or [])
    input_tokens = getattr(model, "input_token_limit", "-")
    output_tokens = getattr(model, "output_token_limit", "-")
    print(f"{name} | actions: {actions or '-'} | input: {input_tokens} | output: {output_tokens}")


def main() -> None:
    args = parse_args()
    client = genai.Client(api_key=get_api_key())

    found = False
    for model in sorted(client.models.list(), key=usable_model_name):
        actions = getattr(model, "supported_actions", None) or []
        if args.all or args.action in actions:
            found = True
            print_model(model, args.details)

    if not found:
        print("Tidak ada model yang cocok dengan filter.")


if __name__ == "__main__":
    main()
