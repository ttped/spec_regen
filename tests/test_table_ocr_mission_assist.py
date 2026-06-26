"""
test_table_ocr_mission_assist.py

Sandbox for OCR'ing a SINGLE table image with a vision-capable model served
through the Mission Assist API. This is the starting point for the planned
table revamp (replace the IRIS Excel/JSON ingest with an internal LLM that
reads the table crop directly).

Lives in: root/tests/
Run from project root.

    # OCR one table crop, print the model's transcription
    python -m tests.test_table_ocr_mission_assist path/to/table_crop.png

    # Discover what the endpoint actually calls its model (handy because the
    # in-API model name for "Gemma 4 31b" is unknown until you can query it)
    python -m tests.test_table_ocr_mission_assist --list-models

    # Force a specific output shape and save it
    python -m tests.test_table_ocr_mission_assist crop.png --format json --out table.json

Connection details mirror utils._call_mission_assist():
    api_url = {base_url}/bae-api-{model_segment}/v1     headers = {"apikey": <key>}
    openai.OpenAI(base_url=api_url, api_key=<key>, default_headers=headers)

Two unknowns you'll need to confirm once you have API access (both overridable
via flags / .env, see CONFIG below):
  1. MODEL_SEGMENT — the "{segment}" in the bae-api-{segment} URL path. For the
     new Gemma 4 31b model this is a guess; use --list-models or ask the MA team.
  2. The in-API model id — auto-resolved via client.models.list() (first model),
     same as the main pipeline. Override with --model-id if needed.
"""

import os
import sys
import base64
import json
import mimetypes
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# =============================================================================
# .env loading (same pattern as the rest of the project)
# =============================================================================

def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            # Strip surrounding quotes the .env sometimes uses
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_ROOT_DIR = Path(__file__).resolve().parent.parent
_load_env(_ROOT_DIR / ".env")


# =============================================================================
# CONFIG — defaults come from .env, every value is overridable on the CLI
# =============================================================================

# Base host for Mission Assist (NOT the per-model /bae-api-... path — that is
# assembled below). e.g. "http://devmissionassist.api.us.baesystems.com"
DEFAULT_BASE_URL = (
    os.environ.get("MA_BASE_URL")
    or os.environ.get("MISSION_ASSIST_BASE_URL")
    or "http://devmissionassist.api.us.baesystems.com"
)

DEFAULT_API_KEY = (
    os.environ.get("MA_API_KEY")
    or os.environ.get("MISSION_ASSIST_API_KEY")
    or os.environ.get("LLM_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

# !!! BEST GUESS — confirm with --list-models or the MA team !!!
# This is the "{segment}" in /bae-api-{segment}/v1. The old code used the model
# name directly (e.g. "gemma3-27b"), special-casing "gpt-oss" -> "gptoss".
DEFAULT_MODEL_SEGMENT = os.environ.get("MA_MODEL_SEGMENT", "gemma4-31b")

DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_TOKENS = 8192


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = (
    "You are a meticulous OCR engine specialized in transcribing tables from "
    "images. You reproduce the table exactly as printed: every cell, in the "
    "original row and column order, with no reordering, no invented values, and "
    "no commentary. Empty cells are preserved as empty strings."
)

# Ask for the same shape the docx writer already understands
# ({"columns": [...], "rows": [[...]]}) so this drops into the pipeline later.
JSON_INSTRUCTION = (
    "Transcribe the table in this image to JSON with EXACTLY this schema:\n"
    '{\n'
    '  "columns": ["<header 1>", "<header 2>", ...],\n'
    '  "rows": [["<r1c1>", "<r1c2>", ...], ["<r2c1>", ...], ...]\n'
    "}\n\n"
    "Rules:\n"
    "- Keep the original left-to-right column order and top-to-bottom row order.\n"
    "- If the table has no clear header row, use empty strings for columns and "
    "put every printed row in rows.\n"
    "- Every row must have the same number of entries as columns; use \"\" for "
    "blank/merged cells.\n"
    "- Return ONLY the JSON object, no markdown fences, no explanation."
)

MARKDOWN_INSTRUCTION = (
    "Transcribe the table in this image to a GitHub-flavored Markdown table, "
    "preserving the original row and column order exactly. Return ONLY the "
    "Markdown table, nothing else."
)


# =============================================================================
# Image encoding
# =============================================================================

def encode_image_data_url(image_path: str) -> str:
    """Read an image file and return an OpenAI-style data: URL (base64)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Table image not found: {path}")

    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        # Fall back to PNG; most YOLO crops are .png
        mime = "image/png"

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# =============================================================================
# Mission Assist client (mirrors utils._call_mission_assist)
# =============================================================================

def build_client(base_url: str, api_key: str, model_segment: str, timeout: float):
    """
    Construct an OpenAI client pointed at a specific Mission Assist model path.

    Returns (client, api_url). Importing openai lazily keeps the rest of this
    file usable (e.g. encode_image_data_url) even where openai isn't installed.
    """
    import openai

    if model_segment == "gpt-oss":
        model_segment = "gptoss"

    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    api_url = f"{base_url.rstrip('/')}/bae-api-{model_segment}/v1"
    client = openai.OpenAI(
        base_url=api_url,
        api_key=api_key,
        default_headers={"apikey": api_key},
        timeout=timeout,
    )
    return client, api_url


def resolve_model_id(client, api_url: str, override: Optional[str] = None) -> str:
    """Use the caller's model id if given, else the endpoint's first model."""
    if override:
        return override
    print(f"  [MA] Resolving model at: {api_url}")
    models = client.models.list()
    if not models.data:
        raise ValueError(f"No models available at endpoint: {api_url}")
    model_id = models.data[0].id
    print(f"  [MA] Using model id: {model_id}")
    return model_id


def list_models(base_url: str, api_key: str, model_segment: str, timeout: float) -> List[str]:
    """Print and return the model ids the endpoint exposes (discovery helper)."""
    client, api_url = build_client(base_url, api_key, model_segment, timeout)
    models = client.models.list()
    ids = [m.id for m in models.data]
    print(f"  [MA] Endpoint {api_url} exposes {len(ids)} model(s):")
    for i in ids:
        print(f"        - {i}")
    return ids


# =============================================================================
# The actual OCR call
# =============================================================================

def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model wrapped its output."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    return s.strip()


def ocr_table(
    image_path: str,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model_segment: str = DEFAULT_MODEL_SEGMENT,
    model_id: Optional[str] = None,
    output_format: str = "json",
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Send one table image to the Mission Assist vision model and return the
    raw transcription string (JSON text or Markdown, per output_format).
    """
    if not api_key:
        raise ValueError(
            "No API key. Set MA_API_KEY (or LLM_API_KEY) in .env or pass --api-key."
        )

    data_url = encode_image_data_url(image_path)
    instruction = JSON_INSTRUCTION if output_format == "json" else MARKDOWN_INSTRUCTION

    client, api_url = build_client(base_url, api_key, model_segment, timeout)
    resolved_model = resolve_model_id(client, api_url, override=model_id)

    print(f"  [MA] OCR'ing {image_path} (format={output_format}) ...")
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def ocr_table_to_dict(image_path: str, **kwargs) -> Dict:
    """
    Convenience wrapper: OCR a table as JSON and parse into a Python dict
    ({"columns": [...], "rows": [[...]]}). Raises if the model returns non-JSON.
    """
    kwargs["output_format"] = "json"
    raw = ocr_table(image_path, **kwargs)
    return json.loads(_strip_json_fences(raw))


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OCR a single table image via the Mission Assist vision API."
    )
    parser.add_argument("image", nargs="?", help="Path to the table image (e.g. a YOLO crop).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Mission Assist host.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key (apikey header).")
    parser.add_argument("--model-segment", default=DEFAULT_MODEL_SEGMENT,
                        help="The {segment} in /bae-api-{segment}/v1 (confirm with --list-models).")
    parser.add_argument("--model-id", default=None,
                        help="Exact in-API model id; if omitted, the endpoint's first model is used.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json",
                        help="Output shape to request from the model.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--out", default=None, help="Optional path to save the transcription.")
    parser.add_argument("--list-models", action="store_true",
                        help="List the model ids the endpoint exposes, then exit.")
    args = parser.parse_args(argv)

    if args.list_models:
        list_models(args.base_url, args.api_key, args.model_segment, args.timeout)
        return 0

    if not args.image:
        parser.error("an image path is required (or use --list-models)")

    result = ocr_table(
        args.image,
        base_url=args.base_url,
        api_key=args.api_key,
        model_segment=args.model_segment,
        model_id=args.model_id,
        output_format=args.format,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 60)
    print("MODEL OUTPUT")
    print("=" * 60)
    print(result)

    if args.format == "json":
        # Best-effort parse so you immediately see if it's well-formed.
        try:
            parsed = json.loads(_strip_json_fences(result))
            cols = len(parsed.get("columns", []))
            rows = len(parsed.get("rows", []))
            print(f"\n  [OK] Parsed JSON: {cols} columns x {rows} rows")
        except json.JSONDecodeError as e:
            print(f"\n  [WARN] Output was not valid JSON: {e}")

    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"  [Saved] {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
