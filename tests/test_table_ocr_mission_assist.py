"""
test_table_ocr_mission_assist.py

Dead-simple tool to test OCR'ing an image with the configured vision LLM.
Sends an image + a prompt and prints the model's reply. Uses the SAME unified
LLM config + call path as the pipeline (utils.call_llm_vision), so one set of
LLM_* env vars drives both.

Run from project root:
    python -m tests.test_table_ocr_mission_assist path/to/image.png
    python -m tests.test_table_ocr_mission_assist table.png --prompt "Transcribe this table as a markdown table."

Config comes from LLM_* env vars (same as the pipeline), overridable by flag.
"""

import os
import sys
import argparse
from pathlib import Path

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "app"))   # utils


# --- Defaults from the unified LLM_* env (Mission Assist Gemma 4 31B fallbacks) ---
PROVIDER = os.environ.get("LLM_PROVIDER", "mission_assist")
HOST     = os.environ.get("LLM_BASE_URL", "https://devmissionassist.api.us.baesystems.com")
SEGMENT  = os.environ.get("LLM_URL_SEGMENT") or os.environ.get("MA_URL_SEGMENT") or "bae-api-gemma-4-31B"
MODEL    = os.environ.get("LLM_MODEL", "/genai/Gemma-4-31B-IT")
API_KEY  = os.environ.get("LLM_API_KEY") or os.environ.get("MA_API_KEY") or ""
CA_CERT  = os.environ.get("LLM_CA_CERT") or os.environ.get("MA_CA_CERT") or ""
MAX_SIDE = int(os.environ.get("LLM_MAX_IMAGE_SIDE", "2048"))
PROMPT   = "Transcribe all text in this image exactly as it appears. If there is no text, briefly describe the image."


def _cfg(host, segment, model, api_key, ca_cert, max_side, provider=PROVIDER):
    """Assemble the unified LLM config dict the pipeline uses."""
    return {
        "provider": provider,
        "model": model,
        "base_url": host,
        "segment": segment,
        "api_key": api_key,
        "ca_cert": ca_cert,
        "max_image_side": max_side,
        "timeout": 180.0,
    }


def ocr_image(image_path, prompt=PROMPT, host=HOST, segment=SEGMENT,
              model=MODEL, api_key=API_KEY, ca_cert=CA_CERT, max_side=MAX_SIDE):
    """Send one image to the configured vision model and return its reply."""
    from utils import call_llm_vision
    cfg = _cfg(host, segment, model, api_key, ca_cert, max_side)
    return call_llm_vision(prompt, image_path, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test OCR'ing an image via the configured vision LLM.")
    ap.add_argument("image", help="Path to the image to OCR.")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL, help="e.g. /genai/Gemma-4-31B-IT")
    ap.add_argument("--segment", default=SEGMENT, help="URL segment, e.g. bae-api-gemma-4-31B")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--ca-cert", default=CA_CERT, help="Path to CA pem (e.g. CA03_Base64_FullRootChain.pem).")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE, help="Downscale long edge to N px (0 = original).")
    args = ap.parse_args()

    print(ocr_image(args.image, prompt=args.prompt, host=args.host, segment=args.segment,
                    model=args.model, api_key=args.api_key, ca_cert=args.ca_cert, max_side=args.max_side))
