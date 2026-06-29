"""
test_table_ocr_mission_assist.py

Dead-simple tool to test OCR'ing an image with a vision model on the Mission
Assist API. Send an image + a prompt, print what the model replies. Start with
a normal image to confirm the connection works, then point it at a table crop.

Run from project root:
    python -m tests.test_table_ocr_mission_assist path/to/image.png
    python -m tests.test_table_ocr_mission_assist table.png --prompt "Transcribe this table as a markdown table."

Connection mirrors the official MA "OpenAI Connector" example:
    OpenAI(base_url="https://devmissionassist.api.us.baesystems.com/<segment>/v1",
           default_headers={"apikey": KEY},
           http_client=httpx.Client(verify=CA_CERT))
    ... .chat.completions.create(model="/genai/...", messages=[...])

Vision-capable models (per MA docs — pass the model name directly, no lookup):
    Gemma 3 27B    segment=bae-api-gemma3   model=/genai/gemma-3-27b      (default)
    Llama 4 Scout  segment=bae-api-llama4   model=/genai/Llama-4-Scout-17B
"""

import os
import base64
import mimetypes
import argparse
from pathlib import Path


# --- Settings (override via environment or CLI flags) ----------------------
HOST    = os.environ.get("MA_HOST", "https://devmissionassist.api.us.baesystems.com")
SEGMENT = os.environ.get("MA_SEGMENT", "bae-api-gemma3")   # Gemma 3 27B is vision-capable
MODEL   = os.environ.get("MA_MODEL", "/genai/gemma-3-27b")
API_KEY = os.environ.get("MA_API_KEY") or os.environ.get("LLM_API_KEY") or ""
CA_CERT = os.environ.get("MA_CA_CERT", "")                 # e.g. CA03_Base64_FullRootChain.pem ("" = system default)
PROMPT  = "Transcribe all text in this image exactly as it appears. If there is no text, briefly describe the image."


def ocr_image(image_path, prompt=PROMPT, host=HOST, segment=SEGMENT,
              model=MODEL, api_key=API_KEY, ca_cert=CA_CERT):
    """Send one image to the Mission Assist vision model and return its reply."""
    import httpx
    from openai import OpenAI

    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

    client = OpenAI(
        api_key=api_key,
        base_url=f"{host}/{segment}/v1",
        default_headers={"apikey": api_key},
        http_client=httpx.Client(verify=ca_cert) if ca_cert else None,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        temperature=0.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test OCR'ing an image via the Mission Assist API.")
    ap.add_argument("image", help="Path to the image to OCR.")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL, help="e.g. /genai/gemma-3-27b")
    ap.add_argument("--segment", default=SEGMENT, help="URL segment, e.g. bae-api-gemma3")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--ca-cert", default=CA_CERT, help="Path to CA pem (e.g. CA03_Base64_FullRootChain.pem).")
    args = ap.parse_args()

    print(ocr_image(args.image, prompt=args.prompt, host=args.host, segment=args.segment,
                    model=args.model, api_key=args.api_key, ca_cert=args.ca_cert))
