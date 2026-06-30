import json
import os
import re
import requests
from typing import Optional, Any, Dict, List
import openai


def clean_json_string(text: str) -> str:
    """Remove trailing commas from JSON (common OCR output issue)."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def repair_unescaped_quotes(text: str, error_pos: int) -> str:
    """
    Attempt to fix an unescaped quote at or near the error position.
    
    This is a heuristic - it looks for quotes that appear to be inside
    a string value (between structural quotes) and escapes them.
    """
    # Find the problematic area - look backwards for the opening quote
    # and forwards for what should be the closing quote
    
    # Simple approach: escape the quote at error_pos if it looks like
    # it's inside a string (preceded by text, not by : or [)
    
    if error_pos >= len(text):
        return text
    
    # Look at character before the error position
    # If we see a pattern like: "text "word" more" 
    # The middle quotes need escaping
    
    search_start = max(0, error_pos - 100)
    search_end = min(len(text), error_pos + 100)
    
    # Find all quotes in this region and try escaping the one at/near error_pos
    region_start = text.rfind('"', search_start, error_pos)
    if region_start == -1:
        return text
    
    # Check if there's a quote right at or just before error_pos that needs escaping
    for offset in range(0, 10):
        check_pos = error_pos - offset
        if check_pos < 0:
            break
        if text[check_pos] == '"' and check_pos > 0 and text[check_pos-1] != '\\':
            # Check if this looks like a mid-string quote (not structural)
            # Structural quotes are preceded by : , [ { or whitespace after these
            prev_significant = check_pos - 1
            while prev_significant > 0 and text[prev_significant] in ' \t\n\r':
                prev_significant -= 1
            
            prev_char = text[prev_significant] if prev_significant >= 0 else ''
            
            # If preceded by a letter, number, or punctuation (not JSON structural)
            # it's probably a quote inside text that needs escaping
            if prev_char not in ':,[{':
                # Escape this quote
                return text[:check_pos] + '\\' + text[check_pos:]
    
    return text


def load_json_with_recovery(file_path: str) -> Any:
    """
    Load JSON file, attempting to fix common issues like trailing commas
    and unescaped quotes from OCR.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        json.JSONDecodeError if parsing fails even after cleanup attempts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First try normal parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [JSON] Parse error at position {e.pos}: {e.msg}")
        
        # Show context around error
        start = max(0, e.pos - 50)
        end = min(len(content), e.pos + 50)
        context = content[start:end].replace('\n', '\\n').replace('\r', '\\r')
        print(f"  [JSON] Context: ...{context}...")
        
        # Try cleaning trailing commas
        print(f"  [JSON] Attempting repairs...")
        cleaned = clean_json_string(content)
        
        try:
            result = json.loads(cleaned)
            print(f"  [JSON] Fixed with trailing comma removal")
            return result
        except json.JSONDecodeError as e2:
            pass
        
        # Try fixing unescaped quotes (multiple attempts)
        repaired = cleaned
        for attempt in range(5):  # Try up to 5 quote repairs
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as eq:
                if "Expecting ',' delimiter" in eq.msg or "Expecting ':' delimiter" in eq.msg:
                    repaired = repair_unescaped_quotes(repaired, eq.pos)
                else:
                    break
        
        # Final attempt
        try:
            result = json.loads(repaired)
            print(f"  [JSON] Fixed with quote escaping")
            return result
        except json.JSONDecodeError as e_final:
            print(f"  [JSON] All repairs failed: {e_final.msg} at position {e_final.pos}")
            start = max(0, e_final.pos - 50)
            end = min(len(repaired), e_final.pos + 50)
            context = repaired[start:end].replace('\n', '\\n').replace('\r', '\\r')
            print(f"  [JSON] Final context: ...{context}...")
            raise


def _find_balanced_json(text: str, start_char: str, end_char: str) -> Optional[str]:
    """
    Find a balanced JSON object or array by counting braces/brackets.
    
    Args:
        text: The text to search
        start_char: '{' for objects, '[' for arrays
        end_char: '}' for objects, ']' for arrays
    
    Returns:
        The balanced JSON string or None
    """
    start_idx = text.find(start_char)
    if start_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:i+1]
    
    return None


def _extract_json_from_llm_string(text: str) -> Optional[str]:
    """
    Attempts to extract a JSON string from LLM output.
    Handles markdown code blocks and nested JSON structures.
    """
    if not text:
        return None

    text = text.strip()

    # First, try to extract from markdown code blocks
    # Look for ```json ... ``` blocks
    json_block_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_block_match:
        content = json_block_match.group(1).strip()
        if _is_valid_json(content):
            return content
        # If not valid, try to find balanced JSON within it
        for start, end in [('{', '}'), ('[', ']')]:
            balanced = _find_balanced_json(content, start, end)
            if balanced and _is_valid_json(balanced):
                return balanced

    # Try generic ``` ... ``` blocks
    generic_block_match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if generic_block_match:
        content = generic_block_match.group(1).strip()
        if _is_valid_json(content):
            return content
        for start, end in [('{', '}'), ('[', ']')]:
            balanced = _find_balanced_json(content, start, end)
            if balanced and _is_valid_json(balanced):
                return balanced

    # Check if entire text is valid JSON
    if _is_valid_json(text):
        return text

    # Try to find balanced JSON object in raw text
    balanced_obj = _find_balanced_json(text, '{', '}')
    if balanced_obj and _is_valid_json(balanced_obj):
        return balanced_obj

    # Try to find balanced JSON array in raw text
    balanced_arr = _find_balanced_json(text, '[', ']')
    if balanced_arr and _is_valid_json(balanced_arr):
        return balanced_arr

    # Last resort: try to repair common issues
    repaired = _try_repair_json(text)
    if repaired:
        return repaired

    print(f"Debug: Could not extract valid JSON from: '{text[:300]}...'")
    return None


def _try_repair_json(text: str) -> Optional[str]:
    """
    Try to repair common JSON issues from LLM output.
    """
    # Find potential JSON start
    obj_start = text.find('{')
    arr_start = text.find('[')
    
    if obj_start == -1 and arr_start == -1:
        return None
    
    # Determine which comes first
    if obj_start == -1:
        start_idx = arr_start
        end_char = ']'
    elif arr_start == -1:
        start_idx = obj_start
        end_char = '}'
    else:
        start_idx = min(obj_start, arr_start)
        end_char = '}' if obj_start < arr_start else ']'
    
    # Extract from start to end of text
    potential = text[start_idx:]
    
    # Try adding missing closing brace/bracket
    for num_closes in range(1, 4):
        candidate = potential + (end_char * num_closes)
        if _is_valid_json(candidate):
            return candidate
    
    # Try removing trailing garbage after last valid close
    for i in range(len(potential) - 1, 0, -1):
        candidate = potential[:i+1]
        if _is_valid_json(candidate):
            return candidate
    
    return None


def _is_valid_json(text: str) -> bool:
    """Helper function to check if a string is valid JSON."""
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _attempt_json_repair(json_string: str, error: json.JSONDecodeError) -> Optional[str]:
    """
    Tries to repair a JSON string by escaping an unescaped quote at the error position.
    """
    if "char" in error.msg:
        error_pos = error.pos
        repaired_string = f"{json_string[:error_pos]}\\{json_string[error_pos:]}"
        print(f"--- INFO: Attempting JSON repair at position {error_pos}. ---")
        return repaired_string
    return None


def save_results_to_json(results: List[Dict[str, Any]], file_path: str):
    """
    Saves the classification results to a JSON file.
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Output successfully saved to {file_path}")


def load_pages_from_json(file_path: str) -> Dict[str, str]:
    """
    Reads page data from a JSON file, intelligently handling two different formats.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Returning empty pages.")
        return {}

    print(f"Reading page data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from {file_path}. Returning empty pages.")
        return {}

    pages_output = {}

    # Check for Format 1 ("Old" format with a "pages" key)
    if "pages" in data and isinstance(data["pages"], dict):
        print(f"  - Detected 'pages' key format in {file_path}.")
        for page_num, content in data["pages"].items():
            if isinstance(content, str):
                pages_output[str(page_num)] = content
        return pages_output

    # Check for Format 2 ("New" format with nested "page_text")
    elif all(isinstance(val, dict) and "page_text" in val for val in data.values()):
        print(f"  - Detected nested 'page_text' format in {file_path}.")
        for page_num, content_dict in data.items():
            page_key = str(content_dict.get('page_Id', page_num))
            pages_output[page_key] = content_dict.get("page_text", "")
        return pages_output
        
    else:
        print(f"Warning: Unrecognized JSON structure in {file_path}. No pages loaded.")
        return {}


def call_llm(prompt: str, cfg: Dict, timeout: float = None) -> Optional[str]:
    """
    Text completion through the single configured LLM provider.

    `cfg` is the one LLM config dict (see simple_pipeline.LLM_CONFIG) with keys:
        provider, model, base_url, api_key, segment, ca_cert, timeout, ...

    Per-provider meaning of `base_url`:
      - transformers : local filesystem path to the model directory
      - llama_server : root URL of the running llama-server (e.g. http://localhost:8080)
      - ollama       : Ollama host
      - mission_assist : API host; combined with `segment` -> host/segment/v1
    """
    provider = cfg["provider"]
    timeout = timeout if timeout is not None else float(cfg.get("timeout") or 120.0)

    if provider == "ollama":
        return _call_ollama(cfg, prompt, timeout)
    elif provider == "mission_assist":
        if not cfg.get("api_key"):
            raise ValueError("api_key is required for mission_assist provider")
        return _call_mission_assist(cfg, prompt, timeout)
    elif provider == "transformers":
        return _call_transformers(prompt, model_path=cfg["base_url"])
    elif provider == "llama_server":
        return _call_llama_server(cfg, prompt, timeout)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Use 'ollama', 'mission_assist', 'transformers', or 'llama_server'."
        )


def call_llm_vision(prompt: str, image_path: str, cfg: Dict, timeout: float = None) -> Optional[str]:
    """
    Vision completion (prompt + image) through the same configured provider.

    Works for mission_assist and any OpenAI-compatible endpoint (llama_server,
    ollama). The image is downscaled to cfg['max_image_side'] and sent as a
    data: URL. The 'transformers' provider does not support vision here.
    """
    provider = cfg["provider"]
    if provider == "transformers":
        raise NotImplementedError("Vision is not supported for the 'transformers' provider.")
    timeout = timeout if timeout is not None else float(cfg.get("timeout") or 180.0)

    data_url = _encode_image_data_url(image_path, int(cfg.get("max_image_side") or 2048))
    client, model = _openai_client(cfg, timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _call_ollama(cfg: Dict, prompt: str, timeout: float = 120.0) -> Optional[str]:
    """Sends a prompt to Ollama and expects a JSON object in the response."""
    base_url = cfg["base_url"]
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    api_url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("message", {}).get("content")


def _call_llama_server(cfg: Dict, prompt: str, timeout: float = 120.0) -> Optional[str]:
    """
    Sends a prompt to a llama.cpp `llama-server` via its OpenAI-compatible
    `/v1/chat/completions` endpoint, requesting JSON output.
    """
    base_url = cfg["base_url"]
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


_MISSION_ASSIST_SYSTEM_PROMPT = """You are an expert-level JSON generation API. Your sole purpose is to respond with a single, valid JSON object or array.

**CRITICAL RESPONSE DIRECTIVES:**

1.  **JSON ONLY:** Your entire response MUST be a single, valid JSON object or array, enclosed in a markdown code block.
2.  **NO EXTRA TEXT:** You MUST NOT include any other text, explanations, reasoning, or conversational filler. The response must contain ONLY the JSON.
3.  **STRICT SYNTAX:** The response must start *exactly* with ```json on a new line and end *exactly* with ``` on a new line. There must be no characters before the opening ```json or after the closing ```.

**DO NOT INCLUDE YOUR REASONING OR THOUGHTS IN THE FINAL RESPONSE.**
"""


def _mission_assist_base_url(cfg: Dict) -> str:
    """Build the Mission Assist per-model API base URL (host/segment/v1) from cfg."""
    segment = cfg.get("segment")
    if not segment:
        model = cfg["model"]
        if model.startswith("/"):
            # A full model id ("/genai/...") can't be turned into a path segment.
            raise ValueError(
                f"Mission Assist URL segment not set for model '{model}'. "
                f"Set LLM_URL_SEGMENT (e.g. 'bae-api-gemma-4-31B')."
            )
        segment = "bae-api-gptoss" if model == "gpt-oss" else f"bae-api-{model}"

    base_url = (cfg.get("base_url") or "").strip()
    if base_url.startswith("http://"):            # Mission Assist is HTTPS-only
        base_url = "https://" + base_url[len("http://"):]
    elif not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    return f"{base_url.rstrip('/')}/{segment}/v1"


def _openai_client(cfg: Dict, timeout: float):
    """
    Return (client, model_id) for an OpenAI-compatible endpoint — shared by the
    text and vision paths. Handles Mission Assist (apikey header, optional CA
    bundle, per-model URL segment) and generic endpoints (llama_server, ollama).
    """
    api_key = cfg.get("api_key") or "not-needed"
    http_client = None
    if cfg.get("ca_cert"):
        import httpx
        http_client = httpx.Client(verify=cfg["ca_cert"], timeout=timeout)

    if cfg["provider"] == "mission_assist":
        api_url = _mission_assist_base_url(cfg)
        client = openai.OpenAI(
            base_url=api_url,
            api_key=api_key,
            default_headers={"apikey": cfg.get("api_key") or ""},
            http_client=http_client,
            timeout=timeout,
        )
        model = cfg["model"]
        if not model.startswith("/"):            # resolve short ids via the endpoint
            model = _resolve_mission_assist_model(client, api_url)
        return client, model

    # Generic OpenAI-compatible endpoint (llama_server, ollama)
    base_url = (cfg["base_url"] or "").strip()
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = openai.OpenAI(base_url=base_url, api_key=api_key, http_client=http_client, timeout=timeout)
    return client, cfg["model"]


def _call_mission_assist(cfg: Dict, prompt: str, timeout: float = 120.0) -> Optional[str]:
    """Mission Assist text completion with a JSON-enforcing system prompt."""
    client, model = _openai_client(cfg, timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _MISSION_ASSIST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=8192,
    )
    return response.choices[0].message.content


def _encode_image_data_url(image_path: str, max_side: int = 2048) -> str:
    """Downscale (long edge) + base64-encode an image as a data: URL for vision calls."""
    import base64
    import io
    import mimetypes
    if max_side and max_side > 0:
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        raw, mime = buf.getvalue(), "image/jpeg"
    else:
        with open(image_path, "rb") as fh:
            raw = fh.read()
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# Cache for resolved Mission Assist model names (keyed by API URL)
_mission_assist_model_cache: Dict[str, str] = {}


def _resolve_mission_assist_model(client: openai.OpenAI, api_url: str) -> str:
    """
    Resolve the actual model name from a Mission Assist endpoint.
    Caches the result so models.list() is only called once per endpoint.
    """
    if api_url in _mission_assist_model_cache:
        return _mission_assist_model_cache[api_url]
    
    print(f"  [LLM] Resolving model for endpoint: {api_url}")
    available_models = client.models.list()
    if not available_models.data:
        raise ValueError(f"No models available at endpoint: {api_url}")
    
    model_name = available_models.data[0].id
    _mission_assist_model_cache[api_url] = model_name
    print(f"  [LLM] Resolved model: {model_name}")
    return model_name


# =============================================================================
# Local transformers provider
# =============================================================================

# Cache for loaded transformers models, keyed by local model path.
# Loading multi-GB weights on every call is not viable, so the bundle
# (processor + model) is held at module scope for the life of the process.
_transformers_cache: Dict[str, Any] = {}


def _load_transformers_model(model_path: str) -> Dict[str, Any]:
    """
    Load a local transformers model and processor, caching the result.

    Returns a dict with 'processor' and 'model' keys. Subsequent calls
    with the same path return the cached bundle without reloading weights.
    """
    if model_path in _transformers_cache:
        return _transformers_cache[model_path]

    print(f"  [LLM] Loading transformers model from {model_path} (one-time)...")
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0", #"auto"
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )

    # Diagnostics — confirm where the model actually landed.
    device_info = model.hf_device_map if hasattr(model, "hf_device_map") else model.device
    print(f"  [LLM] Device map: {device_info}")
    print(f"  [LLM] Model dtype: {next(model.parameters()).dtype}")
    print(f"  [LLM] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [LLM] CUDA device: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [LLM] VRAM: {vram_gb:.1f} GB")

    bundle = {"processor": processor, "model": model}
    _transformers_cache[model_path] = bundle
    print(f"  [LLM] Model loaded and cached.")
    return bundle


def _call_transformers(
    prompt: str,
    model_path: str,
    max_new_tokens: int = 1024,
) -> Optional[str]:
    """
    Run a prompt through a locally-loaded transformers model.

    The model is loaded once per process and cached in `_transformers_cache`.
    Uses greedy decoding (do_sample=False) to match the temperature=0 behavior
    of the other providers. Thinking mode is disabled — for structured
    extraction tasks (classify, title) it adds latency without improving
    output quality, and Gemma E4B follows the user prompt directly without
    needing a JSON-enforcement system prompt.
    """
    bundle = _load_transformers_model(model_path)
    processor = bundle["processor"]
    model = bundle["model"]

    messages = [
        {"role": "user", "content": prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)