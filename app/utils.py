import json
import os
import re
import requests
from typing import Optional, Any, Dict, List
from langchain_openai import OpenAI
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


def call_llm(
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str = None,
    provider: str = "ollama",
    timeout: float = 120.0
) -> Optional[str]:
    """
    Sends a prompt to either Ollama or Mission Assist API.
    """
    if provider == "ollama":
        return _call_ollama(prompt, model_name, base_url, timeout=timeout)
    elif provider == "mission_assist":
        if not api_key:
            raise ValueError("api_key is required for mission_assist provider")
        return _call_mission_assist(prompt, model_name, base_url, api_key, timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'mission_assist'")


def _call_ollama(
    prompt: str,
    model_name: str,
    base_url: str,
    timeout: float = 120.0
) -> Optional[str]:
    """
    Sends a prompt to Ollama and expects a JSON object in the response.
    """
    # Ensure URL has a scheme — requests requires it
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    
    api_url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()

    if response.status_code == 200:
        response_data = response.json()
        raw_llm_content = response_data.get("message", {}).get("content")
        return raw_llm_content


def _call_mission_assist(
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str,
    timeout: float = 120.0
) -> Optional[str]:
    """
    Sends a prompt to Mission Assist API using the openai client and chat completions endpoint.
    """
    if model_name == "gpt-oss":
        url_model_segment = "gptoss"
    else:
        url_model_segment = model_name

    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    
    api_url = f"{base_url.rstrip('/')}/bae-api-{url_model_segment}/v1"
    headers = {"apikey": api_key}

    client = openai.OpenAI(
        default_headers=headers,
        api_key=api_key,
        base_url=api_url,
        timeout=timeout
    )

    # Resolve model name once and cache it
    client_model_name = _resolve_mission_assist_model(client, api_url)

    system_prompt = """You are an expert-level JSON generation API. Your sole purpose is to respond with a single, valid JSON object or array.

**CRITICAL RESPONSE DIRECTIVES:**

1.  **JSON ONLY:** Your entire response MUST be a single, valid JSON object or array, enclosed in a markdown code block.
2.  **NO EXTRA TEXT:** You MUST NOT include any other text, explanations, reasoning, or conversational filler. The response must contain ONLY the JSON.
3.  **STRICT SYNTAX:** The response must start *exactly* with ```json on a new line and end *exactly* with ``` on a new line. There must be no characters before the opening ```json or after the closing ```.

**DO NOT INCLUDE YOUR REASONING OR THOUGHTS IN THE FINAL RESPONSE.**
"""
    
    response = client.chat.completions.create(
        model=client_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=8192
    )
    
    return response.choices[0].message.content


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