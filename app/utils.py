import json
import os
import re
import requests
from typing import Optional, Any, Dict, List
from langchain_openai import OpenAI
import openai

def _attempt_json_repair(json_string: str, error: json.JSONDecodeError) -> Optional[str]:
    """
    Tries to repair a JSON string by escaping an unescaped quote at the error position.
    
    Args:
        json_string: The malformed JSON string.
        error: The JSONDecodeError exception object.
        
    Returns:
        A repaired JSON string or None if the error is not a simple quote issue.
    """
    # The error message for an unescaped quote often includes "char".
    # We focus on this common, fixable error.
    if "char" in error.msg:
        # Point to the character that caused the error
        error_pos = error.pos
        
        # Insert a backslash before the problematic character (likely a quote)
        repaired_string = f"{json_string[:error_pos]}\\{json_string[error_pos:]}"
        
        print(f"--- INFO: Attempting JSON repair at position {error_pos}. ---")
        return repaired_string
        
    return None

def save_results_to_json(results: List[Dict[str, Any]], file_path: str):
    """
    Saves the classification results to a JSON file.

    Args:
        results (List[Dict[str, Any]]): A list of dictionaries containing the classification results.
        file_path (str): The path to the output JSON file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the results to the JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Output successfully saved to {file_path}")

def load_pages_from_json(file_path: str) -> Dict[str, str]:
    """
    Reads page data from a JSON file, intelligently handling two different formats.

    Format 1 ("Old"): A top-level "pages" key containing a dictionary of page_num: page_text.
    Format 2 ("New"): A root object where each key is a page number and the value is a
                     dictionary containing a "page_text" key.

    Args:
        file_path: The path to the source OCR JSON file.

    Returns:
        A standardized dictionary mapping page numbers (str) to page text (str).
        Returns an empty dictionary if the file is invalid or contains no pages.
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
    # This uses the logic from your original function.
    elif all(isinstance(val, dict) and "page_text" in val for val in data.values()):
        print(f"  - Detected nested 'page_text' format in {file_path}.")
        for page_num, content_dict in data.items():
            # Use page_Id if available, otherwise use the key.
            page_key = str(content_dict.get('page_Id', page_num))
            pages_output[page_key] = content_dict.get("page_text", "")
        return pages_output
        
    else:
        print(f"Warning: Unrecognized JSON structure in {file_path}. No pages loaded.")
        return {}

def _extract_json_from_llm_string(text: str) -> Optional[str]:
    """
    Attempts to extract a JSON string from LLM output.
    Handles markdown code blocks and tries to find the main JSON object or array.
    """
    if not text:
        return None

    text = text.strip()

    # Try to find complete JSON code blocks with proper closing
    match_json_block = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if match_json_block:
        return match_json_block.group(1).strip()

    # Try to find incomplete JSON code blocks (missing closing backticks)
    match_incomplete_json = re.search(r"```json\s*(\[.*?\]|\{.*?\})", text, re.DOTALL)
    if match_incomplete_json:
        potential_json = match_incomplete_json.group(1).strip()
        if _is_valid_json(potential_json):
            return potential_json

    # Try generic code blocks
    match_block = re.search(r"```\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if match_block:
        return match_block.group(1).strip()

    # Check if entire text is valid JSON
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        if _is_valid_json(text):
            return text
            
    # Try to extract JSON object or array from anywhere in the text
    # First try arrays
    array_matches = re.finditer(r'\[.*?\]', text, re.DOTALL)
    for match in array_matches:
        potential_json = match.group(0).strip()
        if _is_valid_json(potential_json):
            return potential_json
    
    # Then try objects
    object_matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    for match in object_matches:
        potential_json = match.group(0).strip()
        if _is_valid_json(potential_json):
            return potential_json
        
    print(f"Debug: Could not identify a clear JSON structure in text: '{text[:500]}...'")
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


def call_llm(
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str = None,
    provider: str = "ollama"
) -> Optional[str]:
    """
    Sends a prompt to either Ollama or Mission Assist API.
    
    Args:
        prompt: The prompt to send
        model_name: Model name to use
        base_url: Base URL for the API
        api_key: API key (required for mission_assist)
        provider: Either "ollama" or "mission_assist"
    
    Returns:
        Raw LLM response content as string
    """
    if provider == "ollama":
        return _call_ollama(prompt, model_name, base_url)
    elif provider == "mission_assist":
        if not api_key:
            raise ValueError("api_key is required for mission_assist provider")
        return _call_mission_assist(prompt, model_name, base_url, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'mission_assist'")


def _call_ollama(
    prompt: str,
    model_name: str,
    base_url: str
) -> Optional[str]:
    """
    Sends a prompt to Ollama and expects a JSON object in the response.
    """
    api_url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = requests.post(api_url, json=payload, timeout=600)
    response.raise_for_status()

    if response.status_code == 200:
        response_data = response.json()
        raw_llm_content = response_data.get("message", {}).get("content")
        return raw_llm_content


def _call_mission_assist(
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str
) -> Optional[str]:
    """
    Sends a prompt to Mission Assist API using the openai client and chat completions endpoint.
    """
    # The URL for gpt-oss is a special case and does not contain a hyphen in its name.
    if model_name == "gpt-oss":
        url_model_segment = "gptoss"
    else:
        url_model_segment = model_name

    api_url = f"{base_url.rstrip('/')}/bae-api-{url_model_segment}/v1"
    headers = {"apikey": api_key}

    client = openai.OpenAI(
        default_headers=headers,
        api_key=api_key,
        base_url=api_url
    )

    # Fetch the available model from the endpoint to ensure the correct ID is used.
    available_models = client.models.list()
    if not available_models.data:
        raise ValueError(f"No models available at endpoint: {api_url}")
    
    client_model_name = available_models.data[0].id
    #print(f"Using Mission Assist model: {client_model_name}")

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