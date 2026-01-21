"""
classify_agent.py - Determines where main content begins in a document.

This agent classifies pages to find where actual content starts, handling:
- Standard format: Title Page → Table of Contents → Content
- No ToC: Title Page → Content directly  
- No Title: Content starts on page 1
- Stub documents: Single page redirects to another document

OPTIMIZATION: Uses regex-based fast-path detection for Table of Contents pages
to reduce reliance on smaller LLMs. A page with 10+ section numbers (like 1.1.1,
2.3.4) is automatically classified as TOC without LLM calls.

The flattened text is only used internally for LLM classification - the raw
OCR structure is preserved for downstream processing.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .utils import _extract_json_from_llm_string, call_llm, save_results_to_json
except ImportError:
    from utils import _extract_json_from_llm_string, call_llm, save_results_to_json


# =============================================================================
# REGEX-BASED FAST-PATH CLASSIFICATION
# =============================================================================

# Pattern for hierarchical section numbers: 1.1, 1.1.1, 2.3.4.5, etc.
# Must have at least one dot to distinguish from simple page numbers
SECTION_NUMBER_PATTERN = re.compile(
    r'\b(\d{1,3}(?:\.\d{1,3}){1,5})\b'  # e.g., 1.1, 1.1.1, 2.3.4.5
)

# Pattern for TOC-style entries: section number followed by title and page number
# e.g., "1.1 Introduction ..... 5" or "3.2.1 Requirements 12"
TOC_ENTRY_PATTERN = re.compile(
    r'\b\d{1,2}(?:\.\d{1,2}){1,4}\s+[A-Z][A-Za-z\s\-/&]+[\s\.]*\d{1,4}\b'
)

# Pattern for "Table of Contents", "Contents", "TOC" headers
TOC_HEADER_PATTERN = re.compile(
    r'\b(?:TABLE\s+OF\s+CONTENTS?|CONTENTS?|TOC)\b',
    re.IGNORECASE
)

# Pattern for "List of Figures" / "Table of Figures" / "List of Tables" / "Table of Tables"
LIST_OF_PATTERN = re.compile(
    r'\b(?:LIST\s+OF|TABLE\s+OF)\s+(?:FIGURES?|TABLES?|ILLUSTRATIONS?)\b',
    re.IGNORECASE
)

# Minimum threshold for section numbers to auto-classify as TOC
MIN_SECTION_NUMBERS_FOR_TOC = 10

# Minimum threshold for TOC-style entries
MIN_TOC_ENTRIES_FOR_TOC = 8


def fast_classify_page(page_text: str) -> Optional[str]:
    """
    Attempts to classify a page using regex patterns (fast-path).
    
    Returns:
        - "TABLE_OF_CONTENTS" if high confidence TOC detected
        - "TITLE_PAGE" if high confidence title page detected
        - None if uncertain (fall back to LLM)
    
    This function is designed to be CONSERVATIVE - it only returns a 
    classification when very confident. Uncertain cases go to the LLM.
    """
    if not page_text or len(page_text) < 50:
        return None
    
    text_upper = page_text.upper()
    
    # ==========================================================================
    # Check for Table of Contents (most reliable fast-path)
    # ==========================================================================
    
    # Count hierarchical section numbers (1.1, 1.2.3, etc.)
    section_numbers = SECTION_NUMBER_PATTERN.findall(page_text)
    unique_section_numbers = set(section_numbers)
    
    # Count TOC-style entries (section + title + page number)
    toc_entries = TOC_ENTRY_PATTERN.findall(page_text)
    
    # Check for explicit TOC header
    has_toc_header = bool(TOC_HEADER_PATTERN.search(page_text))
    
    # Check for List of Figures/Tables header (still TOC-like)
    has_list_of_header = bool(LIST_OF_PATTERN.search(page_text))
    
    # Decision logic for TOC:
    # 1. Many unique section numbers (10+) → definitely TOC
    # 2. TOC header + several section numbers (5+) → definitely TOC  
    # 3. Many TOC-style entries (8+) → definitely TOC
    # 4. List of Figures/Tables header → likely TOC (but let LLM confirm for edge cases)
    
    if len(unique_section_numbers) >= MIN_SECTION_NUMBERS_FOR_TOC:
        return "TABLE_OF_CONTENTS"
    
    if has_toc_header and len(unique_section_numbers) >= 5:
        return "TABLE_OF_CONTENTS"
    
    if len(toc_entries) >= MIN_TOC_ENTRIES_FOR_TOC:
        return "TABLE_OF_CONTENTS"
    
    # For List of Figures/Tables, we have the header but numbering is simpler
    # Still auto-classify if we have the header AND multiple numbered items
    if has_list_of_header:
        # Look for figure/table numbers: Figure 1, Table 2, etc.
        fig_tab_numbers = re.findall(r'\b(?:Figure|Table|Fig\.?)\s*\d+', page_text, re.IGNORECASE)
        if len(fig_tab_numbers) >= 5:
            return "TABLE_OF_CONTENTS"
    
    # ==========================================================================
    # Check for obvious Title Page characteristics  
    # ==========================================================================
    
    # Title pages often have these markers
    title_indicators = [
        r'\bDISTRIBUTION\s+STATEMENT\b',
        r'\bAPPROVED\s+FOR\s+(?:PUBLIC\s+)?RELEASE\b',
        r'\bUNCLASSIFIED\b',
        r'\bCONTROLLED\s+UNCLASSIFIED\b',
        r'\bEXPORT\s+CONTROLLED\b',
        r'\bPROPRIETARY\b',
        r'\bCONFIDENTIAL\b',
        r'\bSECRET\b',
        r'\bCONTRACT\s+(?:NO\.?|NUMBER)',
        r'\bPREPARED\s+(?:BY|FOR)\b',
        r'\bSPECIFICATION\s+(?:NO\.?|NUMBER)',
        r'\bREVISION\s+[A-Z0-9]+\b',
    ]
    
    title_score = sum(1 for pattern in title_indicators if re.search(pattern, text_upper))
    
    # If we have 3+ title indicators and very few section numbers, likely title page
    if title_score >= 3 and len(unique_section_numbers) <= 2:
        return "TITLE_PAGE"
    
    # ==========================================================================
    # No confident fast-path classification - return None to use LLM
    # ==========================================================================
    return None


def get_fast_classification_reason(page_text: str) -> str:
    """
    Returns a human-readable reason for the fast classification.
    Used for debugging/logging.
    """
    section_numbers = SECTION_NUMBER_PATTERN.findall(page_text)
    unique_section_numbers = set(section_numbers)
    toc_entries = TOC_ENTRY_PATTERN.findall(page_text)
    has_toc_header = bool(TOC_HEADER_PATTERN.search(page_text))
    has_list_of_header = bool(LIST_OF_PATTERN.search(page_text))
    
    if len(unique_section_numbers) >= MIN_SECTION_NUMBERS_FOR_TOC:
        return f"Found {len(unique_section_numbers)} unique section numbers"
    
    if has_toc_header and len(unique_section_numbers) >= 5:
        return f"TOC header + {len(unique_section_numbers)} section numbers"
    
    if len(toc_entries) >= MIN_TOC_ENTRIES_FOR_TOC:
        return f"Found {len(toc_entries)} TOC-style entries"
    
    if has_list_of_header:
        fig_tab_numbers = re.findall(r'\b(?:Figure|Table|Fig\.?)\s*\d+', page_text, re.IGNORECASE)
        if len(fig_tab_numbers) >= 5:
            return f"List of Figures/Tables header + {len(fig_tab_numbers)} items"
    
    return "No fast-path match"


# =============================================================================
# ORIGINAL FUNCTIONS (with fast-path integration)
# =============================================================================

def flatten_page_text(page_dict: Dict) -> str:
    """
    Flattens a page_dict into a single text string for LLM classification.
    This is ONLY used for classification - not for downstream processing.
    """
    if not page_dict:
        return ""
    
    # Handle wrapped structure
    if 'page_dict' in page_dict and isinstance(page_dict['page_dict'], dict):
        page_dict = page_dict['page_dict']
    
    text_list = page_dict.get('text', [])
    if isinstance(text_list, list):
        return " ".join(str(t) for t in text_list)
    return str(text_list) if text_list else ""


def get_page_type_with_llm(page_text: str, llm_config: Dict) -> str:
    """
    Uses an LLM to classify a single page into one of four types:
    - TITLE_PAGE
    - TABLE_OF_CONTENTS
    - CONTENT_BODY  
    - AMBIGUOUS
    """
    prompt = f"""
You are a document page classifier. Your job is to classify the following page text into one of four specific types:

1.  `TITLE_PAGE`: The page is a cover/title page. It typically contains a document title, document numbers, dates, approval information, distribution statements, warnings, or organizational logos/names. It does NOT contain the main body content or a list of sections.

2.  `TABLE_OF_CONTENTS`: The page is a list of sections, figures, or tables, often with page numbers. It shows the document structure but does NOT contain full paragraphs of body text.

3.  `CONTENT_BODY`: The page contains full, descriptive paragraphs and sentences. It is the main substance of the document. A heading like "1.0 SCOPE" or "1. INTRODUCTION" might be present, followed by paragraph text explaining the content.

4.  `AMBIGUOUS`: The page is unclear, mostly empty, or doesn't fit the other categories.

Analyze the text and respond with a single JSON object containing one key, "page_type", with one of the four values.

Page Text:
---
{page_text[:3000]}
---

Example Output:
{{"page_type": "CONTENT_BODY"}}
"""
    try:
        llm_response = call_llm(
            prompt,
            llm_config['model_name'],
            llm_config['base_url'],
            llm_config['api_key'],
            llm_config['provider']
        )
    except Exception as e:
        print(f"    [Error] LLM Call failed: {e}")
        return "AMBIGUOUS"
    
    if not llm_response:
        return "AMBIGUOUS"

    # Try to extract JSON from the response
    json_string = _extract_json_from_llm_string(llm_response)
    
    # If extraction returned nothing, try the raw response (sometimes models output just JSON)
    if not json_string:
        json_string = llm_response

    try:
        response_data = json.loads(json_string)
        return response_data.get("page_type", "AMBIGUOUS")
    except Exception as e:
        return "AMBIGUOUS"


def classify_page_smart(page_text: str, llm_config: Dict) -> Tuple[str, str]:
    """
    Smart classification that tries fast-path first, then falls back to LLM.
    
    Returns:
        Tuple of (page_type, method) where method is "fast" or "llm"
    """
    # Try fast-path first
    fast_result = fast_classify_page(page_text)
    if fast_result:
        return (fast_result, "fast")
    
    # Fall back to LLM
    llm_result = get_page_type_with_llm(page_text, llm_config)
    return (llm_result, "llm")


def check_if_stub_document(page_text: str, llm_config: Dict) -> Optional[str]:
    """
    Checks if a single-page document is a stub/redirect.
    Returns the redirect target if it is, None otherwise.
    """
    prompt = f"""
You are analyzing a single-page document to determine if it is a "stub" or "redirect" document.

A stub document is one that does NOT contain actual content, but instead tells the reader to refer to a different document, website, or location for the real information.

Common patterns include:
- "This document has been superseded by..."
- "Refer to [document name/number] for..."
- "See [URL or location] for current information"
- "This specification is now maintained at..."
- "Replaced by..."

Analyze the text and respond with a JSON object:
- If it IS a stub/redirect: {{"is_stub": true, "redirect_to": "the document or location it refers to"}}
- If it is NOT a stub: {{"is_stub": false, "redirect_to": null}}

Page Text:
---
{page_text[:2000]}
---
"""
    try:
        llm_response = call_llm(
            prompt,
            llm_config['model_name'],
            llm_config['base_url'],
            llm_config['api_key'],
            llm_config['provider']
        )
    except Exception as e:
        print(f"    [Error] Stub check LLM call failed: {e}")
        return None
    
    if not llm_response:
        return None

    json_string = _extract_json_from_llm_string(llm_response)
    if not json_string:
        json_string = llm_response

    try:
        response_data = json.loads(json_string)
        if response_data.get("is_stub", False):
            return response_data.get("redirect_to", "unknown")
        return None
    except Exception:
        return None


def classify_page_wrapper(args: Tuple[int, str, Dict]) -> Dict:
    """
    Wrapper for parallel execution of page classification.
    Now uses smart classification with fast-path.
    """
    page_num, page_text, llm_config = args
    page_type, method = classify_page_smart(page_text, llm_config)
    return {'page': page_num, 'type': page_type, 'method': method}


def load_pages_for_classification(input_path: str) -> List[Tuple[int, str]]:
    """
    Loads raw OCR and returns a list of (page_id, flattened_text) for classification.
    """
    if not os.path.exists(input_path):
        return []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  - [Error] Failed to load input file {input_path}: {e}")
        print(f"  - [Action] Skipping classification for this file.")
        return []
    
    pages = []
    
    if isinstance(data, dict):
        for key, val in data.items():
            try:
                page_id = int(key)
            except ValueError:
                digits = re.findall(r'\d+', str(key))
                page_id = int(digits[0]) if digits else 0
            
            # Get page_dict if wrapped
            page_dict = val.get('page_dict', val) if isinstance(val, dict) else val
            flat_text = flatten_page_text(page_dict) if isinstance(page_dict, dict) else ""
            pages.append((page_id, flat_text))
            
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            page_id = item.get('page_Id') or item.get('page_num') or (idx + 1)
            try:
                page_id = int(page_id)
            except (ValueError, TypeError):
                page_id = idx + 1
            
            page_dict = item.get('page_dict', item) if isinstance(item, dict) else item
            flat_text = flatten_page_text(page_dict) if isinstance(page_dict, dict) else ""
            pages.append((page_id, flat_text))
    
    pages.sort(key=lambda x: x[0])
    return pages


def find_content_start_page(
    input_path: str, 
    llm_config: Dict, 
    max_workers: int = 4,
    max_pages_to_check: int = 20
) -> Dict[str, Any]:
    """
    Analyzes the first N pages to find where main content begins.
    
    Uses a two-tier approach:
    1. Fast-path regex classification for obvious TOC/Title pages
    2. LLM classification for uncertain pages
    
    Returns a dict with:
        - content_start_page: int
        - document_type: "standard" | "no_toc" | "no_title" | "stub" | "unknown"
        - is_stub: bool
        - stub_redirect: str or None
        - page_classifications: list of {page, type, method} for debugging
    """
    pages = load_pages_for_classification(input_path)
    
    if not pages:
        print(f"  - [Warning] No pages found in {input_path}")
        return {
            "content_start_page": 1,
            "document_type": "unknown",
            "is_stub": False,
            "stub_redirect": None,
            "page_classifications": []
        }
    
    total_pages = len(pages)
    print(f"  - Document has {total_pages} page(s)")
    
    # Special case: single page document - check if it's a stub
    if total_pages == 1:
        page_id, page_text = pages[0]
        print(f"  - Single page document, checking if stub...")
        
        stub_redirect = check_if_stub_document(page_text, llm_config)
        if stub_redirect:
            print(f"  - STUB DOCUMENT detected. Redirects to: {stub_redirect}")
            return {
                "content_start_page": page_id,
                "document_type": "stub",
                "is_stub": True,
                "stub_redirect": stub_redirect,
                "page_classifications": [{'page': page_id, 'type': 'STUB', 'method': 'llm'}]
            }
        
        # Not a stub - classify the single page
        page_type, method = classify_page_smart(page_text, llm_config)
        print(f"    Page {page_id}: {page_type} [{method}]")
        
        return {
            "content_start_page": page_id,
            "document_type": "no_title" if page_type == "CONTENT_BODY" else "unknown",
            "is_stub": False,
            "stub_redirect": None,
            "page_classifications": [{'page': page_id, 'type': page_type, 'method': method}]
        }
    
    # Multi-page document: classify first N pages (INCLUDING page 1)
    pages_to_classify = pages[:max_pages_to_check]
    
    print(f"  - Classifying pages 1-{len(pages_to_classify)} (fast-path + LLM fallback)...")
    
    # ==========================================================================
    # Phase 1: Fast-path classification (synchronous, very fast)
    # ==========================================================================
    page_classifications = []
    pages_needing_llm = []
    
    for page_id, page_text in pages_to_classify:
        fast_result = fast_classify_page(page_text)
        if fast_result:
            reason = get_fast_classification_reason(page_text)
            print(f"    Page {page_id}: {fast_result} [fast: {reason}]")
            page_classifications.append({
                'page': page_id, 
                'type': fast_result, 
                'method': 'fast'
            })
        else:
            pages_needing_llm.append((page_id, page_text))
    
    # ==========================================================================
    # Phase 2: LLM classification for uncertain pages (parallel)
    # ==========================================================================
    if pages_needing_llm:
        print(f"  - {len(pages_needing_llm)} page(s) need LLM classification...")
        
        tasks = [(page_id, page_text, llm_config) for page_id, page_text in pages_needing_llm]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(classify_page_wrapper, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    page_classifications.append(result)
                    print(f"    Page {result['page']}: {result['type']} [llm]")
                except Exception as exc:
                    page_num = future_to_task[future][0]
                    print(f"    Page {page_num}: ERROR - {exc}")
                    page_classifications.append({
                        'page': page_num, 
                        'type': 'AMBIGUOUS',
                        'method': 'error'
                    })
    
    # Sort by page number (results may arrive out of order)
    page_classifications.sort(key=lambda x: x['page'])
    
    # Log summary
    fast_count = sum(1 for p in page_classifications if p.get('method') == 'fast')
    llm_count = sum(1 for p in page_classifications if p.get('method') == 'llm')
    print(f"  - Classification summary: {fast_count} fast-path, {llm_count} LLM")
    
    # Determine content start page and document type
    return analyze_page_sequence(page_classifications)


def analyze_page_sequence(page_classifications: List[Dict]) -> Dict[str, Any]:
    """
    Analyzes the sequence of page classifications to determine:
    - Where content starts
    - What type of document structure this is
    
    Handles:
    - Standard: TITLE -> TOC -> CONTENT
    - No ToC: TITLE -> CONTENT
    - No Title: CONTENT starts on page 1
    - Various edge cases
    """
    if not page_classifications:
        return {
            "content_start_page": 1,
            "document_type": "unknown",
            "is_stub": False,
            "stub_redirect": None,
            "page_classifications": []
        }
    
    # Check if page 1 is content (no title page case)
    first_page = page_classifications[0]
    if first_page['type'] == 'CONTENT_BODY':
        print(f"  - Page 1 is content. No title page detected.")
        return {
            "content_start_page": first_page['page'],
            "document_type": "no_title",
            "is_stub": False,
            "stub_redirect": None,
            "page_classifications": page_classifications
        }
    
    # Look for transitions in the page sequence
    for i in range(1, len(page_classifications)):
        prev_type = page_classifications[i-1]['type']
        curr_type = page_classifications[i]['type']
        curr_page = page_classifications[i]['page']
        
        # Standard case: TOC -> CONTENT
        if prev_type == 'TABLE_OF_CONTENTS' and curr_type == 'CONTENT_BODY':
            print(f"  - Transition: TOC -> Content at page {curr_page}")
            return {
                "content_start_page": curr_page,
                "document_type": "standard",
                "is_stub": False,
                "stub_redirect": None,
                "page_classifications": page_classifications
            }
        
        # No ToC case: TITLE -> CONTENT
        if prev_type == 'TITLE_PAGE' and curr_type == 'CONTENT_BODY':
            print(f"  - Transition: Title -> Content at page {curr_page} (no ToC)")
            return {
                "content_start_page": curr_page,
                "document_type": "no_toc",
                "is_stub": False,
                "stub_redirect": None,
                "page_classifications": page_classifications
            }
        
        # AMBIGUOUS -> CONTENT (treat ambiguous as possible title/intro)
        if prev_type == 'AMBIGUOUS' and curr_type == 'CONTENT_BODY':
            print(f"  - Transition: Ambiguous -> Content at page {curr_page}")
            return {
                "content_start_page": curr_page,
                "document_type": "no_toc",
                "is_stub": False,
                "stub_redirect": None,
                "page_classifications": page_classifications
            }
    
    # Fallback: find the first CONTENT_BODY page
    for classification in page_classifications:
        if classification['type'] == 'CONTENT_BODY':
            print(f"  - Fallback: First content page is {classification['page']}")
            return {
                "content_start_page": classification['page'],
                "document_type": "unknown",
                "is_stub": False,
                "stub_redirect": None,
                "page_classifications": page_classifications
            }
    
    # Last resort: if we only have TITLE/TOC/AMBIGUOUS, start after page 1
    # This handles cases where classification might have missed content pages
    if len(page_classifications) > 1:
        # Start at page 2 if page 1 looks like a title
        if first_page['type'] in ('TITLE_PAGE', 'AMBIGUOUS'):
            fallback_page = page_classifications[1]['page']
        else:
            fallback_page = first_page['page']
    else:
        fallback_page = first_page['page']
    
    print(f"  - [Warning] No clear content found. Defaulting to page {fallback_page}")
    return {
        "content_start_page": fallback_page,
        "document_type": "unknown",
        "is_stub": False,
        "stub_redirect": None,
        "page_classifications": page_classifications
    }


def run_classification_on_file(
    input_path: str, 
    output_path: str, 
    llm_config: Dict, 
    max_workers: int = 4
) -> int:
    """
    Runs classification and saves a result file with document structure info.
    Returns the content start page for backward compatibility.
    """
    result = find_content_start_page(input_path, llm_config, max_workers)
    
    # Add source file info
    result["source_file"] = os.path.basename(input_path)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
    print(f"  - Classification saved to {output_path}")
    print(f"  - Document type: {result['document_type']}, Content starts: page {result['content_start_page']}")
    
    return result['content_start_page']


def load_content_start_page(classification_path: str, default: int = 1) -> int:
    """
    Loads the content start page from a classification result file.
    Default changed to 1 (safer for documents without title pages).
    """
    if not os.path.exists(classification_path):
        return default
    
    try:
        with open(classification_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('content_start_page', default)
    except (json.JSONDecodeError, KeyError):
        return default


def load_classification_result(classification_path: str) -> Optional[Dict]:
    """
    Loads the full classification result for more detailed handling.
    """
    if not os.path.exists(classification_path):
        return None
    
    try:
        with open(classification_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


# =============================================================================
# TESTING / DEBUGGING UTILITIES
# =============================================================================

def test_fast_classification(text: str) -> None:
    """
    Test the fast classification on a text sample.
    Useful for debugging/tuning thresholds.
    """
    print("=" * 60)
    print("FAST CLASSIFICATION TEST")
    print("=" * 60)
    
    # Find section numbers
    section_numbers = SECTION_NUMBER_PATTERN.findall(text)
    unique_sections = set(section_numbers)
    print(f"\nSection numbers found ({len(section_numbers)} total, {len(unique_sections)} unique):")
    for s in sorted(unique_sections)[:20]:
        print(f"  - {s}")
    if len(unique_sections) > 20:
        print(f"  ... and {len(unique_sections) - 20} more")
    
    # Find TOC entries
    toc_entries = TOC_ENTRY_PATTERN.findall(text)
    print(f"\nTOC-style entries found ({len(toc_entries)}):")
    for entry in toc_entries[:10]:
        print(f"  - {entry}")
    if len(toc_entries) > 10:
        print(f"  ... and {len(toc_entries) - 10} more")
    
    # Check headers
    has_toc = bool(TOC_HEADER_PATTERN.search(text))
    has_list_of = bool(LIST_OF_PATTERN.search(text))
    print(f"\nHeaders detected:")
    print(f"  - TOC header: {has_toc}")
    print(f"  - List of Figures/Tables header: {has_list_of}")
    
    # Final result
    result = fast_classify_page(text)
    reason = get_fast_classification_reason(text)
    print(f"\nFAST CLASSIFICATION RESULT: {result or 'None (needs LLM)'}")
    print(f"REASON: {reason}")


if __name__ == '__main__':
    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3",
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test" and len(sys.argv) > 2:
            # Test fast classification on a file
            with open(sys.argv[2], 'r') as f:
                test_text = f.read()
            test_fast_classification(test_text)
        else:
            input_file = sys.argv[1]
            output_file = sys.argv[2] if len(sys.argv) > 2 else "classification_result.json"
            run_classification_on_file(input_file, output_file, llm_config)
    else:
        print("Usage:")
        print("  python classify_agent.py <input.json> [output.json]")
        print("  python classify_agent.py --test <text_file>  # Test fast classification")