"""
validation_agent.py - Validates extracted sections against Table of Contents.

This module compares the sections found in the final output against
what was listed in the original document's Table of Contents.

Outputs a diagnostic report with:
- Total match percentage
- Sections found in TOC but missing from output
- Sections found in output but not in TOC
- Summary statistics
"""

import os
import json
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# SECTION NUMBER PARSING - GREEDY VERSION FOR TOC
# =============================================================================
# Since we KNOW we're looking at TOC pages (already classified by classify_agent),
# we can be much more aggressive/greedy in finding section numbers.
# TOC entries typically look like:
#   1.1.1 ............ Some Title
#   1.1.1    Section Title
#   1.1.1 Section Title .... 5
# There are usually 10-30+ sections per TOC page.

# GREEDY pattern - finds section numbers ANYWHERE in text
# Matches: 1.1, 1.1.1, 2.3.4.5, etc. (must have at least one dot)
# Does NOT require start of line - OCR often mangles line breaks
SECTION_NUMBER_PATTERN_GREEDY = re.compile(
    r'(?<![0-9])(\d{1,2}(?:\.\d{1,3}){1,6})(?![0-9])',
    # (?<![0-9]) - not preceded by digit (avoids partial matches)
    # \d{1,2} - first component 1-2 digits
    # (?:\.\d{1,3}){1,6} - 1-6 more components of .X, .XX, or .XXX
    # (?![0-9]) - not followed by digit
)

# Pattern for top-level sections (just "1", "2", "3", etc.)
# These need SOME context to avoid matching page numbers everywhere
# Look for digit followed by space(s) and letter ON SAME LINE (title start)
TOP_LEVEL_SECTION_PATTERN = re.compile(
    r'(?<![0-9.])(\d{1,2})[ \t]+[A-Za-z]',  # "1 Introduction", "2 Scope"
    # No newline requirement - just needs letter after to confirm it's a section
    # Use [ \t]+ instead of \s+ to avoid matching across newlines
)


def normalize_section_number(raw: str) -> str:
    if not raw: return ""
    normalized = raw.strip().replace(',', '.').replace('..', '.')
    # Fix 1-1-3 -> 1.1.3
    while True:
        new_normalized = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
        if new_normalized == normalized: break
        normalized = new_normalized
    return normalized.rstrip('.')


def is_valid_section_number_strict(num: str) -> bool:
    """
    STRICT validation - used for non-TOC contexts.
    Filters out things that look like page numbers, dates, decimals, etc.
    """
    if not num:
        return False
    
    normalized = normalize_section_number(num)
    
    # Must have at least one digit
    if not any(c.isdigit() for c in normalized):
        return False
    
    # Split into parts
    parts = normalized.split('.')
    
    # Single numbers are valid (like "1", "2", "3")
    # But filter out likely page numbers (high single numbers at depth 1)
    if len(parts) == 1:
        try:
            val = int(parts[0])
            # Section numbers rarely go above 20 at the top level
            # Page numbers are often higher
            if val > 30:
                return False
            return True
        except ValueError:
            return False
    
    # For hierarchical numbers, check each part
    for part in parts:
        try:
            val = int(part)
            # Individual parts shouldn't be too large
            # (filters out things like 4.88 which are decimals)
            if val > 50:
                return False
        except ValueError:
            # Non-numeric part (like 'A' in '3.A') - could be valid
            if not re.match(r'^[A-Za-z]$', part):
                return False
    
    return True


def is_valid_section_number_greedy(num: str) -> bool:
    """
    GREEDY validation for TOC context - much more permissive.
    Since we know we're in a TOC, most X.Y.Z patterns are valid sections.
    """
    if not num:
        return False
    
    normalized = normalize_section_number(num)
    
    # Must have at least one digit
    if not any(c.isdigit() for c in normalized):
        return False
    
    parts = normalized.split('.')
    
    # Must have at least 2 parts for greedy mode (1.1 minimum)
    # This avoids matching standalone numbers like page numbers
    if len(parts) < 2:
        return False
    
    # Check each part is a reasonable number
    for part in parts:
        try:
            val = int(part)
            # Be generous - allow up to 999 for each component
            # Real sections can have high numbers like 10.2.15.3
            if val > 999:
                return False
        except ValueError:
            # Non-numeric part (like 'A' in '3.A') - allow single letters
            if not re.match(r'^[A-Za-z]{1,2}$', part):
                return False
    
    return True


# Keep old function name for compatibility with other code
def is_valid_section_number(num: str) -> bool:
    """Wrapper that uses strict validation by default."""
    return is_valid_section_number_strict(num)


def is_likely_decimal(num: str) -> bool:
    """
    Check if a number looks like a decimal value rather than a section number.
    E.g., 4.88, 2.50, 3.75
    
    STRICT version - used for non-TOC contexts.
    """
    parts = normalize_section_number(num).split('.')
    
    if len(parts) != 2:
        return False
    
    try:
        second_val = int(parts[1])
        # High second component (>30) with 2 digits suggests decimal
        if second_val > 30 and len(parts[1]) >= 2:
            return True
        # Trailing zero like 2.50
        if parts[1].endswith('0') and len(parts[1]) >= 2 and second_val > 30:
            return True
        # Leading zero like 1.05
        if parts[1].startswith('0') and len(parts[1]) >= 2:
            return True
    except ValueError:
        pass
    
    return False


def is_likely_decimal_greedy(num: str) -> bool:
    """
    GREEDY version for TOC - reject obvious decimals but be permissive.
    In TOC context, we want to accept more patterns as valid sections.
    
    Rejects: 4.88, 12.50, 1.05, 2.99, 3.75 (look like prices/measurements)
    Accepts: 1.1, 3.10, 3.21, 4.12, 10.1 (look like sections)
    """
    parts = normalize_section_number(num).split('.')
    
    # Only applies to X.Y format (two parts)
    if len(parts) != 2:
        return False
    
    try:
        first_val = int(parts[0])
        second_val = int(parts[1])
        second_str = parts[1]
        
        # Leading zero like 1.05, 2.00, 3.08 - likely decimal
        if second_str.startswith('0') and len(second_str) >= 2:
            return True
        
        # Two-digit second component that's high (>=50) suggests decimal
        # Section numbers can have X.21, X.32, X.45 etc.
        # But X.88, X.99, X.75 are almost certainly decimals
        if len(second_str) == 2 and second_val >= 50:
            return True
        
        # Exact cents-like patterns that are very common: .25, .50, .75, .99
        if len(second_str) == 2 and second_val in [25, 75, 99]:
            return True
            
    except ValueError:
        pass
    
    return False


# =============================================================================
# TOC PARSING - GREEDY EXTRACTION
# =============================================================================

def extract_sections_from_toc_text(toc_text: str) -> Set[str]:
    """
    Extract section numbers from TOC page text using GREEDY matching.
    
    Since we know this is TOC text (already classified), we can be aggressive
    in finding section numbers. TOC pages typically have 10-30+ sections.
    
    Args:
        toc_text: Flattened text from TOC pages
        
    Returns:
        Set of normalized section numbers found
    """
    sections = set()
    
    if not toc_text:
        return sections
    
    # ==========================================================================
    # PRIMARY: Find ALL X.Y.Z patterns anywhere in text (greedy)
    # This catches most section numbers without requiring newlines
    # ==========================================================================
    greedy_matches = SECTION_NUMBER_PATTERN_GREEDY.findall(toc_text)
    
    for match in greedy_matches:
        normalized = normalize_section_number(match)
        if normalized and is_valid_section_number_greedy(normalized):
            if not is_likely_decimal_greedy(normalized):
                sections.add(normalized)
    
    # ==========================================================================
    # SECONDARY: Top-level sections (1, 2, 3 followed by title text)
    # These don't have dots so need some context to avoid page numbers
    # ==========================================================================
    top_level_matches = TOP_LEVEL_SECTION_PATTERN.findall(toc_text)
    
    for match in top_level_matches:
        normalized = normalize_section_number(match)
        if normalized:
            try:
                val = int(normalized)
                # Top-level sections are usually 1-30
                if 1 <= val <= 30:
                    sections.add(normalized)
            except ValueError:
                pass
    
    return sections
    
    return sections


def load_toc_pages_text(raw_ocr_path: str, classification_path: str) -> Tuple[str, bool]:
    """
    Load and combine text from TOC pages.
    
    Uses classification results to identify which pages are TOC,
    then extracts their text from the raw OCR.
    
    Returns:
        Tuple of (toc_text, has_toc_in_classification)
        - toc_text: Combined text from TOC pages
        - has_toc_in_classification: True if classify_agent found TOC pages
    """
    # Load classification to find TOC pages
    toc_page_numbers = set()
    explicit_toc_found = False  # Track if classify_agent explicitly found TOC
    content_start_page = 1
    
    if os.path.exists(classification_path):
        try:
            with open(classification_path, 'r', encoding='utf-8') as f:
                classification = json.load(f)
            
            content_start_page = classification.get('content_start_page', 1)
            
            # Find all pages classified as TOC
            for page_class in classification.get('page_classifications', []):
                p_type = page_class.get('type')
                if p_type == 'TABLE_OF_CONTENTS':
                    toc_page_numbers.add(page_class.get('page'))
                    explicit_toc_found = True  # classify_agent found TOC!
                # Also include ambiguous pages as candidates if no explicit TOC found yet
                elif p_type == 'AMBIGUOUS' and not toc_page_numbers:
                    toc_page_numbers.add(page_class.get('page'))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    [Warning] Could not parse classification file: {e}")
    
    # If no explicit TOC pages found, assume pages 2 to content_start-1 are TOC
    if not toc_page_numbers and content_start_page > 2:
        toc_page_numbers = set(range(2, content_start_page))
    
    if not toc_page_numbers:
        print("    [Warning] No TOC pages identified")
        return "", False
    
    # Load raw OCR and extract TOC page text
    if not os.path.exists(raw_ocr_path):
        print(f"    [Warning] Raw OCR file not found: {raw_ocr_path}")
        return "", explicit_toc_found
    
    try:
        with open(raw_ocr_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    [Warning] Could not load raw OCR: {e}")
        return "", explicit_toc_found
    
    toc_texts = []
    
    # Handle different data structures
    if isinstance(raw_data, dict):
        for key, val in raw_data.items():
            try:
                page_num = int(key)
            except ValueError:
                digits = re.findall(r'\d+', str(key))
                page_num = int(digits[0]) if digits else 0
            
            if page_num in toc_page_numbers:
                page_dict = val.get('page_dict', val) if isinstance(val, dict) else val
                if isinstance(page_dict, dict):
                    text_list = page_dict.get('text', [])
                    if isinstance(text_list, list):
                        toc_texts.append(" ".join(str(t) for t in text_list))
                        
    elif isinstance(raw_data, list):
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            page_num = item.get('page_Id') or item.get('page_num') or 0
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                continue
            
            if page_num in toc_page_numbers:
                page_dict = item.get('page_dict', item)
                if isinstance(page_dict, dict):
                    text_list = page_dict.get('text', [])
                    if isinstance(text_list, list):
                        toc_texts.append(" ".join(str(t) for t in text_list))
    
    return "\n".join(toc_texts), explicit_toc_found


# =============================================================================
# OUTPUT PARSING
# =============================================================================

def extract_sections_from_output(output_path: str) -> Set[str]:
    """
    Extract section numbers from the final repaired output.
    
    Args:
        output_path: Path to the repaired JSON file
        
    Returns:
        Set of normalized section numbers found
    """
    sections = set()
    
    if not os.path.exists(output_path):
        print(f"    [Warning] Output file not found: {output_path}")
        return sections
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    [Warning] Could not load output file: {e}")
        return sections
    
    # Handle different data structures
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
    elif isinstance(data, list):
        elements = data
    else:
        elements = []
    
    for element in elements:
        if element.get('type') == 'section':
            section_num = element.get('section_number', '')
            if section_num:
                normalized = normalize_section_number(section_num)
                if normalized and is_valid_section_number(normalized):
                    sections.add(normalized)
    
    return sections


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

@dataclass
class ValidationResult:
    """Results of comparing TOC sections against extracted sections."""
    toc_sections: Set[str] = field(default_factory=set)
    output_sections: Set[str] = field(default_factory=set)
    matched_sections: Set[str] = field(default_factory=set)
    in_toc_not_output: Set[str] = field(default_factory=set)  # Missing from extraction
    in_output_not_toc: Set[str] = field(default_factory=set)  # Extra in extraction
    match_percentage: float = 0.0
    toc_coverage: float = 0.0  # What % of TOC sections did we find?
    precision: float = 0.0     # What % of our sections are in TOC?
    has_toc: bool = False      # Did this document have a TOC? (based on classification)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "has_toc": self.has_toc,
                "toc_section_count": len(self.toc_sections),
                "output_section_count": len(self.output_sections),
                "matched_count": len(self.matched_sections),
                "missing_count": len(self.in_toc_not_output),
                "extra_count": len(self.in_output_not_toc),
                "match_percentage": round(self.match_percentage, 2),
                "toc_coverage": round(self.toc_coverage, 2),
                "precision": round(self.precision, 2),
                "note": "has_toc is now based on classify_agent finding TABLE_OF_CONTENTS pages"
            },
            "toc_sections": sorted(self.toc_sections, key=section_sort_key),
            "output_sections": sorted(self.output_sections, key=section_sort_key),
            "matched_sections": sorted(self.matched_sections, key=section_sort_key),
            "missing_from_output": sorted(self.in_toc_not_output, key=section_sort_key),
            "extra_in_output": sorted(self.in_output_not_toc, key=section_sort_key)
        }


def section_sort_key(section: str) -> Tuple:
    """
    Sort key for section numbers that handles hierarchical numbering.
    '1' < '1.1' < '1.2' < '1.10' < '2' < '2.1'
    """
    parts = section.split('.')
    result = []
    for part in parts:
        try:
            result.append((0, int(part)))  # Numeric parts
        except ValueError:
            result.append((1, part))  # Alpha parts sort after numeric
    return tuple(result)


def compare_sections(toc_sections: Set[str], output_sections: Set[str], has_toc_from_classification: bool = None) -> ValidationResult:
    """
    Compare TOC sections against output sections.
    
    Args:
        toc_sections: Section numbers found in TOC
        output_sections: Section numbers found in output
        has_toc_from_classification: Whether classify_agent found TOC pages.
            If provided, this overrides the default behavior of checking
            if toc_sections is non-empty.
        
    Returns:
        ValidationResult with comparison details
    """
    # Determine has_toc: prefer classification result, fallback to checking toc_sections
    if has_toc_from_classification is not None:
        has_toc = has_toc_from_classification
    else:
        has_toc = len(toc_sections) > 0
    
    result = ValidationResult(
        toc_sections=toc_sections,
        output_sections=output_sections,
        has_toc=has_toc
    )
    
    # Find matches, missing, and extra
    result.matched_sections = toc_sections & output_sections
    result.in_toc_not_output = toc_sections - output_sections
    result.in_output_not_toc = output_sections - toc_sections
    
    # Calculate metrics (only meaningful if we have a TOC)
    if result.has_toc:
        total_unique = len(toc_sections | output_sections)
        if total_unique > 0:
            result.match_percentage = (len(result.matched_sections) / total_unique) * 100
        
        if len(toc_sections) > 0:
            result.toc_coverage = (len(result.matched_sections) / len(toc_sections)) * 100
        
        if len(output_sections) > 0:
            result.precision = (len(result.matched_sections) / len(output_sections)) * 100
    
    return result


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def run_validation(
    raw_ocr_path: str,
    classification_path: str,
    output_path: str,
    validation_output_path: str
) -> ValidationResult:
    """
    Run validation comparing TOC sections against extracted sections.
    
    Args:
        raw_ocr_path: Path to raw OCR JSON
        classification_path: Path to classification JSON (to find TOC pages)
        output_path: Path to final repaired output JSON
        validation_output_path: Path to write validation report
        
    Returns:
        ValidationResult
    """
    print(f"  - Extracting sections from TOC pages...")
    toc_text, has_toc_from_classification = load_toc_pages_text(raw_ocr_path, classification_path)
    toc_sections = extract_sections_from_toc_text(toc_text)
    print(f"    Found {len(toc_sections)} sections in TOC text")
    print(f"    Classification reports TOC present: {has_toc_from_classification}")
    
    print(f"  - Extracting sections from output...")
    output_sections = extract_sections_from_output(output_path)
    print(f"    Found {len(output_sections)} sections in output")
    
    print(f"  - Comparing sections...")
    # Pass has_toc_from_classification to use classify_agent's determination
    result = compare_sections(toc_sections, output_sections, has_toc_from_classification)
    
    # Print summary
    print(f"  - Results:")
    if result.has_toc:
        print(f"      TOC sections:     {len(result.toc_sections)}")
        print(f"      Output sections:  {len(result.output_sections)}")
        print(f"      Matched:          {len(result.matched_sections)}")
        print(f"      Missing (in TOC, not output): {len(result.in_toc_not_output)}")
        print(f"      Extra (in output, not TOC):   {len(result.in_output_not_toc)}")
        print(f"      TOC Coverage:     {result.toc_coverage:.1f}%")
        print(f"      Precision:        {result.precision:.1f}%")
    else:
        print(f"      [No TOC detected] - Document has {len(result.output_sections)} sections")
        print(f"      (Statistics not included in aggregate totals)")
    
    # Write validation report
    os.makedirs(os.path.dirname(validation_output_path) or '.', exist_ok=True)
    with open(validation_output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"  - Validation report saved to: {validation_output_path}")
    
    return result


def run_validation_on_file(
    stem: str,
    raw_ocr_dir: str,
    results_dir: str
) -> Optional[ValidationResult]:
    """
    Convenience function to run validation for a single document.
    
    Args:
        stem: Document stem (filename without extension)
        raw_ocr_dir: Directory containing raw OCR files
        results_dir: Directory containing pipeline results
        
    Returns:
        ValidationResult or None if validation couldn't run
    """
    raw_ocr_path = os.path.join(raw_ocr_dir, f"{stem}.json")
    classification_path = os.path.join(results_dir, f"{stem}_classification.json")
    
    # Try to find the best output file (prefer most processed)
    output_candidates = [
        os.path.join(results_dir, f"{stem}_with_tables.json"),
        os.path.join(results_dir, f"{stem}_with_assets.json"),
        os.path.join(results_dir, f"{stem}_repaired.json"),
        os.path.join(results_dir, f"{stem}_organized.json"),
    ]
    
    output_path = None
    for candidate in output_candidates:
        if os.path.exists(candidate):
            output_path = candidate
            break
    
    if output_path is None:
        print(f"  - [Warning] No output file found for {stem}")
        return None
    
    validation_output_path = os.path.join(results_dir, f"{stem}_validation.json")
    
    return run_validation(
        raw_ocr_path,
        classification_path,
        output_path,
        validation_output_path
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 4:
        raw_ocr = sys.argv[1]
        classification = sys.argv[2]
        output = sys.argv[3]
        validation_out = sys.argv[4] if len(sys.argv) > 4 else "validation_report.json"
        
        run_validation(raw_ocr, classification, output, validation_out)
    else:
        print("Usage: python validation_agent.py <raw_ocr.json> <classification.json> <output.json> [validation_report.json]")
        print("")
        print("This tool compares sections found in the Table of Contents against")
        print("sections extracted by the pipeline, producing a diagnostic report.")
        print("")
        print("The report includes:")
        print("  - Match percentage and coverage metrics")
        print("  - Sections in TOC but missing from output")
        print("  - Sections in output but not in TOC")