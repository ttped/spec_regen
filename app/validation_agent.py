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
# SECTION NUMBER PARSING (reused patterns from other modules)
# =============================================================================

# Pattern for hierarchical section numbers in TOC
# Matches: 1, 1.1, 1.1.1, 2.3.4.5, etc.
# Must be at or near the start of a line (after optional whitespace)
SECTION_NUMBER_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d{1,2}(?:\.\d{1,2}){0,5})(?=\s|\.{2,}|$)',
    re.MULTILINE
)

# Alternative pattern for sections that might have titles right after
SECTION_WITH_TITLE_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d{1,2}(?:\.\d{1,2}){0,5})\s+[A-Z]',
    re.MULTILINE
)


def normalize_section_number(raw: str) -> str:
    """
    Normalize a section number for comparison.
    - Remove trailing dots
    - Normalize separators (comma -> dot, hyphen -> dot)
    - Strip whitespace
    """
    if not raw:
        return ""
    
    normalized = raw.strip()
    
    # Replace comma with period
    normalized = normalized.replace(',', '.')
    
    # Replace hyphen between digits with period
    while True:
        new_normalized = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
        if new_normalized == normalized:
            break
        normalized = new_normalized
    
    # Remove trailing dots
    normalized = normalized.rstrip('.')
    
    # Clean up double dots
    while '..' in normalized:
        normalized = normalized.replace('..', '.')
    
    return normalized


def is_valid_section_number(num: str) -> bool:
    """
    Check if a string looks like a valid section number.
    Filters out things that look like page numbers, dates, etc.
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


def is_likely_decimal(num: str) -> bool:
    """
    Check if a number looks like a decimal value rather than a section number.
    E.g., 4.88, 2.50, 3.75
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


# =============================================================================
# TOC PARSING
# =============================================================================

def extract_sections_from_toc_text(toc_text: str) -> Set[str]:
    """
    Extract section numbers from TOC page text.
    
    Args:
        toc_text: Flattened text from TOC pages
        
    Returns:
        Set of normalized section numbers found
    """
    sections = set()
    
    # Try multiple patterns
    # Pattern 1: Section number at start of line
    matches1 = SECTION_NUMBER_PATTERN.findall(toc_text)
    
    # Pattern 2: Section number followed by title
    matches2 = SECTION_WITH_TITLE_PATTERN.findall(toc_text)
    
    # Combine all matches
    all_matches = set(matches1) | set(matches2)
    
    for match in all_matches:
        normalized = normalize_section_number(match)
        
        if not normalized:
            continue
            
        if not is_valid_section_number(normalized):
            continue
            
        if is_likely_decimal(normalized):
            continue
        
        sections.add(normalized)
    
    return sections


def load_toc_pages_text(raw_ocr_path: str, classification_path: str) -> str:
    """
    Load and combine text from TOC pages.
    
    Uses classification results to identify which pages are TOC,
    then extracts their text from the raw OCR.
    """
    # Load classification to find TOC pages
    toc_page_numbers = set()
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
                # ADD THIS:
                elif p_type == 'AMBIGUOUS' and not toc_page_numbers:
                    # Keep ambiguous pages as candidates if no explicit TOC found yet
                    toc_page_numbers.add(page_class.get('page'))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    [Warning] Could not parse classification file: {e}")
    
    # If no explicit TOC pages found, assume pages 2 to content_start-1 are TOC
    if not toc_page_numbers and content_start_page > 2:
        toc_page_numbers = set(range(2, content_start_page))
    
    if not toc_page_numbers:
        print("    [Warning] No TOC pages identified")
        return ""
    
    # Load raw OCR and extract TOC page text
    if not os.path.exists(raw_ocr_path):
        print(f"    [Warning] Raw OCR file not found: {raw_ocr_path}")
        return ""
    
    try:
        with open(raw_ocr_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    [Warning] Could not load raw OCR: {e}")
        return ""
    
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
    
    return "\n".join(toc_texts)


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
    has_toc: bool = False      # Did this document have a TOC?
    
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
                "precision": round(self.precision, 2)
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


def compare_sections(toc_sections: Set[str], output_sections: Set[str]) -> ValidationResult:
    """
    Compare TOC sections against output sections.
    
    Args:
        toc_sections: Section numbers found in TOC
        output_sections: Section numbers found in output
        
    Returns:
        ValidationResult with comparison details
    """
    result = ValidationResult(
        toc_sections=toc_sections,
        output_sections=output_sections,
        has_toc=len(toc_sections) > 0
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
    toc_text = load_toc_pages_text(raw_ocr_path, classification_path)
    toc_sections = extract_sections_from_toc_text(toc_text)
    print(f"    Found {len(toc_sections)} sections in TOC")
    
    print(f"  - Extracting sections from output...")
    output_sections = extract_sections_from_output(output_path)
    print(f"    Found {len(output_sections)} sections in output")
    
    print(f"  - Comparing sections...")
    result = compare_sections(toc_sections, output_sections)
    
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