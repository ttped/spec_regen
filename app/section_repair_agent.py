"""
section_repair_agent.py - Validates and repairs section numbering sequences.

This module detects when section numbers break the expected hierarchical pattern,
which often indicates false positives (e.g., list items detected as sections).

Key concepts:
- Section numbers follow a hierarchical pattern: 1, 1.1, 1.2, 2, 2.1, 2.1.1, etc.
- A valid transition must follow logical rules (can't jump from 1.2.3 to 1 and back)
- OCR errors are tolerated (3.A instead of 3.4) but structural violations are flagged
- False positives are demoted back to content (merged with previous section)

The repair is conservative: when in doubt, leave it alone to avoid error propagation.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SectionNumber:
    """Parsed representation of a section number like '3.1.2' or '3.A'"""
    raw: str
    parts: List[str] = field(default_factory=list)
    depth: int = 0
    is_valid: bool = True
    
    def __post_init__(self):
        self.parts = self._parse_parts(self.raw)
        self.depth = len(self.parts)
        self.is_valid = self.depth > 0
    
    @staticmethod
    def _parse_parts(raw: str) -> List[str]:
        """Parse '3.1.2' into ['3', '1', '2'], handling OCR errors like '3.A'"""
        if not raw:
            return []
        # Split on dots, filter empty parts
        parts = [p.strip() for p in raw.split('.') if p.strip()]
        return parts
    
    def get_numeric_parts(self) -> List[Optional[int]]:
        """Convert parts to integers where possible, None for non-numeric (OCR errors)"""
        result = []
        for part in self.parts:
            try:
                result.append(int(part))
            except ValueError:
                # Could be OCR error like 'A' instead of '4'
                # Try to extract any digits
                digits = re.findall(r'\d+', part)
                if digits:
                    result.append(int(digits[0]))
                else:
                    result.append(None)
        return result
    
    def __repr__(self):
        return f"SectionNumber('{self.raw}' -> {self.parts})"


@dataclass 
class TransitionAnalysis:
    """Analysis of transition between two consecutive section numbers"""
    from_section: SectionNumber
    to_section: SectionNumber
    is_valid: bool = True
    violation_type: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0, lower = less certain it's a violation
    
    def __repr__(self):
        status = "VALID" if self.is_valid else f"INVALID ({self.violation_type})"
        return f"Transition({self.from_section.raw} -> {self.to_section.raw}): {status}"


def analyze_transition(prev: SectionNumber, curr: SectionNumber) -> TransitionAnalysis:
    """
    Analyze if the transition from prev to curr section number is valid.
    
    Valid transitions:
    - Same depth, increment last part: 1.1 -> 1.2
    - Go deeper by 1: 1.1 -> 1.1.1  
    - Go shallower and increment: 1.1.1 -> 1.2, or 1.1.1 -> 2
    - Next major section: 1.x.y -> 2, 2.x.y -> 3
    
    Invalid transitions (likely false positives):
    - Sudden reset to 1 in middle of document: 3.1.2 -> 1 (when we're past section 1)
    - Jump backwards: 3.1 -> 2.1
    - Massive depth change: 1.1 -> 1.1.1.1.1 (skipped 2 levels)
    """
    analysis = TransitionAnalysis(prev, curr)
    
    prev_parts = prev.get_numeric_parts()
    curr_parts = curr.get_numeric_parts()
    
    # Handle None values (OCR errors) - be lenient
    prev_clean = [p for p in prev_parts if p is not None]
    curr_clean = [p for p in curr_parts if p is not None]
    
    if not prev_clean or not curr_clean:
        # Can't analyze if we don't have numbers - assume valid
        return analysis
    
    prev_depth = len(prev_clean)
    curr_depth = len(curr_clean)
    prev_major = prev_clean[0]  # First number (e.g., 3 in 3.1.2)
    curr_major = curr_clean[0]
    
    # Rule 1: Check for valid major section progression
    # Going from any X.y.z to (X+1) is always valid (next major section)
    if curr_depth == 1 and curr_major == prev_major + 1:
        # This is a valid transition: 1.x.y -> 2, 2.x.y -> 3, etc.
        return analysis
    
    # Rule 2: Check for suspicious reset - going BACKWARDS in major section
    # e.g., 3.1.2 -> 1 or 3.1.2 -> 2 (when we're in section 3)
    if curr_depth == 1 and curr_major < prev_major:
        # Going from section 3.x.y back to section 1 or 2? Very suspicious.
        # This is almost certainly a list item
        analysis.is_valid = False
        analysis.violation_type = "suspicious_reset"
        analysis.confidence = 0.95
        return analysis
    
    # Rule 3: Check for suspicious reset within same major section
    # e.g., 2.1.3 -> 1 or 2.3 -> 1 (staying at or going to depth 1, but major didn't increment)
    if prev_depth >= 2 and curr_depth == 1 and curr_major <= prev_major:
        # We were deep (2.1.3) and jumped to just "1" or "2"
        # but the major number didn't actually progress
        if curr_major < prev_major:
            # Definitely wrong: 3.1 -> 1
            analysis.is_valid = False
            analysis.violation_type = "suspicious_reset"
            analysis.confidence = 0.95
            return analysis
        elif curr_major == prev_major and curr_major <= 3:
            # Ambiguous: 2.1 -> 2 could be valid (back to section 2)
            # But 2.1 -> 1 is definitely a list
            # For now, allow same-major transitions
            pass
    
    # Rule 4: Check depth jumps (going too deep too fast)
    depth_change = curr_depth - prev_depth
    
    if depth_change > 1:
        # Jumping more than one level deeper is suspicious
        # e.g., 1.1 -> 1.1.1.1 (skipped 1.1.1)
        analysis.is_valid = False
        analysis.violation_type = "depth_jump"
        analysis.confidence = 0.7
        return analysis
    
    # Rule 5: Check for backwards progression at same depth
    # e.g., 3.2 -> 3.1 or 2.1.4 -> 2.1.2
    if curr_depth == prev_depth and curr_depth >= 2:
        # Compare the prefix (all but last)
        if curr_clean[:-1] == prev_clean[:-1]:
            # Same prefix, check last number
            if curr_clean[-1] < prev_clean[-1]:
                # Going backwards: 3.1.4 -> 3.1.2
                analysis.is_valid = False
                analysis.violation_type = "backwards_subsection"
                analysis.confidence = 0.85
                return analysis
    
    # Rule 6: Check for impossible prefix changes at same depth
    # e.g., 2.3.1 -> 3.1.2 (changed prefix but stayed at depth 3)
    if curr_depth == prev_depth and curr_depth >= 2:
        # If we stay at the same depth, prefix should match or follow logically
        if curr_clean[:-1] != prev_clean[:-1]:
            # Prefix changed - is it a valid parent change?
            # Valid: 2.1.3 -> 2.2.1 (parent incremented)
            # Invalid: 2.1.3 -> 1.1.1 (went backwards)
            if curr_clean[0] < prev_clean[0]:
                analysis.is_valid = False
                analysis.violation_type = "backwards_major_section"
                analysis.confidence = 0.9
                return analysis
    
    return analysis


def find_violations(sections: List[Dict]) -> List[Tuple[int, TransitionAnalysis]]:
    """
    Scan through sections and identify potentially invalid transitions.
    
    Returns list of (index, analysis) tuples for violations.
    """
    violations = []
    
    # Filter to just section elements
    section_elements = [(i, s) for i, s in enumerate(sections) if s.get('type') == 'section']
    
    if len(section_elements) < 2:
        return violations
    
    for i in range(1, len(section_elements)):
        prev_idx, prev_section = section_elements[i - 1]
        curr_idx, curr_section = section_elements[i]
        
        prev_num = SectionNumber(prev_section.get('section_number', ''))
        curr_num = SectionNumber(curr_section.get('section_number', ''))
        
        if not prev_num.is_valid or not curr_num.is_valid:
            continue
        
        analysis = analyze_transition(prev_num, curr_num)
        
        if not analysis.is_valid:
            violations.append((curr_idx, analysis))
    
    return violations


def detect_list_sequences(elements: List[Dict]) -> List[List[int]]:
    """
    Detect sequences that look like lists (1, 2, 3 or a, b, c patterns)
    occurring in the middle of document sections.
    
    Returns list of index sequences that appear to be list items.
    """
    list_sequences = []
    
    # Get section elements with their positions
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if len(section_elements) < 3:
        return list_sequences
    
    # Track the "expected" context - what major section are we in?
    context_major = None
    context_depth = 0
    
    i = 0
    while i < len(section_elements):
        idx, section = section_elements[i]
        num = SectionNumber(section.get('section_number', ''))
        parts = num.get_numeric_parts()
        clean_parts = [p for p in parts if p is not None]
        
        if not clean_parts:
            i += 1
            continue
        
        curr_major = clean_parts[0]
        curr_depth = len(clean_parts)
        
        # If this looks like a legitimate major section update, track it
        if curr_depth >= 2 or (curr_depth == 1 and (context_major is None or curr_major == context_major + 1)):
            context_major = curr_major
            context_depth = max(context_depth, curr_depth)
            i += 1
            continue
        
        # Check if this starts a suspicious sequence (1, 2, 3... in middle of doc)
        if curr_depth == 1 and curr_major == 1 and context_major is not None and context_major > 1:
            # This looks like "1" appearing after we've established higher sections
            # Look ahead to see if there's a sequence
            sequence = [idx]
            j = i + 1
            expected_next = 2
            
            while j < len(section_elements):
                next_idx, next_section = section_elements[j]
                next_num = SectionNumber(next_section.get('section_number', ''))
                next_parts = next_num.get_numeric_parts()
                next_clean = [p for p in next_parts if p is not None]
                
                if not next_clean:
                    j += 1
                    continue
                
                next_major = next_clean[0]
                next_depth = len(next_clean)
                
                # Does this continue the list sequence?
                if next_depth == 1 and next_major == expected_next:
                    sequence.append(next_idx)
                    expected_next += 1
                    j += 1
                # Does this return to normal section numbering?
                elif next_depth >= 2 or next_major > context_major:
                    # End of list sequence
                    break
                else:
                    # Ambiguous - stop here
                    break
            
            if len(sequence) >= 2:
                # Found a list sequence
                list_sequences.append(sequence)
                i = j  # Skip past the sequence
                continue
        
        i += 1
    
    return list_sequences


def find_violation_runs(violations: List[Tuple[int, TransitionAnalysis]], elements: List[Dict]) -> List[List[int]]:
    """
    Group consecutive violations into runs.
    
    A run of violations (like 1, 2, 3 appearing in sequence) is more likely
    to be a list than a single violation.
    """
    if not violations:
        return []
    
    runs = []
    current_run = [violations[0][0]]
    
    for i in range(1, len(violations)):
        prev_idx = violations[i - 1][0]
        curr_idx = violations[i][0]
        
        # Check if these violations are close together in the element list
        # (allowing for some content blocks in between)
        gap = curr_idx - prev_idx
        
        if gap <= 3:  # Violations within 3 elements of each other
            current_run.append(curr_idx)
        else:
            if len(current_run) >= 1:
                runs.append(current_run)
            current_run = [curr_idx]
    
    if current_run:
        runs.append(current_run)
    
    return runs


def demote_section_to_content(section: Dict) -> Dict:
    """
    Convert a section element back to an unassigned_text_block.
    
    The section number and topic become part of the content.
    """
    section_num = section.get('section_number', '')
    topic = section.get('topic', '')
    content = section.get('content', '')
    
    # Reconstruct the original line
    text_parts = []
    if section_num:
        text_parts.append(section_num)
    if topic:
        text_parts.append(topic)
    
    header_text = ' '.join(text_parts)
    
    if content:
        full_content = f"{header_text} {content}" if header_text else content
    else:
        full_content = header_text
    
    return {
        "type": "unassigned_text_block",
        "content": full_content.strip(),
        "page_number": section.get('page_number'),
        "bbox": section.get('bbox'),
        "_demoted_from_section": section_num,  # Track for debugging
        "_original_topic": topic
    }


def merge_consecutive_content_blocks(elements: List[Dict]) -> List[Dict]:
    """
    After demoting sections, merge consecutive unassigned_text_blocks.
    """
    if not elements:
        return elements
    
    merged = []
    i = 0
    
    while i < len(elements):
        current = elements[i]
        
        if current.get('type') == 'unassigned_text_block':
            # Collect consecutive content blocks
            content_parts = [current.get('content', '')]
            bboxes = [current.get('bbox')]
            page = current.get('page_number')
            
            j = i + 1
            while j < len(elements) and elements[j].get('type') == 'unassigned_text_block':
                content_parts.append(elements[j].get('content', ''))
                bboxes.append(elements[j].get('bbox'))
                j += 1
            
            # Merge into single block
            merged_content = ' '.join(p for p in content_parts if p)
            merged_bbox = merge_bboxes([b for b in bboxes if b])
            
            merged.append({
                "type": "unassigned_text_block",
                "content": merged_content,
                "page_number": page,
                "bbox": merged_bbox
            })
            
            i = j
        else:
            merged.append(current)
            i += 1
    
    return merged


def merge_bboxes(bboxes: List[Dict]) -> Optional[Dict]:
    """Merge multiple bounding boxes into one encompassing bbox."""
    valid_bboxes = [b for b in bboxes if b is not None]
    if not valid_bboxes:
        return None
    
    left = min(b['left'] for b in valid_bboxes)
    top = min(b['top'] for b in valid_bboxes)
    right = max(b.get('right', b['left'] + b.get('width', 0)) for b in valid_bboxes)
    bottom = max(b.get('bottom', b['top'] + b.get('height', 0)) for b in valid_bboxes)
    
    return {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "right": right,
        "bottom": bottom
    }


def attach_content_to_previous_section(elements: List[Dict]) -> List[Dict]:
    """
    Attach unassigned_text_blocks to the preceding section as content.
    
    This should be run after demoting false positive sections.
    """
    if not elements:
        return elements
    
    result = []
    i = 0
    
    while i < len(elements):
        current = elements[i]
        
        if current.get('type') == 'section':
            # Collect any following content blocks
            content_parts = []
            if current.get('content'):
                content_parts.append(current['content'])
            
            content_bboxes = [current.get('bbox')]
            
            j = i + 1
            while j < len(elements) and elements[j].get('type') == 'unassigned_text_block':
                content_parts.append(elements[j].get('content', ''))
                content_bboxes.append(elements[j].get('bbox'))
                j += 1
            
            # Update section with merged content
            section_copy = current.copy()
            section_copy['content'] = ' '.join(p for p in content_parts if p)
            section_copy['bbox'] = merge_bboxes(content_bboxes)
            
            result.append(section_copy)
            i = j
        else:
            result.append(current)
            i += 1
    
    return result


def repair_sections(elements: List[Dict], confidence_threshold: float = 0.7) -> Tuple[List[Dict], Dict]:
    """
    Main repair function: detect and fix section numbering violations.
    
    Args:
        elements: List of document elements (sections, text blocks, figures, tables)
        confidence_threshold: Only repair violations with confidence >= this value
    
    Returns:
        Tuple of (repaired_elements, repair_report)
    """
    report = {
        "total_sections_before": sum(1 for e in elements if e.get('type') == 'section'),
        "violations_found": 0,
        "list_sequences_found": 0,
        "sections_demoted": 0,
        "violation_details": [],
        "list_sequences": []
    }
    
    # Find individual violations using transition analysis
    violations = find_violations(elements)
    report["violations_found"] = len(violations)
    
    # Find list sequences (1, 2, 3 patterns)
    list_sequences = detect_list_sequences(elements)
    report["list_sequences_found"] = len(list_sequences)
    report["list_sequences"] = list_sequences
    
    # Determine which indices to demote
    indices_to_demote = set()
    
    # Add violations that meet confidence threshold
    for idx, analysis in violations:
        report["violation_details"].append({
            "index": idx,
            "from": analysis.from_section.raw,
            "to": analysis.to_section.raw,
            "type": analysis.violation_type,
            "confidence": analysis.confidence
        })
        
        if analysis.confidence >= confidence_threshold:
            indices_to_demote.add(idx)
    
    # Add all items from detected list sequences
    for sequence in list_sequences:
        for idx in sequence:
            indices_to_demote.add(idx)
    
    # Also find runs of violations (adjacent violations are likely related)
    runs = find_violation_runs(violations, elements)
    for run in runs:
        if len(run) >= 2:  # A run of 2+ violations is almost certainly related
            for idx in run:
                indices_to_demote.add(idx)
    
    report["sections_demoted"] = len(indices_to_demote)
    
    # Apply demotions
    repaired = []
    for i, element in enumerate(elements):
        if i in indices_to_demote:
            demoted = demote_section_to_content(element)
            repaired.append(demoted)
        else:
            repaired.append(element)
    
    # Re-attach content to sections
    repaired = attach_content_to_previous_section(repaired)
    
    report["total_sections_after"] = sum(1 for e in repaired if e.get('type') == 'section')
    
    return repaired, report


def run_section_repair(input_path: str, output_path: str, confidence_threshold: float = 0.7):
    """
    Main entry point: load elements, repair sections, save results.
    """
    print(f"  - Loading elements from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both old and new format
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
        is_new_format = True
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
        is_new_format = False
    
    print(f"  - Found {len(elements)} elements")
    
    # Run repair
    repaired_elements, report = repair_sections(elements, confidence_threshold)
    
    # Report results
    print(f"  - Sections before repair: {report['total_sections_before']}")
    print(f"  - Violations found: {report['violations_found']}")
    print(f"  - Sections demoted: {report['sections_demoted']}")
    print(f"  - Sections after repair: {report['total_sections_after']}")
    
    if report['violation_details']:
        print(f"  - Violation details:")
        for v in report['violation_details'][:10]:  # Show first 10
            print(f"      {v['from']} -> {v['to']}: {v['type']} (confidence: {v['confidence']:.2f})")
        if len(report['violation_details']) > 10:
            print(f"      ... and {len(report['violation_details']) - 10} more")
    
    # Save results
    if is_new_format:
        output_data = {
            "page_metadata": page_metadata,
            "elements": repaired_elements,
            "_repair_report": report
        }
    else:
        output_data = repaired_elements
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"  - Repaired elements saved to: {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        run_section_repair(input_file, output_file, threshold)
    else:
        print("Usage: python section_repair_agent.py <input.json> <output.json> [confidence_threshold]")
        print("")
        print("This tool validates section numbering and demotes false positives (like list items)")
        print("back to content blocks.")
        print("")
        print("Arguments:")
        print("  input.json           - File with organized sections (from section_processor)")
        print("  output.json          - Output file with repaired sections")
        print("  confidence_threshold - Only fix violations with confidence >= this (default: 0.7)")