"""
section_repair_agent.py - Validates and repairs section numbering sequences.

This module detects when section numbers break the expected hierarchical pattern,
which often indicates false positives (e.g., list items or table rows detected as sections).

Key principles:
1. HIGH-DEPTH sections (with dots like 3.1.2) are almost always VALID - trust them
2. SIMPLE numbers (no dots like "8" or "12") are the source of most errors
3. Use "position tracking" - we know roughly where we are in the document
4. A simple number is valid if it's the logical next major section (current_major + 1)
5. A simple number is suspicious if it's far from our current position

The repair is conservative: when in doubt, leave it alone.
All text is preserved - demoted sections become regular content blocks.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any, Set
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
        parts = [p.strip() for p in raw.split('.') if p.strip()]
        return parts
    
    def get_numeric_parts(self) -> List[Optional[int]]:
        """Convert parts to integers where possible, None for non-numeric (OCR errors)"""
        result = []
        for part in self.parts:
            try:
                result.append(int(part))
            except ValueError:
                digits = re.findall(r'\d+', part)
                if digits:
                    result.append(int(digits[0]))
                else:
                    result.append(None)
        return result
    
    def get_major(self) -> Optional[int]:
        """Get the first (major) section number, or None if not parseable"""
        parts = self.get_numeric_parts()
        return parts[0] if parts and parts[0] is not None else None
    
    def is_simple_number(self) -> bool:
        """Check if this is a simple number with no dots (like '8' or '10')"""
        return self.depth == 1
    
    def has_dots(self) -> bool:
        """Check if this has hierarchical structure (dots)"""
        return self.depth >= 2
    
    def __repr__(self):
        return f"SectionNumber('{self.raw}' -> {self.parts})"


@dataclass 
class TransitionAnalysis:
    """Analysis of transition between two consecutive section numbers"""
    from_section: SectionNumber
    to_section: SectionNumber
    is_valid: bool = True
    violation_type: Optional[str] = None
    confidence: float = 1.0
    reason: str = ""
    
    def __repr__(self):
        status = "VALID" if self.is_valid else f"INVALID ({self.violation_type})"
        return f"Transition({self.from_section.raw} -> {self.to_section.raw}): {status}"


@dataclass
class DocumentPosition:
    """
    Tracks our current "position" in the document structure.
    Like a Kalman filter - we know roughly where we are.
    """
    current_major: int = 0  # The major section we're currently in (e.g., 3 for 3.1.2)
    max_major_seen: int = 0  # Highest major section we've confidently seen
    last_valid_section: Optional[SectionNumber] = None
    
    def update(self, section: SectionNumber):
        """Update position based on a validated section."""
        major = section.get_major()
        if major is not None:
            self.current_major = major
            self.max_major_seen = max(self.max_major_seen, major)
        self.last_valid_section = section
    
    def is_reasonable_next_major(self, major: int) -> bool:
        """
        Check if a simple number is a reasonable next major section.
        
        Valid: current_major + 1 (the logical next section)
        Also valid: current_major (staying in same major, e.g., after subsections)
        """
        if major == self.current_major + 1:
            return True  # Perfect: 3.x.x -> 4
        if major == self.current_major:
            return True  # Okay: back to major after subsections
        return False
    
    def distance_from_current(self, major: int) -> int:
        """How far is this major section from where we currently are?"""
        return abs(major - self.current_major)


def analyze_simple_number(
    curr: SectionNumber,
    position: DocumentPosition
) -> TransitionAnalysis:
    """
    Analyze a simple number (no dots) against our current position.
    
    Simple numbers are the main source of false positives (list items, table rows).
    They're valid if they're the logical next major section.
    """
    analysis = TransitionAnalysis(
        from_section=position.last_valid_section or SectionNumber(""),
        to_section=curr
    )
    
    curr_major = curr.get_major()
    if curr_major is None:
        return analysis  # Can't analyze, assume valid
    
    # Check if this is a reasonable progression
    if position.is_reasonable_next_major(curr_major):
        return analysis  # Valid
    
    # Calculate how far off we are
    distance = position.distance_from_current(curr_major)
    
    # Unrealistically large numbers (like 300, 502) are almost always wrong
    if curr_major > 50:
        analysis.is_valid = False
        analysis.violation_type = "unrealistic_number"
        analysis.confidence = 0.95
        analysis.reason = f"Section {curr_major} is unrealistically large"
        return analysis
    
    # Going backwards significantly is suspicious
    if curr_major < position.current_major:
        # How far back?
        if curr_major <= 3 and position.current_major >= 3:
            # Small number (1, 2, 3) appearing after we're past section 3
            # This is likely a list item
            analysis.is_valid = False
            analysis.violation_type = "suspicious_reset"
            analysis.confidence = 0.90
            analysis.reason = f"Reset to {curr_major} after reaching section {position.current_major}"
            return analysis
        elif distance >= 3:
            # Going back 3+ sections is suspicious
            analysis.is_valid = False
            analysis.violation_type = "backwards_jump"
            analysis.confidence = 0.85
            analysis.reason = f"Jump backwards from {position.current_major} to {curr_major}"
            return analysis
    
    # Jumping forward too much is suspicious
    if curr_major > position.current_major + 1:
        skip_count = curr_major - position.current_major - 1
        
        if skip_count >= 5:
            # Skipping 5+ sections (e.g., 3 -> 9)
            analysis.is_valid = False
            analysis.violation_type = "large_forward_jump"
            analysis.confidence = 0.90
            analysis.reason = f"Skipped {skip_count} sections: {position.current_major} to {curr_major}"
            return analysis
        elif skip_count >= 2:
            # Skipping 2-4 sections - suspicious but not certain
            analysis.is_valid = False
            analysis.violation_type = "forward_jump"
            analysis.confidence = 0.75
            analysis.reason = f"Skipped {skip_count} sections: {position.current_major} to {curr_major}"
            return analysis
    
    return analysis  # Valid


def analyze_hierarchical_section(
    curr: SectionNumber,
    position: DocumentPosition
) -> TransitionAnalysis:
    """
    Analyze a hierarchical section (has dots, like 3.1.2).
    
    These are almost always valid because people don't accidentally type dots.
    We only flag obvious problems.
    """
    analysis = TransitionAnalysis(
        from_section=position.last_valid_section or SectionNumber(""),
        to_section=curr
    )
    
    curr_major = curr.get_major()
    if curr_major is None:
        return analysis  # Can't analyze, assume valid
    
    # Hierarchical sections are trusted, but check for obvious issues
    
    # Unrealistically large major section
    if curr_major > 50:
        analysis.is_valid = False
        analysis.violation_type = "unrealistic_number"
        analysis.confidence = 0.90
        analysis.reason = f"Major section {curr_major} is unrealistically large"
        return analysis
    
    # Major section going backwards significantly
    if curr_major < position.current_major - 1 and position.current_major >= 3:
        # e.g., we're in section 5, and see 2.1.1
        # This could be valid (document reorganization) but is suspicious
        analysis.is_valid = False
        analysis.violation_type = "backwards_major"
        analysis.confidence = 0.70  # Lower confidence - might be valid
        analysis.reason = f"Major section {curr_major} after reaching {position.current_major}"
        return analysis
    
    return analysis  # Valid - trust hierarchical sections


def find_violations(elements: List[Dict]) -> List[Tuple[int, TransitionAnalysis]]:
    """
    Scan through sections and identify potentially invalid ones.
    
    Uses position tracking to determine if sections make sense in context.
    Compares against the last VALID section, not just the previous one.
    """
    violations = []
    position = DocumentPosition()
    
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if not section_elements:
        return violations
    
    for i, (idx, section) in enumerate(section_elements):
        curr_num = SectionNumber(section.get('section_number', ''))
        
        if not curr_num.is_valid:
            continue
        
        if i == 0:
            # First section - establish baseline
            position.update(curr_num)
            continue
        
        # Analyze based on whether it has dots or not
        if curr_num.is_simple_number():
            analysis = analyze_simple_number(curr_num, position)
        else:
            analysis = analyze_hierarchical_section(curr_num, position)
        
        if not analysis.is_valid:
            violations.append((idx, analysis))
            # Don't update position - this section is suspect
        else:
            # Valid section - update our position
            position.update(curr_num)
    
    return violations


def detect_list_sequences(elements: List[Dict]) -> List[List[int]]:
    """
    Detect sequences that look like lists (1, 2, 3 patterns)
    occurring in the middle of document sections.
    """
    list_sequences = []
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if len(section_elements) < 3:
        return list_sequences
    
    # Track context
    context_major = 0
    
    i = 0
    while i < len(section_elements):
        idx, section = section_elements[i]
        num = SectionNumber(section.get('section_number', ''))
        
        if not num.is_valid:
            i += 1
            continue
        
        curr_major = num.get_major()
        if curr_major is None:
            i += 1
            continue
        
        # Hierarchical sections update context
        if num.has_dots():
            context_major = max(context_major, curr_major)
            i += 1
            continue
        
        # Simple number that progresses normally
        if curr_major == context_major + 1:
            context_major = curr_major
            i += 1
            continue
        
        # Check for list sequence: small numbers (1, 2, 3...) after we're past them
        if curr_major <= 3 and context_major > curr_major:
            sequence = [idx]
            j = i + 1
            expected_next = curr_major + 1
            
            while j < len(section_elements):
                next_idx, next_section = section_elements[j]
                next_num = SectionNumber(next_section.get('section_number', ''))
                
                if not next_num.is_valid:
                    j += 1
                    continue
                
                next_major = next_num.get_major()
                if next_major is None:
                    j += 1
                    continue
                
                # Continues the sequence?
                if next_num.is_simple_number() and next_major == expected_next:
                    sequence.append(next_idx)
                    expected_next += 1
                    j += 1
                elif next_num.has_dots() or next_major > context_major:
                    # Back to real sections
                    break
                else:
                    break
            
            if len(sequence) >= 2:
                list_sequences.append(sequence)
                i = j
                continue
        
        i += 1
    
    return list_sequences


def demote_section_to_content(section: Dict) -> Dict:
    """
    Convert a section element back to an unassigned_text_block.
    ALL TEXT IS PRESERVED - nothing is deleted.
    """
    section_num = section.get('section_number', '')
    topic = section.get('topic', '')
    content = section.get('content', '')
    
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
        "_demoted_from_section": section_num,
        "_original_topic": topic
    }


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
    Preserves the original section bbox.
    """
    if not elements:
        return elements
    
    result = []
    i = 0
    
    while i < len(elements):
        current = elements[i]
        
        if current.get('type') == 'section':
            content_parts = []
            if current.get('content'):
                content_parts.append(current['content'])
            
            original_bbox = current.get('bbox')
            
            j = i + 1
            while j < len(elements) and elements[j].get('type') == 'unassigned_text_block':
                content_parts.append(elements[j].get('content', ''))
                j += 1
            
            section_copy = current.copy()
            section_copy['content'] = '\n\n'.join(p for p in content_parts if p)
            section_copy['bbox'] = original_bbox
            
            result.append(section_copy)
            i = j
        else:
            result.append(current)
            i += 1
    
    return result


def repair_sections(elements: List[Dict], confidence_threshold: float = 0.7) -> Tuple[List[Dict], Dict]:
    """
    Main repair function: detect and fix section numbering violations.
    
    Strategy:
    1. Trust hierarchical sections (with dots) - they're almost always valid
    2. Scrutinize simple numbers (no dots) - they're the source of errors
    3. Track position in document to determine what's reasonable
    4. Compare against last VALID section, not last seen
    
    Args:
        elements: List of document elements
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
    
    # Find violations using position tracking
    violations = find_violations(elements)
    report["violations_found"] = len(violations)
    
    # Find list sequences
    list_sequences = detect_list_sequences(elements)
    report["list_sequences_found"] = len(list_sequences)
    report["list_sequences"] = list_sequences
    
    # Determine which indices to demote
    indices_to_demote = set()
    
    # Add violations that meet confidence threshold
    for idx, analysis in violations:
        report["violation_details"].append({
            "index": idx,
            "from": analysis.from_section.raw if analysis.from_section else "",
            "to": analysis.to_section.raw,
            "type": analysis.violation_type,
            "confidence": analysis.confidence,
            "reason": analysis.reason
        })
        
        if analysis.confidence >= confidence_threshold:
            indices_to_demote.add(idx)
    
    # Add list sequences
    for sequence in list_sequences:
        for idx in sequence:
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
    
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
        is_new_format = True
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
        is_new_format = False
    
    print(f"  - Found {len(elements)} elements")
    
    repaired_elements, report = repair_sections(elements, confidence_threshold)
    
    print(f"  - Sections before repair: {report['total_sections_before']}")
    print(f"  - Violations found: {report['violations_found']}")
    print(f"  - List sequences found: {report['list_sequences_found']}")
    print(f"  - Sections demoted: {report['sections_demoted']}")
    print(f"  - Sections after repair: {report['total_sections_after']}")
    
    if report['violation_details']:
        print(f"  - Violation details:")
        for v in report['violation_details'][:10]:
            print(f"      {v['to']}: {v['type']} (conf: {v['confidence']:.2f}) - {v['reason']}")
        if len(report['violation_details']) > 10:
            print(f"      ... and {len(report['violation_details']) - 10} more")
    
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
        print("This tool validates section numbering and demotes false positives")
        print("back to content blocks. All text is preserved - nothing is deleted.")