"""
section_repair_agent.py - Validates and repairs section numbering sequences.

This module detects when section numbers break the expected hierarchical pattern,
which often indicates false positives (e.g., list items detected as sections).

Key concepts:
- Section numbers follow a hierarchical pattern: 1, 1.1, 1.2, 2, 2.1, 2.1.1, etc.
- A valid transition must follow logical rules (can't jump from 1.2.3 to 1 and back)
- OCR errors are tolerated (3.A instead of 3.4) but structural violations are flagged
- False positives are demoted back to content (merged with previous section)

Detection strategies:
- Track "high water mark" - the highest valid major section seen
- Single digits (no periods) that jump significantly are suspicious
- Avoid cascade deletions by validating against last VALID section, not last seen

The repair is conservative: when in doubt, leave it alone to avoid error propagation.
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
    
    def get_major(self) -> Optional[int]:
        """Get the first (major) section number, or None if not parseable"""
        parts = self.get_numeric_parts()
        return parts[0] if parts and parts[0] is not None else None
    
    def is_simple_number(self) -> bool:
        """Check if this is a simple number with no dots (like '8' or '10')"""
        return self.depth == 1
    
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
    reason: str = ""  # Human-readable explanation
    
    def __repr__(self):
        status = "VALID" if self.is_valid else f"INVALID ({self.violation_type})"
        return f"Transition({self.from_section.raw} -> {self.to_section.raw}): {status}"


def analyze_transition(
    prev: SectionNumber, 
    curr: SectionNumber,
    high_water_major: int = 0,
    context_depth: int = 0
) -> TransitionAnalysis:
    """
    Analyze if the transition from prev to curr section number is valid.
    
    Args:
        prev: Previous section number
        curr: Current section number being evaluated
        high_water_major: Highest major section number seen so far (e.g., 3 if we've seen 3.x.x)
        context_depth: Deepest nesting level we've seen recently
    
    Valid transitions:
    - Same depth, increment last part: 1.1 -> 1.2
    - Go deeper by 1: 1.1 -> 1.1.1  
    - Go shallower and increment: 1.1.1 -> 1.2, or 1.1.1 -> 2
    - Next major section: 1.x.y -> 2, 2.x.y -> 3
    
    Invalid transitions (likely false positives):
    - Sudden reset to 1 in middle of document: 3.1.2 -> 1 (when we're past section 1)
    - Simple number that skips too many: 3.1 -> 8 (skipping 4, 5, 6, 7)
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
    
    # Use high water mark if provided
    effective_high_water = max(high_water_major, prev_major)
    
    # ==========================================================================
    # Rule 1: Check for valid major section progression
    # Going from any X.y.z to (X+1) is always valid (next major section)
    # ==========================================================================
    if curr_depth == 1 and curr_major == prev_major + 1:
        return analysis  # Valid: 1.x.y -> 2, 2.x.y -> 3
    
    # ==========================================================================
    # Rule 2: Simple numbers (no dots) that jump too far are suspicious
    # e.g., 3.1 -> 8 or 3.1.2 -> 10
    # ==========================================================================
    if curr.is_simple_number():
        # How big is the jump from the current context?
        jump_size = curr_major - effective_high_water
        
        # If we're jumping more than 1 ahead of our high water mark, suspicious
        if jump_size > 1:
            # The further the jump, the more confident we are it's wrong
            if jump_size >= 5:
                analysis.is_valid = False
                analysis.violation_type = "large_jump"
                analysis.confidence = 0.95
                analysis.reason = f"Simple number {curr_major} jumps {jump_size} sections ahead of high water {effective_high_water}"
                return analysis
            elif jump_size >= 3:
                analysis.is_valid = False
                analysis.violation_type = "medium_jump"
                analysis.confidence = 0.85
                analysis.reason = f"Simple number {curr_major} jumps {jump_size} sections ahead"
                return analysis
            elif jump_size == 2 and prev_depth >= 2:
                # Smaller jump but we're in a subsection - more suspicious
                analysis.is_valid = False
                analysis.violation_type = "skip_in_subsection"
                analysis.confidence = 0.75
                analysis.reason = f"Simple number {curr_major} skips from {prev.raw} (expected {effective_high_water + 1})"
                return analysis
    
    # ==========================================================================
    # Rule 3: Check for suspicious reset - going BACKWARDS in major section
    # e.g., 3.1.2 -> 1 or 3.1.2 -> 2 (when we're in section 3)
    # ==========================================================================
    if curr_depth == 1 and curr_major < effective_high_water:
        # Going from section 3.x.y back to section 1 or 2? Very suspicious.
        # This is almost certainly a list item or table row number
        gap = effective_high_water - curr_major
        
        if gap >= 2:
            # Big gap - very confident this is wrong
            analysis.is_valid = False
            analysis.violation_type = "suspicious_reset"
            analysis.confidence = 0.95
            analysis.reason = f"Reset to {curr_major} after reaching section {effective_high_water}"
            return analysis
        elif gap == 1 and prev_depth >= 2:
            # Smaller gap but we're in a subsection
            analysis.is_valid = False
            analysis.violation_type = "suspicious_reset"
            analysis.confidence = 0.85
            analysis.reason = f"Reset to {curr_major} from {prev.raw} (high water: {effective_high_water})"
            return analysis
    
    # ==========================================================================
    # Rule 4: Check depth jumps (going too deep too fast)
    # BUT: Sections with high depth (like 4.2.2.2.1) are almost certainly real
    # because people don't accidentally type multiple dots
    # ==========================================================================
    depth_change = curr_depth - prev_depth
    
    if depth_change > 1:
        # Jumping more than one level deeper is suspicious
        # e.g., 1.1 -> 1.1.1.1 (skipped 1.1.1)
        
        # EXCEPTION: High-depth sections are almost always real
        # Nobody accidentally types "4.2.2.2.1" - that's intentional
        if curr_depth >= 3:
            # Check if the major section is progressing logically
            if curr_major == prev_major or curr_major == prev_major + 1:
                # This looks valid - don't flag it
                # e.g., 4.6 -> 5.1.1 or 5 -> 5.1.1 (after 5.1 was OCR'd as $.1)
                pass
            elif curr_depth >= 4:
                # Very deep sections (depth 4+) like 4.2.2.2 are almost always valid
                # Even if there's a major section change, trust it
                pass
            else:
                # Major section jump AND depth jump, but only depth 3
                # Still somewhat suspicious but lower confidence
                analysis.is_valid = False
                analysis.violation_type = "depth_jump"
                analysis.confidence = 0.5  # Low confidence - likely still valid
                analysis.reason = f"Depth jump of {depth_change} levels with major section change"
                return analysis
        else:
            # Shallow section (depth 1-2) with depth jump - more suspicious
            analysis.is_valid = False
            analysis.violation_type = "depth_jump"
            analysis.confidence = 0.7
            analysis.reason = f"Depth jump of {depth_change} levels"
            return analysis
    
    # ==========================================================================
    # Rule 5: Check for backwards progression at same depth
    # e.g., 3.2 -> 3.1 or 2.1.4 -> 2.1.2
    # ==========================================================================
    if curr_depth == prev_depth and curr_depth >= 2:
        # Compare the prefix (all but last)
        if curr_clean[:-1] == prev_clean[:-1]:
            # Same prefix, check last number
            if curr_clean[-1] < prev_clean[-1]:
                # Going backwards: 3.1.4 -> 3.1.2
                analysis.is_valid = False
                analysis.violation_type = "backwards_subsection"
                analysis.confidence = 0.85
                analysis.reason = f"Backwards: {curr.raw} < {prev.raw}"
                return analysis
    
    # ==========================================================================
    # Rule 6: Check for backwards major section at depth > 1
    # e.g., 2.3.1 -> 1.1.2 
    # ==========================================================================
    if curr_depth >= 2 and curr_major < prev_major:
        analysis.is_valid = False
        analysis.violation_type = "backwards_major_section"
        analysis.confidence = 0.9
        analysis.reason = f"Major section went backwards: {curr_major} < {prev_major}"
        return analysis
    
    return analysis


def find_violations(
    elements: List[Dict],
    high_water_tracking: bool = True,
    max_realistic_section: int = 50
) -> Tuple[List[Tuple[int, TransitionAnalysis]], Dict[int, int]]:
    """
    Scan through sections and identify potentially invalid transitions.
    
    Uses high water mark tracking to catch jumps like 3.1 -> 8.
    Compares against the LAST VALID section to prevent cascade errors.
    
    Args:
        elements: List of document elements
        high_water_tracking: If True, track the highest major section seen
        max_realistic_section: Maximum realistic major section number (default 50).
                               Numbers above this are almost certainly not real sections.
    
    Returns:
        Tuple of (violations list, high_water_at_index dict)
    """
    violations = []
    violation_indices = set()  # Track which indices are violations
    high_water_at_index = {}  # Track high water mark at each index
    
    # Filter to just section elements
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if len(section_elements) < 2:
        return violations, high_water_at_index
    
    # Track state
    high_water_major = 0
    max_depth_seen = 0
    last_valid_idx = None  # Track the last section that wasn't flagged as a violation
    last_valid_num = None
    
    for i in range(len(section_elements)):
        curr_idx, curr_section = section_elements[i]
        curr_num = SectionNumber(curr_section.get('section_number', ''))
        
        if not curr_num.is_valid:
            continue
        
        curr_major = curr_num.get_major()
        
        # Record high water at this index (before potential update)
        high_water_at_index[curr_idx] = high_water_major
        
        # =======================================================================
        # Pre-check: Unrealistically large section numbers are almost always wrong
        # Documents rarely go past 20-30 sections, let alone 100+
        # =======================================================================
        if curr_major is not None and curr_major > max_realistic_section:
            analysis = TransitionAnalysis(
                from_section=last_valid_num or SectionNumber(""),
                to_section=curr_num,
                is_valid=False,
                violation_type="unrealistic_section_number",
                confidence=0.95,
                reason=f"Section {curr_major} exceeds realistic maximum ({max_realistic_section})"
            )
            violations.append((curr_idx, analysis))
            violation_indices.add(curr_idx)
            continue  # Don't update any state for this
        
        if i == 0:
            # First section - establish baseline
            if curr_major is not None:
                high_water_major = curr_major
                max_depth_seen = curr_num.depth
            last_valid_idx = i
            last_valid_num = curr_num
            continue
        
        # =======================================================================
        # Compare against LAST VALID section, not just the previous one
        # This prevents cascade errors where a bad section causes good ones to fail
        # =======================================================================
        if last_valid_num is None:
            # No valid section yet - use the immediate previous for comparison
            prev_idx, prev_section = section_elements[i - 1]
            prev_num = SectionNumber(prev_section.get('section_number', ''))
        else:
            prev_num = last_valid_num
        
        if not prev_num.is_valid:
            # Still no valid previous - just accept this one
            last_valid_idx = i
            last_valid_num = curr_num
            if curr_major is not None and curr_major > high_water_major:
                high_water_major = curr_major
            max_depth_seen = max(max_depth_seen, curr_num.depth)
            continue
        
        # Analyze transition with context
        analysis = analyze_transition(
            prev_num, 
            curr_num,
            high_water_major=high_water_major,
            context_depth=max_depth_seen
        )
        
        if not analysis.is_valid:
            violations.append((curr_idx, analysis))
            violation_indices.add(curr_idx)
            # Don't update last_valid - keep using the previous valid one
        else:
            # This section is valid - update tracking state
            last_valid_idx = i
            last_valid_num = curr_num
            if curr_major is not None and curr_major > high_water_major:
                high_water_major = curr_major
            max_depth_seen = max(max_depth_seen, curr_num.depth)
    
    return violations, high_water_at_index


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
    context_major = 0
    
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
        
        # If this looks like a legitimate section (has depth or progresses normally), track it
        if curr_depth >= 2:
            context_major = max(context_major, curr_major)
            i += 1
            continue
        
        # Simple numbers that progress normally update context
        if curr_depth == 1 and curr_major == context_major + 1:
            context_major = curr_major
            i += 1
            continue
        
        # Check if this starts a suspicious sequence (1, 2, 3... in middle of doc)
        if curr_depth == 1 and curr_major <= 3 and context_major > curr_major:
            # This looks like a small number appearing after we've established higher sections
            # Look ahead to see if there's a sequence
            sequence = [idx]
            j = i + 1
            expected_next = curr_major + 1
            
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
                    # End of list sequence - this looks like we're back to real sections
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


def detect_isolated_simple_numbers(
    elements: List[Dict],
    violations: List[Tuple[int, TransitionAnalysis]],
    high_water_at_index: Dict[int, int]
) -> Set[int]:
    """
    Find isolated simple numbers (like 8, 10, 100) that don't fit the context.
    
    These are often table row numbers or list items that weren't caught by
    the sequence detector.
    """
    isolated = set()
    violation_indices = {idx for idx, _ in violations}
    
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    for idx, section in section_elements:
        num = SectionNumber(section.get('section_number', ''))
        
        if not num.is_simple_number():
            continue
        
        major = num.get_major()
        if major is None:
            continue
        
        # Get the high water mark at this point
        high_water = high_water_at_index.get(idx, 0)
        
        # If this simple number is way beyond or way below expectations, flag it
        if major > high_water + 2:
            # e.g., we're at section 3 and see "10" - that's 7 sections ahead
            isolated.add(idx)
        elif major < high_water - 1 and high_water >= 3:
            # e.g., we're at section 5 and see "2" - already flagged by transition analysis
            # but double-check it's in violations
            if idx not in violation_indices:
                isolated.add(idx)
    
    return isolated


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
    ALL TEXT IS PRESERVED - nothing is deleted.
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
    
    This preserves the original section bbox (header position only).
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
            
            # Preserve original section bbox - don't merge with content
            original_bbox = current.get('bbox')
            
            j = i + 1
            while j < len(elements) and elements[j].get('type') == 'unassigned_text_block':
                content_parts.append(elements[j].get('content', ''))
                j += 1
            
            # Update section with merged content but keep original bbox
            section_copy = current.copy()
            section_copy['content'] = '\n\n'.join(p for p in content_parts if p)
            section_copy['bbox'] = original_bbox  # Keep original header bbox
            
            result.append(section_copy)
            i = j
        else:
            result.append(current)
            i += 1
    
    return result


def repair_sections(elements: List[Dict], confidence_threshold: float = 0.7) -> Tuple[List[Dict], Dict]:
    """
    Main repair function: detect and fix section numbering violations.
    
    Uses multiple strategies:
    1. Transition analysis with high water mark tracking
    2. List sequence detection (1, 2, 3 patterns)
    3. Isolated simple number detection
    4. Violation run grouping
    
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
        "isolated_numbers_found": 0,
        "sections_demoted": 0,
        "violation_details": [],
        "list_sequences": []
    }
    
    # Find individual violations using transition analysis with high water tracking
    violations, high_water_at_index = find_violations(elements)
    report["violations_found"] = len(violations)
    
    # Find list sequences (1, 2, 3 patterns)
    list_sequences = detect_list_sequences(elements)
    report["list_sequences_found"] = len(list_sequences)
    report["list_sequences"] = list_sequences
    
    # Find isolated simple numbers that don't fit
    isolated_numbers = detect_isolated_simple_numbers(elements, violations, high_water_at_index)
    report["isolated_numbers_found"] = len(isolated_numbers)
    
    # Determine which indices to demote
    indices_to_demote = set()
    
    # Add violations that meet confidence threshold
    for idx, analysis in violations:
        report["violation_details"].append({
            "index": idx,
            "from": analysis.from_section.raw,
            "to": analysis.to_section.raw,
            "type": analysis.violation_type,
            "confidence": analysis.confidence,
            "reason": analysis.reason
        })
        
        if analysis.confidence >= confidence_threshold:
            indices_to_demote.add(idx)
    
    # Add all items from detected list sequences
    for sequence in list_sequences:
        for idx in sequence:
            indices_to_demote.add(idx)
    
    # Add isolated simple numbers
    indices_to_demote.update(isolated_numbers)
    
    # Also find runs of violations (adjacent violations are likely related)
    runs = find_violation_runs(violations, elements)
    for run in runs:
        if len(run) >= 2:  # A run of 2+ violations is almost certainly related
            for idx in run:
                indices_to_demote.add(idx)
    
    report["sections_demoted"] = len(indices_to_demote)
    
    # Apply demotions - ALL TEXT IS PRESERVED
    repaired = []
    for i, element in enumerate(elements):
        if i in indices_to_demote:
            demoted = demote_section_to_content(element)
            repaired.append(demoted)
        else:
            repaired.append(element)
    
    # Re-attach content to sections (preserving original bbox)
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
    print(f"  - List sequences found: {report['list_sequences_found']}")
    print(f"  - Isolated numbers found: {report['isolated_numbers_found']}")
    print(f"  - Sections demoted: {report['sections_demoted']}")
    print(f"  - Sections after repair: {report['total_sections_after']}")
    
    if report['violation_details']:
        print(f"  - Violation details:")
        for v in report['violation_details'][:10]:  # Show first 10
            reason = v.get('reason', v['type'])
            print(f"      {v['from']} -> {v['to']}: {reason} (confidence: {v['confidence']:.2f})")
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
        print("back to content blocks. All text is preserved - nothing is deleted.")
        print("")
        print("Arguments:")
        print("  input.json           - File with organized sections (from section_processor)")
        print("  output.json          - Output file with repaired sections")
        print("  confidence_threshold - Only fix violations with confidence >= this (default: 0.7)")