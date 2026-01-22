"""
section_repair_agent.py - Validates and repairs section numbering sequences.

This module detects when section numbers break the expected hierarchical pattern,
which often indicates false positives (e.g., list items or table rows detected as sections).

Key principles:
1. HIGH-DEPTH sections (with dots like 3.1.2) are almost always VALID - trust them
2. SIMPLE numbers (no dots like "8" or "12") are the source of most errors
3. Use observation history to detect when position state has been corrupted
4. A simple number alone should NOT update major section state with high confidence
5. Multiple corroborating observations are needed to confirm a major section change
6. TOC matching provides additional confidence boost for hierarchical sections

The repair is conservative: when in doubt, leave it alone.
All text is preserved - demoted sections become regular content blocks.

IMPROVED APPROACH (v2):
- Track observation history, not just current position
- Single-digit sections have LOW confidence for updating major section
- Hierarchical sections (3.1.2) have HIGH confidence
- When many high-confidence sections contradict the position, reset position
- Implements a "soft Kalman filter" where observations are weighted by confidence

IMPROVED APPROACH (v3):
- Optional TOC matching to boost confidence for sections found in TOC
- Only applies to hierarchical sections (depth >= 2) to avoid false positives
- Protects legitimate sections from being incorrectly demoted
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque


# =============================================================================
# TOC EXTRACTION HELPERS
# =============================================================================

def normalize_section_for_comparison(raw: str) -> str:
    """
    Normalize a section number for comparison with TOC.
    Handles OCR errors and formatting variations.
    """
    if not raw:
        return ""
    
    normalized = raw.strip()
    
    # Replace comma with period (common OCR error)
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


def load_toc_sections_from_classification(classification_path: str, raw_ocr_path: str) -> Set[str]:
    """
    Load section numbers from TOC pages.
    
    Args:
        classification_path: Path to classification JSON (identifies TOC pages)
        raw_ocr_path: Path to raw OCR JSON (contains page text)
        
    Returns:
        Set of normalized section numbers found in TOC
    """
    toc_sections = set()
    
    # Load classification to find TOC pages
    if not os.path.exists(classification_path):
        return toc_sections
    
    try:
        with open(classification_path, 'r', encoding='utf-8') as f:
            classification = json.load(f)
    except (json.JSONDecodeError, OSError):
        return toc_sections
    
    # Find TOC page numbers
    toc_page_numbers = set()
    for page_class in classification.get('page_classifications', []):
        if page_class.get('type') == 'TABLE_OF_CONTENTS':
            toc_page_numbers.add(page_class.get('page'))
    
    if not toc_page_numbers:
        return toc_sections
    
    # Load raw OCR
    if not os.path.exists(raw_ocr_path):
        return toc_sections
    
    try:
        with open(raw_ocr_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return toc_sections
    
    # Extract text from TOC pages
    toc_texts = []
    
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
    
    # Extract section numbers from TOC text
    toc_text = "\n".join(toc_texts)
    
    # Pattern for hierarchical section numbers
    section_pattern = re.compile(
        r'(?:^|\s)(\d{1,2}(?:\.\d{1,2}){1,5})(?=\s|\.{2,}|$)',
        re.MULTILINE
    )
    
    matches = section_pattern.findall(toc_text)
    for match in matches:
        normalized = normalize_section_for_comparison(match)
        if normalized and '.' in normalized:  # Only hierarchical (has dots)
            toc_sections.add(normalized)
    
    return toc_sections


def is_section_in_toc(section_number: str, toc_sections: Set[str]) -> bool:
    """
    Check if a section number appears in the TOC.
    
    Only returns True for hierarchical sections (depth >= 2) to avoid
    false positives from simple numbers like 1, 2, 3.
    """
    if not toc_sections:
        return False
    
    normalized = normalize_section_for_comparison(section_number)
    
    # Only match hierarchical sections (must have at least one dot)
    if '.' not in normalized:
        return False
    
    return normalized in toc_sections


@dataclass
class SectionNumber:
    """Parsed representation of a section number like '3.1.2' or '3.A'"""
    raw: str
    parts: List[str] = field(default_factory=list)
    depth: int = 0
    is_valid: bool = True
    normalized: str = ""  # The normalized version with consistent separators
    
    def __post_init__(self):
        self.normalized, self.parts = self._parse_parts(self.raw)
        self.depth = len(self.parts)
        self.is_valid = self.depth > 0
    
    @staticmethod
    def _parse_parts(raw: str) -> Tuple[str, List[str]]:
        """
        Parse '3.1.2' into ['3', '1', '2'], handling OCR errors.
        
        OCR commonly confuses separators:
        - '1,1.3' should be '1.1.3'
        - '1-1-3' should be '1.1.3'
        - '3,2,1' should be '3.2.1'
        
        Returns (normalized_string, parts_list)
        """
        if not raw:
            return "", []
        
        # Normalize separators: treat , and - as . (common OCR errors)
        # But be careful: don't convert standalone hyphens in things like "A-1"
        # We want to convert "1,1.3" -> "1.1.3" and "1-1-3" -> "1.1.3"
        normalized = raw
        
        # Replace comma with period
        normalized = normalized.replace(',', '.')
        
        # Replace hyphen with period, but only when between digits
        # This avoids breaking things like "A-1" or "Phase-2"
        # Use a loop to handle consecutive replacements like "3-2-1"
        while True:
            new_normalized = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
            if new_normalized == normalized:
                break
            normalized = new_normalized
        
        parts = [p.strip() for p in normalized.split('.') if p.strip()]
        return normalized, parts
    
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
        if self.raw != self.normalized:
            return f"SectionNumber('{self.raw}' -> '{self.normalized}' -> {self.parts})"
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
class Observation:
    """A single observation of a section number with its confidence."""
    section: SectionNumber
    element_index: int
    confidence: float  # How confident we are this is a real section
    major: Optional[int] = None
    
    def __post_init__(self):
        self.major = self.section.get_major()


@dataclass
class DocumentPositionTracker:
    """
    Improved position tracker that uses observation history.
    
    Key insight: Position should be updated based on WEIGHTED observations,
    not just the most recent one. A single "4" shouldn't override many "3.x.x" sections.
    """
    # History of ACCEPTED observations (not rejected ones)
    observation_history: List[Observation] = field(default_factory=list)
    
    # Current estimated major section (weighted by observation confidence)
    current_major: int = 0
    
    # Track what we've confidently seen
    max_confident_major: int = 0
    
    # Recent observations for detecting contradictions
    recent_window: int = 10  # Look at last N observations
    
    def add_observation(self, obs: Observation):
        """Add an observation and update position estimate."""
        self.observation_history.append(obs)
        
        if obs.major is None:
            return
            
        # High-confidence observations (hierarchical sections) update position strongly
        if obs.confidence >= 0.8:
            self.current_major = obs.major
            self.max_confident_major = max(self.max_confident_major, obs.major)
        elif obs.confidence >= 0.5:
            # Medium confidence: only update if it's a reasonable progression
            if obs.major == self.current_major + 1 or obs.major == self.current_major:
                self.current_major = obs.major
                self.max_confident_major = max(self.max_confident_major, obs.major)
        # Low confidence observations don't update position
    
    def get_recent_major_distribution(self) -> Dict[int, float]:
        """
        Get weighted distribution of major sections from recent observations.
        This helps detect when we've strayed from the true position.
        """
        recent = self.observation_history[-self.recent_window:] if self.observation_history else []
        
        distribution = {}
        for obs in recent:
            if obs.major is not None:
                if obs.major not in distribution:
                    distribution[obs.major] = 0.0
                distribution[obs.major] += obs.confidence
        
        return distribution
    
    def get_most_likely_major(self) -> int:
        """
        Get the most likely current major section based on recent observations.
        Weighted by confidence.
        """
        dist = self.get_recent_major_distribution()
        if not dist:
            return self.current_major
        
        # Find the major with highest weighted count
        return max(dist.items(), key=lambda x: x[1])[0]
    
    def detect_position_corruption(self) -> bool:
        """
        Detect if the current position seems corrupted.
        
        Signs of corruption:
        - Many high-confidence observations at a different major than current_major
        - Recent observations strongly disagree with current_major
        """
        dist = self.get_recent_major_distribution()
        if not dist:
            return False
        
        # If current_major has very low weight compared to another major
        current_weight = dist.get(self.current_major, 0)
        total_weight = sum(dist.values())
        
        if total_weight > 2:  # Need enough observations
            # Find the dominant major
            dominant_major = max(dist.items(), key=lambda x: x[1])[0]
            dominant_weight = dist[dominant_major]
            
            # If dominant is different and much stronger
            if dominant_major != self.current_major:
                if dominant_weight > current_weight * 2 and dominant_weight > 1.5:
                    return True
        
        return False
    
    def reset_to_likely_position(self):
        """Reset position to the most likely major based on observations."""
        likely = self.get_most_likely_major()
        print(f"      [Position Reset] {self.current_major} -> {likely} (based on observation history)")
        self.current_major = likely
    
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


def calculate_title_confidence(topic: str) -> float:
    """
    Calculate confidence based on whether the topic text looks like a real title.
    
    Real titles tend to be:
    - Short (1-4 words is ideal)
    - Noun phrases ("System Requirements", "Introduction")
    - Not sentence-like
    
    False positives tend to be:
    - Long (7+ words)
    - Sentence-like ("The numbers in parenthesis refer to...")
    - Start with articles or have conjunctions
    
    Returns a multiplier (0.0 to 1.0) to apply to the base confidence.
    """
    if not topic:
        # Empty topic (just a section number) is valid
        return 1.0
    
    # Clean up the topic
    topic = topic.strip()
    
    # Count words (split on whitespace)
    words = topic.split()
    word_count = len(words)
    
    # === Word count based confidence ===
    if word_count == 0:
        word_confidence = 1.0
    elif word_count <= 4:
        word_confidence = 1.0  # Ideal length
    elif word_count <= 6:
        word_confidence = 0.85  # Slightly long but possible
    elif word_count <= 8:
        word_confidence = 0.65  # Suspicious - getting sentence-like
    elif word_count <= 10:
        word_confidence = 0.45  # Very suspicious
    else:
        word_confidence = 0.25  # Almost certainly not a title
    
    # === Sentence-like indicators ===
    sentence_penalty = 1.0
    
    # Starts with common sentence starters (articles)
    sentence_starters = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'there']
    first_word = words[0].lower() if words else ""
    if first_word in sentence_starters:
        sentence_penalty *= 0.7
    
    # Starts with imperative verbs (common in instructions/list items, not titles)
    imperative_starters = ['see', 'refer', 'note', 'ensure', 'verify', 'check', 'use',
                          'apply', 'follow', 'review', 'contact', 'consult', 'consider']
    if first_word in imperative_starters:
        sentence_penalty *= 0.5  # Strong penalty - very likely not a title
    
    # Contains sentence-like conjunctions/connectors mid-text
    # These rarely appear in titles but often in sentences
    sentence_connectors = ['that', 'which', 'where', 'when', 'because', 'since', 
                          'although', 'however', 'therefore', 'furthermore',
                          'refers', 'refer', 'shown', 'listed', 'described',
                          'following', 'below', 'above', 'details', 'additional']
    topic_lower = topic.lower()
    for connector in sentence_connectors:
        # Check if connector appears as a word (not part of another word)
        if re.search(r'\b' + connector + r'\b', topic_lower):
            sentence_penalty *= 0.6
            break  # Only apply once
    
    # Ends with prepositions (might be truncated sentence)
    ending_preps = ['to', 'for', 'with', 'from', 'by', 'at', 'in', 'on', 'of']
    last_word = words[-1].lower() if words else ""
    if last_word in ending_preps and word_count > 3:
        sentence_penalty *= 0.8
    
    # Contains parenthetical references like "(see section 4.2)"
    if re.search(r'\([^)]*\)', topic):
        sentence_penalty *= 0.7
    
    # Very long words or technical gibberish detection
    # Titles usually have normal-length words
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 12:  # Unusually long average word length
            sentence_penalty *= 0.8
    
    # === Positive indicators (boost confidence) ===
    title_boost = 1.0
    
    # All caps or title case is common for section headers
    if topic.isupper() or topic.istitle():
        title_boost = 1.1  # Small boost
    
    # Common section title words
    title_keywords = ['introduction', 'scope', 'requirements', 'overview', 'summary',
                     'description', 'specification', 'interface', 'design', 'test',
                     'verification', 'validation', 'general', 'system', 'software',
                     'hardware', 'performance', 'functional', 'applicable', 'documents',
                     'definitions', 'acronyms', 'abbreviations', 'references', 'appendix']
    for keyword in title_keywords:
        if keyword in topic_lower:
            title_boost = min(title_boost * 1.1, 1.2)  # Cap at 1.2
            break
    
    # Combine all factors
    final_confidence = word_confidence * sentence_penalty * title_boost
    
    # Clamp to [0.1, 1.0] range (never completely zero out)
    return max(0.1, min(1.0, final_confidence))


def calculate_section_confidence(
    section: SectionNumber, 
    topic: str = "",
    toc_sections: Optional[Set[str]] = None
) -> float:
    """
    Calculate how confident we are that this is a real section header.
    
    Combines multiple factors:
    1. Depth-based confidence (hierarchical numbers are more trustworthy)
    2. Title-based confidence (short, noun-phrase-like topics are more trustworthy)
    3. TOC matching (if section appears in TOC, boost confidence by 25%)
    
    High confidence:
    - Deep hierarchical numbers (3.1.2.1) - very unlikely to be list items
    - Medium depth (3.1.2) - quite likely real
    - Section found in TOC - strong signal it's real
    
    Medium confidence:
    - Shallow hierarchical (3.1) - could be outline or real
    
    Low confidence:
    - Simple numbers (4) - could easily be list items
    """
    if not section.is_valid:
        return 0.0
    
    major = section.get_major()
    if major is None:
        return 0.0
    
    depth = section.depth
    
    # Depth-based confidence
    if depth >= 4:
        depth_confidence = 0.95  # 3.1.2.1 - almost certainly real
    elif depth == 3:
        depth_confidence = 0.90  # 3.1.2 - very likely real
    elif depth == 2:
        depth_confidence = 0.80  # 3.1 - likely real
    else:  # depth == 1 (simple number)
        # Simple numbers are suspicious
        # Smaller numbers are more suspicious (1, 2, 3 are common list items)
        if major <= 3:
            depth_confidence = 0.30  # Very suspicious
        elif major <= 10:
            depth_confidence = 0.40  # Suspicious
        elif major <= 20:
            depth_confidence = 0.35  # Suspicious (could be table row)
        else:
            depth_confidence = 0.20  # Very suspicious (large numbers are rarely real sections)
    
    # Title-based confidence
    title_confidence = calculate_title_confidence(topic)
    
    # Combine: use geometric mean to balance both factors
    # But weight depth more heavily for hierarchical sections (they're reliable)
    if depth >= 2:
        # For hierarchical sections, title is less important (0.7 depth, 0.3 title)
        combined = (depth_confidence ** 0.7) * (title_confidence ** 0.3)
    else:
        # For simple numbers, title matters more (0.5 depth, 0.5 title)
        combined = (depth_confidence ** 0.5) * (title_confidence ** 0.5)
    
    # TOC matching boost (only for hierarchical sections, depth >= 2)
    # Simple numbers (1, 2, 3) are too common in text to trust TOC matching
    if toc_sections and depth >= 2:
        if is_section_in_toc(section.raw, toc_sections):
            # Boost confidence by 25%, but cap at 0.98
            combined = min(0.98, combined * 1.25)
    
    return combined


def analyze_simple_number(
    curr: SectionNumber,
    tracker: DocumentPositionTracker,
    combined_confidence: float = 1.0,
    topic: str = ""
) -> TransitionAnalysis:
    """
    Analyze a simple number (no dots) against our current position.
    
    Simple numbers are the main source of false positives (list items, table rows).
    They're valid if they're the logical next major section AND have reasonable confidence.
    """
    from_section = tracker.observation_history[-1].section if tracker.observation_history else SectionNumber("")
    
    analysis = TransitionAnalysis(
        from_section=from_section,
        to_section=curr
    )
    
    curr_major = curr.get_major()
    if curr_major is None:
        return analysis  # Can't analyze, assume valid
    
    # === NEW: Direct rejection based on low confidence ===
    # If the combined confidence (depth + title) is very low, reject immediately
    # This catches cases like "4 The numbers in parenthesis refer to..."
    if combined_confidence < 0.40:
        analysis.is_valid = False
        analysis.violation_type = "low_confidence_title"
        analysis.confidence = 0.85
        title_preview = topic[:50] + "..." if len(topic) > 50 else topic
        analysis.reason = f"Simple number with suspicious title (conf={combined_confidence:.2f}): '{title_preview}'"
        return analysis
    
    # Check for position corruption first
    if tracker.detect_position_corruption():
        tracker.reset_to_likely_position()
    
    # Check if this is a reasonable progression
    if tracker.is_reasonable_next_major(curr_major):
        return analysis  # Valid
    
    # Calculate how far off we are
    distance = tracker.distance_from_current(curr_major)
    
    # Unrealistically large numbers (like 300, 502) are almost always wrong
    if curr_major > 50:
        analysis.is_valid = False
        analysis.violation_type = "unrealistic_number"
        analysis.confidence = 0.95
        analysis.reason = f"Section {curr_major} is unrealistically large"
        return analysis
    
    # Going backwards significantly is suspicious
    if curr_major < tracker.current_major:
        # Check observation history - if we've seen many sections at the lower major,
        # this might actually be valid
        dist = tracker.get_recent_major_distribution()
        lower_weight = dist.get(curr_major, 0)
        
        # If there's strong evidence we're actually at this lower major
        if lower_weight > 1.5:
            # This is probably valid - position was corrupted
            return analysis
        
        # How far back?
        if curr_major <= 3 and tracker.current_major >= 3:
            # Small number (1, 2, 3) appearing after we're past section 3
            # This is likely a list item
            analysis.is_valid = False
            analysis.violation_type = "suspicious_reset"
            analysis.confidence = 0.90
            analysis.reason = f"Reset to {curr_major} after reaching section {tracker.current_major}"
            return analysis
        elif distance >= 3:
            # Going back 3+ sections is suspicious
            analysis.is_valid = False
            analysis.violation_type = "backwards_jump"
            analysis.confidence = 0.85
            analysis.reason = f"Jump backwards from {tracker.current_major} to {curr_major}"
            return analysis
    
    # Jumping forward too much is suspicious
    if curr_major > tracker.current_major + 1:
        skip_count = curr_major - tracker.current_major - 1
        
        if skip_count >= 5:
            # Skipping 5+ sections (e.g., 3 -> 9)
            analysis.is_valid = False
            analysis.violation_type = "large_forward_jump"
            analysis.confidence = 0.90
            analysis.reason = f"Skipped {skip_count} sections: {tracker.current_major} to {curr_major}"
            return analysis
        elif skip_count >= 2:
            # Skipping 2-4 sections - suspicious but not certain
            analysis.is_valid = False
            analysis.violation_type = "forward_jump"
            analysis.confidence = 0.75
            analysis.reason = f"Skipped {skip_count} sections: {tracker.current_major} to {curr_major}"
            return analysis
    
    return analysis  # Valid


def analyze_hierarchical_section(
    curr: SectionNumber,
    tracker: DocumentPositionTracker
) -> TransitionAnalysis:
    """
    Analyze a hierarchical section (has dots, like 3.1.2).
    
    These are almost always valid because people don't accidentally type dots.
    We only flag obvious problems.
    
    IMPORTANT CHANGE: Hierarchical sections can CORRECT a corrupted position.
    If we see many 3.x.x sections after "6", the 6 was probably wrong.
    """
    from_section = tracker.observation_history[-1].section if tracker.observation_history else SectionNumber("")
    
    analysis = TransitionAnalysis(
        from_section=from_section,
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
    
    # Check for position corruption BEFORE rejecting backwards sections
    if tracker.detect_position_corruption():
        tracker.reset_to_likely_position()
    
    # Major section going backwards significantly
    # BUT: be more lenient if this hierarchical section matches recent observations
    if curr_major < tracker.current_major - 1 and tracker.current_major >= 3:
        # Check if we have evidence that we're actually at this major
        dist = tracker.get_recent_major_distribution()
        curr_major_weight = dist.get(curr_major, 0)
        current_weight = dist.get(tracker.current_major, 0)
        
        # If more evidence supports curr_major than current position
        if curr_major_weight >= current_weight:
            # This is probably valid - position was corrupted by false positives
            return analysis  # Valid
        
        # Otherwise, this is suspicious
        analysis.is_valid = False
        analysis.violation_type = "backwards_major"
        analysis.confidence = 0.70  # Lower confidence - might be valid
        analysis.reason = f"Major section {curr_major} after reaching {tracker.current_major}"
        return analysis
    
    return analysis  # Valid - trust hierarchical sections


def find_violations(elements: List[Dict]) -> Tuple[List[Tuple[int, TransitionAnalysis]], DocumentPositionTracker]:
    """
    Scan through sections and identify potentially invalid ones.
    
    Uses position tracking with observation history to determine if sections make sense.
    Returns both violations and the tracker for debugging.
    """
    violations = []
    tracker = DocumentPositionTracker()
    
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if not section_elements:
        return violations, tracker
    
    for i, (idx, section) in enumerate(section_elements):
        curr_num = SectionNumber(section.get('section_number', ''))
        topic = section.get('topic', '')
        
        if not curr_num.is_valid:
            continue
        
        # Calculate confidence for this section (now includes title analysis)
        confidence = calculate_section_confidence(curr_num, topic)
        
        if i == 0:
            # First section - establish baseline
            obs = Observation(curr_num, idx, confidence)
            tracker.add_observation(obs)
            continue
        
        # Analyze based on whether it has dots or not
        if curr_num.is_simple_number():
            # Pass confidence and topic for direct low-confidence rejection
            analysis = analyze_simple_number(curr_num, tracker, confidence, topic)
        else:
            analysis = analyze_hierarchical_section(curr_num, tracker)
        
        if not analysis.is_valid:
            violations.append((idx, analysis))
            # Don't add to observation history - this section is suspect
            # BUT we do create a low-confidence observation for tracking purposes
            obs = Observation(curr_num, idx, confidence * 0.1)  # Very low confidence
            tracker.add_observation(obs)
        else:
            # Valid section - add to observation history
            obs = Observation(curr_num, idx, confidence)
            tracker.add_observation(obs)
    
    return violations, tracker


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


def detect_false_positive_chains(elements: List[Dict], tracker: DocumentPositionTracker) -> Set[int]:
    """
    Detect chains of false positive simple numbers that corrupted the position.
    
    Pattern: Section 3.2.1 -> [4] -> [5] -> [6] -> 3.2.3 (valid!)
    The [4], [5], [6] are likely false positives that should be demoted.
    
    Strategy: 
    1. Find sequences of consecutive simple numbers that "jump ahead"
    2. Check if hierarchical sections continue at the previous major after the jump
    3. If so, the simple numbers are likely false positives
    
    Key insight: If after 3.2.1 we see 4, 5, 6 then 3.2.3, the 4, 5, 6 are false positives
    because real section 4 wouldn't be followed by 3.2.3.
    """
    false_positives = set()
    
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if len(section_elements) < 5:
        return false_positives
    
    # Pass 1: Find the "established" major section before any suspicious jumps
    # by looking at hierarchical sections
    established_majors = []  # List of (position_in_list, major, is_hierarchical)
    
    for i, (idx, section) in enumerate(section_elements):
        num = SectionNumber(section.get('section_number', ''))
        if not num.is_valid:
            continue
        
        major = num.get_major()
        if major is None:
            continue
        
        established_majors.append((i, major, num.has_dots(), idx))
    
    # Pass 2: Look for pattern: hierarchical at major N, then simple numbers > N, then hierarchical at N again
    i = 0
    while i < len(established_majors) - 2:
        pos_i, major_i, is_hier_i, idx_i = established_majors[i]
        
        if not is_hier_i:
            i += 1
            continue
        
        # Found a hierarchical section at major N
        # Look for simple numbers that jump ahead
        suspicious_simples = []
        j = i + 1
        
        while j < len(established_majors):
            pos_j, major_j, is_hier_j, idx_j = established_majors[j]
            
            if not is_hier_j and major_j > major_i:
                # Simple number that jumped ahead of our established major
                suspicious_simples.append((j, major_j, idx_j))
                j += 1
            elif is_hier_j and major_j == major_i:
                # Found a hierarchical section back at our established major!
                # The simple numbers between are likely false positives
                if len(suspicious_simples) >= 1:
                    # Mark all the suspicious simples as false positives
                    for _, _, simple_idx in suspicious_simples:
                        false_positives.add(simple_idx)
                break
            elif is_hier_j:
                # Hierarchical at different major - reset
                break
            else:
                j += 1
        
        i += 1
    
    # Pass 3: Additional heuristic - sequences of 3+ consecutive simple numbers
    # that span multiple major sections are suspicious
    consecutive_simples = []
    
    for i, (idx, section) in enumerate(section_elements):
        num = SectionNumber(section.get('section_number', ''))
        if not num.is_valid:
            continue
        
        if num.is_simple_number():
            consecutive_simples.append((idx, num.get_major()))
        else:
            # Check the consecutive sequence
            if len(consecutive_simples) >= 3:
                majors = [m for _, m in consecutive_simples]
                # If they span 3+ different values (like 4, 5, 6), mark as suspicious
                if max(majors) - min(majors) >= 2:
                    # Check if context suggests these are false positives
                    # Look at what comes after
                    next_hier_major = None
                    for k in range(i, min(i + 5, len(section_elements))):
                        next_num = SectionNumber(section_elements[k][1].get('section_number', ''))
                        if next_num.has_dots():
                            next_hier_major = next_num.get_major()
                            break
                    
                    # If next hierarchical is at a lower major, these are false positives
                    if next_hier_major is not None and next_hier_major < max(majors):
                        for simple_idx, _ in consecutive_simples:
                            false_positives.add(simple_idx)
            
            consecutive_simples = []
    
    return false_positives


def detect_sandwich_false_positives(elements: List[Dict]) -> Set[int]:
    """
    Detect sections that are "sandwiched" between other sections in a way that
    violates logical numbering flow. These are almost certainly false positives.
    
    IMPROVED VERSION: Now catches ANY out-of-place section, not just simple numbers.
    
    Pattern examples:
    - 3.2.3.4 -> 3 -> 3.2.3.5  (the "3" is sandwiched)
    - 3.2.3.4 -> 4 -> 3.2.3.5  (the "4" is sandwiched)  
    - 3.1.1 -> 3 -> 4 -> 3.1.2  (both "3" and "4" are sandwiched)
    - 3.1.1 -> 4.88 -> 3.1.2   (decimal "4.88" is sandwiched - NEW!)
    - 3.1.1 -> 20 -> 3.1.2     (random number sandwiched)
    - 3.1.5 -> 2.5 -> 3.1.6    (decimal masquerading as section)
    
    Key insight: If the MAJOR section number before and after a candidate are the
    same (or form a logical sequence), but the candidate has a DIFFERENT major,
    it's likely a false positive - especially if it looks like a decimal.
    """
    false_positives = set()
    
    section_elements = [(i, s) for i, s in enumerate(elements) if s.get('type') == 'section']
    
    if len(section_elements) < 3:
        return false_positives
    
    # Build a list of (index, section_number, depth, major, parts, raw)
    section_info = []
    for idx, section in section_elements:
        num = SectionNumber(section.get('section_number', ''))
        if num.is_valid:
            section_info.append((
                idx,
                num,
                num.depth,
                num.get_major(),
                num.get_numeric_parts(),
                num.raw
            ))
    
    if len(section_info) < 3:
        return false_positives
    
    # ==========================================================================
    # Pass 1: Detect sections sandwiched between IMMEDIATE neighbors of same major
    # This is now more conservative - only looks at immediate context
    # ==========================================================================
    for i in range(1, len(section_info) - 1):
        idx, num, depth, major, parts, raw = section_info[i]
        
        # Skip if current section's major is None
        if major is None:
            continue
        
        # Get IMMEDIATE neighbors (more conservative than looking at 8)
        prev_info = section_info[i - 1]
        next_info = section_info[i + 1]
        
        prev_major = prev_info[3]
        next_major = next_info[3]
        prev_depth = prev_info[2]
        next_depth = next_info[2]
        
        if prev_major is None or next_major is None:
            continue
        
        # Only flag if BOTH immediate neighbors are hierarchical AND same major
        # AND current is different - this is a clear sandwich
        if prev_depth >= 2 and next_depth >= 2:
            if prev_major == next_major and major != prev_major:
                # Extra validation: look one more step to confirm pattern
                # This prevents flagging at legitimate transition boundaries
                confirmed = False
                
                if i >= 2 and i < len(section_info) - 2:
                    prev_prev_major = section_info[i - 2][3]
                    next_next_major = section_info[i + 2][3]
                    
                    # If pattern is X, X, [current], X, X - definitely sandwiched
                    if prev_prev_major == prev_major and next_next_major == next_major:
                        confirmed = True
                    # If pattern is X, X, [current], X, Y where Y = X+1, still likely sandwiched
                    elif prev_prev_major == prev_major:
                        confirmed = True
                else:
                    # At edges, trust the immediate neighbors
                    confirmed = True
                
                if confirmed:
                    false_positives.add(idx)
                    continue
        
        # Case 2: Current major is WAY off (like 20 in the middle of section 3-5)
        # Only flag if it's drastically out of range
        if prev_depth >= 2 and next_depth >= 2:
            expected_range = range(min(prev_major, next_major), max(prev_major, next_major) + 2)
            if major not in expected_range and abs(major - prev_major) > 2 and abs(major - next_major) > 2:
                false_positives.add(idx)
                continue
    
    # ==========================================================================
    # Pass 2: Detect decimal-like section numbers that are contextually wrong
    # Numbers like 4.88, 2.50, etc. that slipped through initial detection
    # ==========================================================================
    for i in range(len(section_info)):
        idx, num, depth, major, parts, raw = section_info[i]
        
        # Skip if major is None
        if major is None:
            continue
        
        # Check for decimal-like patterns (X.YY where YY > 30)
        if depth == 2 and len(parts) == 2:
            if parts[1] is not None and parts[1] > 30:
                # Looks like a decimal (4.88, 3.75, etc.)
                # Check context - is it surrounded by sections with different major?
                context_majors = []
                for j in range(max(0, i - 3), min(len(section_info), i + 4)):
                    if j != i and section_info[j][2] >= 2:  # hierarchical neighbor
                        neighbor_major = section_info[j][3]
                        if neighbor_major is not None:
                            context_majors.append(neighbor_major)
                
                if context_majors:
                    dominant_context = max(set(context_majors), key=context_majors.count)
                    # If context is different major, this is likely a decimal
                    if dominant_context != major:
                        false_positives.add(idx)
                        continue
                    # Even same major - 3.75 in section 3 content is suspicious
                    # Check if neighboring sections have reasonable subsection numbers
                    neighbor_subsections = []
                    for j in range(max(0, i - 2), min(len(section_info), i + 3)):
                        if j != i:
                            neighbor_parts = section_info[j][4]
                            if len(neighbor_parts) >= 2 and neighbor_parts[1] is not None:
                                neighbor_subsections.append(neighbor_parts[1])
                    
                    if neighbor_subsections:
                        max_neighbor = max(neighbor_subsections)
                        # If neighbors have small subsection numbers but this has 75, 88, etc.
                        if parts[1] > max_neighbor + 20:
                            false_positives.add(idx)
                            continue
    
    # ==========================================================================
    # Pass 3: Original sandwich detection for simple numbers between hierarchical sequences
    # ==========================================================================
    for i in range(1, len(section_info) - 1):
        idx, num, depth, major, parts, raw = section_info[i]
        
        # Only check simple numbers (depth 1)
        if depth != 1:
            continue
        
        # Look backward for a hierarchical section
        prev_hier = None
        for j in range(i - 1, max(0, i - 5) - 1, -1):
            if section_info[j][2] >= 2:  # depth >= 2
                prev_hier = section_info[j]
                break
        
        if prev_hier is None:
            continue
        
        prev_idx, prev_num, prev_depth, prev_major, prev_parts, prev_raw = prev_hier
        
        # Look forward for a hierarchical section
        next_hier = None
        for j in range(i + 1, min(len(section_info), i + 6)):
            if section_info[j][2] >= 2:  # depth >= 2
                next_hier = section_info[j]
                break
        
        if next_hier is None:
            continue
        
        next_idx, next_num, next_depth, next_major, next_parts, next_raw = next_hier
        
        # Check if prev and next form a sequence at the same depth
        if prev_depth == next_depth and prev_depth >= 2:
            if len(prev_parts) == len(next_parts) >= 2:
                # Check if all but the last part match
                prefix_matches = all(
                    prev_parts[k] == next_parts[k] 
                    for k in range(len(prev_parts) - 1)
                    if prev_parts[k] is not None and next_parts[k] is not None
                )
                
                # Check if the last part increments (with some tolerance for gaps)
                last_increments = (
                    prev_parts[-1] is not None and 
                    next_parts[-1] is not None and
                    0 < next_parts[-1] - prev_parts[-1] <= 5  # Allow gaps
                )
                
                if prefix_matches and last_increments:
                    # This section is sandwiched between a sequence - mark it
                    for k in range(i, len(section_info)):
                        if section_info[k][0] == next_idx:
                            break
                        if section_info[k][2] == 1:  # simple number
                            false_positives.add(section_info[k][0])
        
        # Also check: same major on both sides
        if prev_major is not None and next_major is not None:
            if prev_major == next_major and prev_depth >= 2 and next_depth >= 2:
                false_positives.add(idx)
    
    # ==========================================================================
    # Pass 4: Catch sequences of sandwiched items ONLY when they return to same major
    # e.g., 3.1.1 -> [garbage] -> 3.1.2  (garbage is sandwiched)
    # But NOT: 3.1.1 -> 4.1.1 -> 4.1.2 (this is legitimate progression)
    # ==========================================================================
    i = 0
    while i < len(section_info) - 2:
        # Find start of a hierarchical section
        if section_info[i][2] < 2:  # not hierarchical
            i += 1
            continue
        
        start_idx, start_num, start_depth, start_major, start_parts, start_raw = section_info[i]
        
        # Skip if start_major is None
        if start_major is None:
            i += 1
            continue
        
        # Look ahead to see if we return to the same major after some interruption
        # This catches: 3.1.1 -> [junk] -> 3.1.2
        # But NOT: 3.1.1 -> 4.1.1 -> 4.1.2 (legitimate forward progress)
        
        suspicious_sequence = []
        found_return_to_start = False
        j = i + 1
        
        while j < len(section_info):
            curr_info = section_info[j]
            curr_major = curr_info[3]
            curr_depth = curr_info[2]
            
            # Skip None majors
            if curr_major is None:
                j += 1
                continue
            
            # If we find a hierarchical section that RETURNS to start_major
            # after seeing some non-matching sections, those are sandwiched
            if curr_depth >= 2 and curr_major == start_major:
                if len(suspicious_sequence) > 0:
                    # Verify that the suspicious sections are actually wrong
                    # They should have different majors than start_major
                    for susp_info in suspicious_sequence:
                        susp_major = susp_info[3]
                        if susp_major != start_major:
                            false_positives.add(susp_info[0])
                    found_return_to_start = True
                break
            
            # If we see a legitimate next major (start_major + 1) with hierarchy,
            # that's normal document flow - STOP looking for sandwiches
            elif curr_depth >= 2 and curr_major == start_major + 1:
                # This is legitimate forward progress, not a sandwich situation
                break
            
            # If we see a major MORE than +1 ahead, that's suspicious
            # e.g., going from 3.x directly to 5.x
            elif curr_depth >= 2 and curr_major > start_major + 1:
                # Could be a problem, but don't mark as sandwich - let other passes handle
                break
            
            else:
                # Non-hierarchical or same major simple number - potentially sandwiched
                # But only if we later return to start_major
                if curr_major != start_major and curr_major != start_major + 1:
                    suspicious_sequence.append(curr_info)
            
            j += 1
        
        i += 1
    
    return false_positives


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


def repair_sections(
    elements: List[Dict], 
    confidence_threshold: float = 0.7,
    toc_sections: Optional[Set[str]] = None
) -> Tuple[List[Dict], Dict]:
    """
    Main repair function: detect and fix section numbering violations.
    
    IMPROVED STRATEGY (v2):
    1. Trust hierarchical sections (with dots) - they're almost always valid
    2. Scrutinize simple numbers (no dots) - they're the source of errors
    3. Track position with OBSERVATION HISTORY, not just current state
    4. Detect when simple numbers have corrupted position and recover
    5. Use confidence-weighted position updates
    6. (v3) Use TOC matching to boost confidence for legitimate sections
    
    Args:
        elements: List of document elements
        confidence_threshold: Only repair violations with confidence >= this value
        toc_sections: Optional set of section numbers from TOC for confidence boost
    
    Returns:
        Tuple of (repaired_elements, repair_report)
    """
    report = {
        "total_sections_before": sum(1 for e in elements if e.get('type') == 'section'),
        "violations_found": 0,
        "list_sequences_found": 0,
        "false_positive_chains_found": 0,
        "sandwich_false_positives_found": 0,
        "sections_demoted": 0,
        "toc_sections_available": len(toc_sections) if toc_sections else 0,
        "toc_protected_sections": 0,
        "violation_details": [],
        "list_sequences": [],
        "sandwich_patterns": [],
        "position_resets": 0
    }
    
    # Find violations using improved position tracking
    violations, tracker = find_violations(elements)
    report["violations_found"] = len(violations)
    
    # Find list sequences
    list_sequences = detect_list_sequences(elements)
    report["list_sequences_found"] = len(list_sequences)
    report["list_sequences"] = list_sequences
    
    # Find false positive chains (simple numbers that corrupted position)
    false_positive_chains = detect_false_positive_chains(elements, tracker)
    report["false_positive_chains_found"] = len(false_positive_chains)
    
    # Find sandwich false positives (simple numbers between hierarchical sequences)
    sandwich_false_positives = detect_sandwich_false_positives(elements)
    report["sandwich_false_positives_found"] = len(sandwich_false_positives)
    
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
    
    # Add false positive chains
    for idx in false_positive_chains:
        indices_to_demote.add(idx)
    
    # Add sandwich false positives
    for idx in sandwich_false_positives:
        indices_to_demote.add(idx)
    
    # ==========================================================================
    # TOC PROTECTION: Remove indices from demotion if they match TOC entries
    # Only protects hierarchical sections (depth >= 2) to avoid false positives
    # ==========================================================================
    toc_protected = set()
    if toc_sections:
        for idx in list(indices_to_demote):
            element = elements[idx]
            if element.get('type') == 'section':
                section_num = element.get('section_number', '')
                if is_section_in_toc(section_num, toc_sections):
                    # This section is in the TOC - protect it from demotion
                    indices_to_demote.remove(idx)
                    toc_protected.add(idx)
        
        report["toc_protected_sections"] = len(toc_protected)
        if toc_protected:
            # Log which sections were protected
            protected_nums = []
            for idx in toc_protected:
                protected_nums.append(elements[idx].get('section_number', ''))
            report["toc_protected_list"] = protected_nums
    
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
    
    # ==========================================================================
    # FINAL CLEANUP PASS: Simple sequential scan to catch remaining anomalies
    # This catches things like 2.5 appearing between 3.2.x sections
    # ==========================================================================
    final_demotions, final_details = final_sequence_cleanup(repaired)
    
    if final_demotions:
        report["final_cleanup_demotions"] = len(final_demotions)
        report["final_cleanup_details"] = final_details
        
        # Apply final demotions
        final_repaired = []
        for i, element in enumerate(repaired):
            if i in final_demotions:
                demoted = demote_section_to_content(element)
                final_repaired.append(demoted)
            else:
                final_repaired.append(element)
        
        # Re-attach content again after final demotions
        repaired = attach_content_to_previous_section(final_repaired)
    
    report["total_sections_after"] = sum(1 for e in repaired if e.get('type') == 'section')
    
    return repaired, report


def final_sequence_cleanup(elements: List[Dict]) -> Tuple[Set[int], List[Dict]]:
    """
    Final cleanup pass: simple sequential scan to catch any remaining out-of-sequence sections.
    
    This is intentionally CONSERVATIVE - only catches obvious misfits like:
    - 2.5 appearing between 3.2.1 and 3.2.2 (same major on both sides, different in middle)
    - 4.88 appearing where neighbors have small subsection numbers (decimal detection)
    
    This should NOT flag:
    - Legitimate transitions (3.x -> 4.x -> 5.x)
    - Sections at transition boundaries
    
    Returns:
        Tuple of (indices_to_demote, details_list)
    """
    to_demote = set()
    details = []
    
    # Get all sections with their indices
    sections = [(i, e) for i, e in enumerate(elements) if e.get('type') == 'section']
    
    if len(sections) < 3:
        return to_demote, details
    
    # Parse all section numbers
    parsed_sections = []
    for idx, section in sections:
        num = SectionNumber(section.get('section_number', ''))
        if num.is_valid:
            parsed_sections.append((idx, num, section))
    
    if len(parsed_sections) < 3:
        return to_demote, details
    
    # Scan through looking for anomalies
    for i in range(1, len(parsed_sections) - 1):
        idx, num, section = parsed_sections[i]
        major = num.get_major()
        
        if major is None:
            continue
        
        # Get IMMEDIATE neighbors only (1-2 on each side) for sandwich detection
        # Using a smaller window is more conservative
        prev_major = parsed_sections[i-1][1].get_major()
        next_major = parsed_sections[i+1][1].get_major()
        
        if prev_major is None or next_major is None:
            continue
        
        is_misfit = False
        reason = ""
        
        # =======================================================================
        # CONSERVATIVE Case 1: Sandwiched between SAME major
        # e.g., 3.2.1 -> 2.5 -> 3.2.2 (prev=3, curr=2, next=3)
        # This is a clear misfit - something with different major between same majors
        # =======================================================================
        if prev_major == next_major and major != prev_major:
            # Extra check: make sure this isn't at a legitimate boundary
            # Look a bit further to confirm the pattern
            extended_prev = [parsed_sections[j][1].get_major() for j in range(max(0, i-3), i)]
            extended_next = [parsed_sections[j][1].get_major() for j in range(i+1, min(len(parsed_sections), i+4))]
            extended_prev = [m for m in extended_prev if m is not None]
            extended_next = [m for m in extended_next if m is not None]
            
            # Only flag if extended context also shows this is out of place
            if extended_prev and extended_next:
                prev_consistent = all(m == prev_major for m in extended_prev[-2:]) if len(extended_prev) >= 2 else True
                next_consistent = all(m == next_major for m in extended_next[:2]) if len(extended_next) >= 2 else True
                
                if prev_consistent and next_consistent:
                    is_misfit = True
                    reason = f"major {major} sandwiched between {prev_major} sections"
        
        # =======================================================================
        # CONSERVATIVE Case 2: Decimal-like subsection (X.YY where YY > 50)
        # Only flag very obvious decimals, and only if context confirms
        # =======================================================================
        parts = num.get_numeric_parts()
        if not is_misfit and len(parts) == 2 and parts[1] is not None and parts[1] > 50:
            # Check immediate neighbor subsections
            prev_parts = parsed_sections[i-1][1].get_numeric_parts()
            next_parts = parsed_sections[i+1][1].get_numeric_parts()
            
            neighbor_subs = []
            if len(prev_parts) >= 2 and prev_parts[1] is not None:
                neighbor_subs.append(prev_parts[1])
            if len(next_parts) >= 2 and next_parts[1] is not None:
                neighbor_subs.append(next_parts[1])
            
            if neighbor_subs and max(neighbor_subs) < 30:
                # Neighbors have small subsections, we have 50+, likely decimal
                is_misfit = True
                reason = f"subsection {parts[1]} looks like decimal (neighbors: {neighbor_subs})"
        
        if is_misfit:
            to_demote.add(idx)
            details.append({
                "index": idx,
                "section": num.raw,
                "reason": reason,
                "prev_major": prev_major,
                "next_major": next_major
            })
    
    return to_demote, details
    
    return to_demote, details


def run_section_repair(
    input_path: str, 
    output_path: str, 
    confidence_threshold: float = 0.7,
    classification_path: str = None,
    raw_ocr_path: str = None
):
    """
    Main entry point: load elements, repair sections, save results.
    
    Args:
        input_path: Path to organized JSON
        output_path: Path to save repaired JSON
        confidence_threshold: Minimum confidence to apply repairs
        classification_path: Optional path to classification JSON (for TOC extraction)
        raw_ocr_path: Optional path to raw OCR JSON (for TOC extraction)
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
    
    # Load TOC sections if paths provided
    toc_sections = None
    if classification_path and raw_ocr_path:
        toc_sections = load_toc_sections_from_classification(classification_path, raw_ocr_path)
        if toc_sections:
            print(f"  - Loaded {len(toc_sections)} sections from TOC for confidence boost")
    
    repaired_elements, report = repair_sections(elements, confidence_threshold, toc_sections)
    
    print(f"  - Sections before repair: {report['total_sections_before']}")
    print(f"  - Violations found: {report['violations_found']}")
    print(f"  - List sequences found: {report['list_sequences_found']}")
    print(f"  - False positive chains found: {report['false_positive_chains_found']}")
    print(f"  - Sandwich false positives found: {report['sandwich_false_positives_found']}")
    
    # Log TOC protection
    if report.get('toc_protected_sections', 0) > 0:
        print(f"  - TOC-protected sections (saved from demotion): {report['toc_protected_sections']}")
    
    print(f"  - Sections demoted: {report['sections_demoted']}")
    
    # Log final cleanup if any
    if report.get('final_cleanup_demotions', 0) > 0:
        print(f"  - Final cleanup demotions: {report['final_cleanup_demotions']}")
        for detail in report.get('final_cleanup_details', [])[:5]:
            print(f"      {detail['section']}: {detail['reason']}")
        if len(report.get('final_cleanup_details', [])) > 5:
            print(f"      ... and {len(report['final_cleanup_details']) - 5} more")
    
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
        print("")
        print("IMPROVEMENTS in v2:")
        print("  - Uses observation history instead of just current position")
        print("  - Simple numbers (4, 5, 6) have LOW confidence for position updates")
        print("  - Hierarchical sections (3.1.2) have HIGH confidence")
        print("  - Detects when position has been corrupted by false positives")
        print("  - Can recover from corrupted position state")