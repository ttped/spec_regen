"""
feature_extractor.py - Extract ML features from processed documents.

This module generates training data for a classifier to determine if a
detected "section" is a true section or a false positive.

Key insight: Features should measure SELF-CONSISTENCY within a document,
not absolute values. Section formatting is consistent within a document
but varies across documents.

Output: CSV with features + metadata for manual labeling

UPDATES (v2):
- Added vertical_gap_from_prev_line: Distance (px and pct) from previous line
- Added newlines_before_section_num: Count of newlines preceding the section number
- Removed is_sandwiched and sandwich_same_neighbors (negative correlation)
- Added manual labeling helper features (see MANUAL LABELING FEATURES section)

UPDATES (v3):
- Added SpaCy linguistic features for title vs sentence detection
- Added enhanced blank/empty title features
- Added relative position within page features (for late-page junk detection)
- Added height-based features to detect text that spans multiple lines
"""

import os
import json
import re
import csv
import statistics
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

# =============================================================================
# SPACY INITIALIZATION (required for linguistic features)
# =============================================================================

import spacy
#_nlp = spacy.load("en_core_web_sm")
_nlp = spacy.load(os.path.join(os.path.dirname(__file__), "model_en"))

# =============================================================================
# DEFAULT PATHS (same as simple_pipeline.py)
# =============================================================================

DEFAULT_RAW_OCR_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
DEFAULT_RESULTS_DIR = "results_simple"


# =============================================================================
# SECTION NUMBER PARSING (from section_repair_agent.py)
# =============================================================================

@dataclass
class ParsedSection:
    """Parsed representation of a section number."""
    raw: str
    normalized: str = ""
    parts: List[str] = field(default_factory=list)
    depth: int = 0
    is_valid: bool = True
    
    def __post_init__(self):
        self.normalized, self.parts = self._parse(self.raw)
        self.depth = len(self.parts)
        self.is_valid = self.depth > 0
    
    @staticmethod
    def _parse(raw: str) -> Tuple[str, List[str]]:
        if not raw:
            return "", []
        
        # Normalize separators
        normalized = raw.replace(',', '.').strip()
        while True:
            new = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
            if new == normalized:
                break
            normalized = new
        
        while '..' in normalized:
            normalized = normalized.replace('..', '.')
        
        # Strip leading and trailing dots (handles cases like ".3.0" -> "3.0")
        normalized = normalized.strip('.')
        
        if not normalized:
            return "", []
        
        parts = normalized.split('.')
        # Validate parts
        valid_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Accept digits or single letters
            if p.isdigit() or (len(p) <= 2 and p.isalpha()):
                valid_parts.append(p)
            else:
                break
        
        return '.'.join(valid_parts), valid_parts
    
    def get_major(self) -> Optional[int]:
        """Get the major section number (first component)."""
        if not self.parts:
            return None
        try:
            return int(self.parts[0])
        except ValueError:
            return None
    
    def get_numeric_parts(self) -> List[Optional[int]]:
        """Get all parts as integers (None for non-numeric)."""
        result = []
        for p in self.parts:
            try:
                result.append(int(p))
            except ValueError:
                result.append(None)
        return result
    
    def get_parent(self) -> Optional[str]:
        """Get the parent section number (e.g., '1.2' for '1.2.3')."""
        if len(self.parts) <= 1:
            return None
        return '.'.join(self.parts[:-1])


# =============================================================================
# DOCUMENT STATISTICS (for normalization)
# =============================================================================

@dataclass
class DocumentStats:
    """Statistics computed from all sections in a document for normalization."""
    # Bounding box stats
    section_x_positions: List[float] = field(default_factory=list)
    section_y_positions: List[float] = field(default_factory=list)
    section_widths: List[float] = field(default_factory=list)
    section_heights: List[float] = field(default_factory=list)
    
    # Vertical gaps between consecutive sections (NEW)
    vertical_gaps: List[float] = field(default_factory=list)
    
    # By depth
    x_by_depth: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Title stats
    title_lengths: List[int] = field(default_factory=list)
    title_word_counts: List[int] = field(default_factory=list)
    
    # Section number stats
    depths: List[int] = field(default_factory=list)
    major_sections_seen: Set[int] = field(default_factory=set)
    all_section_numbers: Set[str] = field(default_factory=set)
    
    # Page stats
    page_width: float = 0
    page_height: float = 0
    total_pages: int = 0
    
    # Computed after collection
    x_median: float = 0
    x_std: float = 0
    y_median: float = 0
    y_std: float = 0
    title_len_median: float = 0
    title_len_std: float = 0
    depth_median: float = 0
    
    # Vertical gap stats (NEW)
    vertical_gap_median: float = 0
    vertical_gap_std: float = 0
    
    def compute_stats(self):
        """Compute derived statistics after collecting all data."""
        if self.section_x_positions:
            self.x_median = statistics.median(self.section_x_positions)
            self.x_std = statistics.stdev(self.section_x_positions) if len(self.section_x_positions) > 1 else 1
        
        if self.section_y_positions:
            self.y_median = statistics.median(self.section_y_positions)
            self.y_std = statistics.stdev(self.section_y_positions) if len(self.section_y_positions) > 1 else 1
        
        if self.title_lengths:
            self.title_len_median = statistics.median(self.title_lengths)
            self.title_len_std = statistics.stdev(self.title_lengths) if len(self.title_lengths) > 1 else 1
        
        if self.depths:
            self.depth_median = statistics.median(self.depths)
        
        # Compute vertical gap stats (NEW)
        if self.vertical_gaps:
            self.vertical_gap_median = statistics.median(self.vertical_gaps)
            self.vertical_gap_std = statistics.stdev(self.vertical_gaps) if len(self.vertical_gaps) > 1 else 1
    
    def get_x_median_for_depth(self, depth: int) -> Optional[float]:
        """Get median X position for sections at a specific depth."""
        positions = self.x_by_depth.get(depth, [])
        if positions:
            return statistics.median(positions)
        return None
    
    def get_x_std_for_depth(self, depth: int) -> Optional[float]:
        """Get std dev of X position for sections at a specific depth."""
        positions = self.x_by_depth.get(depth, [])
        if len(positions) > 1:
            return statistics.stdev(positions)
        return None


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features_for_section(
    section: Dict,
    idx: int,
    all_sections: List[Dict],
    doc_stats: DocumentStats,
    toc_sections: Set[str],
    doc_name: str
) -> Dict[str, Any]:
    """
    Extract all features for a single section.
    
    Returns a dict with:
    - Metadata (for validation, not ML features)
    - Features (normalized, for ML)
    - Label placeholder
    """
    features = {}
    
    # =========================================================================
    # METADATA (not features, for human validation)
    # =========================================================================
    features['_doc_name'] = doc_name
    features['_index'] = idx
    features['_section_number_raw'] = section.get('section_number', '')  # Original OCR'd string
    features['_title'] = section.get('topic', '')[:100]  # Truncate for CSV
    features['_title_full_length'] = len(section.get('topic', '') or '')  # Full length for reference
    features['_page'] = section.get('page_number', 0)
    features['_label'] = ''  # To be filled manually: 1=valid, 0=false positive
    
    # =========================================================================
    # SECTION NUMBER FEATURES
    # =========================================================================
    parsed = ParsedSection(section.get('section_number', ''))
    
    # Add normalized version to metadata too
    features['_section_number_normalized'] = parsed.normalized
    
    features['section_depth'] = parsed.depth
    features['section_depth_vs_median'] = parsed.depth - doc_stats.depth_median if doc_stats.depth_median else 0
    
    major = parsed.get_major()
    features['section_major'] = major if major is not None else -1
    features['section_major_seen_before'] = 1 if major in doc_stats.major_sections_seen else 0
    
    # Is this a simple number (no dots)?
    features['is_simple_number'] = 1 if parsed.depth == 1 else 0
    
    # Check if parent exists
    parent = parsed.get_parent()
    features['parent_exists'] = 1 if parent and parent in doc_stats.all_section_numbers else 0
    
    # Numeric parts analysis
    numeric_parts = parsed.get_numeric_parts()
    features['has_non_numeric_part'] = 1 if None in numeric_parts else 0
    
    # Check for suspicious subsection numbers (like X.88, X.999)
    max_subsection = 0
    for p in numeric_parts[1:]:  # Skip major
        if p is not None:
            max_subsection = max(max_subsection, p)
    features['max_subsection_value'] = max_subsection
    features['subsection_looks_like_decimal'] = 1 if max_subsection >= 50 else 0
    
    # =========================================================================
    # SECTION NUMBER STRING ANALYSIS (new features)
    # =========================================================================
    raw_section_num = section.get('section_number', '')
    
    # Count of alpha characters in section number
    # Previously: rejected if > 2 alpha chars. Now it's a feature.
    alpha_count = sum(1 for c in raw_section_num if c.isalpha())
    features['section_num_alpha_count'] = alpha_count
    features['section_num_too_many_alpha'] = 1 if alpha_count > 2 else 0
    
    # Is this a pure digit section number? (no dots)
    # Previously: rejected if pure digits > 3 chars. Now it's a feature.
    digits_only = raw_section_num.replace(' ', '')
    features['section_num_is_pure_digits'] = 1 if digits_only.isdigit() else 0
    features['section_num_pure_digit_length'] = len(digits_only) if digits_only.isdigit() else 0
    features['section_num_pure_digits_too_long'] = 1 if digits_only.isdigit() and len(digits_only) > 3 else 0
    
    # Mixed alpha+digit without dots (like "A1" or "B2")
    # Previously: rejected. Now it's a feature.
    has_alpha = any(c.isalpha() for c in raw_section_num)
    has_digit = any(c.isdigit() for c in raw_section_num)
    has_dot = '.' in raw_section_num
    features['section_num_mixed_no_dot'] = 1 if has_alpha and has_digit and not has_dot else 0
    
    # Length to depth ratio: "1" = 1/1 = 1.0, "1.2.3" = 5/3 = 1.67, "1.10.11" = 7/3 = 2.33
    # Optimal is ~1.0-1.5, anything > 2 is suspicious
    section_num_length = len(raw_section_num.replace(' ', ''))  # Length without spaces
    if parsed.depth > 0:
        features['section_num_length_to_depth_ratio'] = section_num_length / parsed.depth
    else:
        features['section_num_length_to_depth_ratio'] = 0
    
    # Flag if ratio is suspicious (> 2.0 means long numbers like 1.10.100)
    features['section_num_ratio_suspicious'] = 1 if features['section_num_length_to_depth_ratio'] > 2.0 else 0
    
    # Raw section number length
    features['section_num_raw_length'] = section_num_length
    
    # Count of dots in raw string (should equal depth - 1 for valid sections)
    dot_count = raw_section_num.count('.')
    features['section_num_dot_count'] = dot_count
    features['section_num_dots_match_depth'] = 1 if dot_count == parsed.depth - 1 else 0
    
    # Check for non-standard separators (OCR errors: hyphens, commas, spaces)
    features['section_num_has_hyphen'] = 1 if '-' in raw_section_num else 0
    features['section_num_has_comma'] = 1 if ',' in raw_section_num else 0
    features['section_num_has_space'] = 1 if ' ' in raw_section_num else 0
    
    # Count non-standard characters (anything that's not digit or dot)
    non_standard_chars = sum(1 for c in raw_section_num if not c.isdigit() and c != '.')
    features['section_num_non_standard_char_count'] = non_standard_chars
    
    # Check if section number contains letters (like "3.A" or "A.1")
    features['section_num_has_letters'] = 1 if any(c.isalpha() for c in raw_section_num) else 0
    
    # Check for double dots (OCR error: "1..2")
    features['section_num_has_double_dot'] = 1 if '..' in raw_section_num else 0
    
    # Check if starts/ends with dot (malformed: ".1.2" or "1.2.")
    features['section_num_starts_with_dot'] = 1 if raw_section_num.startswith('.') else 0
    features['section_num_ends_with_dot'] = 1 if raw_section_num.endswith('.') else 0
    
    # =========================================================================
    # DETECTION CONTEXT FEATURES (from greedy section_processor)
    # =========================================================================
    # These features capture the context in which the section number was found
    # Useful for ML to determine if this is a real section or false positive
    
    detection_ctx = section.get('detection_context', {})
    
    # Was there whitespace before the section number? (suggests line start)
    features['had_leading_whitespace'] = 1 if detection_ctx.get('had_leading_whitespace', False) else 0
    features['leading_whitespace_len'] = detection_ctx.get('leading_whitespace_len', 0)
    
    # Was there text BEFORE the section number? (suggests mid-line, likely false positive)
    features['had_text_before_number'] = 1 if detection_ctx.get('had_text_before_number', False) else 0
    
    # Length of text before the number (0 if at line start)
    text_before = detection_ctx.get('text_before_number', '')
    features['text_before_number_len'] = len(text_before)
    
    # Check what kind of text was before (if any)
    if text_before:
        # Common prefixes that might indicate a reference, not a section
        reference_words = ['see', 'refer', 'section', 'para', 'paragraph', 'item', 'ref', 'per']
        features['text_before_is_reference'] = 1 if any(w in text_before.lower() for w in reference_words) else 0
    else:
        features['text_before_is_reference'] = 0
    
    # Original line length (very long lines might indicate paragraph text, not headers)
    features['original_line_length'] = detection_ctx.get('original_line_length', 0)
    
    # =========================================================================
    # NEW: NEWLINES BEFORE SECTION NUMBER
    # =========================================================================
    # Count newlines in detection context - new sections often have whitespace above
    # This captures whether the section number was preceded by blank lines
    original_line = detection_ctx.get('original_line', '')
    
    # Count leading newlines in the original line (if captured)
    # This would need to be captured in section_processor if not already
    features['newlines_before_section_num'] = detection_ctx.get('newlines_before', 0)
    
    # =========================================================================
    # LINE POSITION FEATURES (inferred from bbox)
    # =========================================================================
    bbox = section.get('bbox', {})
    if bbox:
        x_pos = bbox.get('left', 0)
        
        # Check if X position is at/near the left margin
        # Sections typically start at consistent left positions
        # If X is much further right than typical, might be mid-line text
        if doc_stats.section_x_positions:
            min_x = min(doc_stats.section_x_positions)
            # Is this section's X position near the minimum (left margin)?
            features['x_near_left_margin'] = 1 if x_pos <= min_x * 1.5 else 0
            features['x_distance_from_min'] = x_pos - min_x
        else:
            features['x_near_left_margin'] = 0
            features['x_distance_from_min'] = 0
        
        # Check if X position suggests indentation (deeper sections are often indented)
        # Compare X position to what's expected for this depth
        depth_x_median = doc_stats.get_x_median_for_depth(parsed.depth)
        if depth_x_median is not None:
            features['x_matches_depth_indent'] = 1 if abs(x_pos - depth_x_median) < 50 else 0
        else:
            features['x_matches_depth_indent'] = 0
    else:
        features['x_near_left_margin'] = 0
        features['x_distance_from_min'] = 0
        features['x_matches_depth_indent'] = 0
    
    # =========================================================================
    # SEQUENCE FEATURES (Kalman-like)
    # =========================================================================
    if idx > 0:
        prev_section = all_sections[idx - 1]
        prev_parsed = ParsedSection(prev_section.get('section_number', ''))
        prev_major = prev_parsed.get_major()
        
        # Gap from previous major
        if major is not None and prev_major is not None:
            features['major_gap_from_prev'] = major - prev_major
        else:
            features['major_gap_from_prev'] = 0
        
        # Depth change
        features['depth_change_from_prev'] = parsed.depth - prev_parsed.depth
        
        # Is this a logical next section?
        # e.g., 1.1 -> 1.2, or 1.1.3 -> 1.1.4, or 1.1.9 -> 1.2
        features['is_logical_next'] = _is_logical_next(prev_parsed, parsed)
    else:
        features['major_gap_from_prev'] = 0
        features['depth_change_from_prev'] = 0
        features['is_logical_next'] = 1  # First section is always "logical"
    
    # Look ahead for context
    if idx < len(all_sections) - 1:
        next_section = all_sections[idx + 1]
        next_parsed = ParsedSection(next_section.get('section_number', ''))
        next_major = next_parsed.get_major()
        
        if major is not None and next_major is not None:
            features['major_gap_to_next'] = next_major - major
        else:
            features['major_gap_to_next'] = 0
        
        # NEW: Is the NEXT section a logical successor to THIS one?
        # This helps identify if current section "fits" in the sequence
        features['next_is_logical_successor'] = _is_logical_next(parsed, next_parsed)
    else:
        features['major_gap_to_next'] = 0
        features['next_is_logical_successor'] = 0
    
    # =========================================================================
    # EXTENDED SEQUENCE FEATURES (replaces problematic sandwich features)
    # =========================================================================
    # Instead of sandwich detection (which had negative correlation),
    # focus on sequence continuity and logical flow
    
    features['extended_sequence_score'] = _compute_extended_sequence_score(idx, all_sections)
    
    # =========================================================================
    # NEW: VERTICAL GAP FEATURES (distance from previous line)
    # =========================================================================
    # New sections tend to have MORE whitespace above them
    # Numbers embedded in text have LESS whitespace above
    
    bbox = section.get('bbox', {})
    prev_bottom = None
    
    if idx > 0:
        prev_section = all_sections[idx - 1]
        prev_bbox = prev_section.get('bbox', {})
        if prev_bbox:
            prev_bottom = prev_bbox.get('top', 0) + prev_bbox.get('height', 0)
    
    if bbox and prev_bottom is not None:
        current_top = bbox.get('top', 0)
        
        # Only compute if on same page (cross-page gaps are meaningless)
        prev_page = all_sections[idx - 1].get('page_number', 0) if idx > 0 else 0
        curr_page = section.get('page_number', 0)
        
        if prev_page == curr_page and current_top >= prev_bottom:
            vertical_gap_px = current_top - prev_bottom
            features['vertical_gap_from_prev_px'] = vertical_gap_px
            
            # Normalize by page height
            if doc_stats.page_height > 0:
                features['vertical_gap_from_prev_pct'] = vertical_gap_px / doc_stats.page_height
            else:
                features['vertical_gap_from_prev_pct'] = 0
            
            # Compare to document median vertical gap
            if doc_stats.vertical_gap_std and doc_stats.vertical_gap_std > 0:
                features['vertical_gap_z_score'] = (vertical_gap_px - doc_stats.vertical_gap_median) / doc_stats.vertical_gap_std
            else:
                features['vertical_gap_z_score'] = 0
            
            # Flag: is this gap larger than typical? (suggests new section)
            features['has_large_vertical_gap'] = 1 if vertical_gap_px > doc_stats.vertical_gap_median * 1.5 else 0
            
            # Flag: is this gap smaller than typical? (suggests embedded in text)
            features['has_small_vertical_gap'] = 1 if vertical_gap_px < doc_stats.vertical_gap_median * 0.5 else 0
        else:
            # Cross-page or invalid gap
            features['vertical_gap_from_prev_px'] = -1  # Sentinel for "not applicable"
            features['vertical_gap_from_prev_pct'] = -1
            features['vertical_gap_z_score'] = 0
            features['has_large_vertical_gap'] = 0
            features['has_small_vertical_gap'] = 0
    else:
        features['vertical_gap_from_prev_px'] = -1
        features['vertical_gap_from_prev_pct'] = -1
        features['vertical_gap_z_score'] = 0
        features['has_large_vertical_gap'] = 0
        features['has_small_vertical_gap'] = 0
    
    # =========================================================================
    # BOUNDING BOX FEATURES (normalized to page dimensions)
    # =========================================================================
    # Page dimensions come from image_meta.render_raw.width_px/height_px
    # This allows proper normalization as percentages (0-1) that are
    # comparable across documents with different page sizes
    
    bbox = section.get('bbox', {})
    
    if bbox:
        x_pos = bbox.get('left', 0)
        y_pos = bbox.get('top', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        right_pos = x_pos + width
        bottom_pos = y_pos + height
        
        # Raw pixel values (for reference)
        features['bbox_left_px'] = x_pos
        features['bbox_top_px'] = y_pos
        features['bbox_width_px'] = width
        features['bbox_height_px'] = height
        
        # =====================================================================
        # NORMALIZED POSITIONS (as percentage of page, 0.0 to 1.0)
        # =====================================================================
        # These are comparable across documents with different page sizes
        
        if doc_stats.page_width > 0:
            features['bbox_left_pct'] = x_pos / doc_stats.page_width
            features['bbox_right_pct'] = right_pos / doc_stats.page_width
            features['bbox_width_pct'] = width / doc_stats.page_width
            features['bbox_center_x_pct'] = (x_pos + width / 2) / doc_stats.page_width
        else:
            features['bbox_left_pct'] = 0
            features['bbox_right_pct'] = 0
            features['bbox_width_pct'] = 0
            features['bbox_center_x_pct'] = 0
        
        if doc_stats.page_height > 0:
            features['bbox_top_pct'] = y_pos / doc_stats.page_height
            features['bbox_bottom_pct'] = bottom_pos / doc_stats.page_height
            features['bbox_height_pct'] = height / doc_stats.page_height
            features['bbox_center_y_pct'] = (y_pos + height / 2) / doc_stats.page_height
        else:
            features['bbox_top_pct'] = 0
            features['bbox_bottom_pct'] = 0
            features['bbox_height_pct'] = 0
            features['bbox_center_y_pct'] = 0
        
        # =====================================================================
        # DEVIATION FROM DOCUMENT NORMS (z-scores)
        # =====================================================================
        # How does this section's position compare to other sections in same doc?
        
        if doc_stats.x_std and doc_stats.x_std > 0:
            features['x_deviation_from_median'] = (x_pos - doc_stats.x_median) / doc_stats.x_std
        else:
            features['x_deviation_from_median'] = 0
        
        if doc_stats.y_std and doc_stats.y_std > 0:
            features['y_deviation_from_median'] = (y_pos - doc_stats.y_median) / doc_stats.y_std
        else:
            features['y_deviation_from_median'] = 0
        
        # Deviation from depth-specific median (sections at same depth should align)
        depth_x_median = doc_stats.get_x_median_for_depth(parsed.depth)
        depth_x_std = doc_stats.get_x_std_for_depth(parsed.depth)
        if depth_x_median is not None and depth_x_std and depth_x_std > 0:
            features['x_deviation_from_depth_median'] = (x_pos - depth_x_median) / depth_x_std
        else:
            features['x_deviation_from_depth_median'] = 0
        
        # =====================================================================
        # POSITION FLAGS
        # =====================================================================
        # Is this near the left margin? (typical for section headers)
        features['is_near_left_margin'] = 1 if features.get('bbox_left_pct', 1) < 0.15 else 0
        
        # Is this indented? (might indicate subsection or list item)
        features['is_indented'] = 1 if features.get('bbox_left_pct', 0) > 0.10 else 0
        
        # Is this in the header zone? (top 10% of page)
        features['is_in_header_zone'] = 1 if features.get('bbox_top_pct', 1) < 0.10 else 0
        
        # Is this in the footer zone? (bottom 10% of page)
        features['is_in_footer_zone'] = 1 if features.get('bbox_top_pct', 0) > 0.90 else 0
        
        # Legacy compatibility
        features['x_position_relative'] = features.get('bbox_left_pct', 0)
        features['y_position_relative'] = features.get('bbox_top_pct', 0)
        features['width_relative'] = features.get('bbox_width_pct', 0)
        
    else:
        # No bbox data available
        features['bbox_left_px'] = 0
        features['bbox_top_px'] = 0
        features['bbox_width_px'] = 0
        features['bbox_height_px'] = 0
        features['bbox_left_pct'] = 0
        features['bbox_right_pct'] = 0
        features['bbox_width_pct'] = 0
        features['bbox_center_x_pct'] = 0
        features['bbox_top_pct'] = 0
        features['bbox_bottom_pct'] = 0
        features['bbox_height_pct'] = 0
        features['bbox_center_y_pct'] = 0
        features['x_deviation_from_median'] = 0
        features['y_deviation_from_median'] = 0
        features['x_deviation_from_depth_median'] = 0
        features['is_near_left_margin'] = 0
        features['is_indented'] = 0
        features['is_in_header_zone'] = 0
        features['is_in_footer_zone'] = 0
        features['x_position_relative'] = 0
        features['y_position_relative'] = 0
        features['width_relative'] = 0
    
    # =========================================================================
    # TITLE FEATURES
    # =========================================================================
    title = section.get('topic', '') or ''
    
    features['title_length_chars'] = len(title)
    features['title_length_words'] = len(title.split()) if title else 0
    
    # Deviation from document median title length
    if doc_stats.title_len_std and doc_stats.title_len_std > 0:
        features['title_length_z_score'] = (len(title) - doc_stats.title_len_median) / doc_stats.title_len_std
    else:
        features['title_length_z_score'] = 0
    
    # Title characteristics
    features['title_is_empty'] = 1 if not title.strip() else 0
    features['title_starts_capital'] = 1 if title and title[0].isupper() else 0
    features['title_is_all_caps'] = 1 if title and title.isupper() else 0
    features['title_is_title_case'] = 1 if title and title.istitle() else 0
    
    # Sentence-like indicators (from calculate_title_confidence)
    first_word_lower = title.lower().split()[0] if title.split() else ''
    features['title_starts_with_article'] = 1 if first_word_lower in ['the', 'a', 'an', 'this', 'that'] else 0
    
    imperative_starters = ['see', 'refer', 'note', 'ensure', 'verify', 'check', 'use',
                          'apply', 'follow', 'review', 'contact', 'consult', 'consider']
    features['title_starts_with_imperative'] = 1 if first_word_lower in imperative_starters else 0
    
    # Title starts with month name (likely a date, not a section title)
    # Previously this was a hard filter, now it's a feature for ML
    months_full = ['january', 'february', 'march', 'april', 'may', 'june', 
                   'july', 'august', 'september', 'october', 'november', 'december']
    months_abbrev = ['jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec']
    all_months = months_full + months_abbrev
    
    features['title_starts_with_month'] = 1 if first_word_lower.rstrip('-.') in all_months else 0
    
    # =========================================================================
    # DATE PATTERN DETECTION
    # =========================================================================
    # Pattern: section_number="17" + title="Jan-1999" or "December-97"
    # This is a common false positive where dates get split into section + title
    
    section_num = section.get('section_number', '')
    
    # Check if section number could be a day (1-31)
    is_possible_day = False
    try:
        num_val = int(section_num.replace('.', '').replace('-', '').replace(',', ''))
        is_possible_day = 1 <= num_val <= 31
    except ValueError:
        pass
    features['section_num_is_possible_day'] = 1 if is_possible_day else 0
    
    # Check if title starts with month (full or abbreviated, with optional separator)
    # Patterns: "Jan-1999", "January 1999", "Dec-97", "December, 1997"
    title_lower = title.lower().strip()
    title_starts_month = False
    for month in all_months:
        if title_lower.startswith(month):
            title_starts_month = True
            break
    features['title_starts_with_month_pattern'] = 1 if title_starts_month else 0
    
    # Check if title contains year pattern after month
    # Matches: "Jan-1999", "January 1999", "Dec-97", "Mar, 2001"
    year_after_month_pattern = re.search(
        r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)'
        r'[\s,\-\.]*'
        r'(\d{2,4})($|\s|[,\.])',
        title_lower
    )
    features['title_is_month_year'] = 1 if year_after_month_pattern else 0
    
    # Combined date flag: section looks like day AND title looks like month-year
    features['looks_like_date'] = 1 if (is_possible_day and (title_starts_month or year_after_month_pattern)) else 0
    
    # Also check for other date patterns in title
    # e.g., "1999", "12/25/1999", "1999-01-15"
    has_date_pattern = bool(re.search(
        r'(^|\s)(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|'  # 12/25/1999 or 12-25-99
        r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'  # 1999-12-25
        r'(^(19|20)\d{2}($|\s|[,\.]))',  # Standalone year like 1999 or 2001
        title
    ))
    features['title_has_date_pattern'] = 1 if has_date_pattern else 0
    
    # Contains parenthetical
    features['title_has_parenthetical'] = 1 if re.search(r'\([^)]*\)', title) else 0
    
    # Common section keywords
    title_keywords = ['introduction', 'scope', 'requirements', 'overview', 'summary',
                     'description', 'specification', 'interface', 'design', 'test',
                     'verification', 'validation', 'general', 'system', 'software',
                     'hardware', 'performance', 'functional', 'applicable', 'documents',
                     'definitions', 'acronyms', 'abbreviations', 'references', 'appendix']
    features['title_has_section_keyword'] = 1 if any(kw in title.lower() for kw in title_keywords) else 0
    
    # Average word length
    words = title.split()
    features['title_avg_word_length'] = sum(len(w) for w in words) / len(words) if words else 0
    
    # =========================================================================
    # TITLE LENGTH FLAGS (detecting missed sections / absorbed content)
    # =========================================================================
    # When a section header is missed, the "title" can become absurdly long
    # because it absorbs body content. Normal titles are 1-6 words typically.
    
    # Absolute length flags
    features['title_suspiciously_long'] = 1 if len(title) > 100 else 0  # > 100 chars
    features['title_very_long'] = 1 if len(title) > 200 else 0  # > 200 chars
    features['title_absurdly_long'] = 1 if len(title) > 500 else 0  # > 500 chars - almost certainly wrong
    
    # Word count flags
    features['title_too_many_words'] = 1 if features['title_length_words'] > 10 else 0
    features['title_sentence_length'] = 1 if features['title_length_words'] > 15 else 0  # Definitely a sentence
    
    # Check for newlines in title (suggests multiple lines were captured)
    features['title_has_newline'] = 1 if '\n' in title else 0
    features['title_newline_count'] = title.count('\n')
    
    # Check for multiple sentences (multiple periods followed by space+capital)
    sentence_breaks = len(re.findall(r'\.\s+[A-Z]', title))
    features['title_sentence_break_count'] = sentence_breaks
    features['title_has_multiple_sentences'] = 1 if sentence_breaks > 0 else 0
    
    # =========================================================================
    # LINGUISTIC FEATURES (SpaCy) - Title vs Sentence Detection
    # =========================================================================
    # These features help distinguish real section titles from sentences/phrases
    # that happened to start with a number-like pattern.
    #
    # Key insight: Section titles are typically noun phrases ("Scope", "System Overview")
    # while false positives are often sentences ("This system shall...", "Send data...")
    
    if title.strip():
        doc = _nlp(title.strip()[:200])  # Limit length for efficiency
        
        # 1. ROOT POS: Titles are usually NOUN/PROPN. Sentences have VERB/AUX roots.
        # "Scope" -> Root is NOUN (Good)
        # "This system shall..." -> Root is VERB (Bad)
        root_tokens = [token for token in doc if token.dep_ == "ROOT"]
        root_pos = root_tokens[0].pos_ if root_tokens else "NOUN"
        
        features['lang_root_is_verb'] = 1 if root_pos in ["VERB", "AUX"] else 0
        features['lang_root_is_noun'] = 1 if root_pos in ["NOUN", "PROPN"] else 0
        
        # 2. FINITE VERBS: Titles rarely have conjugated verbs.
        # "shall be", "must verify", "is connected" -> likely a sentence
        has_finite_verb = any(
            t.pos_ in ["VERB", "AUX"] and t.tag_ in ["VB", "VBD", "VBP", "VBZ", "MD"]
            for t in doc
        )
        features['lang_has_finite_verb'] = 1 if has_finite_verb else 0
        
        # Also check for any verb (including infinitives)
        features['lang_has_verb'] = 1 if any(t.pos_ in ["VERB", "AUX"] for t in doc) else 0
        
        # 3. STARTS WITH DETERMINER: "The", "A", "This"
        # Titles CAN start with "The" but it's a mild negative signal
        features['lang_starts_det'] = 1 if doc[0].pos_ == "DET" else 0
        
        # 4. IMPERATIVE CHECK: "Send serial data..." (Starts with verb base form)
        # If first word is VERB and has no subject, it's likely an instruction
        first_token = doc[0]
        is_imperative = (
            first_token.pos_ == "VERB" and 
            first_token.tag_ == "VB" and
            not any(t.dep_ == "nsubj" for t in doc)
        )
        features['lang_starts_imperative'] = 1 if is_imperative else 0
        
        # 5. SUBJECT-VERB STRUCTURE: Full sentences have subjects
        has_subject = any(t.dep_ in ["nsubj", "nsubjpass"] for t in doc)
        features['lang_has_subject'] = 1 if has_subject else 0
        
        # 6. SENTENCE COMPLETENESS: Does it look like a complete sentence?
        # (Has subject AND verb AND potentially object)
        features['lang_is_complete_sentence'] = 1 if (has_subject and has_finite_verb) else 0
        
        # 7. MODAL VERBS: "shall", "must", "should" indicate requirements, not titles
        has_modal = any(t.tag_ == "MD" for t in doc)
        features['lang_has_modal'] = 1 if has_modal else 0
        
        # 8. Word count from spaCy (excludes punctuation)
        word_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        features['lang_word_count'] = len(word_tokens)
    else:
        # Empty title - set all linguistic features to 0
        features['lang_root_is_verb'] = 0
        features['lang_root_is_noun'] = 0
        features['lang_has_finite_verb'] = 0
        features['lang_has_verb'] = 0
        features['lang_starts_det'] = 0
        features['lang_starts_imperative'] = 0
        features['lang_has_subject'] = 0
        features['lang_is_complete_sentence'] = 0
        features['lang_has_modal'] = 0
        features['lang_word_count'] = 0
    
    # =========================================================================
    # ENHANCED BLANK/EMPTY TITLE FEATURES
    # =========================================================================
    # Sections with blank titles are almost always false positives
    # These features help the model strongly penalize blank titles
    
    title_stripped = title.strip()
    
    # Various "empty" conditions
    features['title_is_whitespace_only'] = 1 if title and not title_stripped else 0
    features['title_is_truly_empty'] = 1 if not title else 0
    features['title_is_blank'] = 1 if not title_stripped else 0  # Combined: empty or whitespace
    
    # Very short titles (1-2 chars) are suspicious
    features['title_is_very_short'] = 1 if 0 < len(title_stripped) <= 2 else 0
    
    # Title is just punctuation or symbols
    features['title_is_punctuation_only'] = 1 if title_stripped and all(c in string.punctuation for c in title_stripped) else 0
    
    # Title is just numbers (might be a misidentified number)
    features['title_is_numeric_only'] = 1 if title_stripped and title_stripped.replace('.', '').replace('-', '').isdigit() else 0
    
    # Combined "useless title" flag
    features['title_is_useless'] = 1 if (
        features['title_is_blank'] or 
        features['title_is_very_short'] or 
        features['title_is_punctuation_only'] or
        features['title_is_numeric_only']
    ) else 0
    
    # =========================================================================
    # RELATIVE HEIGHT FEATURES (Multi-line text detection)
    # =========================================================================
    # True section headers are typically single-line
    # False positives that capture paragraphs span multiple lines
    # This uses bbox height relative to typical line height
    
    bbox = section.get('bbox', {})
    if bbox and doc_stats.section_heights:
        section_height = bbox.get('height', 0)
        median_height = statistics.median(doc_stats.section_heights)
        
        if median_height > 0:
            features['relative_height'] = section_height / median_height
            features['height_is_multiline'] = 1 if section_height > median_height * 1.8 else 0
            features['height_is_very_tall'] = 1 if section_height > median_height * 3 else 0
        else:
            features['relative_height'] = 1.0
            features['height_is_multiline'] = 0
            features['height_is_very_tall'] = 0
    else:
        features['relative_height'] = 1.0
        features['height_is_multiline'] = 0
        features['height_is_very_tall'] = 0
    
    # =========================================================================
    # POSITION WITHIN PAGE FEATURES (Late-page junk detection)
    # =========================================================================
    # Observation: After section 6-ish, there tends to be more junk numbers
    # (page numbers, table data, reference numbers, etc.)
    # 
    # Simple approach: track vertical position within the page
    # Sections in the bottom half of a page are more likely to be false positives
    # (especially if they don't fit the sequence pattern)
    
    if bbox and doc_stats.page_height > 0:
        y_pos = bbox.get('top', 0)
        
        # Vertical position as percentage of page (0 = top, 1 = bottom)
        features['page_position_y_pct'] = y_pos / doc_stats.page_height
        
        # Is this in the bottom third of the page?
        features['is_in_bottom_third'] = 1 if features['page_position_y_pct'] > 0.67 else 0
        
        # Is this in the bottom quarter?
        features['is_in_bottom_quarter'] = 1 if features['page_position_y_pct'] > 0.75 else 0
        
        # Combined risk: late in page AND doesn't fit sequence
        late_page = features['page_position_y_pct'] > 0.67
        bad_sequence = features.get('is_logical_next', 1) == 0 and features.get('next_is_logical_successor', 1) == 0
        features['late_page_bad_sequence'] = 1 if (late_page and bad_sequence) else 0
        
        # Late page with blank/useless title = very suspicious
        features['late_page_useless_title'] = 1 if (late_page and features.get('title_is_useless', 0)) else 0
    else:
        features['page_position_y_pct'] = 0.5  # Default to middle
        features['is_in_bottom_third'] = 0
        features['is_in_bottom_quarter'] = 0
        features['late_page_bad_sequence'] = 0
        features['late_page_useless_title'] = 0
    
    # =========================================================================
    # SECTION NUMBER MAGNITUDE FEATURES
    # =========================================================================
    # Major sections > 6 are less common and numbers after that are more likely
    # to be false positives (page numbers, figure numbers, etc.)
    # 
    # NOTE: This is a soft signal, not a hard rule. Some docs have 10+ major sections.
    
    if major is not None:
        features['major_section_gt_6'] = 1 if major > 6 else 0
        features['major_section_gt_10'] = 1 if major > 10 else 0
        features['major_section_gt_20'] = 1 if major > 20 else 0  # Almost certainly wrong
    else:
        features['major_section_gt_6'] = 0
        features['major_section_gt_10'] = 0
        features['major_section_gt_20'] = 0
    
    # =========================================================================
    # CONTENT FEATURES (if available)
    # =========================================================================
    content = section.get('content', '') or ''
    features['content_length_chars'] = len(content)
    features['content_length_words'] = len(content.split()) if content else 0
    features['content_is_empty'] = 1 if not content.strip() else 0
    
    # Ratio of title to content length (high ratio = suspicious, title too long relative to content)
    if features['content_length_chars'] > 0:
        features['title_to_content_ratio'] = features['title_length_chars'] / features['content_length_chars']
    else:
        features['title_to_content_ratio'] = 0 if features['title_length_chars'] == 0 else 999  # No content but has title
    
    # =========================================================================
    # TOC MATCHING FEATURES
    # =========================================================================
    if toc_sections:
        features['in_toc_exact'] = 1 if parsed.normalized in toc_sections else 0
        features['toc_fuzzy_score'] = _calculate_toc_fuzzy_score(parsed.normalized, toc_sections)
        
        # Check if parent is in TOC
        if parent:
            features['parent_in_toc'] = 1 if parent in toc_sections else 0
        else:
            features['parent_in_toc'] = 0
    else:
        features['in_toc_exact'] = 0
        features['toc_fuzzy_score'] = 0.5  # Neutral
        features['parent_in_toc'] = 0
    
    # =========================================================================
    # CONTEXT FEATURES
    # =========================================================================
    features['page_number'] = section.get('page_number', 0)
    features['page_number_normalized'] = section.get('page_number', 0) / doc_stats.total_pages if doc_stats.total_pages else 0
    
    # Position in sequence
    features['sequence_position'] = idx
    features['sequence_position_normalized'] = idx / len(all_sections) if all_sections else 0
    
    # Sections on same page
    same_page_count = sum(1 for s in all_sections if s.get('page_number') == section.get('page_number'))
    features['sections_on_same_page'] = same_page_count
    
    # =========================================================================
    # MANUAL LABELING HELPER FEATURES
    # =========================================================================
    # These features help with manual labeling but are also useful for ML
    # They represent "soft" indicators that a human might use to judge validity
    
    # 1. Bidirectional logical sequence: both prev->this AND this->next are logical
    #    Strong indicator of a valid section in an intact sequence
    if idx > 0 and idx < len(all_sections) - 1:
        features['in_logical_sequence'] = 1 if (features['is_logical_next'] and features['next_is_logical_successor']) else 0
    else:
        features['in_logical_sequence'] = features.get('is_logical_next', 0)
    
    # 2. Kalman-style prediction confidence
    #    How well does this section fit the expected pattern?
    features['sequence_fit_score'] = _compute_sequence_fit_score(idx, all_sections, parsed)
    
    # 3. Structural consistency: does this section's format match others at same depth?
    features['format_consistency_score'] = _compute_format_consistency_score(section, all_sections, parsed)
    
    # 4. Title quality score: how "title-like" is the title?
    features['title_quality_score'] = _compute_title_quality_score(title)
    
    # 5. Combined confidence score (weighted combination of positive indicators)
    features['combined_confidence'] = _compute_combined_confidence(features)
    
    return features


def _is_logical_next(prev: ParsedSection, curr: ParsedSection) -> int:
    """Check if curr is a logical next section after prev."""
    if not prev.is_valid or not curr.is_valid:
        return 0
    
    prev_parts = prev.get_numeric_parts()
    curr_parts = curr.get_numeric_parts()
    
    if not prev_parts or not curr_parts:
        return 0
    
    # Same depth, last component incremented by 1
    if len(prev_parts) == len(curr_parts):
        if prev_parts[:-1] == curr_parts[:-1]:  # Same prefix
            if prev_parts[-1] is not None and curr_parts[-1] is not None:
                if curr_parts[-1] == prev_parts[-1] + 1:
                    return 1
    
    # Going deeper (e.g., 1.1 -> 1.1.1)
    if len(curr_parts) == len(prev_parts) + 1:
        if curr_parts[:-1] == prev_parts:
            if curr_parts[-1] == 1:  # First subsection
                return 1
    
    # Going up and incrementing (e.g., 1.1.9 -> 1.2)
    if len(curr_parts) < len(prev_parts):
        # Check if it's incrementing at the right level
        prefix_len = len(curr_parts) - 1
        if prefix_len >= 0:
            # Check prefix matches
            if prev_parts[:prefix_len] == curr_parts[:prefix_len]:
                # Check last part increments
                if prev_parts[prefix_len] is not None and curr_parts[-1] is not None:
                    if curr_parts[-1] == prev_parts[prefix_len] + 1:
                        return 1
    
    return 0


def _compute_extended_sequence_score(idx: int, all_sections: List[Dict]) -> float:
    """
    Compute a sequence consistency score looking at extended neighborhood.
    Higher score = more consistent with surrounding sections (likely valid).
    
    This replaces the sandwich detection which had negative correlation.
    """
    if len(all_sections) < 3:
        return 0.5  # Neutral for very short sequences
    
    curr_parsed = ParsedSection(all_sections[idx].get('section_number', ''))
    curr_major = curr_parsed.get_major()
    
    if curr_major is None:
        return 0.3  # Low score for unparseable
    
    score = 0.0
    checks = 0
    
    # Check logical progression with immediate neighbors
    if idx > 0:
        prev_parsed = ParsedSection(all_sections[idx - 1].get('section_number', ''))
        if _is_logical_next(prev_parsed, curr_parsed):
            score += 1.0
        checks += 1
    
    if idx < len(all_sections) - 1:
        next_parsed = ParsedSection(all_sections[idx + 1].get('section_number', ''))
        if _is_logical_next(curr_parsed, next_parsed):
            score += 1.0
        checks += 1
    
    # Check if major section is consistent with neighbors (within reasonable range)
    prev_majors = []
    for i in range(max(0, idx - 3), idx):
        p = ParsedSection(all_sections[i].get('section_number', ''))
        m = p.get_major()
        if m is not None:
            prev_majors.append(m)
    
    next_majors = []
    for i in range(idx + 1, min(len(all_sections), idx + 4)):
        p = ParsedSection(all_sections[i].get('section_number', ''))
        m = p.get_major()
        if m is not None:
            next_majors.append(m)
    
    if prev_majors:
        # Is current major within reasonable range of previous majors?
        max_prev = max(prev_majors)
        min_prev = min(prev_majors)
        if min_prev <= curr_major <= max_prev + 2:  # Allow +2 for progression
            score += 0.5
        checks += 1
    
    if next_majors:
        # Is current major consistent with next majors?
        min_next = min(next_majors)
        max_next = max(next_majors)
        if curr_major - 1 <= min_next <= curr_major + 2:
            score += 0.5
        checks += 1
    
    return score / checks if checks > 0 else 0.5


def _calculate_toc_fuzzy_score(section_number: str, toc_sections: Set[str]) -> float:
    """Calculate fuzzy match score against TOC."""
    if not toc_sections or not section_number:
        return 0.5
    
    if section_number in toc_sections:
        return 1.0
    
    parts = section_number.split('.')
    
    # Check prefix matching
    best_match = 0
    for toc_entry in toc_sections:
        toc_parts = toc_entry.split('.')
        match_count = 0
        for p1, p2 in zip(parts, toc_parts):
            try:
                if int(p1) == int(p2):
                    match_count += 1
                else:
                    break
            except ValueError:
                if p1.upper() == p2.upper():
                    match_count += 1
                else:
                    break
        best_match = max(best_match, match_count)
    
    if best_match >= 2:
        return 0.7 + 0.1 * min(best_match - 1, 3)
    elif best_match == 1:
        return 0.4
    else:
        return 0.1


def _compute_sequence_fit_score(idx: int, all_sections: List[Dict], parsed: ParsedSection) -> float:
    """
    Compute how well this section fits the expected sequence pattern.
    Uses a simple Kalman-like prediction approach.
    """
    if idx == 0:
        # First section: check if it's a reasonable starting point (1, 1.1, etc.)
        if parsed.normalized in ['1', '1.1', '1.0']:
            return 1.0
        major = parsed.get_major()
        if major is not None and major <= 3:
            return 0.7
        return 0.3
    
    score = 0.0
    
    # Check logical next
    prev_parsed = ParsedSection(all_sections[idx - 1].get('section_number', ''))
    if _is_logical_next(prev_parsed, parsed):
        score += 0.5
    
    # Check if major section makes sense in context
    prev_major = prev_parsed.get_major()
    curr_major = parsed.get_major()
    
    if prev_major is not None and curr_major is not None:
        gap = curr_major - prev_major
        if gap == 0:
            score += 0.3  # Same major, likely subsection
        elif gap == 1:
            score += 0.4  # Next major section
        elif gap > 1:
            score -= 0.1 * (gap - 1)  # Penalize jumps
    
    # Check depth consistency
    if idx >= 2:
        depths = [ParsedSection(s.get('section_number', '')).depth for s in all_sections[max(0, idx-3):idx]]
        if depths:
            avg_depth = sum(depths) / len(depths)
            depth_diff = abs(parsed.depth - avg_depth)
            if depth_diff <= 1:
                score += 0.2
    
    return max(0, min(1, score))


def _compute_format_consistency_score(section: Dict, all_sections: List[Dict], parsed: ParsedSection) -> float:
    """
    Check if this section's format matches others at the same depth.
    """
    depth = parsed.depth
    same_depth_sections = [s for s in all_sections 
                          if ParsedSection(s.get('section_number', '')).depth == depth
                          and s != section]
    
    if not same_depth_sections:
        return 0.5  # No comparison available
    
    score = 0.0
    checks = 0
    
    # Check X position consistency
    bbox = section.get('bbox', {})
    if bbox:
        curr_x = bbox.get('left', 0)
        other_xs = [s.get('bbox', {}).get('left', 0) for s in same_depth_sections if s.get('bbox')]
        if other_xs:
            median_x = statistics.median(other_xs)
            if abs(curr_x - median_x) < 50:  # Within 50px
                score += 0.5
            checks += 1
    
    # Check title style consistency
    title = section.get('topic', '')
    if title:
        curr_is_caps = title.isupper()
        curr_is_title = title.istitle()
        
        other_titles = [s.get('topic', '') for s in same_depth_sections if s.get('topic')]
        if other_titles:
            caps_count = sum(1 for t in other_titles if t.isupper())
            title_count = sum(1 for t in other_titles if t.istitle())
            
            if caps_count > len(other_titles) / 2 and curr_is_caps:
                score += 0.3
            elif title_count > len(other_titles) / 2 and curr_is_title:
                score += 0.3
            checks += 1
    
    return score / checks if checks > 0 else 0.5


def _compute_title_quality_score(title: str) -> float:
    """
    Compute how "title-like" the title text is.
    Higher scores indicate more typical section titles.
    """
    if not title or not title.strip():
        return 0.0
    
    score = 0.0
    
    # Length check (typical titles are 2-50 chars)
    length = len(title)
    if 2 <= length <= 50:
        score += 0.3
    elif 50 < length <= 100:
        score += 0.1
    
    # Word count check (typical titles are 1-8 words)
    word_count = len(title.split())
    if 1 <= word_count <= 8:
        score += 0.3
    elif 8 < word_count <= 15:
        score += 0.1
    
    # Starts with capital
    if title[0].isupper():
        score += 0.1
    
    # No sentence-ending punctuation mid-title
    if not re.search(r'\.\s+[A-Z]', title):
        score += 0.1
    
    # Contains section keyword
    keywords = ['introduction', 'scope', 'requirements', 'overview', 'summary',
               'description', 'specification', 'interface', 'design', 'test',
               'general', 'system', 'software', 'hardware', 'definitions', 'references']
    if any(kw in title.lower() for kw in keywords):
        score += 0.2
    
    return min(1.0, score)


def _compute_combined_confidence(features: Dict) -> float:
    """
    Compute a combined confidence score from multiple positive indicators.
    This serves as a "manual labeling helper" feature.
    
    Updated v3: Added linguistic features, blank title penalties, and late-page penalties.
    """
    score = 0.0
    
    # =======================================================================
    # POSITIVE INDICATORS
    # =======================================================================
    if features.get('is_logical_next'):
        score += 0.15
    if features.get('next_is_logical_successor'):
        score += 0.15
    if features.get('in_toc_exact'):
        score += 0.2
    if features.get('parent_exists'):
        score += 0.1
    if features.get('title_has_section_keyword'):
        score += 0.1
    if features.get('title_quality_score', 0) > 0.5:
        score += 0.1
    if features.get('has_large_vertical_gap'):
        score += 0.05
    if not features.get('had_text_before_number'):
        score += 0.1
    
    # Linguistic positive: root is noun (title-like)
    if features.get('lang_root_is_noun'):
        score += 0.05
    
    # =======================================================================
    # NEGATIVE INDICATORS
    # =======================================================================
    if features.get('looks_like_date'):
        score -= 0.3
    if features.get('title_absurdly_long'):
        score -= 0.2
    if features.get('subsection_looks_like_decimal'):
        score -= 0.15
    if features.get('had_text_before_number'):
        score -= 0.1
    if features.get('has_small_vertical_gap'):
        score -= 0.05
    
    # --- BLANK TITLE PENALTIES (new in v3) ---
    if features.get('title_is_blank'):
        score -= 0.3  # Strong penalty for blank titles
    if features.get('title_is_useless'):
        score -= 0.2  # Penalty for useless titles (punctuation only, etc.)
    
    # --- LINGUISTIC PENALTIES (new in v3) ---
    if features.get('lang_root_is_verb'):
        score -= 0.1  # Likely a sentence, not a title
    if features.get('lang_has_finite_verb'):
        score -= 0.1  # Has conjugated verb
    if features.get('lang_is_complete_sentence'):
        score -= 0.15  # Definitely looks like a sentence
    if features.get('lang_has_modal'):
        score -= 0.1  # "shall", "must" = requirement text, not title
    if features.get('lang_starts_imperative'):
        score -= 0.1  # Instruction, not title
    
    # --- LATE PAGE PENALTIES (new in v3) ---
    if features.get('late_page_bad_sequence'):
        score -= 0.15  # Late in page AND doesn't fit sequence
    if features.get('late_page_useless_title'):
        score -= 0.2  # Late in page with blank/useless title
    
    # --- HEIGHT PENALTIES (new in v3) ---
    if features.get('height_is_multiline'):
        score -= 0.1  # Spans multiple lines = probably not a header
    if features.get('height_is_very_tall'):
        score -= 0.15  # Very tall = definitely not a single-line header
    
    # --- MAJOR SECTION PENALTIES (new in v3) ---
    if features.get('major_section_gt_20'):
        score -= 0.2  # Major section > 20 is almost certainly wrong
    elif features.get('major_section_gt_10'):
        score -= 0.1  # Major section > 10 is suspicious
    
    return max(0, min(1, score))


# =============================================================================
# MAIN PROCESSING (using pandas)
# =============================================================================

def collect_document_stats(elements: List[Dict], page_metadata: Dict) -> DocumentStats:
    """Collect statistics from all sections in a document."""
    stats = DocumentStats()
    
    # Get page dimensions from metadata
    if page_metadata:
        first_page_meta = next(iter(page_metadata.values()), {})
        stats.page_width = first_page_meta.get('page_width', 0)
        stats.page_height = first_page_meta.get('page_height', 0)
        stats.total_pages = len(page_metadata)
    
    # First pass: collect basic stats
    sections_with_bbox = []
    for elem in elements:
        if elem.get('type') != 'section':
            continue
        
        parsed = ParsedSection(elem.get('section_number', ''))
        if not parsed.is_valid:
            continue
        
        # Section number stats
        stats.depths.append(parsed.depth)
        stats.all_section_numbers.add(parsed.normalized)
        
        major = parsed.get_major()
        if major is not None:
            stats.major_sections_seen.add(major)
        
        # Bounding box stats
        bbox = elem.get('bbox', {})
        if bbox:
            x = bbox.get('left', 0)
            y = bbox.get('top', 0)
            
            stats.section_x_positions.append(x)
            stats.section_y_positions.append(y)
            stats.section_widths.append(bbox.get('width', 0))
            stats.section_heights.append(bbox.get('height', 0))
            
            # By depth
            stats.x_by_depth[parsed.depth].append(x)
            
            # Save for vertical gap computation
            sections_with_bbox.append({
                'page': elem.get('page_number', 0),
                'top': y,
                'bottom': y + bbox.get('height', 0)
            })
        
        # Title stats
        title = elem.get('topic', '') or ''
        stats.title_lengths.append(len(title))
        stats.title_word_counts.append(len(title.split()) if title else 0)
    
    # Second pass: compute vertical gaps between consecutive sections on same page
    sections_with_bbox.sort(key=lambda x: (x['page'], x['top']))
    for i in range(1, len(sections_with_bbox)):
        prev = sections_with_bbox[i - 1]
        curr = sections_with_bbox[i]
        
        if prev['page'] == curr['page'] and curr['top'] >= prev['bottom']:
            gap = curr['top'] - prev['bottom']
            stats.vertical_gaps.append(gap)
    
    stats.compute_stats()
    return stats


def load_toc_sections(classification_path: str, raw_ocr_path: str) -> Set[str]:
    """Load TOC sections from classification and raw OCR."""
    toc_sections = set()
    
    if not os.path.exists(classification_path) or not os.path.exists(raw_ocr_path):
        return toc_sections
    
    try:
        with open(classification_path, 'r') as f:
            classification = json.load(f)
    except:
        return toc_sections
    
    # Find TOC pages
    toc_pages = set()
    for pc in classification.get('page_classifications', []):
        if pc.get('type') == 'TABLE_OF_CONTENTS':
            toc_pages.add(pc.get('page'))
    
    if not toc_pages:
        return toc_sections
    
    # Load raw OCR and extract section numbers from TOC pages
    try:
        with open(raw_ocr_path, 'r') as f:
            raw_data = json.load(f)
    except:
        return toc_sections
    
    # Extract text from TOC pages
    toc_text = ""
    if isinstance(raw_data, dict):
        for key, val in raw_data.items():
            try:
                page_num = int(key)
            except:
                continue
            if page_num in toc_pages:
                page_dict = val.get('page_dict', val) if isinstance(val, dict) else {}
                if isinstance(page_dict, dict):
                    text_list = page_dict.get('text', [])
                    if isinstance(text_list, list):
                        toc_text += " ".join(str(t) for t in text_list) + "\n"
    
    # Extract section numbers
    pattern = re.compile(r'(?<![0-9])(\d{1,2}(?:\.\d{1,3}){1,6})(?![0-9])')
    for match in pattern.findall(toc_text):
        normalized = match.replace(',', '.').strip('.')
        if normalized:
            toc_sections.add(normalized)
    
    return toc_sections


def process_document(
    organized_path: str,
    classification_path: str,
    raw_ocr_path: str,
    doc_name: str
) -> List[Dict]:
    """Process a single document and extract features for all sections."""
    
    # Load organized data
    if not os.path.exists(organized_path):
        print(f"  [Skip] {doc_name}: organized file not found")
        return []
    
    try:
        with open(organized_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [Error] {doc_name}: {e}")
        return []
    
    if isinstance(data, dict):
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
    
    # Filter to sections only
    sections = [e for e in elements if e.get('type') == 'section']
    
    if not sections:
        print(f"  [Skip] {doc_name}: no sections found")
        return []
    
    # Collect document statistics
    doc_stats = collect_document_stats(elements, page_metadata)
    
    # Load TOC sections
    toc_sections = load_toc_sections(classification_path, raw_ocr_path)
    
    # Extract features for each section
    all_features = []
    for idx, section in enumerate(sections):
        features = extract_features_for_section(
            section, idx, sections, doc_stats, toc_sections, doc_name
        )
        all_features.append(features)
    
    print(f"  [OK] {doc_name}: {len(sections)} sections")
    return all_features


# Check for pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Warning] pandas not installed. Some features will be limited.")
    print("  Install with: pip install pandas")


def process_directory(
    results_dir: str = DEFAULT_RESULTS_DIR,
    raw_ocr_dir: str = DEFAULT_RAW_OCR_DIR,
    output_csv: str = None
) -> 'pd.DataFrame':
    """
    Process all documents in a results directory.
    
    Args:
        results_dir: Directory containing pipeline output files
        raw_ocr_dir: Directory containing raw OCR JSON files
        output_csv: Path to save CSV (optional)
        
    Returns:
        pandas DataFrame with all features
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    all_features = []
    
    # Find all organized files
    organized_files = []
    for f in os.listdir(results_dir):
        if f.endswith('_organized.json'):
            stem = f.replace('_organized.json', '')
            organized_files.append((stem, f))
    
    if not organized_files:
        print(f"No organized files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Processing {len(organized_files)} documents...")
    
    for stem, organized_file in sorted(organized_files):
        organized_path = os.path.join(results_dir, organized_file)
        classification_path = os.path.join(results_dir, f"{stem}_classification.json")
        raw_ocr_path = os.path.join(raw_ocr_dir, f"{stem}.json")
        
        features = process_document(organized_path, classification_path, raw_ocr_path, stem)
        all_features.extend(features)
    
    if not all_features:
        print("No features extracted!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns: metadata first, then features
    meta_cols = [c for c in df.columns if c.startswith('_')]
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    df = df[meta_cols + sorted(feature_cols)]
    
    # Preserve existing labels if output file exists
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if '_label' in existing_df.columns:
                # Get labeled rows
                labeled_df = existing_df[existing_df['_label'].notna() & (existing_df['_label'] != '')]
                
                if len(labeled_df) > 0:
                    print(f"  Found {len(labeled_df)} existing labels to preserve")
                    
                    # Create composite key for matching using CONTENT-BASED keys only
                    # NOTE: We intentionally exclude '_index' because it shifts when sections
                    # are added/removed, causing labels to be applied to wrong rows.
                    # Using content-based keys ensures labels survive re-runs of the pipeline.
                    primary_key_cols = ['_doc_name', '_section_number_raw', '_page', '_title']
                    
                    # Check which key columns exist in both dataframes
                    old_key_cols = []
                    for col in primary_key_cols:
                        if col in existing_df.columns:
                            old_key_cols.append(col)
                        elif col == '_section_number_raw' and '_section_number' in existing_df.columns:
                            # Handle old column name
                            old_key_cols.append('_section_number')
                    
                    new_key_cols = [c for c in primary_key_cols if c in df.columns]
                    
                    if len(old_key_cols) >= 3 and len(new_key_cols) >= 3:
                        # Create composite key for existing labels
                        labeled_df = labeled_df.copy()
                        labeled_df['_merge_key'] = labeled_df[old_key_cols].fillna('').astype(str).agg('|'.join, axis=1)
                        
                        # For duplicates in labeled data, they should have the same label
                        # Use first label encountered for each key (duplicates should match anyway)
                        label_map = {}
                        for _, row in labeled_df.iterrows():
                            key = row['_merge_key']
                            if key not in label_map:
                                label_map[key] = row['_label']
                        
                        # Create composite key for new data
                        df['_merge_key'] = df[new_key_cols].fillna('').astype(str).agg('|'.join, axis=1)
                        
                        # Count labels before merge
                        labels_before = df['_label'].notna().sum() if '_label' in df.columns else 0
                        
                        # Apply labels - this preserves ALL rows, just adds labels where keys match
                        df['_label'] = df['_merge_key'].map(label_map).fillna(
                            df['_label'] if '_label' in df.columns else pd.NA
                        )
                        
                        labels_after = df['_label'].notna().sum()
                        
                        # Clean up merge key
                        df = df.drop(columns=['_merge_key'])
                        
                        preserved = labels_after - labels_before
                        print(f"  Preserved {preserved} labels from existing file")
                        
                        # Check for labels that couldn't be matched
                        unique_old_keys = set(labeled_df['_merge_key'])
                        unique_new_keys = set(df[new_key_cols].fillna('').astype(str).agg('|'.join, axis=1))
                        unmatched_keys = unique_old_keys - unique_new_keys
                        
                        if unmatched_keys:
                            print(f"  Warning: {len(unmatched_keys)} labeled rows from old file could not be matched")
                            print(f"           (sections may have been removed or changed)")
                    else:
                        print(f"  Warning: Not enough key columns for matching (old: {old_key_cols}, new: {new_key_cols})")
                else:
                    print("  No existing labels found to preserve")
            else:
                print("  Existing file has no _label column")
                
        except Exception as e:
            print(f"  Warning: Could not load existing CSV for label preservation: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows (sections): {len(df)}")
    print(f"Documents processed:   {df['_doc_name'].nunique()}")
    print(f"Feature columns:       {len(feature_cols)}")
    print(f"Metadata columns:      {len(meta_cols)}")
    
    # Label summary
    if '_label' in df.columns:
        labeled_count = df['_label'].notna() & (df['_label'] != '')
        labeled_count = labeled_count.sum()
        print(f"Rows with labels:      {labeled_count} / {len(df)} ({100*labeled_count/len(df):.1f}%)")
    
    print(f"\nNew features in this version:")
    new_features = [
        'vertical_gap_from_prev_px', 'vertical_gap_from_prev_pct', 'vertical_gap_z_score',
        'has_large_vertical_gap', 'has_small_vertical_gap',
        'newlines_before_section_num', 'next_is_logical_successor',
        'in_logical_sequence', 'sequence_fit_score', 'format_consistency_score',
        'title_quality_score', 'combined_confidence', 'extended_sequence_score'
    ]
    for f in new_features:
        if f in feature_cols:
            print(f"  - {f}")
    
    print(f"\nRemoved features (negative correlation):")
    removed_features = ['is_sandwiched', 'sandwich_same_neighbors']
    for f in removed_features:
        print(f"  - {f}")
    
    # Save to CSV if path provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    
    return df


def load_training_data(csv_path: str) -> 'pd.DataFrame':
    """
    Load previously saved training data CSV.
    
    Useful for resuming labeling or training a model.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    df = pd.read_csv(csv_path)
    
    # Summary
    total = len(df)
    labeled = df['_label'].notna().sum()
    unlabeled = total - labeled
    
    print(f"Loaded {total} rows from {csv_path}")
    print(f"  Labeled:   {labeled} ({100*labeled/total:.1f}%)")
    print(f"  Unlabeled: {unlabeled} ({100*unlabeled/total:.1f}%)")
    
    if labeled > 0:
        positives = (df['_label'] == 1).sum()
        negatives = (df['_label'] == 0).sum()
        print(f"  Positives (valid sections): {positives}")
        print(f"  Negatives (false positives): {negatives}")
    
    return df


def get_feature_columns(df: 'pd.DataFrame') -> List[str]:
    """Get list of feature column names (excludes metadata columns)."""
    return [c for c in df.columns if not c.startswith('_')]


def get_labeled_data(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Filter to only rows that have been labeled."""
    return df[df['_label'].notna()].copy()


def get_unlabeled_data(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Filter to only rows that need labeling."""
    return df[df['_label'].isna()].copy()


def summary_by_document(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Get summary statistics grouped by document."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required")
    
    summary = df.groupby('_doc_name').agg({
        '_section_number_raw': 'count',
        '_label': lambda x: x.notna().sum(),
        'section_depth': 'mean',
        'in_toc_exact': 'mean',
        'combined_confidence': 'mean',
    }).rename(columns={
        '_section_number_raw': 'num_sections',
        '_label': 'num_labeled',
        'section_depth': 'avg_depth',
        'in_toc_exact': 'toc_match_rate',
        'combined_confidence': 'avg_confidence',
    })
    
    return summary.round(2)


# =============================================================================
# CLI
# =============================================================================

def print_feature_correlations(df: 'pd.DataFrame', min_samples: int = 50) -> None:
    """
    Print correlations of features with the _label column.
    Useful for identifying which features are predictive.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required")
    
    labeled = df[df['_label'].notna()].copy()
    if len(labeled) < min_samples:
        print(f"Need at least {min_samples} labeled samples, have {len(labeled)}")
        return
    
    labeled['_label'] = labeled['_label'].astype(float)
    
    feature_cols = get_feature_columns(df)
    correlations = []
    
    for col in feature_cols:
        if labeled[col].std() > 0:  # Skip constant columns
            corr = labeled['_label'].corr(labeled[col])
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nFeature Correlations with Label (sorted by |correlation|):")
    print("=" * 60)
    print(f"{'Feature':<45} {'Correlation':>12}")
    print("-" * 60)
    
    for col, corr in correlations:
        indicator = "+++" if corr > 0.3 else "++" if corr > 0.2 else "+" if corr > 0.1 else \
                   "---" if corr < -0.3 else "--" if corr < -0.2 else "-" if corr < -0.1 else ""
        print(f"{col:<45} {corr:>10.3f} {indicator}")


def analyze_title_vocabulary(df: 'pd.DataFrame', top_n: int = 25):
    """
    Analyzes and prints the most common words in titles for valid (1) 
    and invalid (0) sections.
    """
    if not HAS_PANDAS: return

    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    # Filter labeled data (0 or 1)
    labeled = df[df['_label'].isin([0, 1])].copy()
    
    if len(labeled) == 0:
        print("No labeled data (0 or 1) found for vocabulary analysis.")
        return

    # Basic stopwords to ignore
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'by', 'at', 
        'is', 'are', 'be', 'as', 'from', 'that', 'this', 'it', 'not', 'have', 'has'
    }

    def get_tokens(titles):
        tokens = []
        for t in titles:
            if not isinstance(t, str): continue
            # Remove punctuation (except internal hyphens sometimes useful, but let's strip all for now)
            clean = t.lower().translate(str.maketrans('', '', string.punctuation))
            # Split and filter
            words = [w for w in clean.split() if w not in stopwords and len(w) > 2 and not w.isdigit()]
            tokens.extend(words)
        return tokens

    # --- Valid Sections (Label = 1) ---
    valid_titles = labeled[labeled['_label'] == 1]['_title']
    valid_tokens = get_tokens(valid_titles)
    valid_counts = Counter(valid_tokens).most_common(top_n)

    print(f"\nTop {top_n} words in VALID sections (Label=1):")
    print(f"Total valid sections analyzed: {len(valid_titles)}")
    print("-" * 60)
    print(f"{'Word':<20} {'Count':<10} {'Freq %':<10}")
    print("-" * 60)
    
    total_valid = len(valid_titles) if len(valid_titles) > 0 else 1
    for word, count in valid_counts:
        pct = (count / total_valid) * 100
        print(f"{word:<20} {count:<10} {pct:.1f}%")

    # --- Invalid Sections (Label = 0) ---
    invalid_titles = labeled[labeled['_label'] == 0]['_title']
    invalid_tokens = get_tokens(invalid_titles)
    invalid_counts = Counter(invalid_tokens).most_common(top_n)

    print(f"\nTop {top_n} words in INVALID sections (Label=0):")
    print(f"Total invalid sections analyzed: {len(invalid_titles)}")
    print("-" * 60)
    print(f"{'Word':<20} {'Count':<10} {'Freq %':<10}")
    print("-" * 60)
    
    total_invalid = len(invalid_titles) if len(invalid_titles) > 0 else 1
    for word, count in invalid_counts:
        pct = (count / total_invalid) * 100
        print(f"{word:<20} {count:<10} {pct:.1f}%")


def print_feature_correlations(df: 'pd.DataFrame', min_samples: int = 20) -> None:
    """
    Print correlations of features with the _label column.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required")
    
    # Ensure we only have numeric types for correlation
    # Filter for labels 0 and 1 (exclude -1 if it slipped through)
    labeled = df[df['_label'].isin([0, 1])].copy()
    
    if len(labeled) < min_samples:
        print(f"Need at least {min_samples} labeled samples (0 or 1), have {len(labeled)}")
        return
    
    labeled['_label'] = labeled['_label'].astype(float)
    
    # Get feature columns (ignore metadata starting with _)
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    
    # Further filter to only numeric columns to avoid errors
    numeric_cols = labeled[feature_cols].select_dtypes(include=['number']).columns
    
    correlations = []
    
    for col in numeric_cols:
        if labeled[col].std() > 0:  # Skip constant columns
            corr = labeled['_label'].corr(labeled[col])
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n" + "="*60)
    print("FEATURE CORRELATIONS (Label 0 vs 1)")
    print("="*60)
    print(f"{'Feature':<45} {'Correlation':>12}")
    print("-" * 60)
    
    for col, corr in correlations:
        indicator = "+++" if corr > 0.5 else "++" if corr > 0.3 else "+" if corr > 0.1 else \
                   "---" if corr < -0.5 else "--" if corr < -0.3 else "-" if corr < -0.1 else ""
        print(f"{col:<45} {corr:>10.3f} {indicator}")


if __name__ == '__main__':
    import argparse
    import numpy as np  # Needed for explicit numeric filtering if not imported
    
    parser = argparse.ArgumentParser(
        description="Extract ML features from processed documents for training a section classifier."
    )
    
    parser.add_argument(
        "--raw_ocr_dir",
        type=str,
        default=DEFAULT_RAW_OCR_DIR,
        help=f"Directory containing raw OCR JSON files (default: {DEFAULT_RAW_OCR_DIR})"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing pipeline results (default: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <results_dir>/training_features.csv)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to CSV, just print summary"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.no_save:
        output_csv = None
    elif args.output:
        output_csv = args.output
    else:
        output_csv = os.path.join(args.results_dir, "training_features.csv")
    
    # Run extraction
    df = process_directory(args.results_dir, args.raw_ocr_dir, output_csv)
    
    # =========================================================================
    # POST-PROCESSING ANALYSIS
    # =========================================================================
    if not df.empty and '_label' in df.columns:
        
        # 1. Load data explicitly to ensure we have the latest (including preserved labels)
        if output_csv and os.path.exists(output_csv):
            analysis_df = pd.read_csv(output_csv)
        else:
            analysis_df = df
            
        # 2. Filter: Remove rows where label is missing or -1 (or any negative number)
        #    We only want confirmed valid (1) and confirmed invalid (0)
        labeled_df = analysis_df[analysis_df['_label'].isin([0, 1])]
        
        if not labeled_df.empty:
            # Set pandas display to show everything if needed (though we print manually mostly)
            pd.set_option('display.max_rows', None)
            
            # 3. Print correlations (automatically handles non-numeric removal inside function)
            print_feature_correlations(labeled_df)
            
            # 4. Analyze vocabulary for valid vs invalid titles
            analyze_title_vocabulary(labeled_df, top_n=25)
            
        else:
            print("\n[Info] No labeled data (0 or 1) found. Open the CSV, label some rows, and run again to see analysis.")