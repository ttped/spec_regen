"""
feature_extractor.py - Extract ML features from processed documents.

This module generates training data for a classifier to determine if a
detected "section" is a true section or a false positive.

Key insight: Features should measure SELF-CONSISTENCY within a document,
not absolute values. Section formatting is consistent within a document
but varies across documents.

Output: CSV with features + metadata for manual labeling
"""

import os
import json
import re
import csv
import statistics
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


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
        normalized = normalized.rstrip('.')
        
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
    
    # Look ahead for sandwich detection
    if idx < len(all_sections) - 1:
        next_section = all_sections[idx + 1]
        next_parsed = ParsedSection(next_section.get('section_number', ''))
        next_major = next_parsed.get_major()
        
        if major is not None and next_major is not None:
            features['major_gap_to_next'] = next_major - major
        else:
            features['major_gap_to_next'] = 0
    else:
        features['major_gap_to_next'] = 0
    
    # =========================================================================
    # SANDWICH FEATURES
    # =========================================================================
    # Check if this section is "sandwiched" between sections with different major
    features['is_sandwiched'] = 0
    features['sandwich_same_neighbors'] = 0
    
    if 0 < idx < len(all_sections) - 1:
        prev_parsed = ParsedSection(all_sections[idx - 1].get('section_number', ''))
        next_parsed = ParsedSection(all_sections[idx + 1].get('section_number', ''))
        
        prev_major = prev_parsed.get_major()
        next_major = next_parsed.get_major()
        
        if prev_major is not None and next_major is not None and major is not None:
            # Sandwiched between same major but this one is different
            if prev_major == next_major and major != prev_major:
                features['is_sandwiched'] = 1
                features['sandwich_same_neighbors'] = 1
    
    # Extended sandwich check (look at 2-3 neighbors each side)
    features['extended_sandwich_score'] = _compute_extended_sandwich_score(idx, all_sections)
    
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
        pass  # Complex logic, skip for now
    
    return 0


def _compute_extended_sandwich_score(idx: int, all_sections: List[Dict]) -> float:
    """
    Compute a sandwich score looking at extended neighborhood.
    Higher score = more likely to be sandwiched (false positive).
    """
    if idx == 0 or idx >= len(all_sections) - 1:
        return 0.0
    
    curr_parsed = ParsedSection(all_sections[idx].get('section_number', ''))
    curr_major = curr_parsed.get_major()
    
    if curr_major is None:
        return 0.0
    
    # Look at 3 neighbors on each side
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
    
    if not prev_majors or not next_majors:
        return 0.0
    
    # Check how "out of place" current major is
    all_neighbor_majors = prev_majors + next_majors
    if curr_major in all_neighbor_majors:
        return 0.0  # Current major appears in neighbors, not sandwiched
    
    # Check if neighbors are consistent with each other
    unique_neighbor_majors = set(all_neighbor_majors)
    if len(unique_neighbor_majors) == 1:
        # All neighbors have same major, current is different -> strong sandwich
        return 1.0
    elif len(unique_neighbor_majors) == 2:
        # Neighbors have 2 majors, could be at boundary
        return 0.3
    else:
        return 0.1


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
    
    # Collect from all sections
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
        
        # Title stats
        title = elem.get('topic', '') or ''
        stats.title_lengths.append(len(title))
        stats.title_word_counts.append(len(title.split()) if title else 0)
    
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


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Install with: pip install pandas")


def process_directory(
    results_dir: str = None,
    raw_ocr_dir: str = None,
    output_csv: str = None
) -> 'pd.DataFrame':
    """
    Process all documents in a directory and return a pandas DataFrame.
    
    If output_csv already exists, labels from the existing file will be preserved
    and merged back into the new DataFrame (matched by doc_name, section_number, 
    title, and page).
    
    Args:
        results_dir: Directory containing *_organized.json and *_classification.json
                    (default: results_simple)
        raw_ocr_dir: Directory containing raw OCR JSON files
                    (default: iris_ocr/CM_Spec_OCR_and_figtab_output/raw_data_advanced)
        output_csv: Optional path to save CSV (if None, just returns DataFrame)
    
    Returns:
        pandas DataFrame with all features
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    # Use defaults if not specified
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if raw_ocr_dir is None:
        raw_ocr_dir = DEFAULT_RAW_OCR_DIR
    
    print(f"Results dir: {results_dir}")
    print(f"Raw OCR dir: {raw_ocr_dir}")
    print()
    
    all_features = []
    
    # Find all organized files
    organized_files = []
    for f in os.listdir(results_dir):
        if f.endswith('_organized.json'):
            stem = f.replace('_organized.json', '')
            organized_files.append((stem, os.path.join(results_dir, f)))
    
    print(f"Found {len(organized_files)} organized files")
    
    for stem, organized_path in sorted(organized_files):
        classification_path = os.path.join(results_dir, f"{stem}_classification.json")
        raw_ocr_path = os.path.join(raw_ocr_dir, f"{stem}.json")
        
        features = process_document(organized_path, classification_path, raw_ocr_path, stem)
        all_features.extend(features)
    
    if not all_features:
        print("No features extracted!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns: metadata first (prefixed with _), then features
    meta_cols = [c for c in df.columns if c.startswith('_')]
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    df = df[meta_cols + feature_cols]
    
    # =========================================================================
    # PRESERVE EXISTING LABELS
    # =========================================================================
    # Primary key for matching: (_doc_name, _section_number_raw, _title, _page)
    primary_key_cols = ['_doc_name', '_section_number_raw', '_title', '_page']
    
    if output_csv and os.path.exists(output_csv):
        print(f"\nFound existing CSV: {output_csv}")
        try:
            existing_df = pd.read_csv(output_csv)
            
            # Check if _label column exists and has any non-empty values
            if '_label' in existing_df.columns:
                # Get rows with labels (not empty/NaN)
                labeled_mask = existing_df['_label'].notna() & (existing_df['_label'] != '')
                labeled_df = existing_df[labeled_mask]
                
                if len(labeled_df) > 0:
                    print(f"  Found {len(labeled_df)} existing labels to preserve")
                    
                    # Create a mapping from primary key to label
                    # Handle potential column name differences (old vs new)
                    old_key_cols = []
                    for col in primary_key_cols:
                        if col in existing_df.columns:
                            old_key_cols.append(col)
                        elif col == '_section_number_raw' and '_section_number' in existing_df.columns:
                            # Handle old column name
                            old_key_cols.append('_section_number')
                    
                    if len(old_key_cols) >= 3:  # Need at least 3 key columns to match
                        # Create composite key for existing labels
                        labeled_df = labeled_df.copy()
                        labeled_df['_merge_key'] = labeled_df[old_key_cols].astype(str).agg('|'.join, axis=1)
                        label_map = dict(zip(labeled_df['_merge_key'], labeled_df['_label']))
                        
                        # Create composite key for new data
                        new_key_cols = [c for c in primary_key_cols if c in df.columns]
                        df['_merge_key'] = df[new_key_cols].astype(str).agg('|'.join, axis=1)
                        
                        # Merge labels
                        labels_before = df['_label'].notna().sum() if '_label' in df.columns else 0
                        df['_label'] = df['_merge_key'].map(label_map).fillna(df['_label'])
                        labels_after = df['_label'].notna().sum()
                        
                        # Clean up merge key
                        df = df.drop(columns=['_merge_key'])
                        
                        preserved = labels_after - labels_before
                        print(f"  Preserved {preserved} labels from existing file")
                        
                        # Check for labels that couldn't be matched (rows removed)
                        unmatched = len(labeled_df) - preserved
                        if unmatched > 0:
                            print(f"  Warning: {unmatched} labeled rows from old file no longer exist in new data")
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
    print(f"Metadata columns:      {meta_cols}")
    
    # Label summary
    if '_label' in df.columns:
        labeled_count = df['_label'].notna() & (df['_label'] != '')
        labeled_count = labeled_count.sum()
        print(f"Rows with labels:      {labeled_count} / {len(df)} ({100*labeled_count/len(df):.1f}%)")
    
    print(f"\nFeatures: {feature_cols}")
    
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
        '_section_number': 'count',
        '_label': lambda x: x.notna().sum(),
        'section_depth': 'mean',
        'is_sandwiched': 'sum',
        'in_toc_exact': 'mean',
    }).rename(columns={
        '_section_number': 'num_sections',
        '_label': 'num_labeled',
        'section_depth': 'avg_depth',
        'is_sandwiched': 'sandwiched_count',
        'in_toc_exact': 'toc_match_rate',
    })
    
    return summary.round(2)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract ML features from processed documents for training a section classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Use pipeline defaults
    python feature_extractor.py
    
    # Use pipeline defaults, save to specific CSV
    python feature_extractor.py -o training_data.csv
    
    # Custom directories
    python feature_extractor.py --raw_ocr_dir ./my_ocr --results_dir ./my_results
    
    # Just print summary, don't save
    python feature_extractor.py --no-save

INTERACTIVE USAGE:
    from feature_extractor import process_directory, load_training_data
    
    # Extract features (uses pipeline defaults)
    df = process_directory()
    
    # Or with custom paths
    df = process_directory('./my_results', './my_ocr')
    
    # Explore
    df.head()
    df.describe()
    
    # Save
    df.to_csv('training_data.csv', index=False)
    
    # Reload later
    df = load_training_data('training_data.csv')

LABELING:
    Open the CSV and fill in '_label' column:
        1 = valid section
        0 = false positive
        """
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
    
    # If not saving, offer interactive hint
    if args.no_save and not df.empty:
        print("\nDataFrame created but not saved (--no-save flag)")
        print("To save: df.to_csv('output.csv', index=False)")