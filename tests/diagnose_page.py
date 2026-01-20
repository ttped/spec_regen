#!/usr/bin/env python3
"""
diagnose_page.py - Analyze a specific page from raw OCR to debug section detection issues.

Usage:
    python diagnose_page.py <raw_ocr.json> <page_number>
    
Example:
    python diagnose_page.py raw_data/mydoc.json 16
"""

import sys
import os
import json
import re
from typing import Dict, List, Optional, Tuple


def get_page_dict(data, page_num: int) -> Optional[Dict]:
    """Extract page_dict for a specific page number."""
    
    if isinstance(data, dict):
        # Try direct key access
        for key in [str(page_num), page_num, f"page_{page_num}"]:
            if key in data:
                page_obj = data[key]
                if isinstance(page_obj, dict):
                    return page_obj.get('page_dict', page_obj)
        
        # Search through all keys
        for key, val in data.items():
            if isinstance(val, dict):
                pid = val.get('page_Id') or val.get('page_num')
                if pid is not None:
                    try:
                        if int(pid) == page_num:
                            return val.get('page_dict', val)
                    except:
                        pass
                        
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                pid = item.get('page_Id') or item.get('page_num')
                if pid is not None:
                    try:
                        if int(pid) == page_num:
                            return item.get('page_dict', item)
                    except:
                        pass
    
    return None


def reconstruct_lines(page_dict: Dict) -> List[Dict]:
    """Reconstruct lines from word-level OCR data."""
    if not page_dict.get('text'):
        return []
    
    text_list = page_dict.get('text', [])
    block_nums = page_dict.get('block_num', [])
    line_nums = page_dict.get('line_num', [])
    par_nums = page_dict.get('par_num', [0] * len(text_list))
    tops = page_dict.get('top', [])
    lefts = page_dict.get('left', [])
    
    print(f"\n  Raw data lengths:")
    print(f"    text:      {len(text_list)}")
    print(f"    block_num: {len(block_nums)}")
    print(f"    line_num:  {len(line_nums)}")
    print(f"    par_num:   {len(par_nums)}")
    print(f"    top:       {len(tops)}")
    print(f"    left:      {len(lefts)}")
    
    if not block_nums or not line_nums:
        print("\n  WARNING: Missing block_num or line_num - cannot reconstruct lines properly!")
        return [{"text": " ".join(str(t) for t in text_list), "words": text_list}]
    
    lines = []
    current_words = []
    current_indices = []
    
    last_block = block_nums[0] if block_nums else 0
    last_par = par_nums[0] if par_nums else 0
    last_line = line_nums[0] if line_nums else 0
    
    for i in range(len(text_list)):
        if i >= len(block_nums) or i >= len(line_nums):
            break
        
        block = block_nums[i]
        par = par_nums[i] if i < len(par_nums) else 0
        line = line_nums[i]
        word = str(text_list[i])
        top = tops[i] if i < len(tops) else None
        left = lefts[i] if i < len(lefts) else None
        
        if block != last_block or par != last_par or line != last_line:
            # Save current line
            if current_words:
                lines.append({
                    "text": " ".join(current_words),
                    "words": current_words.copy(),
                    "indices": current_indices.copy(),
                    "block": last_block,
                    "par": last_par,
                    "line": last_line
                })
            current_words = [word]
            current_indices = [i]
            last_block, last_par, last_line = block, par, line
        else:
            current_words.append(word)
            current_indices.append(i)
    
    # Last line
    if current_words:
        lines.append({
            "text": " ".join(current_words),
            "words": current_words.copy(),
            "indices": current_indices.copy(),
            "block": last_block,
            "par": last_par,
            "line": last_line
        })
    
    return lines


def check_section_header(line_text: str) -> Tuple[bool, str]:
    """Check if line is a section header and return reason."""
    
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', line_text)
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', line_text)
    
    if match:
        potential_num, full_topic = match.groups()
        match_type = "pattern1 (num + text)"
    elif match_no_title:
        potential_num = match_no_title.group(1)
        full_topic = ""
        match_type = "pattern2 (num only)"
    else:
        return False, "no regex match"

    MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    if full_topic and any(full_topic.lower().startswith(m) for m in MONTHS):
        return False, "topic starts with month"

    if not any(c.isdigit() for c in potential_num):
        return False, f"no digits in '{potential_num}'"

    if sum(c.isalpha() for c in potential_num) > 2:
        return False, f"too many alpha chars in '{potential_num}'"
        
    if len(potential_num) > 20:
        return False, f"too long: {len(potential_num)} chars"

    if potential_num.isalpha():
        return False, "pure alphabetic"

    if potential_num.isdigit() and len(potential_num) > 3:
        return False, f"pure digits > 3 chars: '{potential_num}'"

    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        return False, "mixed alpha+digit without dot"

    section_num = potential_num.strip().rstrip('.')
    return True, f"ACCEPTED via {match_type} -> section '{section_num}'"


def analyze_page(filepath: str, page_num: int):
    """Analyze a specific page from raw OCR."""
    
    print(f"{'='*70}")
    print(f"Analyzing page {page_num} from: {filepath}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    page_dict = get_page_dict(data, page_num)
    
    if page_dict is None:
        print(f"ERROR: Could not find page {page_num} in the data")
        print(f"\nAvailable structure:")
        if isinstance(data, dict):
            print(f"  Dict with keys: {list(data.keys())[:20]}...")
        elif isinstance(data, list):
            print(f"  List with {len(data)} items")
            if data:
                print(f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'not a dict'}")
        return
    
    print(f"\nPage dict keys: {list(page_dict.keys())}")
    
    # Reconstruct lines
    lines = reconstruct_lines(page_dict)
    
    print(f"\n{'='*70}")
    print(f"Reconstructed {len(lines)} lines:")
    print(f"{'='*70}\n")
    
    # Look for section-like patterns
    section_pattern = re.compile(r'^[0-9]+\.?([0-9]+\.?)*$')
    
    for i, line_info in enumerate(lines):
        text = line_info['text']
        words = line_info['words']
        
        # Check if this could be a section
        is_section, reason = check_section_header(text)
        
        # Highlight potential section lines
        first_word = words[0] if words else ""
        looks_like_section = bool(section_pattern.match(first_word.rstrip('.')))
        
        # Get top position if available
        indices = line_info.get('indices', [])
        tops = page_dict.get('top', [])
        top_val = tops[indices[0]] if indices and indices[0] < len(tops) else "?"
        
        marker = ""
        if is_section:
            marker = " ✓ SECTION"
        elif looks_like_section:
            marker = " ⚠ LOOKS LIKE SECTION BUT REJECTED"
        
        print(f"[{i:3d}] (top={top_val}) block={line_info.get('block')}, par={line_info.get('par')}, line={line_info.get('line')}")
        print(f"      Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        print(f"      Words: {words[:10]}{'...' if len(words) > 10 else ''}")
        if marker:
            print(f"      {marker}: {reason}")
        print()
    
    # Summary: find anything that starts with a digit
    print(f"\n{'='*70}")
    print("Lines starting with digit patterns:")
    print(f"{'='*70}\n")
    
    for i, line_info in enumerate(lines):
        words = line_info['words']
        if words:
            first = words[0]
            if re.match(r'^[0-9\.\$]+$', first):
                text = line_info['text']
                is_section, reason = check_section_header(text)
                status = "✓" if is_section else "✗"
                print(f"  {status} Line {i}: first_word='{first}' -> {reason}")
                print(f"       Full: '{text[:60]}...'")
                print()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    
    filepath = sys.argv[1]
    page_num = int(sys.argv[2])
    
    analyze_page(filepath, page_num)


if __name__ == '__main__':
    main()