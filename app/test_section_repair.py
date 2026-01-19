"""
test_section_repair.py - Test cases for section repair logic

Run with: python test_section_repair.py
"""

from section_repair_agent import (
    SectionNumber, 
    analyze_transition, 
    repair_sections,
    find_violations
)


def test_section_number_parsing():
    """Test that section numbers are parsed correctly."""
    print("Testing SectionNumber parsing...")
    
    cases = [
        ("3.1.2", ["3", "1", "2"], 3),
        ("1", ["1"], 1),
        ("3.A", ["3", "A"], 2),  # OCR error
        ("10.2.1", ["10", "2", "1"], 3),
        ("", [], 0),
    ]
    
    for raw, expected_parts, expected_depth in cases:
        sn = SectionNumber(raw)
        assert sn.parts == expected_parts, f"Failed for {raw}: got {sn.parts}"
        assert sn.depth == expected_depth, f"Failed depth for {raw}: got {sn.depth}"
        print(f"  ✓ '{raw}' -> parts={sn.parts}, depth={sn.depth}")
    
    print("  All parsing tests passed!\n")


def test_valid_transitions():
    """Test transitions that should be considered VALID."""
    print("Testing VALID transitions...")
    
    valid_cases = [
        ("1", "2", "Simple increment"),
        ("1", "1.1", "Go deeper"),
        ("1.1", "1.2", "Same level increment"),
        ("1.1.1", "1.1.2", "Same level increment (deep)"),
        ("1.1.1", "1.2", "Go shallower and increment"),
        ("1.1.1", "2", "Go to next major section"),
        ("2.1.1", "3", "Go to next major section from deep"),
        ("3.A", "3.5", "OCR error tolerance"),
        ("1.2", "2", "Valid major section increment"),
    ]
    
    all_passed = True
    for prev, curr, desc in valid_cases:
        prev_sn = SectionNumber(prev)
        curr_sn = SectionNumber(curr)
        analysis = analyze_transition(prev_sn, curr_sn)
        status = '✓' if analysis.is_valid else '✗'
        print(f"  {status} {prev} -> {curr}: {desc}")
        if not analysis.is_valid:
            print(f"      UNEXPECTED: {analysis.violation_type}")
            all_passed = False
    
    if all_passed:
        print("  All valid transition tests passed!")
    print()


def test_invalid_transitions():
    """Test transitions that should be considered INVALID (false positives)."""
    print("Testing INVALID transitions (likely false positives)...")
    
    invalid_cases = [
        ("3.1.2", "1", "Reset to 1 in middle of doc - likely list item"),
        ("3.1.2", "2", "Reset to 2 in middle of doc - likely list item"),  
        ("3.1", "1", "Reset to 1 - likely list item"),
        ("2.1.1", "1", "Reset after being in section 2"),
        ("1.1", "1.1.1.1", "Skipped levels - depth jump"),
    ]
    
    for prev, curr, description in invalid_cases:
        prev_sn = SectionNumber(prev)
        curr_sn = SectionNumber(curr)
        analysis = analyze_transition(prev_sn, curr_sn)
        status = '✓ DETECTED' if not analysis.is_valid else '✗ MISSED'
        print(f"  {prev} -> {curr}: {status}")
        print(f"      {description}")
        if not analysis.is_valid:
            print(f"      Violation type: {analysis.violation_type}, confidence: {analysis.confidence:.2f}")
    
    print()


def test_full_repair():
    """Test the full repair process on a mock document."""
    print("Testing full repair on mock document...")
    
    # Simulate a document with a list appearing in the middle
    elements = [
        {"type": "section", "section_number": "1", "topic": "Introduction", "content": ""},
        {"type": "section", "section_number": "1.1", "topic": "Overview", "content": "Some overview text"},
        {"type": "section", "section_number": "1.2", "topic": "Scope", "content": ""},
        {"type": "section", "section_number": "2", "topic": "Requirements", "content": ""},
        {"type": "section", "section_number": "2.1", "topic": "General", "content": "The following items are required:"},
        # FALSE POSITIVES - These are list items, not sections!
        {"type": "section", "section_number": "1", "topic": "First item in list", "content": ""},
        {"type": "section", "section_number": "2", "topic": "Second item", "content": ""},
        {"type": "section", "section_number": "3", "topic": "Third item", "content": ""},
        # Back to real sections
        {"type": "section", "section_number": "2.2", "topic": "Specific Requirements", "content": ""},
        {"type": "section", "section_number": "3", "topic": "Testing", "content": ""},
    ]
    
    print(f"\n  Input: {len(elements)} elements")
    for i, e in enumerate(elements):
        if e['type'] == 'section':
            print(f"    [{i}] Section {e['section_number']}: {e['topic']}")
    
    # Find violations
    violations = find_violations(elements)
    print(f"\n  Transition violations found: {len(violations)}")
    for idx, analysis in violations:
        print(f"    Index {idx}: {analysis}")
    
    # Run repair
    repaired, report = repair_sections(elements, confidence_threshold=0.7)
    
    print(f"\n  List sequences detected: {report.get('list_sequences_found', 0)}")
    for seq in report.get('list_sequences', []):
        seq_nums = [elements[i]['section_number'] for i in seq]
        print(f"    Indices {seq}: sections {seq_nums}")
    
    print(f"\n  Repair Report:")
    print(f"    Sections before: {report['total_sections_before']}")
    print(f"    Sections demoted: {report['sections_demoted']}")
    print(f"    Sections after: {report['total_sections_after']}")
    
    print(f"\n  Output: {len(repaired)} elements")
    for e in repaired:
        if e['type'] == 'section':
            print(f"    Section {e['section_number']}: {e['topic']}")
        else:
            content_preview = e.get('content', '')[:50]
            demoted = e.get('_demoted_from_section', '')
            if demoted:
                print(f"    [demoted from '{demoted}']: {content_preview}...")
    
    print()


def test_table_in_section():
    """Test detecting table data masquerading as sections."""
    print("Testing table data detected as sections...")
    
    # Simulate a table with row numbers being detected as sections
    elements = [
        {"type": "section", "section_number": "3", "topic": "Data Table", "content": ""},
        {"type": "section", "section_number": "3.1", "topic": "Table Title", "content": ""},
        # Table rows detected as sections
        {"type": "section", "section_number": "1", "topic": "Row 1 data here", "content": ""},
        {"type": "section", "section_number": "2", "topic": "Row 2 data here", "content": ""},
        {"type": "section", "section_number": "3", "topic": "Row 3 data here", "content": ""},
        {"type": "section", "section_number": "4", "topic": "Row 4 data here", "content": ""},
        # Real section continues
        {"type": "section", "section_number": "3.2", "topic": "Analysis", "content": ""},
    ]
    
    violations = find_violations(elements)
    repaired, report = repair_sections(elements, confidence_threshold=0.7)
    
    print(f"  Violations found: {len(violations)}")
    print(f"  Sections demoted: {report['sections_demoted']}")
    
    print("\n  After repair:")
    for e in repaired:
        if e['type'] == 'section':
            print(f"    Section {e['section_number']}: {e['topic']}")
        else:
            demoted = e.get('_demoted_from_section', '')
            if demoted:
                print(f"    [demoted from {demoted}]: {e.get('content', '')[:40]}...")
    
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Section Repair Agent - Test Suite")
    print("=" * 60)
    print()
    
    test_section_number_parsing()
    test_valid_transitions()
    test_invalid_transitions()
    test_full_repair()
    test_table_in_section()
    
    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)