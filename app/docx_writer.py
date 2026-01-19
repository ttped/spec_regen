import os
import json
from typing import List, Dict, Optional

# Import from the new 'docx' library (python-docx-oss)
import docx
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH


def configure_heading_styles(doc):
    """
    Configure the built-in Heading styles for base formatting.
    
    This preserves ToC functionality since Word's Table of Contents
    is based on these built-in Heading styles.
    
    Note: We set bold here but NOT underline - underline is applied
    selectively to just the title portion (not section number) in
    add_section_heading().
    
    Font sizes by level:
    - Heading 1: 14pt
    - Heading 2: 13pt
    - Heading 3: 12pt
    - Heading 4+: 11pt
    """
    font_sizes = {
        'Heading 1': Pt(14),
        'Heading 2': Pt(13),
        'Heading 3': Pt(12),
        'Heading 4': Pt(11),
        'Heading 5': Pt(11),
        'Heading 6': Pt(11),
        'Heading 7': Pt(11),
        'Heading 8': Pt(11),
        'Heading 9': Pt(11),
    }
    
    for style_name, font_size in font_sizes.items():
        try:
            style = doc.styles[style_name]
            
            # Set font properties (bold but NOT underline - that's applied per-run)
            style.font.bold = True
            style.font.underline = False
            style.font.size = font_size
            style.font.name = 'Calibri'
            
            # Set color to black (remove default blue color that some templates use)
            style.font.color.rgb = None  # Inherit/auto color
            
            # Adjust paragraph spacing
            style.paragraph_format.space_before = Pt(12) if style_name == 'Heading 1' else Pt(8)
            style.paragraph_format.space_after = Pt(6)
            
        except KeyError:
            # Style doesn't exist in this document
            pass


def add_section_heading(doc, section_number: str, topic: str, level: int = 1):
    """
    Adds a section heading with the section number bold (not underlined)
    and the topic bold + underlined.
    
    Uses Word's built-in Heading styles to preserve ToC functionality.
    
    Args:
        doc: The document object
        section_number: The section number (e.g., "3.1.1") - bold only
        topic: The section title (e.g., "System Requirements") - bold + underlined
        level: Heading level 1-9
    """
    # Determine the heading style name
    heading_level = max(1, min(level, 9))
    style_name = f'Heading {heading_level}'
    
    # Create paragraph with heading style (for ToC compatibility)
    p = doc.add_paragraph(style=style_name)
    
    # Add section number run (bold, no underline)
    if section_number:
        run_number = p.add_run(section_number)
        run_number.bold = True
        run_number.underline = False
        
        # Add space between number and topic
        if topic:
            run_space = p.add_run(" ")
            run_space.bold = True
            run_space.underline = False
    
    # Add topic run (bold + underlined)
    if topic:
        run_topic = p.add_run(topic)
        run_topic.bold = True
        run_topic.underline = True
    
    return p


def add_docx_table_from_data(doc, table_data: Dict):
    """
    Adds a table to the docx document from structured data.
    """
    columns = table_data.get('columns', [])
    rows_data = table_data.get('rows', [])
    
    if not columns or not rows_data:
        return

    table = doc.add_table(rows=1, cols=len(columns))
    table.style = 'Table Grid'

    header_cells = table.rows[0].cells
    for i, col_name in enumerate(columns):
        cell = header_cells[i]
        cell.text = col_name
        cell.paragraphs[0].runs[0].font.bold = True

    for row_values in rows_data:
        row_cells = table.add_row().cells
        for i, value in enumerate(row_values):
            if i < len(row_cells):
                row_cells[i].text = str(value)


def add_figure_caption(doc, text: str):
    """
    Adds a true Word caption for a FIGURE, which can be used for a Table of Figures.
    """
    p = doc.add_paragraph(style='Caption')
    p.add_run("Figure ")

    run = p.add_run()
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Figure \\* ARABIC'
    run._r.append(instrText)

    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)

    p.add_run(f": {text}")


def add_table_caption(doc, text: str):
    """
    Adds a true Word caption for a TABLE, which can be used for a Table of Tables.
    """
    p = doc.add_paragraph(style='Caption')
    p.add_run("Table ")

    run = p.add_run()
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Table \\* ARABIC'
    run._r.append(instrText)

    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)

    p.add_run(f": {text}")


def add_bordered_paragraph(doc, text: str):
    """
    Adds a paragraph with a single-line border around it.
    """
    p = doc.add_paragraph(text)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    
    for border_name in ['top', 'left', 'bottom', 'right']:
        border_el = OxmlElement(f'w:{border_name}')
        border_el.set(qn('w:val'), 'single')
        border_el.set(qn('w:sz'), '4')
        border_el.set(qn('w:space'), '1')
        border_el.set(qn('w:color'), 'auto')
        pBdr.append(border_el)
    
    pPr.append(pBdr)


def add_title_page(doc, title_data: Dict):
    """
    Adds a formatted title page to the document using the extracted data.
    
    Layout order:
    1. Document title (centered, bold, large)
    2. Approval status with date (date in red if DRAFT)
    3. Distribution statement box (bordered)
    4. Export warning (in the box)
    5. Destruction notice (in the box)
    6. CONTROLLED BY entries
    7. CUI CATEGORY
    8. POC
    """
    from docx.shared import RGBColor
    
    # --- 1. Document Title ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.add_run('\n\n\n')  # Top spacing
    
    title_text = title_data.get('document_title', '')
    if title_text:
        run = p_title.add_run(title_text)
        run.bold = True
        run.font.size = Pt(16)
    
    doc.add_paragraph()  # Spacing after title
    
    # --- 2. Approval Status (with date potentially in red) ---
    approval_status = title_data.get('approval_status', '')
    approval_date = title_data.get('approval_date', '')
    
    if approval_status:
        p_approval = doc.add_paragraph()
        p_approval.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Check if we need to highlight the date in red
        if approval_date and approval_date in approval_status:
            # Split the approval status around the date
            parts = approval_status.split(approval_date)
            
            # Add text before date
            if parts[0]:
                run_before = p_approval.add_run(parts[0])
            
            # Add date in red (especially for DRAFT)
            run_date = p_approval.add_run(approval_date)
            if 'DRAFT' in approval_date.upper() or 'TBD' in approval_date.upper():
                run_date.font.color.rgb = RGBColor(255, 0, 0)  # Red
                run_date.bold = True
            
            # Add text after date
            if len(parts) > 1 and parts[1]:
                run_after = p_approval.add_run(parts[1])
        else:
            # No date to highlight, just add the whole thing
            p_approval.add_run(approval_status)
    
    doc.add_paragraph()  # Spacing
    
    # --- 3-5. Boxed Content (Distribution, Export Warning, Destruction) ---
    boxed_content = []
    
    distribution = title_data.get('distribution_statement', '')
    if distribution:
        boxed_content.append(distribution)
    
    export_warning = title_data.get('export_warning', '')
    if export_warning:
        boxed_content.append(export_warning)
    
    destruction = title_data.get('destruction_notice', '')
    if destruction:
        boxed_content.append(destruction)
    
    if boxed_content:
        add_bordered_paragraph(doc, '\n\n'.join(boxed_content))
    
    doc.add_paragraph()  # Spacing after box
    
    # --- 6. CONTROLLED BY entries ---
    controlled_by = title_data.get('controlled_by', [])
    
    # Handle both list and string formats
    if isinstance(controlled_by, str) and controlled_by:
        controlled_by = [controlled_by]
    elif not isinstance(controlled_by, list):
        controlled_by = []
    
    for entry in controlled_by:
        if entry:
            p = doc.add_paragraph()
            run_label = p.add_run("CONTROLLED BY: ")
            run_label.bold = True
            p.add_run(str(entry))
    
    # --- 7. CUI Category ---
    cui_category = title_data.get('cui_category', '')
    if cui_category:
        p = doc.add_paragraph()
        run_label = p.add_run("CUI CATEGORY: ")
        run_label.bold = True
        p.add_run(cui_category)
    
    # --- 8. Point of Contact ---
    poc = title_data.get('point_of_contact', '')
    if poc:
        p = doc.add_paragraph()
        run_label = p.add_run("POC: ")
        run_label.bold = True
        p.add_run(poc)
    
    # --- Optional: Document date if present ---
    doc_date = title_data.get('document_date', '')
    if doc_date:
        p = doc.add_paragraph()
        run_label = p.add_run("DATE: ")
        run_label.bold = True
        p.add_run(doc_date)


def add_field(paragraph, field_text: str):
    """
    Adds a Word field (like PAGE or SEQ) to a paragraph.
    """
    run = paragraph.add_run()
    
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = field_text
    run._r.append(instrText)

    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)


def add_caption(doc, text: str):
    """
    Adds a true Word caption to the document.
    """
    p = doc.add_paragraph(style='Caption')
    p.add_run("Figure ")

    run = p.add_run()
    
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Figure \\* ARABIC'
    run._r.append(instrText)

    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar)

    p.add_run(f": {text}")


def create_docx_from_elements(elements: List[Dict], output_filename: str, figures_image_folder: str, part_number: str, title_data: Optional[Dict]):
    """
    Creates a .docx file from document elements.
    Handles 'section', 'unassigned_text_block', 'figure', and 'table' element types.
    
    Section headings use Word's built-in Heading styles (for ToC support)
    but are configured to be bold + underlined.
    """
    doc = docx.Document()
    
    # Configure base Normal style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    
    # Configure Heading styles to be bold + underlined (preserves ToC functionality)
    configure_heading_styles(doc)

    if title_data:
        add_title_page(doc, title_data)
        doc.add_page_break()

    # --- Add Header ---
    section = doc.sections[0]
    header = section.header
    
    for p in header.paragraphs:
        p_element = p._element
        p_element.getparent().remove(p_element)

    cui_p = header.add_paragraph()
    cui_run = cui_p.add_run('CUI')
    cui_run.bold = True
    cui_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    info_p = header.add_paragraph()
    info_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    info_p.add_run(part_number)
    info_p.add_run(" | Page ")
    add_field(info_p, "PAGE")

    # --- Add Footer ---
    footer = section.footer
    for p in footer.paragraphs:
        p_element = p._element
        p_element.getparent().remove(p_element)

    footer_p = footer.add_paragraph()
    footer_run = footer_p.add_run('CUI')
    footer_run.bold = True
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER


    for element in elements:
        element_type = element.get("type")

        if element_type == "section":
            section_number = element.get("section_number", "")
            topic = element.get("topic", "")
            content = element.get("content", "")
            
            # Calculate heading level from section number depth
            level = len(section_number.split('.')) if section_number else 1
            heading_level = max(1, min(level, 9))  # Word supports Heading 1-9
            
            # Add heading with section number (bold) and topic (bold + underlined)
            if section_number or topic:
                add_section_heading(doc, section_number, topic, level=heading_level)
            
            if content:
                doc.add_paragraph(content)
        
        elif element_type == "unassigned_text_block":
            content = element.get("content", "")
            if content:
                doc.add_paragraph(content)

        elif element_type == "figure":
            image_filename = element.get("export", {}).get("image_file")
            if image_filename:
                image_path = os.path.join(figures_image_folder, image_filename)
                if os.path.exists(image_path):
                    doc.add_picture(image_path, width=Inches(6.0))
                    caption_name = element.get('asset_id', 'Untitled Figure')
                    add_figure_caption(doc, caption_name)
                else:
                    p_error = doc.add_paragraph()
                    p_error.add_run(f"[Image not found: {image_filename}]").italic = True
        
        elif element_type == "table":
            caption_name = element.get('asset_id', 'Untitled Table')
            table_data = element.get("table_data")

            if table_data:
                add_docx_table_from_data(doc, table_data)
            else:
                image_filename = element.get("export", {}).get("image_file")
                p_placeholder = doc.add_paragraph()
                run = p_placeholder.add_run(
                    f"[Table '{caption_name}' inserted as image. Structured data not found.]"
                )
                run.italic = True
                if image_filename:
                    image_path = os.path.join(figures_image_folder, image_filename)
                    if os.path.exists(image_path):
                        doc.add_picture(image_path, width=Inches(6.0))
            
            add_table_caption(doc, caption_name)

    doc.save(output_filename)


def run_docx_creation(input_path: str, output_path: str, figures_base_path: str, doc_stem: str, title_data_path: str):
    """
    Loads final elements and title data to generate a DOCX file.
    
    Handles both old format (list of elements) and new format (dict with page_metadata and elements).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}. Skipping DOCX creation for {doc_stem}.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both old and new data formats
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
        print(f"  - Loaded {len(elements)} elements (new format)")
        if page_metadata:
            print(f"  - Page metadata available for {len(page_metadata)} pages")
    else:
        elements = data if isinstance(data, list) else []
        print(f"  - Loaded {len(elements)} elements (legacy format)")
    
    title_data = None
    if os.path.exists(title_data_path):
        with open(title_data_path, 'r', encoding='utf-8') as f:
            title_data_list = json.load(f)
            if title_data_list:
                title_data = title_data_list[0]
    else:
        print(f"  - Warning: Title data file not found at {title_data_path}. Proceeding without a title page.")

    figure_image_folder = os.path.join(figures_base_path, doc_stem)

    create_docx_from_elements(elements, output_path, figure_image_folder, doc_stem, title_data)
    print(f"  - DOCX document successfully created at {output_path}")


if __name__ == '__main__':
    doc_stem = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "..", "results", f"{doc_stem}_repaired.json")
    output_file = os.path.join(script_dir, "..", "results", f"{doc_stem}_final_oss.docx")
    figures_path = os.path.join(script_dir, "..", "iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")

    run_docx_creation(input_file, output_file, figures_path, doc_stem)