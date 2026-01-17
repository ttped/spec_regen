import os
import json
from typing import List, Dict, Optional

# Import from the new 'docx' library (python-docx-oss)
import docx
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_docx_table_from_data(doc, table_data: Dict):
    """
    Adds a table to the docx document from structured data.
    (This function is unchanged)
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
    This function constructs the necessary XML for a 'SEQ Figure' field.

    Args:
        doc: The document object.
        text: The caption text (e.g., "My Figure Name").
    """
    # Create a new paragraph for the caption with the 'Caption' style
    p = doc.add_paragraph(style='Caption')
    
    # Add the "Figure " prefix
    p.add_run("Figure ")

    # --- Create the SEQ field for automatic numbering ---
    run = p.add_run()
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Figure \\* ARABIC'  # Field for figures
    run._r.append(instrText)

    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)

    # Add the rest of the caption text
    p.add_run(f": {text}")


def add_table_caption(doc, text: str):
    """
    Adds a true Word caption for a TABLE, which can be used for a Table of Tables.
    This function constructs the necessary XML for a 'SEQ Table' field.

    Args:
        doc: The document object.
        text: The caption text (e.g., "My Table Name").
    """
    # Create a new paragraph for the caption with the 'Caption' style
    p = doc.add_paragraph(style='Caption')
    
    # Add the "Table " prefix
    p.add_run("Table ")

    # --- Create the SEQ field for automatic numbering ---
    run = p.add_run()
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Table \\* ARABIC'  # Field for tables
    run._r.append(instrText)

    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)

    # Add the rest of the caption text
    p.add_run(f": {text}")


def add_bordered_paragraph(doc, text: str):
    """
    Adds a paragraph with a single-line border around it.
    This is used to create the "box" effect for distribution statements.
    """
    p = doc.add_paragraph(text)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    
    # Define the border style for all four sides
    for border_name in ['top', 'left', 'bottom', 'right']:
        border_el = OxmlElement(f'w:{border_name}')
        border_el.set(qn('w:val'), 'single')  # Style: single line
        border_el.set(qn('w:sz'), '4')         # Width: 1/4 pt
        border_el.set(qn('w:space'), '1')      # Spacing from text
        border_el.set(qn('w:color'), 'auto')   # Color
        pBdr.append(border_el)
    
    pPr.append(pBdr)

def add_title_page(doc, title_data: Dict):
    """
    Adds a formatted title page to the document using the extracted data.
    """
    # 1. Add the Document Title (Bold and Centered)
    # Add vertical spacing to push the title down the page
    p_title = doc.add_paragraph()
    p_title.add_run('\n\n\n\n') 
    
    # Add the title text, bolded and centered
    run = p_title.add_run(title_data.get('document_title', ''))
    run.bold = True
    run.font.size = Pt(18)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add more spacing after the title
    p_title.add_run('\n\n\n\n\n')

    # 2. Add the Distribution Statements in a single box
    boxed_content = []
    if title_data.get('distribution_statement'):
        boxed_content.append(title_data.get('distribution_statement'))
    if title_data.get('export_warning'):
        boxed_content.append(title_data.get('export_warning'))
    if title_data.get('destruction_notice'):
        boxed_content.append(title_data.get('destruction_notice'))

    if boxed_content:
        add_bordered_paragraph(doc, '\n\n'.join(boxed_content))

    # 3. Add the Control Statements as normal text
    doc.add_paragraph() # Add a space
    if title_data.get('approval_status'):
        doc.add_paragraph(title_data.get('approval_status'))
    if title_data.get('controlled_by'):
        doc.add_paragraph(f"CONTROLLED BY: {title_data.get('controlled_by')}")
    if title_data.get('cui_category'):
        doc.add_paragraph(f"CUI CATEGORY: {title_data.get('cui_category')}")
    if title_data.get('point_of_contact'):
        doc.add_paragraph(f"POC: {title_data.get('point_of_contact')}")

def add_field(paragraph, field_text: str):
    """
    Adds a Word field (like PAGE or SEQ) to a paragraph.

    Args:
        paragraph: The paragraph object to which the field will be added.
        field_text: The field code string (e.g., "PAGE").
    """
    run = paragraph.add_run()
    
    # Create the fldChar element and set it to 'begin'
    fldChar_begin = OxmlElement('w:fldChar')
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar_begin)

    # Create the instrText element with the field code
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = field_text
    run._r.append(instrText)

    # Create the fldChar element and set it to 'end'
    fldChar_end = OxmlElement('w:fldChar')
    fldChar_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar_end)

def add_caption(doc, text: str):
    """
    Adds a true Word caption to the document, which can be used for a Table of Figures.
    This function constructs the necessary XML for a SEQ (Sequence) field.

    Args:
        doc: The document object.
        text: The caption text (e.g., "My Figure Name").
    """
    # Create a new paragraph for the caption with the 'Caption' style
    p = doc.add_paragraph(style='Caption')
    
    # Add the "Figure " prefix
    p.add_run("Figure ")

    # --- Create the SEQ field for automatic numbering ---
    # 1. Create the run that will contain the field
    run = p.add_run()
    
    # 2. Create the fldChar element and set it to 'begin'
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar)

    # 3. Create the instrText element with the SEQ field code
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'SEQ Figure \\* ARABIC'
    run._r.append(instrText)

    # 4. Create the fldChar element and set it to 'end'
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar)

    # Add the rest of the caption text (e.g., ": My Figure Name")
    p.add_run(f": {text}")

def create_docx_from_elements(elements: List[Dict], output_filename: str, figures_image_folder: str, part_number: str, title_data: Optional[Dict]):
    """
    Creates a .docx file. It now handles 'unassigned_text_block' by writing
    its content as a plain paragraph.
    """
    doc = docx.Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

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
            level = len(section_number.split('.')) if section_number else 1
            heading_level = max(1, min(level, 9))
            heading_text = f"{section_number} {topic}".strip()
            if heading_text:
                doc.add_heading(heading_text, level=heading_level)
            if content:
                doc.add_paragraph(content)
        
        # --- NEW: Handle unassigned text blocks ---
        elif element_type == "unassigned_text_block":
            content = element.get("content", "")
            if content:
                # Simply add the content as a plain paragraph
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
            table_data = element.get("table_data") # Check for pre-processed data

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
    No longer requires llm_config.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}. Skipping DOCX creation for {doc_stem}.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    
    title_data = None
    if os.path.exists(title_data_path):
        with open(title_data_path, 'r', encoding='utf-8') as f:
            title_data_list = json.load(f)
            if title_data_list:
                title_data = title_data_list[0]
    else:
        print(f"Warning: Title data file not found at {title_data_path}. Proceeding without a title page.")

    figure_image_folder = os.path.join(figures_base_path, doc_stem)

    create_docx_from_elements(elements, output_path, figure_image_folder, doc_stem, title_data)
    print(f"DOCX document successfully created at {output_path}")


if __name__ == '__main__':
    # Example of how to run this module standalone
    doc_stem = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "..", "results", f"{doc_stem}_repaired.json")
    # I've changed the output name to avoid confusion with previous versions
    output_file = os.path.join(script_dir, "..", "results", f"{doc_stem}_final_oss.docx")
    figures_path = os.path.join(script_dir, "..", "iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")

    run_docx_creation(input_file, output_file, figures_path, doc_stem)