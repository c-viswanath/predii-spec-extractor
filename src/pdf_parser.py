import re
import fitz

HEADER_NOISE_PATTERN = re.compile(
    r'2014 F-150 Workshop Manual\s*\n'
    r'(?:2014-\d{2}-\d{2}\s*\n)?'
    r'(?:file:///[^\n]*\n)?'
    r'(?:repair4less\s*\n?)?'
    r'(?:Page \d+ sur \d+\s*\n?)?'
)
SECTION_PATTERN = re.compile(r'SECTION\s+[\d\-]+[A-Z]*:\s+.+', re.IGNORECASE)

class PageContent:
    def __init__(self, page_num: int, text: str, section_hint: str):
        self.page_num = page_num
        self.text = text
        self.section_hint = section_hint

def extract_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    
    for i, page in enumerate(doc):
        raw_text = page.get_text("text")

        # OCR Integration: if the page yields little digital text, scan its images.
        if len(raw_text.strip()) < 300:
            try:
                ocr_tp = page.get_textpage_ocr(flags=0, dpi=150, full=False)
                ocr_text = ocr_tp.extractTEXT()
                if len(ocr_text) > len(raw_text):
                    raw_text = ocr_text
            except Exception:
                pass

        clean_text = _clean_text(raw_text)
        section = _detect_section(clean_text)

        if len(clean_text.strip()) > 50:
            pages.append(PageContent(i + 1, clean_text.strip(), section))

    doc.close()
    return pages

def _clean_text(text: str) -> str:
    text = HEADER_NOISE_PATTERN.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _detect_section(text: str) -> str:
    match = SECTION_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return ""