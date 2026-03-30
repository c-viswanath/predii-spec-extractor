from dataclasses import dataclass
from typing import List, Tuple
from src.pdf_parser import PageContent

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page_num: int
    section: str
    char_start: int

def chunk_pages(pages: List[PageContent], chunk_size: int=800, overlap: int=150) -> List[TextChunk]:
    chunks = []
    for page in pages:
        page_chunks = _sliding_window(page.text, chunk_size, overlap)
        for idx, (text, char_start) in enumerate(page_chunks):
            if len(text.strip()) < 30:
                continue
            chunks.append(TextChunk(chunk_id=f'p{page.page_num}_c{idx}', text=text.strip(), page_num=page.page_num, section=page.section_hint, char_start=char_start))
    print(f'[chunker] Created {len(chunks)} chunks from {len(pages)} pages')
    return chunks

def _sliding_window(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
    results = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            newline_pos = text.rfind('\n', start + size // 2, end)
            if newline_pos != -1:
                end = newline_pos
        chunk = text[start:end]
        if chunk.strip():
            results.append((chunk, start))
        next_start = end - overlap
        if end == length:
            break
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return results