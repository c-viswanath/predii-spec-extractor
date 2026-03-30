import json
import csv
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from src.pdf_parser import extract_pages
from src.chunker import chunk_pages
from src.embedder import VectorStore
from src.extractor import extract_specs

PDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample-service-manual 1.pdf')
VECTORSTORE_PATH = 'outputs/vectorstore.pkl'
JSON_OUTPUT = 'outputs/extracted_specs.json'
CSV_OUTPUT = 'outputs/extracted_specs.csv'
SPEC_DB_PATH = 'outputs/spec_database.json'

DEMO_QUERIES = [
    'Torque for brake caliper anchor plate bolts',
    'Torque for lower control arm bolts',
    'Torque for stabilizer bar bracket nuts',
    'Front wheel alignment camber and caster specification',
    'Ball joint deflection specification limit',
    'Wheel nut torque specification tighten',
    'Minimum Traction-Lok breakaway torque specification',
    'Drive pinion flange runout maximum tolerance',
    'Rear axle differential fluid fill capacity',
    'Wheel and tire anti-seize lubricant part number specification',
]


def build_index(force_rebuild: bool = False) -> VectorStore:
    store = VectorStore()
    if not force_rebuild and Path(VECTORSTORE_PATH).exists():
        store.load(VECTORSTORE_PATH)
        return store
    os.makedirs('outputs', exist_ok=True)
    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=800, overlap=150)
    store.build(chunks)
    store.save(VECTORSTORE_PATH)
    if not Path(SPEC_DB_PATH).exists():
        pre_extract_all_specs(store)
    return store


def pre_extract_all_specs(store: VectorStore):
    from collections import defaultdict
    page_groups = defaultdict(list)
    for chunk in store.chunks:
        page_groups[chunk.page_num].append(chunk)

    all_specs = []
    total = len(page_groups)
    for i, (page_num, chunks) in enumerate(sorted(page_groups.items())):
        if i % 50 == 0:
            print(f'  [pre-extract] Page group {i}/{total}...')
        pairs = [(c, 1.0) for c in chunks]
        try:
            specs = extract_specs(f"Extract all specifications from page {page_num}", pairs)
            all_specs.extend(specs)
        except Exception as e:
            print(f'  [pre-extract] Error on page {page_num}: {e}')

    with open(SPEC_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_specs, f, indent=2, ensure_ascii=False)
    print(f'[pipeline] Pre-extracted {len(all_specs)} specs → {SPEC_DB_PATH}')


def query_specs(question: str, store: VectorStore, top_k: int = 5):
    retrieved = store.search(question, top_k=top_k)
    return extract_specs(question, retrieved)


def run_demo_queries(store: VectorStore):
    all_results = []
    for q in DEMO_QUERIES:
        print(f'\n[pipeline] Query: {q}')
        specs = query_specs(q, store)
        for spec in specs:
            spec['query'] = q
            all_results.append(spec)
            print(f"  ✓ {spec.get('component', '?')} | {spec.get('spec_type', '?')}: {spec.get('value', '?')} {spec.get('unit', '')}")
        if not specs:
            print('  ✗ No specs found')

    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n[pipeline] Saved {len(all_results)} specs → {JSON_OUTPUT}')

    if all_results:
        fields = ['query', 'component', 'spec_type', 'value', 'unit', 'context', 'source_page']
        with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        print(f'[pipeline] Saved CSV → {CSV_OUTPUT}')

    return all_results


if __name__ == '__main__':
    store = build_index()
    run_demo_queries(store)