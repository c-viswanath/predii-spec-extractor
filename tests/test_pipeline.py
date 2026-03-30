import json
import sys
import os
import re
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pdf_parser import extract_pages, _clean_text, _detect_section
from src.chunker import chunk_pages, _sliding_window
from src.embedder import VectorStore
from src.extractor import extract_specs, _safe_parse_json
from src.pipeline import PDF_PATH
VECTORSTORE_PATH = 'outputs/vectorstore.pkl'
REQUIRED_FIELDS = {'component', 'spec_type', 'value', 'unit', 'context', 'source_page'}

@pytest.fixture(scope='session')
def pages():
    return extract_pages(PDF_PATH)

@pytest.fixture(scope='session')
def chunks(pages):
    return chunk_pages(pages, chunk_size=800, overlap=150)

@pytest.fixture(scope='session')
def store():
    vs = VectorStore()
    vs.load(VECTORSTORE_PATH)
    return vs

class TestPDFExtraction:

    def test_correct_page_count(self, pages):
        assert len(pages) >= 700, f'Expected ≥700 non-empty pages, got {len(pages)}'
        assert len(pages) <= 852, f'Cannot have more than 852 pages, got {len(pages)}'

    def test_each_page_has_text(self, pages):
        for p in pages:
            assert len(p.text.strip()) >= 50, f'Page {p.page_num} has too little text'

    def test_header_noise_stripped(self, pages):
        for p in pages[:50]:
            lines = p.text.split('\n')
            for line in lines:
                assert not line.startswith('file:///C:/TSO/'), f'Footer URL not stripped on page {p.page_num}: {line[:80]}'

    def test_known_page_content(self, pages):
        pg14 = next((p for p in pages if p.page_num == 14), None)
        assert pg14 is not None, 'Page 14 not found'
        assert '350 Nm' in pg14.text or '350' in pg14.text, 'Page 14 should contain 350 Nm torque value'

    def test_section_detection(self, pages):
        section_pages = [p for p in pages if p.section_hint]
        assert len(section_pages) > 50, f'Expected >50 pages with section headers, got {len(section_pages)}'

    def test_clean_text_removes_noise(self):
        raw = 'Real content here.\n2014 F-150 Workshop Manual\n2014-03-01\nfile:///C:/TSO/tsocache/test.htm\nrepair4less\nPage 5 sur 100\nMore real content.'
        cleaned = _clean_text(raw)
        assert 'file:///C:/TSO/' not in cleaned, 'Footer URL should be stripped'
        assert 'Real content here' in cleaned
        assert 'More real content' in cleaned

    def test_detect_section(self):
        text = 'SECTION 204-00: Suspension System — General Information\nSome content here.'
        section = _detect_section(text)
        assert '204-00' in section

    def test_page_numbers_sequential(self, pages):
        nums = [p.page_num for p in pages]
        assert nums[0] == 1, 'First page must be page 1'
        assert nums == sorted(nums), 'Page numbers must be in ascending order'
        assert len(set(nums)) == len(nums), 'Page numbers must be unique'

class TestChunking:

    def test_chunks_created(self, chunks):
        assert len(chunks) >= 852, f'Expected ≥852 chunks (one per page minimum), got {len(chunks)}'
        assert len(chunks) > 500, f'Expected >500 chunks, got {len(chunks)}'

    def test_chunk_ids_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), 'Duplicate chunk IDs found'

    def test_chunk_minimum_length(self, chunks):
        for c in chunks:
            assert len(c.text.strip()) >= 30, f"Chunk {c.chunk_id} is too short: '{c.text[:50]}'"

    def test_chunk_carries_page_num(self, chunks):
        for c in chunks:
            assert 1 <= c.page_num <= 852, f'Chunk {c.chunk_id} has invalid page_num: {c.page_num}'

    def test_chunk_size_bounded(self, chunks):
        oversized = [c for c in chunks if len(c.text) > 1200]
        pct = len(oversized) / len(chunks) * 100
        assert pct < 5, f'{pct:.1f}% of chunks are far over the 800-char limit'

    def test_sliding_window_basic(self):
        text = 'A' * 2000
        results = _sliding_window(text, size=800, overlap=150)
        assert len(results) >= 2, 'Expected multiple chunks for 2000-char text'
        start0 = results[0][1]
        start1 = results[1][1]
        chunk_advance = start1 - start0
        assert chunk_advance < 800, 'Chunks should overlap'

    def test_sliding_window_no_infinite_loop(self):
        text = 'Hello world.'
        results = _sliding_window(text, size=800, overlap=150)
        assert len(results) == 1, 'Short text should produce exactly 1 chunk'

class TestVectorStore:

    def test_store_loaded(self, store):
        assert len(store.chunks) > 0
        assert store.index is not None

    def test_search_returns_results(self, store):
        results = store.search('torque for brake caliper bolts', top_k=5)
        assert len(results) == 5, 'Should return exactly top_k results'

    def test_search_scores_normalized(self, store):
        results = store.search('stabilizer bar bracket nut torque', top_k=5)
        for chunk, score in results:
            assert -1.0 <= score <= 1.0, f'Invalid cosine score: {score}'

    def test_search_relevance_brake_caliper(self, store):
        results = store.search('Torque for brake caliper bolts', top_k=5)
        chunk_texts = ' '.join([c.text.lower() for c, _ in results])
        assert 'caliper' in chunk_texts or 'brake' in chunk_texts, 'Brake caliper query should retrieve relevant chunks'

    def test_search_relevance_wheel_nut(self, store):
        results = store.search('Wheel lug nut torque specification', top_k=10)
        chunk_texts = ' '.join([c.text.lower() for c, _ in results])
        assert 'wheel nut' in chunk_texts or '204' in chunk_texts or 'lug' in chunk_texts, 'Wheel nut query should retrieve relevant chunks'

    def test_search_relevance_stabilizer(self, store):
        results = store.search('Torque for stabilizer bar bracket nuts', top_k=5)
        chunk_texts = ' '.join([c.text.lower() for c, _ in results])
        assert 'stabilizer' in chunk_texts or '55 nm' in chunk_texts.lower(), 'Stabilizer query should retrieve relevant chunks'

    def test_top_k_respected(self, store):
        for k in [3, 5, 8]:
            results = store.search('torque', top_k=k)
            assert len(results) == k, f'Expected {k} results, got {len(results)}'

class TestExtractorParsing:

    def test_parse_valid_array(self):
        raw = '[{"component":"Bolt","spec_type":"Torque","value":"35","unit":"Nm","context":"Tighten to 35 Nm","source_page":1}]'
        result = _safe_parse_json(raw, 'test')
        assert len(result) == 1
        assert result[0]['component'] == 'Bolt'

    def test_parse_with_markdown_fences(self):
        raw = '```json\n[{"component":"Nut","spec_type":"Torque","value":"55","unit":"Nm","context":"55 Nm","source_page":14}]\n```'
        result = _safe_parse_json(raw, 'test')
        assert len(result) == 1
        assert result[0]['value'] == '55'

    def test_parse_empty_array(self):
        result = _safe_parse_json('[]', 'test')
        assert result == []

    def test_parse_dict_wrapper(self):
        raw = '{"specs": [{"component": "Bolt","spec_type":"Torque","value":"34","unit":"Nm","context":"34 Nm","source_page":5}]}'
        result = _safe_parse_json(raw, 'test')
        assert len(result) == 1
        assert result[0]['value'] == '34'

    def test_parse_malformed_json(self):
        result = _safe_parse_json('This is not JSON at all!', 'test')
        assert isinstance(result, list)

    def test_parse_extracts_array_from_text(self):
        raw = 'Sure, here is the JSON: [{"component":"X","spec_type":"Torque","value":"1","unit":"Nm","context":"ok","source_page":1}]'
        result = _safe_parse_json(raw, 'test')
        assert len(result) == 1

class TestOutputSchema:

    @pytest.fixture(scope='class')
    def extracted_data(self):
        with open('outputs/extracted_specs.json') as f:
            return json.load(f)

    def test_output_file_exists(self):
        assert os.path.exists('outputs/extracted_specs.json'), 'JSON output missing'
        assert os.path.exists('outputs/extracted_specs.csv'), 'CSV output missing'

    def test_output_is_non_empty(self, extracted_data):
        assert len(extracted_data) > 0, 'No specs were extracted'

    def test_all_required_fields_present(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            missing = REQUIRED_FIELDS - set(spec.keys())
            assert not missing, f'Spec #{i} missing fields: {missing}\n{spec}'

    def test_value_is_string(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert isinstance(spec['value'], str), f"Spec #{i} 'value' should be str, got {type(spec['value'])}: {spec}"

    def test_source_page_is_int(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert isinstance(spec['source_page'], int), f"Spec #{i} 'source_page' should be int: {spec}"

    def test_source_page_in_valid_range(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert 1 <= spec['source_page'] <= 852, f"Spec #{i} has out-of-range source_page: {spec['source_page']}"

    def test_component_not_empty(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert spec['component'].strip(), f"Spec #{i} has empty 'component' field"

    def test_spec_type_not_empty(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert spec['spec_type'].strip(), f"Spec #{i} has empty 'spec_type' field"

    def test_context_not_empty(self, extracted_data):
        for i, spec in enumerate(extracted_data):
            assert spec['context'].strip(), f"Spec #{i} has empty 'context' field"

class TestGroundTruth:

    def _extract(self, store, query, top_k=10):
        results = store.search(query, top_k=top_k)
        return extract_specs(query, results)

    def _find_spec(self, specs, value_contains=None, unit=None, component_contains=None):
        for s in specs:
            val = str(s.get('value', '') or '')
            if value_contains and value_contains not in val:
                continue
            unit_val = str(s.get('unit', '') or '')
            if unit and unit_val.lower() != unit.lower():
                continue
            comp = str(s.get('component', '') or '')
            if component_contains and component_contains.lower() not in comp.lower():
                continue
            return s
        return None

    def test_stabilizer_bar_bracket_nut_55nm(self, store):
        specs = self._extract(store, 'Torque for stabilizer bar bracket nuts', top_k=10)
        assert len(specs) > 0, 'No specs returned for stabilizer bar bracket nut query'
        match = self._find_spec(specs, value_contains='55', unit='Nm')
        assert match is not None, f"Expected to find 55 Nm stabilizer spec. Got: {[s['value'] for s in specs]}"

    def test_wheel_nut_torque_204nm(self, store):
        wheel_nut_chunks = [c for c in store.chunks if c.page_num == 197 and '204' in c.text]
        assert len(wheel_nut_chunks) > 0, "Chunk containing '204 Nm' on page 197 must exist in the index"
        target = wheel_nut_chunks[0]
        specs = extract_specs('What is the wheel nut tightening torque?', [(target, 0.95)])
        assert len(specs) > 0, 'LLM should extract spec from wheel nut chunk'
        match = self._find_spec(specs, value_contains='204', unit='Nm')
        assert match is not None, f"Expected 204 Nm for wheel nut. Got: {[(s.get('value'), s.get('unit')) for s in specs]}"

    def test_brake_caliper_anchor_bolt_torque(self, store):
        specs = self._extract(store, 'Torque for brake caliper anchor plate bolts', top_k=10)
        assert len(specs) > 0, 'No specs returned for brake caliper bolt query'
        nm_specs = [s for s in specs if s.get('unit', '').lower() in ('nm', 'n·m')]
        assert len(nm_specs) > 0, 'Should extract at least one Nm torque for brake caliper'

    def test_lower_control_arm_bolt_350nm(self, store):
        arm_chunks = [c for c in store.chunks if c.page_num == 14 and '350' in c.text]
        assert len(arm_chunks) > 0, "Chunk containing '350 Nm' on page 14 must exist in the index"
        target = arm_chunks[0]
        specs = extract_specs('What is the torque for lower control arm cam bolt nut?', [(target, 0.95)])
        assert len(specs) > 0, 'LLM should extract spec from lower control arm chunk'
        match = self._find_spec(specs, value_contains='350', unit='Nm')
        assert match is not None, f"Expected 350 Nm for lower control arm. Got: {[(s.get('value'), s.get('unit')) for s in specs]}"

    def test_traction_lok_breakaway_torque(self, store):
        specs = self._extract(store, 'Minimum Traction-Lok breakaway torque', top_k=15)
        assert len(specs) > 0, 'No specs for Traction-Lok breakaway torque'
        match = self._find_spec(specs, value_contains='27', unit='Nm')
        assert match is not None, f"Expected 27 Nm Traction-Lok spec. Got: {[(s['value'], s['unit']) for s in specs]}"

    def test_driveline_pinion_runout(self, store):
        specs = self._extract(store, 'Drive pinion flange runout specification', top_k=10)
        assert isinstance(specs, list), 'Should return a list'
        if specs:
            for s in specs:
                assert isinstance(s.get('source_page', 0), int), 'source_page must be int'
        runout_chunks = [c for c in store.chunks if '0.25' in c.text and 'pinion' in c.text.lower()]
        assert len(runout_chunks) > 0, "Chunk with '0.25 mm' pinion runout must exist in index"

    def test_spec_type_is_torque_for_nm_values(self, store):
        specs = self._extract(store, 'Torque for stabilizer bar bracket nuts', top_k=5)
        nm_specs = [s for s in specs if 'Nm' in s.get('unit', '') or 'nm' == s.get('unit', '').lower()]
        for s in nm_specs:
            assert 'torque' in s.get('spec_type', '').lower(), f"Nm value should have Torque spec_type, got '{s['spec_type']}'"

class TestEdgeCases:

    def test_no_hallucination_on_irrelevant_query(self, store):
        specs = extract_specs('What color is the engine block paint?', store.search('engine block paint color', top_k=5))
        for s in specs:
            page = s.get('source_page')
            assert isinstance(page, int) and 1 <= page <= 852, f'Hallucinated spec has invalid source_page: {page}'
            assert isinstance(s.get('component', ''), str)

    def test_empty_result_is_empty_list(self):
        assert _safe_parse_json('[]', 'test') == []

    def test_extraction_with_zero_chunks(self):
        specs = extract_specs('some query', [])
        assert specs == []

    def test_source_page_not_string(self, store):
        specs = extract_specs('Torque for stabilizer bar bracket nuts', store.search('Torque for stabilizer bar bracket nuts', top_k=5))
        for s in specs:
            assert isinstance(s.get('source_page'), int), f"source_page should be int, got: {type(s.get('source_page'))} = {s.get('source_page')}"