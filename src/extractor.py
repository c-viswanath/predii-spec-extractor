import json
import re
import ollama
from typing import List, Dict, Any, Tuple
from src.chunker import TextChunk

MODEL_FAST = "llama3.2:3b"
MODEL_FULL = "llama3.1:8b"
MODEL_TINY = "llama3.2:1b"

SIMPLE_PATTERNS = [
    "torque", "bolt", "nut", "capacity", "fluid", "thickness",
    "clearance", "runout", "preload", "backlash", "sensor",
    "bracket", "caliper", "disc", "rotor", "bearing"
]

SYSTEM_PROMPT = """You are an automotive technical data extraction specialist.
Extract vehicle specifications from Ford F-150 workshop manual text.

OUTPUT RULES:
1. Return ONLY a valid JSON array. No markdown fences, no explanation.
2. Each element must have these fields:
   - "component": string, the part or assembly name
   - "spec_type": string, category (Torque, Fluid Capacity, Clearance, etc.)
   - "value": string, the numeric value (e.g. "350")
   - "unit": string, unit of measurement (e.g. "Nm", "L", "mm")
   - "context": string, the sentence from the text containing the spec
   - "source_page": integer, page number
3. ALL string values MUST be quoted. Never output bare numbers with units.
4. For broad queries like "what are the specs", extract ALL specs from the text.
5. If no specifications exist, return exactly: []
6. Do not invent specifications.
7. Every spec is a flat JSON object. No nesting.

Example:
[{"component": "Wheel Nut", "spec_type": "Torque", "value": "204", "unit": "Nm", "context": "Tighten to 204 Nm.", "source_page": 197}]"""

QA_PROMPT = """You are a Ford F-150 workshop manual assistant. Answer the user's question using ONLY the provided manual excerpts.

Rules:
- Start your answer DIRECTLY with the relevant information. NEVER start with disclaimers like "Unfortunately", "There is no information", or "The excerpts do not contain".
- If the excerpts contain ANY relevant procedure, steps, torque specs, or mentions — present that information immediately.
- For procedures (removal, installation, how-to), list ALL steps exactly as written in the manual. Include torque specifications.
- For factual questions, be concise (2-4 sentences).
- Workshop manuals use generic language like "the specified component" or "reverse the removal procedure" — this IS the answer. Report it exactly as written.
- ONLY say "This information is not covered in the retrieved manual sections." if NONE of the excerpts contain ANY remotely relevant information.
- Do not invent information."""


def _classify_complexity(query):
    q = query.lower().strip()
    broad_keywords = ["all", "specs", "specifications", "everything", "list", "what are"]
    if any(kw in q for kw in broad_keywords):
        return MODEL_FULL
    words = q.split()
    if len(words) <= 8 and any(p in q for p in SIMPLE_PATTERNS):
        return MODEL_FAST
    if any(kw in q for kw in ["compare", "difference", "versus", "vs", "why", "how", "explain"]):
        return MODEL_FULL
    if len(words) <= 12:
        return MODEL_FAST
    return MODEL_FULL


def determine_intent(query: str) -> str:
    prompt_text = f"Does the following query ask for procedural steps, instructions, explanations, or how-tos? Answer ONLY YES or NO.\nQuery: \"{query}\"\nAnswer:"
    prompt_table = f"Does the following query ask for numerical specifications, torques, capacities, clearances, or a list of parts? Answer ONLY YES or NO.\nQuery: \"{query}\"\nAnswer:"
    try:
        res_text = ollama.chat(
            model=MODEL_TINY,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": 0.0, "num_predict": 5},
        )["message"]["content"].strip().upper()
        res_table = ollama.chat(
            model=MODEL_TINY,
            messages=[{"role": "user", "content": prompt_table}],
            options={"temperature": 0.0, "num_predict": 5},
        )["message"]["content"].strip().upper()

        needs_text = "YES" in res_text
        needs_table = "YES" in res_table
        q_lower = query.lower()
        if any(kw in q_lower for kw in ["steps", "how to", "procedure", "removal", "installation", "remove", "install"]):
            needs_text = True
            needs_table = False

        if needs_text and needs_table: return "BOTH"
        if needs_text: return "TEXT"
        if needs_table: return "TABLE"
        return "BOTH"
    except Exception:
        return "BOTH"


def _check_model_available(model):
    try:
        models = ollama.list()
        available = [m.get("name", "") for m in models.get("models", [])]
        return any(model in name for name in available)
    except Exception:
        return False


def extract_specs(query, retrieved_chunks, model=None):
    if not retrieved_chunks:
        return []
    if model is None:
        chosen = _classify_complexity(query)
        if chosen == MODEL_FAST and not _check_model_available(MODEL_FAST):
            chosen = MODEL_FULL
        model = chosen

    context_blocks = []
    for chunk, score in retrieved_chunks:
        block = f"[Page {chunk.page_num} | Section: {chunk.section or 'General'} | Relevance: {score:.3f}]\n{chunk.text}"
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)
    user_message = f"Query: {query}\n\nFord F-150 Workshop Manual excerpts:\n\n{context}\n\nExtract all specifications relevant to the query. Return JSON array only."

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.1, "num_predict": 8192},
    )
    raw = response["message"]["content"].strip()
    specs = _safe_parse_json(raw, query)
    return _validate_specs(specs)


def answer_question(query, retrieved_chunks, model=None):
    if not retrieved_chunks:
        return ""
    if model is None:
        model = _classify_complexity(query)
        if model == MODEL_FAST and not _check_model_available(MODEL_FAST):
            model = MODEL_FULL

    import re as _re
    stop = {"what", "are", "the", "for", "and", "how", "to", "is", "in", "of", "a", "an",
            "i", "it", "do", "does", "this", "that", "these", "those", "can", "we"}
    q_terms = [t for t in _re.sub(r'[^\w\s]', '', query.lower()).split() if len(t) > 2 and t not in stop]

    def _chunk_relevance(chunk):
        text = chunk.text.lower()
        return sum(1 for t in q_terms if t in text)

    ranked = sorted(retrieved_chunks, key=lambda x: _chunk_relevance(x[0]), reverse=True)

    context_blocks = []
    for chunk, score in ranked[:5]:
        context_blocks.append(f"[Page {chunk.page_num}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": QA_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nManual excerpts:\n{context}"},
        ],
        options={"temperature": 0.2, "num_predict": 1024},
    )
    raw = response["message"]["content"].strip()

    # Strip hedging disclaimers if the answer also contains actual procedure content
    import re as _re2
    negation_patterns = ["not covered", "no information", "does not contain", "not available",
                         "not found", "unfortunately", "not explicitly", "not mentioned",
                         "no specific information", "not described"]
    has_disclaimer = any(p in raw.lower() for p in negation_patterns)
    procedure_signals = [r'\d+\.\s', r'removal', r'installation', r'tighten',
                         r'torque', r'position the', r'reverse the']
    has_content = sum(1 for p in procedure_signals if _re2.search(p, raw.lower())) >= 2
    if has_disclaimer and has_content:
        lines = raw.split('\n')
        cleaned = []
        for line in lines:
            ll = line.lower().strip()
            if not ll:
                cleaned.append(line)
                continue
            # Skip lines that are pure disclaimers (contain negation, don't start with a step number)
            if any(p in ll for p in negation_patterns) and not _re2.match(r'^\d+\.', ll):
                continue
            # Skip "However," transition lines
            if ll.startswith('however,') and not _re2.match(r'^\d+\.', ll):
                continue
            cleaned.append(line)
        raw = '\n'.join(cleaned).strip()

    return raw


def _validate_specs(specs):
    valid = []
    for s in specs:
        if not isinstance(s, dict):
            continue
        if not s.get("component") and not s.get("value"):
            continue
        s["component"] = str(s.get("component", "Unknown")).strip()
        s["value"] = str(s.get("value", "")).strip()
        s["spec_type"] = str(s.get("spec_type", "Specification")).strip()
        s["unit"] = str(s.get("unit", "")).strip()
        s["context"] = str(s.get("context", "")).strip()
        try:
            s["source_page"] = int(str(s.get("source_page", 0)).strip().split()[0])
        except (ValueError, TypeError, IndexError):
            s["source_page"] = 0
        valid.append(s)

    seen = set()
    deduped = []
    for s in valid:
        key = (s["component"].lower(), s["value"], s["unit"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def _repair_json(raw):
    repaired = re.sub(
        r':\s*(\d+[\d.\s]*(?:Nm|mm|lb-ft|lb-in|ml|pt|L|in|°)[^",}\]]*)',
        lambda m: ': "' + m.group(1).strip() + '"',
        raw
    )
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    return repaired


def _try_parse(text):
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
            return []
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _repair_truncated_array(text):
    depth = 0
    last_complete_end = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_complete_end = i
    if last_complete_end > 0:
        bracket_start = text.find('[')
        if bracket_start >= 0:
            truncated = text[bracket_start:last_complete_end + 1] + ']'
            return _try_parse(truncated)
    return None


def _safe_parse_json(raw, query):
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    parsed = _try_parse(cleaned)
    if parsed is not None:
        return parsed

    repaired = _repair_json(cleaned)
    parsed = _try_parse(repaired)
    if parsed is not None:
        return parsed

    repaired_trunc = _repair_truncated_array(cleaned)
    if repaired_trunc is not None:
        return repaired_trunc

    array_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if array_match:
        parsed = _try_parse(array_match.group(0))
        if parsed is not None:
            return parsed
        parsed = _try_parse(_repair_json(array_match.group(0)))
        if parsed is not None:
            return parsed

    return []