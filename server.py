import os
import sys
import json
import hashlib

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

from src.embedder import VectorStore
from src.extractor import extract_specs, answer_question, determine_intent
from src.pipeline import VECTORSTORE_PATH, SPEC_DB_PATH

app = FastAPI(title="Spec Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
try:
    store.load(VECTORSTORE_PATH)
    print(f"Loaded {len(store.chunks)} chunks.")
except Exception:
    print("Warning: VectorStore not loaded. Run pipeline.py first.")

spec_database: List[Dict[str, Any]] = []
if os.path.exists(SPEC_DB_PATH):
    with open(SPEC_DB_PATH, "r") as f:
        spec_database = json.load(f)

_cache: Dict[str, Dict[str, Any]] = {}
CACHE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "query_cache.json")
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r") as f:
            _cache = json.load(f)
        print(f"Loaded {len(_cache)} cached queries.")
    except Exception:
        pass

_session_specs: Dict[str, List[Dict[str, Any]]] = {}
_chat_cache_keys: Dict[str, List[str]] = {}
CHAT_CACHE_KEYS_PATH = os.path.join(os.path.dirname(__file__), "outputs", "chat_cache_keys.json")
if os.path.exists(CHAT_CACHE_KEYS_PATH):
    try:
        with open(CHAT_CACHE_KEYS_PATH, "r") as f:
            _chat_cache_keys = json.load(f)
    except Exception:
        pass

CHAT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "outputs", "chat_history.json")
PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample-service-manual 1.pdf")


def _cache_key(query: str, top_k: int, chat_id: str = "") -> str:
    return hashlib.md5(f"{chat_id}:{query.strip().lower()}:{top_k}".encode()).hexdigest()


def _save_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(_cache, f)
    with open(CHAT_CACHE_KEYS_PATH, "w") as f:
        json.dump(_chat_cache_keys, f)


def _search_spec_db(query: str) -> List[Dict[str, Any]]:
    if not spec_database:
        return []
    terms = query.lower().split()
    results = []
    for spec in spec_database:
        text = f"{spec.get('component','')} {spec.get('spec_type','')} {spec.get('value','')} {spec.get('unit','')} {spec.get('context','')}".lower()
        if all(t in text for t in terms):
            results.append(spec)
    return results


def _keyword_search_chunks(query: str, top_k: int = 10):
    if not store.chunks:
        return []
    import re as _re
    stop = {"what", "are", "the", "for", "and", "how", "to", "is", "in", "of", "a", "an",
            "i", "it", "do", "does", "this", "that", "these", "those", "can", "we"}
    raw_terms = _re.sub(r'[^\w\s]', '', query.lower()).split()
    terms = [t for t in raw_terms if len(t) > 2 and t not in stop]
    if not terms:
        return []

    def _variants(term):
        v = {term}
        # Strip common suffixes for stem matching
        for sfx in ['ing', 'tion', 'ed', 'es', 's', 'er', 'ers', 'ment']:
            if term.endswith(sfx) and len(term) - len(sfx) >= 3:
                v.add(term[:-len(sfx)])
        # Strip double-letter variants: calliper -> caliper
        v.add(_re.sub(r'(.)\1', r'\1', term))
        # Short prefix
        if len(term) >= 5:
            v.add(term[:4])
        return v

    term_variants = [(t, _variants(t)) for t in terms]

    scored = []
    for chunk in store.chunks:
        text = chunk.text.lower()
        score = sum(1 for _, variants in term_variants if any(v in text for v in variants))
        if score >= 2:
            scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(chunk, float(score) / len(terms)) for chunk, score in scored[:top_k]]


def _check_session_specs(chat_id: str, query: str) -> List[Dict[str, Any]]:
    if chat_id not in _session_specs:
        return []
    stop_words = {"what", "are", "the", "for", "and", "how", "to", "is", "in", "of", "a", "an", "steps", "procedure"}
    terms = [t for t in query.lower().split() if len(t) > 2 and t not in stop_words]
    if not terms:
        return []
    matches = []
    for spec in _session_specs[chat_id]:
        text = f"{spec.get('component','')} {spec.get('spec_type','')} {spec.get('context','')} {spec.get('value','')} {spec.get('unit','')}".lower()
        if all(t in text for t in terms):
            matches.append(spec)
    return matches


class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    chat_context: Optional[List[str]] = None
    chat_id: Optional[str] = None


class QueryResponse(BaseModel):
    specs: List[Dict[str, Any]]
    context_chunks: List[str]
    answer: str = ""
    cached: bool = False
    source: str = "rag"
    show_table: bool = True


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not store.chunks:
        raise HTTPException(status_code=500, detail="VectorStore not loaded.")

    chat_id = req.chat_id or ""
    key = _cache_key(req.query, req.top_k, chat_id)

    # Tier 1: query cache
    if key in _cache:
        cached_data = _cache[key]
        if isinstance(cached_data, list):
            specs = cached_data
            answer = f"Found {len(specs)} specification(s) matching your query." if specs else ""
            show_table = True
        else:
            specs = cached_data.get("specs", [])
            answer = cached_data.get("answer", "")
            show_table = cached_data.get("show_table", True)
        return {"specs": specs, "context_chunks": [], "answer": answer, "cached": True, "source": "cache", "show_table": show_table}

    # Tier 2: session memory
    if req.chat_id:
        session_matches = _check_session_specs(req.chat_id, req.query)
        if len(session_matches) >= 1:
            summary = f"Found {len(session_matches)} relevant spec(s) from this conversation."
            return {"specs": session_matches, "context_chunks": [], "answer": summary, "cached": True, "source": "session", "show_table": True}

    # Tier 3: pre-extracted spec database
    db_matches = _search_spec_db(req.query)
    if len(db_matches) >= 2:
        if chat_id:
            _session_specs.setdefault(chat_id, []).extend(db_matches)
            if key not in _chat_cache_keys.setdefault(chat_id, []):
                _chat_cache_keys[chat_id].append(key)
        summary = f"Found {len(db_matches)} specification(s) via instant lookup."
        _cache[key] = {"specs": db_matches, "answer": summary, "show_table": True}
        _save_cache()
        return {"specs": db_matches, "context_chunks": [], "answer": summary, "cached": False, "source": "spec_db", "show_table": True}

    # Tier 4: full RAG
    enriched_query = req.query
    if req.chat_context:
        context_str = " | ".join(req.chat_context[-3:])
        enriched_query = f"Previous questions: {context_str}. Current question: {req.query}"

    results = store.search(enriched_query, top_k=req.top_k)

    # Hybrid: supplement semantic results with keyword-matched chunks
    kw_results = _keyword_search_chunks(req.query, top_k=10)
    seen_ids = {chunk.chunk_id for chunk, _ in results}
    for chunk, score in kw_results:
        if chunk.chunk_id not in seen_ids:
            results.append((chunk, score))
            seen_ids.add(chunk.chunk_id)
    results = results[:req.top_k + 10]  # allow slightly more context for hybrid

    chunks_text = [chunk.text for chunk, _ in results]
    intent = determine_intent(req.query)

    specs = []
    answer = ""
    show_table = True

    if intent in ["TABLE", "BOTH"]:
        specs = extract_specs(req.query, results)
    if intent in ["TEXT", "BOTH"]:
        answer = answer_question(req.query, results)
    else:
        answer = f"Found {len(specs)} specification(s) matching your query."
    if intent == "TEXT":
        show_table = False

    if specs:
        stop = {"what", "is", "the", "of", "for", "a", "an", "and", "to", "in", "are"}
        q_terms = set(req.query.lower().split()) - stop
        def relevance(s):
            text = f"{s.get('component','')} {s.get('spec_type','')} {s.get('context','')}".lower()
            return sum(1 for t in q_terms if t in text)
        specs = sorted(specs, key=relevance, reverse=True)[:5]

    not_found_phrases = ["not covered", "not found", "does not contain", "no information", "not available"]
    if specs and answer and any(p in answer.lower() for p in not_found_phrases):
        top_comps = ", ".join(s["component"] for s in specs[:3])
        answer = f"Here are the most relevant specifications I found for your query: {top_comps}."
        if len(specs) > 3:
            answer += f" Showing top {len(specs)} results."

    _cache[key] = {"specs": specs, "answer": answer, "show_table": show_table}
    if chat_id:
        _session_specs.setdefault(chat_id, []).extend(specs)
        if key not in _chat_cache_keys.setdefault(chat_id, []):
            _chat_cache_keys[chat_id].append(key)
    _save_cache()

    return {"specs": specs, "context_chunks": chunks_text, "answer": answer, "cached": False, "source": "rag", "show_table": show_table}


@app.get("/api/chats")
def get_chats():
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            return json.load(f)
    return {"chats": []}


@app.post("/api/chats")
def save_chats(data: dict):
    os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)
    with open(CHAT_HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved"}


@app.delete("/api/cache")
def clear_cache():
    _cache.clear()
    _chat_cache_keys.clear()
    for path in [CACHE_PATH, CHAT_CACHE_KEYS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    return {"status": "cleared"}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    _session_specs.pop(chat_id, None)
    if chat_id in _chat_cache_keys:
        for key in _chat_cache_keys[chat_id]:
            _cache.pop(key, None)
        _chat_cache_keys.pop(chat_id, None)
        _save_cache()
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            data = json.load(f)
        data["chats"] = [c for c in data.get("chats", []) if c.get("id") != chat_id]
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return {"status": "deleted"}
    return {"status": "not_found"}


@app.put("/api/chats/{chat_id}/rename")
def rename_chat(chat_id: str, body: dict):
    new_title = body.get("title", "")
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            data = json.load(f)
        for c in data.get("chats", []):
            if c.get("id") == chat_id:
                c["title"] = new_title
                break
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return {"status": "renamed"}
    return {"status": "not_found"}


@app.post("/api/chats/{chat_id}/autotitle")
def autotitle_chat(chat_id: str, body: dict):
    import ollama as _ollama
    question = body.get("question", "")
    answer = body.get("answer", "")
    prompt = (
        f"Generate a short chat title (4-6 words max, no punctuation) for a conversation about: "
        f"'{question}'. Answer was: '{answer[:120]}'. Reply with ONLY the title, nothing else."
    )
    try:
        res = _ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 20},
        )["message"]["content"].strip().strip("\"'").strip()
        title = res[:50] if res else question[:50]
    except Exception:
        title = question[:50]

    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            data = json.load(f)
        for c in data.get("chats", []):
            if c.get("id") == chat_id:
                c["title"] = title
                break
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
    return {"title": title}


@app.get("/api/pdf")
def serve_pdf():
    if not os.path.exists(PDF_FILE_PATH):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(PDF_FILE_PATH, media_type="application/pdf", headers={"Content-Disposition": "inline"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
