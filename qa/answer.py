# qa/answer.py
from __future__ import annotations

import os
import time
import json
import logging
from typing import List, Sequence, Tuple, Optional, Dict, Set
from collections import Counter as CCounter
import textwrap
import numpy as np
import requests
from numpy.linalg import norm
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval.index_store import RetrievalResult

log = logging.getLogger(__name__)

# ----------------------------
# Config
# ----------------------------
_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4")
_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

_SENTENCE_RERANKER_MODEL = os.getenv(
    "SENTENCE_RERANKER_MODEL", "ibm-granite/granite-embedding-reranker-english-r2"
)
_EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "ibm-granite/granite-embedding-english-r2"
)

# Singletons
_SENT_RERANKER: Optional[CrossEncoder] = None
_EMBEDDER: Optional[SentenceTransformer] = None

# Generic English stopwords (domain-agnostic)
_STOP: Set[str] = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","had",
    "he","her","hers","him","his","i","if","in","into","is","it","its","itself","me",
    "my","myself","of","on","or","our","ours","ourselves","she","so","than","that",
    "the","their","theirs","them","themselves","then","there","these","they","this",
    "those","to","was","we","were","what","when","where","which","who","whom","why",
    "with","you","your","yours","yourself","yourselves","do","does","did","how","will",
    "shall","should","can","could","would","may","might","not","no","yes","up","down",
    "about","over","under","again","further","once"
}

# ----------------------------
# Model loaders
# ----------------------------
def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _ensure_sentence_reranker() -> CrossEncoder:
    global _SENT_RERANKER
    if _SENT_RERANKER is None:
        device = "cuda" if _cuda_ok() else "cpu"
        log.info("Loading sentence reranker: %s (%s)", _SENTENCE_RERANKER_MODEL, device)
        _SENT_RERANKER = CrossEncoder(_SENTENCE_RERANKER_MODEL, device=device)
    return _SENT_RERANKER

def _ensure_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        device = "cuda" if _cuda_ok() else "cpu"
        log.info("Loading embedder: %s (%s)", _EMBED_MODEL, device)
        _EMBEDDER = SentenceTransformer(_EMBED_MODEL, device=device)
    return _EMBEDDER

# ----------------------------
# Text & similarity utils
# ----------------------------
def _simple_sentence_split(s: str) -> List[str]:
    import re
    s = s.replace("\r", "")
    parts = re.split(r"(?m)(?<=\.)\s+|(?<=\?)\s+|(?<=!)\s+|\n{2,}", s)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 1]
    return parts

def _tokens(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) >= 3 and t not in _STOP]

def _content_words(text: str) -> Set[str]:
    return set(_tokens(text))

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    da, db = float(norm(a)), float(norm(b))
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def _char_ngrams(token: str, n: int = 3) -> Set[str]:
    t = f"^{token}$"
    return {t[i : i + n] for i in range(len(t) - n + 1)} if len(t) >= n else {t}

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

# ----------------------------
# Evidence selection (overlap-first; semantic fallback; MMR)
# ----------------------------
def _select_sentences_overlap_then_semantic_mmr(
    question: str,
    results: Sequence[RetrievalResult],
    max_sentences: int = 5,
    df_uninformative_ratio: float = 0.30,   # tokens in >30% of sentences are uninformative (local)
    q_ngram_thresh: float = 0.30,           # fuzzy 3-gram Jaccard threshold for overlap
    min_sem_floor: float = 0.30,            # absolute semantic floor
    top_rel_ratio: float = 0.80,            # keep >= 80% of best semantic sim
    lambda_diversity: float = 0.7,          # MMR tradeoff
) -> Tuple[List[Dict], bool]:
    """
    Returns (selected_sentences, had_overlap_pool)

    Step 1: gather sentences
    Step 2: build informative question tokens (remove stopwords and locally ubiquitous tokens)
    Step 3: build OVERLAP pool via fuzzy 3-gram Jaccard
    Step 4: if no overlap, build SEMANTIC pool via embeddings + dynamic gate
    Step 5: rank chosen pool by cross-encoder; pick with MMR (diversity on embeddings)
    """
    embedder = _ensure_embedder()
    reranker = _ensure_sentence_reranker()

    # candidates
    candidates: List[Tuple[str, str, Dict]] = []
    sent_tokens: List[Set[str]] = []
    for r in results:
        for s in _simple_sentence_split(r.chunk.text):
            candidates.append((s, r.chunk.id, r.chunk.metadata))
            sent_tokens.append(_content_words(s))
    if not candidates:
        return [], False

    # local DF (domain-agnostic)
    df = CCounter()
    for toks in sent_tokens:
        df.update(set(toks))
    n_sent = max(1, len(sent_tokens))
    uninformative: Set[str] = {t for t, c in df.items() if c >= max(2, int(df_uninformative_ratio * n_sent))}

    # informative question tokens (+ 3-grams)
    q_tokens_inf = [t for t in _content_words(question) if t not in uninformative]
    q_token_ngrams = {t: _char_ngrams(t, 3) for t in q_tokens_inf}

    # semantic sims (for fallback + MMR)
    sents = [c[0] for c in candidates]
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=False)[0]
    s_embs = embedder.encode(sents, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
    sims = np.array([_cos_sim(q_emb, e) for e in s_embs], dtype=np.float32)
    best_sim = float(np.max(sims))
    sem_gate = max(min_sem_floor, best_sim * top_rel_ratio)

    # (A) overlap pool
    overlap_idx: List[int] = []
    if q_tokens_inf:
        sent_token_ngrams = [{st: _char_ngrams(st, 3) for st in toks} for toks in sent_tokens]
        for i in range(len(candidates)):
            best_j = 0.0
            for qt_ng in q_token_ngrams.values():
                for st_ng in sent_token_ngrams[i].values():
                    j = _jaccard(qt_ng, st_ng)
                    if j > best_j:
                        best_j = j
                        if best_j >= q_ngram_thresh:
                            break
                if best_j >= q_ngram_thresh:
                    break
            if best_j >= q_ngram_thresh:
                overlap_idx.append(i)

    had_overlap_pool = len(overlap_idx) > 0

    # choose pool
    if had_overlap_pool:
        pool_idx = overlap_idx
    else:
        # semantic fallback ONLY to provide neutral citations; not a license to claim facts
        pool_idx = [i for i, s in enumerate(sims) if s >= sem_gate] or [int(np.argmax(sims))]

    kept_candidates = [candidates[i] for i in pool_idx]
    kept_embs = s_embs[pool_idx]

    # cross-encoder ranking
    pairs = [(question, c[0]) for c in kept_candidates]
    ce_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False).tolist()
    cemin, cemax = float(min(ce_scores)), float(max(ce_scores))
    ce_norm = [(s - cemin) / (cemax - cemin) if cemax > cemin else 0.5 for s in ce_scores]

    # greedy MMR
    selected: List[int] = []
    while len(selected) < max_sentences and len(selected) < len(kept_candidates):
        best_i, best_mmr = -1, -1.0
        for i in range(len(kept_candidates)):
            if i in selected:
                continue
            if not selected:
                div_pen = 0.0
            else:
                div_pen = max(float(_cos_sim(kept_embs[i], kept_embs[j])) for j in selected)
            mmr = lambda_diversity * ce_norm[i] - (1.0 - lambda_diversity) * div_pen
            if mmr > best_mmr:
                best_mmr, best_i = mmr, i
        if best_i == -1:
            break
        selected.append(best_i)

    # one excerpt per chunk
    seen = set()
    out: List[Dict] = []
    for idx in selected:
        sent, cid, md = kept_candidates[idx]
        # if cid in seen:
        #     continue
        seen.add(cid)
        out.append({"sentence": sent, "score": float(ce_scores[idx]), "chunk_id": cid, "metadata": md})
        if len(out) >= max_sentences:
            break

    return out, had_overlap_pool

# ----------------------------
# Generation + warmers
# ----------------------------
def _ollama_generate(system_prompt: str, user_prompt: str, max_tokens: int = 220) -> Optional[str]:
    try:
        payload = {
            "model": _OLLAMA_MODEL,
            "prompt": f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2, "top_p": 0.9},
        }
        r = requests.post(
            f"{_OLLAMA_BASE}/api/generate",
            data=json.dumps(payload),
            timeout=_OLLAMA_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        log.warning("Ollama generation failed: %s", e)
        return None

def warm_answer_models() -> None:
    t0 = time.time()
    _ = _ensure_sentence_reranker()
    _ = _ensure_embedder()
    try:
        requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=3)
    except Exception:
        pass
    log.info("Answer models warmed in %.2fs", time.time() - t0)

# ----------------------------
# Public API
# ----------------------------
def answer_question(question: str, results: List[RetrievalResult]) -> Tuple[str, List[Dict]]:
    """
    Grounded answer with unique, rich citations (no domain hardcoding).
    CRITICAL GUARDRail:
      - If no evidence sentence overlaps (fuzzy) with an informative question token,
        DO NOT CLAIM a fact. Return a neutral 'not explicitly stated' answer.
    """
    if not results:
        return ("Not found in the indexed documents.", [])

    top, had_overlap_pool = _select_sentences_overlap_then_semantic_mmr(question, results, max_sentences=20)
    if not top:
        return ("Not found in the indexed documents.", [])

    # Build evidence lines (safe string ops)
    evidence_lines = []
    for i, d in enumerate(top, 1):
        ev = d.get("sentence", "").replace("\n", " ").strip()

        # d["metadata"] is attached in retrieval; fall back to empty dict
        md = d.get("metadata", {}) or {}

        heading = str(md.get("heading", "") or "").strip()
        section = str(md.get("section", "") or "").strip()
        summary = str(md.get("summary", "") or "").strip()

        note_parts = []
        # Prefer: (Section 5.4: "Employees requesting approval...")
        if heading or section:
            if section and heading:
                note_parts.append(f'Section {section}: "{heading}"')
            elif section:
                note_parts.append(f"Section {section}")
            else:
                note_parts.append(f'"{heading}"')

        if summary:
            # Keep summaries compact; truncate if very long
            summarized = textwrap.shorten(summary, width=240, placeholder="…")
            note_parts.append(f"Summary: {summarized}")

        suffix = ""
        if note_parts:
            suffix = " (" + " | ".join(note_parts) + ")"

        evidence_lines.append(f"{i}. {ev}{suffix}")

    if not had_overlap_pool:
        # No lexical coverage for any informative question token → do not call LLM.
        ans = (
            "The provided documents do not explicitly address this question. "
            "Please review the cited policy for details."
        )
    else:
        # system = (
        #     "You are a precise assistant. Answer ONLY using the provided evidence. "
        #     "Use the policy text verbatim where possible. "
        #     "Focus strictly on the question’s topic; do not add unrelated information. "
        #     "Include any specific identifiers (e.g., form numbers). "
        #     "Write 1–3 concise sentences."
        # )
        
        system = (
            "You are a policy assistant.  Answer the user’s question using only the provided evidence. "
            "When the policy mentions a form or document, include its exact name in your answer. "
            "List each step in order and do not invent any details that are not explicitly in the evidence."
        )

        user = "Question: " + question + "\n\nEvidence:\n" + "\n".join(evidence_lines) + "\n\nAnswer:"
        ans = _ollama_generate(system, user, max_tokens=200) or " ".join(d["sentence"] for d in top[:3])

    # citations
    citations: List[Dict] = []
    for d in top:
        md = d.get("metadata") or {}
        citations.append(
            {
                "id": d["chunk_id"],
                "score": d["score"],
                "excerpt": d["sentence"],
                "section": md.get("section"),
                "heading": md.get("heading"),
                "document_title": md.get("document_title"),
                "source_url": md.get("source_url"),
            }
        )
    return ans, citations

def generate_summary(text: str) -> str:
    system = (
        "You summarise into a concise paragraph including all key steps and conditions. "
        "No fluff, only key facts. "
        "Write as a standalone statement."
    )
    user = f"Summarize succinctly:\n{text[:2500]}"
    out = _ollama_generate(system, user, max_tokens=120)
    if out:
        return out
    parts = _simple_sentence_split(text)
    return " ".join(parts[:2])[:400]
