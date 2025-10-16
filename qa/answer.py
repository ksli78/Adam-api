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
# ----------------------------
# Compression / noise filter (keep only decisive, same-document snippets)
# ----------------------------
import re

_BANNED_SECTION_TITLES = {
    "purpose", "scope", "definitions", "applicable and reference documents",
    "document history", "glossary"
}
_DECISIVE_PATTERNS = [
    r"\bnot\s+permitted\b",
    r"\bmay\s+not\b",
    r"\bprohibited\b",
    r"\bdoes\s+not\s+constitute\s+an\s+approved\s+reason\b",
    r"\bconstitutes\s+immediate\s+termination\b",
    r"\brequires?\b.*\bapproval\b",
]

def _is_banned_section(md: Dict) -> bool:
    h = (md or {}).get("heading") or (md or {}).get("section") or ""
    return bool(h and h.strip().lower() in _BANNED_SECTION_TITLES)

def _decisive_score(text: str) -> int:
    t = text.lower()
    score = 0
    for pat in _DECISIVE_PATTERNS:
        if re.search(pat, t):
            score += 1
    return score

def _compress_policy_snippets(
    question: str,
    snippets: List[Dict],
    *,
    max_keep: int = 2,
    prefer_single_document: bool = True
) -> List[Dict]:
    """
    Each snippet: {'sentence','score','chunk_id','metadata'}.
    Keeps 1–2 decisive, on-topic lines from the best-matching document.
    """
    if not snippets:
        return []

    # 1) Optionally restrict to the single most relevant document
    pool = snippets
    if prefer_single_document:
        by_doc: Dict[str, List[Dict]] = {}
        for s in snippets:
            md = s.get("metadata") or {}
            doc_id = md.get("document_id") or s.get("chunk_id")
            by_doc.setdefault(str(doc_id), []).append(s)
        scored_docs = [(doc, sum(x.get("score", 0.0) for x in lst)) for doc, lst in by_doc.items()]
        top_doc = max(scored_docs, key=lambda x: x[1])[0]
        pool = by_doc[top_doc]

    # 2) Drop boilerplate sections (Scope, Definitions, Purpose…)
    pool = [s for s in pool if not _is_banned_section(s.get("metadata") or {})] or pool

    # 3) Rank by decisive phrases first, then reranker score
    ranked = sorted(
        pool,
        key=lambda s: (_decisive_score(s.get("sentence", "")), float(s.get("score", 0.0))),
        reverse=True,
    )

    # 4) Deduplicate near-identical sentences and cap
    seen = set()
    out: List[Dict] = []
    for s in ranked:
        t = s.get("sentence", "").strip().lower()
        if t in seen:
            continue
        seen.add(t)
        out.append(s)
        if len(out) >= max_keep:
            break
    return out

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
    max_sentences: int = 20,
    df_uninformative_ratio: float = 0.30,
    q_ngram_thresh: float = 0.25,
    min_sem_floor: float = 0.15,
    top_rel_ratio: float = 0.15,
    lambda_diversity: float = 0.7,
) -> Tuple[List[Dict], bool]:
    """
    Returns (selected_sentences, had_overlap_pool)

    Strategy:
      1) Build an overlap pool (fuzzy char-3gram Jaccard).
      2) If the pool is small or from few chunks, AUGMENT with top semantic sentences
         preferring NEW chunks first.
      3) Cross-encoder re-rank for relevance.
      4) Diversify with round-robin across chunks (optionally with MMR inside each round).
    """
    embedder = _ensure_embedder()
    reranker = _ensure_sentence_reranker()

    # ---- candidates ----
    candidates: List[Tuple[str, str, Dict]] = []
    sent_tokens: List[Set[str]] = []
    for r in results:
        for s in _simple_sentence_split(r.chunk.text):
            candidates.append((s, r.chunk.id, r.chunk.metadata))
            sent_tokens.append(_content_words(s))
    if not candidates:
        return [], False

    # ---- local DF to drop ubiquitous tokens ----
    df = CCounter()
    for toks in sent_tokens:
        df.update(set(toks))
    n_sent = max(1, len(sent_tokens))
    uninformative: Set[str] = {t for t, c in df.items() if c >= max(2, int(df_uninformative_ratio * n_sent))}

    # ---- informative question tokens + 3-grams ----
    q_tokens_inf = [t for t in _content_words(question) if t not in uninformative]
    q_token_ngrams = {t: _char_ngrams(t, 3) for t in q_tokens_inf}

    # ---- semantic sims for fallback/augmentation ----
    sents = [c[0] for c in candidates]
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=False)[0]
    s_embs = embedder.encode(sents, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
    sims = np.array([_cos_sim(q_emb, e) for e in s_embs], dtype=np.float32)
    order_by_sem = np.argsort(sims)[::-1].tolist()
    best_sim = float(np.max(sims)) if sims.size else 0.0
    sem_gate = max(min_sem_floor, best_sim * top_rel_ratio)

    # ---- (A) overlap pool via fuzzy n-gram Jaccard ----
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

    # ---- build pool with cross-chunk augmentation ----
    MIN_POOL_SIZE = min(8, max_sentences)
    pool_idx: List[int] = list(dict.fromkeys(overlap_idx)) if had_overlap_pool else [
        i for i in order_by_sem if sims[i] >= sem_gate
    ]

    # Prefer adding items from NEW chunks first
    def chunk_id(i: int) -> str:
        return candidates[i][1]

    have_chunks = {chunk_id(i) for i in pool_idx}
    # First pass: add top semantic items from *new* chunks until threshold
    if len(pool_idx) < MIN_POOL_SIZE:
        for i in order_by_sem:
            cid = chunk_id(i)
            if cid in have_chunks:
                continue
            pool_idx.append(i)
            have_chunks.add(cid)
            if len(pool_idx) >= MIN_POOL_SIZE:
                break
    # Second pass: still short? pad by overall semantics
    if len(pool_idx) < MIN_POOL_SIZE:
        for i in order_by_sem:
            if i not in pool_idx:
                pool_idx.append(i)
            if len(pool_idx) >= MIN_POOL_SIZE:
                break

    kept_candidates = [candidates[i] for i in pool_idx]
    kept_embs = s_embs[pool_idx]

    # ---- cross-encoder re-ranking (relevance) ----
    pairs = [(question, c[0]) for c in kept_candidates]
    try:
        ce_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False).tolist()
    except Exception:
        ce_scores = [float(sims[i]) for i in pool_idx]  # fallback to bi-encoder sims

    # Normalize embeddings for diversity calcs if needed later
    kept_norm = kept_embs.astype(np.float32)
    norms = np.linalg.norm(kept_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    kept_norm = kept_norm / norms

    # ---- round-robin across chunks (guarantees >1 sentence if multiple chunks exist) ----
    # (empirically used in diversification pipelines; simple & effective)  # noqa
    # Build groups by chunk with items sorted by CE score desc
    idx_sorted = sorted(range(len(kept_candidates)), key=lambda i: ce_scores[i], reverse=True)
    groups: Dict[str, List[int]] = {}
    for i in idx_sorted:
        groups.setdefault(kept_candidates[i][1], []).append(i)

    # cycle through groups, pick 1 at a time (optionally MMR inside ties)
    picked: List[int] = []
    per_chunk_cap = 1  # set to 2 if you want more per chunk
    group_order = list(groups.keys())

    while len(picked) < max_sentences:
        made_progress = False
        for g in group_order:
            g_list = groups[g]
            # enforce per-chunk cap
            taken_from_g = sum(1 for p in picked if kept_candidates[p][1] == g)
            if taken_from_g >= per_chunk_cap:
                continue
            # pick next best from this chunk
            if g_list:
                cand_i = g_list.pop(0)
                picked.append(cand_i)
                made_progress = True
                if len(picked) >= max_sentences:
                    break
        if not made_progress:
            break  # exhausted all groups

    # If we still have room (few chunks), fill remaining with global MMR on leftovers
    if len(picked) < max_sentences:
        remaining = [i for i in idx_sorted if i not in picked]
        ce_min, ce_max = min(ce_scores) if ce_scores else 0.0, max(ce_scores) if ce_scores else 1.0
        ce_norm = [(s - ce_min) / (ce_max - ce_min) if ce_max > ce_min else 0.5 for s in ce_scores]
        while remaining and len(picked) < max_sentences:
            best_i, best_mmr = -1, -1e9
            for i in remaining:
                if not picked:
                    div_pen = 0.0
                else:
                    div_pen = max(float(np.dot(kept_norm[i], kept_norm[j])) for j in picked)
                mmr = lambda_diversity * ce_norm[i] - (1.0 - lambda_diversity) * div_pen
                if mmr > best_mmr:
                    best_mmr, best_i = mmr, i
            if best_i == -1:
                break
            picked.append(best_i)
            remaining.remove(best_i)

    # ---- output ----
    out: List[Dict] = []
    for i in picked[:max_sentences]:
        sent, cid, md = kept_candidates[i]
        out.append({"sentence": sent, "score": float(ce_scores[i]), "chunk_id": cid, "metadata": md})
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
            "options": {"num_predict": max_tokens, "temperature": 0.0, "top_p": 0.9},
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
    # shrink noisy evidence to only the decisive 1–2 lines from the best-matching doc
    top = _compress_policy_snippets(question,top,max_keep=2,prefer_single_document=True)

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
        system = (
            "You are a precise assistant. Answer ONLY using the provided evidence. "
            "Use the policy text verbatim where possible. "
            "Focus strictly on the question’s topic; do not add unrelated information. "
            "Include any specific identifiers (e.g., form numbers). "
            "Do not add any extra context, examples, or unrelated policy text. "
            "If the policy forbids something, say so directly and quote the decisive clause verbatim."
            "Write 1–3 concise sentences."
        )
        
        # system = (
        #     "You are a policy assistant.  Answer the user’s question using only the provided evidence. "
        #     "When the policy mentions a form or document, include its exact name in your answer. "
        #     "List each step in order and do not invent any details that are not explicitly in the evidence."
        # )

        user = "Question: " + question + "\n\nEvidence:\n" + "\n".join(evidence_lines) + "\n\nAnswer:"
        #user = "Question: " + question + "\n\nEvidence:\n" + "\n\n".join(r.chunk.text for r in results[:3]) + "\n\nAnswer:"
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
