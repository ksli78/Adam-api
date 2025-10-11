"""
Instruction‑tuned question answering and summarisation.

This module wraps IBM's Granite instruction models to provide two
high‑level capabilities: summarising document chunks and answering user
queries given a set of retrieved contexts.  The model is loaded lazily on
first use and cached globally to avoid redundant initialisation.

Because Granite‑3.3‑8B‑Instruct is a large model, ensure that your
environment has sufficient VRAM (or use a smaller variant) and consider
loading it in 4‑bit or 8‑bit quantisation if running on a consumer GPU.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import config
from retrieval.index_store import RetrievalResult

logger = logging.getLogger(__name__)


_INSTRUCT_TOKENIZER: Optional[AutoTokenizer] = None
_INSTRUCT_MODEL: Optional[AutoModelForCausalLM] = None


class InstructModel:
    """Loader and inference wrapper for Granite instruction models."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.INSTRUCT_MODEL_NAME
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    @staticmethod
    def _load_tokenizer() -> AutoTokenizer:
        global _INSTRUCT_TOKENIZER
        if _INSTRUCT_TOKENIZER is None:
            logger.info("Loading tokenizer for model %s", config.INSTRUCT_MODEL_NAME)
            _INSTRUCT_TOKENIZER = AutoTokenizer.from_pretrained(
                config.INSTRUCT_MODEL_NAME
            )
        return _INSTRUCT_TOKENIZER

    @staticmethod
    def _load_model() -> AutoModelForCausalLM:
        global _INSTRUCT_MODEL
        if _INSTRUCT_MODEL is None:
            logger.info("Loading instruction model %s", config.INSTRUCT_MODEL_NAME)
            # Use bfloat16 if available; fallback to float16
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
            _INSTRUCT_MODEL = AutoModelForCausalLM.from_pretrained(
                config.INSTRUCT_MODEL_NAME,
                device_map="auto",
                torch_dtype=dtype,
            )
        return _INSTRUCT_MODEL

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        """Generate text from a prompt using the instruction model."""
        tokenizer = self.tokenizer
        model = self.model
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        # Set generation parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=False,
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                generation_config=generation_config,
            )
        # Remove the prompt portion from the output
        generated_ids = output_ids[0][input_ids.shape[1] :]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result.strip()


def generate_summary(text: str, *, max_new_tokens: int = 64) -> str:
    """Generate a concise summary of a chunk of text."""
    instruct = InstructModel()
    prompt = (
        "You are a helpful assistant summarising sections of internal policy documents. "
        "Given the following section, produce a concise summary capturing the main points."\
        "\n\nSection:\n"
        f"{text}\n\nSummary:"
    )
    summary = instruct.generate(prompt, max_new_tokens=max_new_tokens)
    return summary


def answer_question(
    question: str,
    contexts: List[RetrievalResult],
    *,
    max_new_tokens: int = 512,
) -> str:
    """Answer a user's question using retrieved context.

    Constructs a prompt that introduces the assistant (ADAM), provides
    references to the top retrieved chunks (including summaries if available)
    and asks the model to answer the question citing sources using square
    brackets with the chunk identifiers.  The model is encouraged to admit
    when it does not know the answer and to avoid fabrication.

    Parameters
    ----------
    question:
        The user's query.
    contexts:
        List of retrieval results (chunks) ranked by relevance.
    max_new_tokens:
        Maximum number of tokens to generate in the answer.

    Returns
    -------
    str
        The model's answer, including citations like ``[doc123-1a2b]``.
    """
    instruct = InstructModel()
    # Construct context string with citations
    context_lines = []
    for res in contexts:
        chunk = res.chunk
        citation = f"[{chunk.id}]"
        # Use summary if available to reduce prompt length
        snippet = chunk.summary or chunk.text
        # Truncate snippet to avoid exceeding the context length; keep first 200 words
        words = snippet.split()
        if len(words) > 200:
            snippet = " ".join(words[:200]) + " ..."
        context_lines.append(f"{citation} {snippet}")
    context_str = "\n".join(context_lines)

    # Compose the final prompt
    prompt = (
        "You are ADAM (Amentum Document Assistance Model), a knowledge engine trained "
        "to answer questions about company policies and procedures. "
        "Use the provided context to answer the question accurately. "
        "Cite the relevant chunks by their identifiers in square brackets (e.g., [doc123-1a2b]). "
        "If the answer is not present in the context, respond with 'I'm not sure based on the provided documents.'\n\n"
        "Context:\n"
        f"{context_str}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = instruct.generate(prompt, max_new_tokens=max_new_tokens)
    return answer
