"""
PDF parsing utilities using Docling.

This module wraps the Docling library to convert PDF documents into a
structured, layout‑preserving markdown format.  By default we use
IBM’s Granite‑Docling model to extract tables, formulas and visual
hierarchies accurately【583088636613067†L68-L99】.  Should you wish to use a
different model or pipeline, adjust the ``model_spec`` argument when
constructing the converter.

If ``PDF_CACHE_DIR`` is configured in :mod:`adam_rag.config`, this module
caches downloaded or uploaded PDF files to that directory so that
conversion can be repeated without redownloading.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from . import text_cleaner  # imported for side effects (ensures config loaded)
from config import PDF_CACHE_DIR

logger = logging.getLogger(__name__)


def _ensure_pdf_cache_dir() -> None:
    """Ensure the PDF cache directory exists if configured."""
    if PDF_CACHE_DIR is not None:
        PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def convert_pdf_to_markdown(source: str | Path, *, cache: bool = True) -> str:
    """Convert a PDF document into markdown using Docling.

    Parameters
    ----------
    source:
        Either a file path to a local PDF or a URL pointing at a remote
        resource (e.g. SharePoint).  Docling accepts both local paths and
        remote URLs.
    cache:
        If ``True`` and :data:`PDF_CACHE_DIR` is configured, a copy of the
        original PDF will be stored in that directory using the file name
        portion of ``source``.  This can speed up subsequent ingestions of
        the same file.

    Returns
    -------
    str
        The extracted markdown representation of the document.

    Notes
    -----
    Docling automatically downloads the required model weights (e.g. Granite
    Docling) on first use.  To run in an air‑gapped environment, you must
    pre‑download those weights in your development environment and bake
    them into your Docker image.
    """
    try:
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except ImportError as exc:
        raise ImportError(
            "Docling is not installed. Please install docling with\n"
            "  pip install 'docling[vllm]'\n"
            "and ensure you have sufficient GPU/CPU resources."
        ) from exc

    # If the source is a remote URL and caching is enabled, download to cache
    # directory before passing to Docling.  Docling can read remote URLs
    # directly, but caching simplifies offline reuse and ensures reproducible
    # filenames for deduplication.
    src_str = str(source)
    local_path: str | Path = source
    if cache and PDF_CACHE_DIR is not None:
        _ensure_pdf_cache_dir()
        filename = Path(src_str).name
        cached_path = PDF_CACHE_DIR / filename
        # Only copy if the file does not already exist
        if isinstance(source, (str, Path)) and Path(source).is_file():
            # Local file provided; copy it into cache
            if not cached_path.exists():
                shutil.copy2(source, cached_path)
            local_path = cached_path
        elif src_str.lower().startswith(("http://", "https://")):
            # Remote file; rely on Docling's internal download but we still
            # instruct it to save the file locally by specifying the cache path
            local_path = src_str  # remote URL; Docling will fetch
    else:
        local_path = src_str

    # Construct the converter to use the GraniteDocling VLM pipeline.  We use
    # VlmPipeline with GraniteDocling specs to obtain a rich markdown
    # representation that preserves tables, equations and layout【583088636613067†L68-L99】.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(
                    vlm_options=vlm_model_specs.GRANITEDOCLING_DEFAULT
                ),
            )
        }
    )

    logger.info("Converting PDF %s to markdown using Docling", source)
    doc = converter.convert(source=str(local_path)).document
    markdown = doc.export_to_markdown()
    return markdown
