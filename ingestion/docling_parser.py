# ingestion/docling_parser.py
from __future__ import annotations

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Windows-safe HF cache defaults (symlink-free)
os.environ.setdefault("HF_HOME", str(Path(".hf").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

def convert_pdf_to_markdown(source: str | Path) -> str:
    """
    Convert a PDF to Markdown using Docling.
    Defaults to the standard pipeline (fast for native PDFs).
    Use VLM only if DOCLING_USE_VLM=1 is set in env.
    """
    from docling.document_converter import DocumentConverter

    use_vlm = os.getenv("DOCLING_USE_VLM", "0") == "1"

    if use_vlm:
        try:
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.document_converter import PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline

            # Try a few well-known specs (version-safe)
            candidates = [
                "GRANITEDOCLING_TRANSFORMERS",
                "SMOLDOCLING_TRANSFORMERS",
                "SMOLDOCLING_ONNX",
            ]
            selected = None
            for name in candidates:
                if hasattr(vlm_model_specs, name):
                    selected = getattr(vlm_model_specs, name)
                    break

            if selected is not None:
                log.info("Using Docling VLM spec: %s", name)
                pipeline_options = VlmPipelineOptions(vlm_options=selected)
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=pipeline_options,
                        )
                    }
                )
            else:
                log.warning("No known VLM spec found; falling back to standard pipeline.")
                converter = DocumentConverter()
        except Exception as e:
            log.warning("VLM pipeline not available (%s). Using standard pipeline.", e)
            converter = DocumentConverter()
    else:
        converter = DocumentConverter()

    res = converter.convert(source=source)
    md = res.document.export_to_markdown()
    return md
