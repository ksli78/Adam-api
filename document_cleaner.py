"""
Document Cleaner Service

Removes noise from extracted documents including:
- CUI/security banners
- Page numbers and headers/footers
- Signature blocks
- Excessive whitespace
- OCR artifacts
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Cleans extracted document text by removing noise patterns.

    Uses configuration file (cleaning_config.yaml) to define patterns
    for removal, making it easy to customize without code changes.
    """

    def __init__(self, config_path: str = "cleaning_config.yaml"):
        """
        Initialize the document cleaner with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.compiled_patterns = self._compile_patterns()
        logger.info(f"DocumentCleaner initialized with {len(self.compiled_patterns)} patterns")

    def _load_config(self) -> Dict[str, Any]:
        """Load cleaning configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded cleaning config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "patterns_to_remove": [
                {"pattern": r"CUI//.*?//CUI", "description": "CUI banners"},
                {"pattern": r"^Page\s+\d+\s+of\s+\d+\s*$", "description": "Page numbers"},
                {"pattern": r"COMPANY\s+CONFIDENTIAL", "description": "Confidential marks"},
            ],
            "skip_lines_containing": ["© Copyright", "All Rights Reserved"],
            "min_section_length": 100,
            "min_line_length": 15,
            "remove_punctuation_heavy_lines": True,
            "punctuation_threshold": 0.5,
            "fix_common_artifacts": {
                "enabled": True,
                "replacements": {"�": "", "\x00": ""}
            }
        }

    def _compile_patterns(self) -> List[Dict[str, Any]]:
        """Compile regex patterns from configuration."""
        compiled = []

        for pattern_config in self.config.get("patterns_to_remove", []):
            pattern_str = pattern_config.get("pattern")
            flags_list = pattern_config.get("flags", [])

            # Build regex flags
            flags = re.IGNORECASE | re.MULTILINE
            if "DOTALL" in flags_list:
                flags |= re.DOTALL

            try:
                compiled_pattern = re.compile(pattern_str, flags)
                compiled.append({
                    "pattern": compiled_pattern,
                    "replace_with": pattern_config.get("replace_with", ""),
                    "description": pattern_config.get("description", "")
                })
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern_str}': {e}")

        return compiled

    def clean(self, text: str) -> str:
        """
        Clean document text by removing noise patterns.

        Args:
            text: Raw extracted text from document

        Returns:
            Cleaned text with noise removed
        """
        if not text or not text.strip():
            return ""

        original_length = len(text)

        # Step 1: Fix common artifacts
        text = self._fix_artifacts(text)

        # Step 2: Apply regex pattern removal
        text = self._remove_patterns(text)

        # Step 3: Clean lines
        text = self._clean_lines(text)

        # Step 4: Remove excessive whitespace
        text = self._normalize_whitespace(text)

        cleaned_length = len(text)
        removed_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0

        logger.debug(f"Cleaned text: {original_length} -> {cleaned_length} chars ({removed_pct:.1f}% removed)")

        return text.strip()

    def _fix_artifacts(self, text: str) -> str:
        """Fix common OCR and extraction artifacts."""
        artifacts_config = self.config.get("fix_common_artifacts", {})

        if not artifacts_config.get("enabled", True):
            return text

        replacements = artifacts_config.get("replacements", {})
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _remove_patterns(self, text: str) -> str:
        """Remove text matching configured patterns."""
        for pattern_info in self.compiled_patterns:
            pattern = pattern_info["pattern"]
            replace_with = pattern_info["replace_with"]

            text = pattern.sub(replace_with, text)

        return text

    def _clean_lines(self, text: str) -> str:
        """Clean individual lines based on rules."""
        lines = text.split('\n')
        cleaned_lines = []

        min_line_length = self.config.get("min_line_length", 15)
        skip_phrases = self.config.get("skip_lines_containing", [])
        remove_punct_heavy = self.config.get("remove_punctuation_heavy_lines", True)
        punct_threshold = self.config.get("punctuation_threshold", 0.5)

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (will be normalized later)
            if not stripped:
                cleaned_lines.append("")
                continue

            # Skip lines containing certain phrases
            if any(phrase.lower() in stripped.lower() for phrase in skip_phrases):
                continue

            # Skip very short lines (likely headers/footers)
            if len(stripped) < min_line_length:
                continue

            # Skip punctuation-heavy lines
            if remove_punct_heavy and self._is_punctuation_heavy(stripped, punct_threshold):
                continue

            cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines)

    def _is_punctuation_heavy(self, line: str, threshold: float) -> bool:
        """Check if line is mostly punctuation (likely noise)."""
        if not line:
            return False

        punct_chars = sum(1 for c in line if not c.isalnum() and not c.isspace())
        total_chars = len(line)

        punct_ratio = punct_chars / total_chars if total_chars > 0 else 0
        return punct_ratio > threshold

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize excessive whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Replace more than 3 newlines with exactly 2 (paragraph break)
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Remove spaces at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # Collapse multiple empty lines to at most 2
        text = re.sub(r'\n\n\n+', '\n\n', text)

        return text

    def clean_section(self, section_text: str, section_title: str = "") -> Dict[str, str]:
        """
        Clean a document section and return structured result.

        Args:
            section_text: Text content of the section
            section_title: Optional section title/heading

        Returns:
            Dict with cleaned_text and metadata
        """
        cleaned = self.clean(section_text)

        # Check if section meets minimum length requirement
        min_length = self.config.get("min_section_length", 100)
        is_valid = len(cleaned) >= min_length

        return {
            "title": section_title,
            "cleaned_text": cleaned,
            "original_length": len(section_text),
            "cleaned_length": len(cleaned),
            "is_valid": is_valid,
            "removed_chars": len(section_text) - len(cleaned)
        }

    def get_section_markers(self) -> List[str]:
        """Get regex patterns for detecting section boundaries."""
        return self.config.get("section_markers", [
            r"^\d+\.\s+",  # 1. Section
            r"^\d+\.\d+\.?\s+",  # 1.1 Subsection
            r"^[A-Z][a-z]+:$",  # Purpose:
            r"^[A-Z][A-Z\s]+$"  # ALL CAPS HEADERS
        ])


# Singleton instance
_cleaner_instance = None


def get_document_cleaner(config_path: str = "cleaning_config.yaml") -> DocumentCleaner:
    """
    Get or create singleton DocumentCleaner instance.

    Args:
        config_path: Path to cleaning configuration file

    Returns:
        DocumentCleaner instance
    """
    global _cleaner_instance

    if _cleaner_instance is None:
        _cleaner_instance = DocumentCleaner(config_path)

    return _cleaner_instance


if __name__ == "__main__":
    # Test the cleaner
    logging.basicConfig(level=logging.DEBUG)

    test_text = """
    CUI//SP-PRVCY//CUI

    COMPANY CONFIDENTIAL

    Time Off Policy

    Page 1 of 10

    1. Purpose

    This policy outlines the time off benefits available to all employees.
    Employees are entitled to PTO as described in section 4.3.

    Printed on: 12/25/2024

    © Copyright 2024 Company Name. All Rights Reserved.

    Signature: ___________________________

    4.3 PTO Accrual

    Full-time employees accrue 15 days of PTO per year.
    Part-time employees accrue PTO on a pro-rated basis.

    Page 2 of 10
    """

    cleaner = get_document_cleaner()
    cleaned = cleaner.clean(test_text)

    print("=== Original ===")
    print(test_text)
    print("\n=== Cleaned ===")
    print(cleaned)
