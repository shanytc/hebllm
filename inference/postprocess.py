#!/usr/bin/env python3
"""
Post-processing utilities for HebLLM output.

Handles text normalization, Hebrew-specific processing,
and output formatting for visual reading order.
"""

import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    return "".join(
        char for char in text
        if unicodedata.category(char) != "Cc" or char in "\n\t"
    )


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, trim lines."""
    lines = text.split("\n")
    normalized = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized.append(line)
    return "\n".join(normalized)


def fix_hebrew_punctuation(text: str) -> str:
    """Fix common Hebrew punctuation issues."""
    # Move punctuation to correct position for RTL text
    # Hebrew text with misplaced periods
    text = re.sub(r'(\.\s*)([א-ת])', r'\2\1', text)
    return text


def detect_language_segments(text: str) -> list[dict]:
    """
    Detect language segments in mixed text.

    Returns:
        List of dicts with 'text', 'language', 'start', 'end'
    """
    segments = []
    current_lang = None
    current_start = 0
    current_text = []

    for i, char in enumerate(text):
        if '\u0590' <= char <= '\u05FF':  # Hebrew
            lang = "hebrew"
        elif char.isalpha():
            lang = "english"
        else:
            lang = current_lang or "neutral"

        if lang != current_lang and current_lang is not None:
            if current_text:
                segments.append({
                    "text": "".join(current_text),
                    "language": current_lang,
                    "start": current_start,
                    "end": i
                })
            current_text = []
            current_start = i

        current_text.append(char)
        if lang != "neutral":
            current_lang = lang

    # Final segment
    if current_text:
        segments.append({
            "text": "".join(current_text),
            "language": current_lang or "neutral",
            "start": current_start,
            "end": len(text)
        })

    return segments


def to_visual_order(text: str) -> str:
    """
    Convert text to visual reading order.

    For mixed Hebrew/English documents, this ensures text
    appears as it would be read visually on the page.
    """
    lines = text.split("\n")
    visual_lines = []

    for line in lines:
        segments = detect_language_segments(line)

        # Check if line is predominantly Hebrew
        hebrew_chars = sum(1 for c in line if '\u0590' <= c <= '\u05FF')
        total_chars = sum(1 for c in line if c.isalpha())

        if total_chars > 0 and hebrew_chars / total_chars > 0.5:
            # Hebrew-dominant: reverse for visual order
            visual_lines.append(line[::-1])
        else:
            visual_lines.append(line)

    return "\n".join(visual_lines)


def clean_ocr_output(text: str) -> str:
    """
    Clean and normalize OCR output.

    Applies standard cleaning operations.
    """
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    return text


def format_for_display(text: str,
                       visual_order: bool = False,
                       max_line_length: int = 80) -> str:
    """
    Format text for terminal/display output.

    Args:
        text: Input text
        visual_order: Convert to visual reading order
        max_line_length: Wrap lines at this length

    Returns:
        Formatted text
    """
    text = clean_ocr_output(text)

    if visual_order:
        text = to_visual_order(text)

    # Word wrap
    if max_line_length:
        wrapped_lines = []
        for line in text.split("\n"):
            if len(line) <= max_line_length:
                wrapped_lines.append(line)
            else:
                words = line.split()
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 <= max_line_length:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_line:
                            wrapped_lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    wrapped_lines.append(" ".join(current_line))

        text = "\n".join(wrapped_lines)

    return text


class OCRPostProcessor:
    """Configurable post-processor for OCR output."""

    def __init__(self,
                 normalize: bool = True,
                 fix_punctuation: bool = True,
                 visual_order: bool = False):
        """
        Initialize post-processor.

        Args:
            normalize: Apply Unicode normalization
            fix_punctuation: Fix Hebrew punctuation
            visual_order: Convert to visual order
        """
        self.normalize = normalize
        self.fix_punctuation = fix_punctuation
        self.visual_order = visual_order

    def __call__(self, text: str) -> str:
        """Process OCR output."""
        if self.normalize:
            text = clean_ocr_output(text)

        if self.fix_punctuation:
            text = fix_hebrew_punctuation(text)

        if self.visual_order:
            text = to_visual_order(text)

        return text


if __name__ == "__main__":
    # Test post-processing
    test_text = """שלום עולם
Hello World
מעורב mixed טקסט"""

    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(clean_ocr_output(test_text))
    print("\nSegments:")
    for seg in detect_language_segments(test_text):
        print(f"  {seg['language']}: {seg['text'][:30]}...")
