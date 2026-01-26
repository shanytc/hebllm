#!/usr/bin/env python3
"""
HebMark - Custom Visual Marker System for Hebrew Text Replacement

Designed for LLM visual decoding. Each marker:
1. Has a distinctive magenta border (easily recognizable)
2. Contains a diamond prefix (◆) + compact alphanumeric ID
3. Maps to Hebrew text + LLM tokens via a JSON mapping

Marker format:  ┏━━━━━━━━━┓
                ┃ ◆A7     ┃
                ┗━━━━━━━━━┛
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable
from abc import ABC, abstractmethod

import fitz  # PyMuPDF


# =============================================================================
# Tokenizer Support
# =============================================================================

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Tokenizer name/identifier."""
        pass


class TiktokenWrapper(BaseTokenizer):
    """Wrapper for OpenAI's tiktoken library."""

    def __init__(self, model: str = "gpt-4"):
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
            self._model = model
        except ImportError:
            raise ImportError("tiktoken not installed. Run: pip install tiktoken")

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    @property
    def name(self) -> str:
        return f"tiktoken:{self._model}"


class SimpleUtf8Tokenizer(BaseTokenizer):
    """
    Simple UTF-8 byte tokenizer - no external dependencies.
    Each token is a byte value (0-255).
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")

    @property
    def name(self) -> str:
        return "utf8-bytes"


class CharacterTokenizer(BaseTokenizer):
    """
    Character-level tokenizer using Unicode code points.
    Simple and reversible for any text.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)

    @property
    def name(self) -> str:
        return "unicode-codepoints"


def get_tokenizer(name: str = "utf8") -> BaseTokenizer:
    """
    Get a tokenizer by name.

    Options:
        - "utf8": Simple UTF-8 byte tokenizer (default, no deps)
        - "char": Unicode codepoint tokenizer
        - "gpt-4": OpenAI GPT-4 tokenizer (requires tiktoken)
        - "gpt-3.5-turbo": OpenAI GPT-3.5 tokenizer (requires tiktoken)
        - "cl100k_base": Claude/GPT-4 base encoding (requires tiktoken)
    """
    if name == "utf8":
        return SimpleUtf8Tokenizer()
    elif name == "char":
        return CharacterTokenizer()
    elif name in ("gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"):
        return TiktokenWrapper(name)
    else:
        # Try as tiktoken model name
        try:
            return TiktokenWrapper(name)
        except Exception:
            raise ValueError(f"Unknown tokenizer: {name}")

# Marker visual constants
MARKER_PREFIX = "\u25c6"  # ◆ Diamond symbol - distinctive and rare
MARKER_BORDER_COLOR = (0.8, 0.2, 0.8)  # Magenta - stands out
MARKER_FILL_COLOR = (1.0, 0.95, 1.0)  # Light magenta background
MARKER_TEXT_COLOR = (0.4, 0.0, 0.4)  # Dark magenta text
MARKER_BORDER_WIDTH = 1.5


@dataclass
class HebMarkEntry:
    """A single marker entry mapping ID to Hebrew text."""
    marker_id: str
    hebrew_text: str
    tokens: list[int] = field(default_factory=list)
    page: int = 0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    block: int = 0
    line: int = 0
    word: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HebMarkEntry":
        return cls(**data)


class HebMarkEncoder:
    """
    Encodes Hebrew text into unique, compact marker IDs.

    Uses base36 encoding for compact representation:
    - 2 chars = 1,296 unique IDs (0-ZZ)
    - 3 chars = 46,656 unique IDs (0-ZZZ)
    """

    BASE36_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        self._counter = 0
        self._text_to_id: dict[str, str] = {}
        self._id_to_entry: dict[str, HebMarkEntry] = {}
        self._tokenizer = tokenizer or SimpleUtf8Tokenizer()

    def _to_base36(self, num: int, min_length: int = 2) -> str:
        """Convert integer to base36 string."""
        if num == 0:
            return "0" * min_length

        result = []
        while num:
            result.append(self.BASE36_CHARS[num % 36])
            num //= 36

        result_str = "".join(reversed(result))
        return result_str.zfill(min_length)

    def _from_base36(self, s: str) -> int:
        """Convert base36 string to integer."""
        result = 0
        for char in s.upper():
            result = result * 36 + self.BASE36_CHARS.index(char)
        return result

    def encode(self, hebrew_text: str, page: int = 0,
               bbox: tuple = (0, 0, 0, 0), block: int = 0,
               line: int = 0, word: int = 0,
               tokens: Optional[list[int]] = None) -> str:
        """
        Encode Hebrew text into a marker ID.

        Same text on same page gets same ID (deduplication).
        Returns the marker ID (without prefix).
        """
        # Create unique key based on text + position
        unique_key = f"{page}:{block}:{line}:{word}:{hebrew_text}"

        if unique_key in self._text_to_id:
            return self._text_to_id[unique_key]

        # Generate new ID
        marker_id = self._to_base36(self._counter)
        self._counter += 1

        # Tokenize if not provided
        if tokens is None:
            tokens = self._tokenizer.encode(hebrew_text)

        # Store mapping
        self._text_to_id[unique_key] = marker_id
        self._id_to_entry[marker_id] = HebMarkEntry(
            marker_id=marker_id,
            hebrew_text=hebrew_text,
            tokens=tokens,
            page=page,
            bbox=bbox,
            block=block,
            line=line,
            word=word
        )

        return marker_id

    def decode(self, marker_id: str) -> Optional[HebMarkEntry]:
        """Decode a marker ID back to Hebrew text entry."""
        # Strip prefix if present
        if marker_id.startswith(MARKER_PREFIX):
            marker_id = marker_id[1:]
        return self._id_to_entry.get(marker_id.upper())

    def get_marker_text(self, marker_id: str) -> str:
        """Get the full marker text with prefix."""
        return f"{MARKER_PREFIX}{marker_id}"

    def get_mapping(self) -> dict[str, dict]:
        """Get the full ID to entry mapping as dict."""
        return {mid: entry.to_dict() for mid, entry in self._id_to_entry.items()}

    def save_mapping(self, path: Path | str) -> None:
        """Save the mapping to a JSON file."""
        path = Path(path)
        mapping = {
            "version": "1.0",
            "prefix": MARKER_PREFIX,
            "tokenizer": self._tokenizer.name,
            "count": len(self._id_to_entry),
            "markers": self.get_mapping()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    def load_mapping(self, path: Path | str) -> None:
        """Load mapping from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._id_to_entry.clear()
        self._text_to_id.clear()

        for marker_id, entry_data in data.get("markers", {}).items():
            entry = HebMarkEntry.from_dict(entry_data)
            self._id_to_entry[marker_id] = entry
            unique_key = f"{entry.page}:{entry.block}:{entry.line}:{entry.word}:{entry.hebrew_text}"
            self._text_to_id[unique_key] = marker_id

        # Update counter to continue from max ID
        if self._id_to_entry:
            max_id = max(self._from_base36(mid) for mid in self._id_to_entry.keys())
            self._counter = max_id + 1


class HebMarkRenderer:
    """Renders HebMark markers onto PDF pages."""

    def __init__(self, encoder: HebMarkEncoder):
        self.encoder = encoder

    def draw_marker(self, page: fitz.Page, bbox: fitz.Rect,
                    marker_id: str, font_size: Optional[float] = None) -> None:
        """
        Draw a marker in place of Hebrew text.

        Args:
            page: PyMuPDF page object
            bbox: Original bounding box of Hebrew text
            marker_id: The marker ID to display
            font_size: Optional font size (auto-calculated if None)
        """
        # First, redact/cover the original Hebrew text with white
        # Use a white filled rectangle to cover the original text
        page.draw_rect(bbox, color=None, fill=(1, 1, 1))

        # Calculate font size based on bbox height if not provided
        box_height = bbox.height
        box_width = bbox.width

        if font_size is None:
            # Font size should fit in the box height with some margin
            font_size = max(6, min(box_height * 0.7, 14))

        # Marker text
        marker_text = self.encoder.get_marker_text(marker_id)

        # Calculate text width to determine if we need to adjust
        # Approximate: each character is ~0.6 * font_size wide
        approx_text_width = len(marker_text) * font_size * 0.6

        # If text would be wider than bbox, expand the marker box
        marker_bbox = fitz.Rect(bbox)
        if approx_text_width > box_width:
            # Expand horizontally, keeping left edge
            marker_bbox.x1 = marker_bbox.x0 + approx_text_width + 4

        # Draw marker background
        page.draw_rect(
            marker_bbox,
            color=MARKER_BORDER_COLOR,
            fill=MARKER_FILL_COLOR,
            width=MARKER_BORDER_WIDTH
        )

        # Draw marker text
        # Position text in center of marker box
        text_point = fitz.Point(
            marker_bbox.x0 + 2,
            marker_bbox.y0 + (marker_bbox.height + font_size) / 2 - 2
        )

        page.insert_text(
            text_point,
            marker_text,
            fontsize=font_size,
            color=MARKER_TEXT_COLOR,
            fontname="helv"  # Helvetica - widely available
        )

    def replace_hebrew_with_markers(self, page: fitz.Page,
                                     hebrew_words: list[dict],
                                     page_num: int = 0) -> list[str]:
        """
        Replace all Hebrew words on a page with markers.

        Args:
            page: PyMuPDF page object
            hebrew_words: List of dicts with 'text', 'bbox', etc.
            page_num: Page number for encoding

        Returns:
            List of marker IDs created
        """
        marker_ids = []

        for word in hebrew_words:
            # Encode the Hebrew text
            marker_id = self.encoder.encode(
                hebrew_text=word['text'],
                page=page_num,
                bbox=(word['bbox'].x0, word['bbox'].y0,
                      word['bbox'].x1, word['bbox'].y1),
                block=word.get('block', 0),
                line=word.get('line', 0),
                word=word.get('word', 0)
            )

            # Draw the marker
            self.draw_marker(page, word['bbox'], marker_id)
            marker_ids.append(marker_id)

        return marker_ids


def generate_llm_prompt(mapping_path: Path | str, compact: bool = False) -> str:
    """
    Generate a prompt explaining the marker system to an LLM.

    Args:
        mapping_path: Path to the mapping JSON file
        compact: If True, generate a more compact mapping format

    This prompt should be included when sending the marked PDF to an LLM
    for processing.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if compact:
        # Compact format: JSON-like for efficiency
        prompt = f"""HEBMARK DECODER: Markers (◆XX) map to Hebrew text.
MAP: {{"""
        pairs = [f'"{mid}":"{e["hebrew_text"]}"'
                 for mid, e in mapping.get("markers", {}).items()]
        prompt += ",".join(pairs) + "}"
        return prompt

    # Verbose format
    prompt = f"""This document contains HebMark markers that replace Hebrew text.

MARKER FORMAT:
- Each marker appears as: {MARKER_PREFIX}XX (diamond symbol + ID)
- Markers have a distinctive magenta border
- The ID maps to original Hebrew text

MARKER MAPPING:
"""

    for marker_id, entry in mapping.get("markers", {}).items():
        prompt += f"  {MARKER_PREFIX}{marker_id} = \"{entry['hebrew_text']}\"\n"

    prompt += """
When you encounter a marker like ◆A7, replace it with the corresponding Hebrew text from the mapping above.
"""

    return prompt


def decode_llm_output(text: str, mapping_path: Path | str) -> str:
    """
    Decode HebMark markers in LLM output back to Hebrew text.

    Args:
        text: Text containing markers like ◆A7
        mapping_path: Path to the mapping JSON file

    Returns:
        Text with markers replaced by original Hebrew
    """
    import re

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    markers = mapping.get("markers", {})

    # Pattern to match markers: ◆ followed by alphanumeric ID
    pattern = rf"{re.escape(MARKER_PREFIX)}([0-9A-Za-z]+)"

    def replace_marker(match):
        marker_id = match.group(1).upper()
        if marker_id in markers:
            return markers[marker_id]["hebrew_text"]
        return match.group(0)  # Keep original if not found

    return re.sub(pattern, replace_marker, text)


def decode_from_tokens(tokens: list[int], tokenizer_name: str = "char") -> str:
    """
    Decode tokens back to Hebrew text.

    Args:
        tokens: List of token IDs
        tokenizer_name: Name of tokenizer used for encoding

    Returns:
        Decoded Hebrew text
    """
    tokenizer = get_tokenizer(tokenizer_name)
    return tokenizer.decode(tokens)


class HebMarkDecoder:
    """
    Utility class for decoding markers in LLM responses.

    Usage:
        decoder = HebMarkDecoder("heb_mapping.json")
        hebrew_text = decoder.decode("The word ◆00 means...")
    """

    def __init__(self, mapping_path: Path | str):
        self.mapping_path = Path(mapping_path)
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.markers = data.get("markers", {})
        self.tokenizer_name = data.get("tokenizer", "utf8")
        self.prefix = data.get("prefix", MARKER_PREFIX)

    def decode_marker(self, marker_id: str) -> Optional[str]:
        """Decode a single marker ID to Hebrew text."""
        marker_id = marker_id.lstrip(self.prefix).upper()
        entry = self.markers.get(marker_id)
        return entry["hebrew_text"] if entry else None

    def decode_text(self, text: str) -> str:
        """Decode all markers in text to Hebrew."""
        import re
        pattern = rf"{re.escape(self.prefix)}([0-9A-Za-z]+)"

        def replace(match):
            result = self.decode_marker(match.group(1))
            return result if result else match.group(0)

        return re.sub(pattern, replace, text)

    def get_entry(self, marker_id: str) -> Optional[dict]:
        """Get full entry for a marker ID including tokens."""
        marker_id = marker_id.lstrip(self.prefix).upper()
        return self.markers.get(marker_id)

    def decode_tokens(self, marker_id: str) -> Optional[str]:
        """Decode marker using its stored tokens."""
        entry = self.get_entry(marker_id)
        if entry and entry.get("tokens"):
            return decode_from_tokens(entry["tokens"], self.tokenizer_name)
        return None


# Convenience function for quick usage
def create_marked_pdf(input_path: str, output_suffix: str = "_marked",
                      tokenizer_name: str = "utf8") -> tuple[str, str]:
    """
    Process a PDF, replacing Hebrew text with markers.

    Args:
        input_path: Path to input PDF
        output_suffix: Suffix for output PDF filename
        tokenizer_name: Tokenizer to use ("utf8", "char", "gpt-4", etc.)

    Returns:
        Tuple of (output_pdf_path, mapping_json_path)
    """
    from hebrew_box_detector import extract_hebrew_words_with_boxes

    input_path = Path(input_path)
    output_pdf_path = input_path.parent / f"{input_path.stem}{output_suffix}.pdf"
    mapping_path = input_path.parent / f"{input_path.stem}_mapping.json"

    # Initialize encoder and renderer with tokenizer
    tokenizer = get_tokenizer(tokenizer_name)
    encoder = HebMarkEncoder(tokenizer=tokenizer)
    renderer = HebMarkRenderer(encoder)

    print(f"Using tokenizer: {tokenizer.name}")

    # Open PDF
    doc = fitz.open(input_path)

    print(f"Processing: {input_path}")
    print(f"Total pages: {len(doc)}")
    print("-" * 50)

    total_markers = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract Hebrew words
        hebrew_words = extract_hebrew_words_with_boxes(page)

        # Replace with markers
        marker_ids = renderer.replace_hebrew_with_markers(
            page, hebrew_words, page_num
        )

        total_markers += len(marker_ids)
        print(f"Page {page_num + 1}: Created {len(marker_ids)} markers")

    print("-" * 50)
    print(f"Total markers created: {total_markers}")

    # Save outputs
    doc.save(output_pdf_path)
    doc.close()

    encoder.save_mapping(mapping_path)

    print(f"Output PDF: {output_pdf_path}")
    print(f"Mapping file: {mapping_path}")

    return str(output_pdf_path), str(mapping_path)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="HebMark - Replace Hebrew text with LLM-readable markers"
    )
    parser.add_argument("pdf_file", help="Input PDF file")
    parser.add_argument(
        "--tokenizer", "-t",
        default="utf8",
        help="Tokenizer to use: utf8 (default), char, gpt-4, gpt-3.5-turbo"
    )
    parser.add_argument(
        "--suffix", "-s",
        default="_marked",
        help="Output file suffix (default: _marked)"
    )

    args = parser.parse_args()

    pdf_path, mapping_path = create_marked_pdf(
        args.pdf_file,
        output_suffix=args.suffix,
        tokenizer_name=args.tokenizer
    )

    print("\n" + "=" * 50)
    print("LLM PROMPT:")
    print("=" * 50)
    print(generate_llm_prompt(mapping_path))
