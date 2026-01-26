#!/usr/bin/env python3
"""
Hebrew Word Bounding Box Detector for PDFs

This script loads a PDF file, detects Hebrew words using bounding boxes,
and exports a new PDF with boxes drawn around all Hebrew text.
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def is_hebrew_char(char: str) -> bool:
    """Check if a character is Hebrew (Unicode range 0x0590-0x05FF)."""
    if not char:
        return False
    code = ord(char)
    # Hebrew Unicode block: U+0590 to U+05FF
    return 0x0590 <= code <= 0x05FF


def contains_hebrew(text: str) -> bool:
    """Check if text contains any Hebrew characters."""
    return any(is_hebrew_char(c) for c in text)


def extract_hebrew_words_with_boxes(page: fitz.Page) -> list[dict]:
    """
    Extract all words containing Hebrew characters with their bounding boxes.

    Args:
        page: A PyMuPDF page object

    Returns:
        List of dicts with 'text' and 'bbox' keys
    """
    hebrew_words = []

    # Get all words with their bounding boxes
    # words is a list of tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    words = page.get_text("words")

    for word_info in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = word_info

        # Check if the word contains Hebrew characters
        if contains_hebrew(text):
            hebrew_words.append({
                'text': text,
                'bbox': fitz.Rect(x0, y0, x1, y1),
                'block': block_no,
                'line': line_no,
                'word': word_no
            })

    return hebrew_words


def draw_bounding_boxes(page: fitz.Page, hebrew_words: list[dict],
                        color: tuple = (1, 0, 0), width: float = 1.0) -> int:
    """
    Draw bounding boxes around Hebrew words on the page.

    Args:
        page: A PyMuPDF page object
        hebrew_words: List of word dicts with 'bbox' key
        color: RGB color tuple (default: red)
        width: Line width for the box

    Returns:
        Number of boxes drawn
    """
    for word in hebrew_words:
        bbox = word['bbox']
        # Draw rectangle around the word
        page.draw_rect(bbox, color=color, width=width)

    return len(hebrew_words)


def process_pdf(input_path: str, output_suffix: str = "_boxes") -> str:
    """
    Process a PDF file, detect Hebrew words, and save with bounding boxes.

    Args:
        input_path: Path to the input PDF file
        output_suffix: Suffix to add to output filename (default: "_boxes")

    Returns:
        Path to the output PDF file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"PDF file not found: {input_path}")

    if not input_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {input_path}")

    # Create output path
    output_path = input_path.parent / f"{input_path.stem}{output_suffix}.pdf"

    # Open the PDF
    doc = fitz.open(input_path)

    total_hebrew_words = 0

    print(f"Processing: {input_path}")
    print(f"Total pages: {len(doc)}")
    print("-" * 50)

    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract Hebrew words with bounding boxes
        hebrew_words = extract_hebrew_words_with_boxes(page)

        # Draw bounding boxes
        boxes_drawn = draw_bounding_boxes(page, hebrew_words)
        total_hebrew_words += boxes_drawn

        print(f"Page {page_num + 1}: Found {boxes_drawn} Hebrew words")

    print("-" * 50)
    print(f"Total Hebrew words detected: {total_hebrew_words}")

    # Save the modified PDF
    doc.save(output_path)
    doc.close()

    print(f"Output saved to: {output_path}")

    return str(output_path)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python hebrew_box_detector.py <pdf_file>")
        print("Example: python hebrew_box_detector.py document.pdf")
        sys.exit(1)

    input_pdf = sys.argv[1]

    try:
        output_path = process_pdf(input_pdf)
        print(f"\nSuccess! Output file: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
