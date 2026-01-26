#!/usr/bin/env python3
"""
Synthetic PDF Data Generator for HebLLM Training

Generates synthetic PDF pages with Hebrew, English, and mixed content,
then applies HebMark markers and saves training data pairs.
"""

import json
import random
import string
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Literal

import fitz  # PyMuPDF
from PIL import Image
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# RTL text support
try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False
    print("Warning: python-bidi not installed. Hebrew text may appear reversed.")
    print("Install with: pip install python-bidi")

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from hebmark import HebMarkEncoder, HebMarkRenderer, get_tokenizer
from hebrew_box_detector import extract_hebrew_words_with_boxes


# =============================================================================
# Text Sources
# =============================================================================

# Sample Hebrew text corpus (expandable)
HEBREW_SAMPLES = [
    "שלום עולם",
    "בראשית ברא אלוהים את השמים ואת הארץ",
    "תודה רבה על העזרה",
    "אני לומד עברית כל יום",
    "הספר הזה מעניין מאוד",
    "מה שלומך היום",
    "אנחנו גרים בתל אביב",
    "העבודה הזאת חשובה מאוד",
    "יש לי שאלה בבקשה",
    "המחשב שלי עובד טוב",
    "הילדים משחקים בחצר",
    "האוכל היה טעים מאוד",
    "השמש זורחת בחוץ",
    "הגשם יורד כל היום",
    "אני אוהב לקרוא ספרים",
    "היא עובדת בבית חולים",
    "הוא מנגן בגיטרה יפה",
    "אנחנו נוסעים לים מחר",
    "התלמידים לומדים בכיתה",
    "המורה מסבירה את השיעור",
]

# Sample English text corpus
ENGLISH_SAMPLES = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming technology",
    "Python is a versatile programming language",
    "Data science combines statistics and computing",
    "Natural language processing enables text analysis",
    "Computer vision helps machines understand images",
    "Deep learning uses neural network architectures",
    "Artificial intelligence is advancing rapidly",
    "Software engineering requires careful planning",
    "The documentation should be clear and concise",
    "Testing ensures code quality and reliability",
    "Version control tracks changes over time",
    "APIs enable communication between services",
    "Databases store and retrieve information",
    "Cloud computing provides scalable resources",
    "Security is essential for all applications",
    "User experience drives product success",
    "Agile methodology promotes iterative development",
    "Open source fosters collaboration and innovation",
]

# Hebrew paragraph templates
HEBREW_PARAGRAPHS = [
    """בימינו, הטכנולוגיה משנה את חיינו בצורה דרמטית. מחשבים, טלפונים חכמים ואינטרנט הפכו לחלק בלתי נפרד מהשגרה היומית שלנו. אנשים מתקשרים זה עם זה בקלות רבה יותר, ומידע זמין בלחיצת כפתור.""",
    """החינוך הוא הבסיס לחברה מתקדמת. בתי ספר ואוניברסיטאות מכשירים את הדור הבא למשימות העתיד. למידה מתמשכת היא המפתח להצלחה בעולם המשתנה.""",
    """הסביבה היא הנכס החשוב ביותר שלנו. עלינו לשמור על הטבע ולהגן על משאבי הטבע לדורות הבאים. שימוש באנרגיה מתחדשת יכול לעזור להפחית את הזיהום.""",
    """המדע מתקדם בקצב מסחרר. גילויים חדשים בתחומי הרפואה, הפיזיקה והביולוגיה משנים את הבנתנו את העולם. מדענים ברחבי העולם עובדים יחד על פתרונות לבעיות האנושות.""",
]

ENGLISH_PARAGRAPHS = [
    """Technology is revolutionizing how we live and work. From smartphones to artificial intelligence, digital innovations are transforming every aspect of our daily lives. The pace of change continues to accelerate.""",
    """Education is the foundation of progress. Schools and universities prepare students for the challenges of tomorrow. Lifelong learning has become essential in our rapidly evolving world.""",
    """Environmental protection is crucial for our future. Climate change poses significant challenges that require global cooperation. Sustainable practices can help preserve our planet for generations to come.""",
    """Scientific research continues to push boundaries. New discoveries in medicine, physics, and biology are expanding our understanding of the universe. International collaboration drives innovation forward.""",
]


def to_visual_hebrew(text: str) -> str:
    """Convert logical Hebrew text to visual order for rendering.

    Reportlab renders text left-to-right, so Hebrew needs to be
    converted to visual order (reversed) for correct display.
    """
    if HAS_BIDI:
        return get_display(text)
    else:
        # Fallback: simple reversal for pure Hebrew text
        # This doesn't handle mixed content well
        return text[::-1]


def is_hebrew_text(text: str) -> bool:
    """Check if text is predominantly Hebrew."""
    hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    alpha_chars = sum(1 for c in text if c.isalpha())
    return alpha_chars > 0 and hebrew_chars / alpha_chars > 0.5


@dataclass
class PDFStyle:
    """Configuration for PDF visual appearance."""
    page_size: tuple = A4
    margin_left: float = 20 * mm
    margin_right: float = 20 * mm
    margin_top: float = 25 * mm
    margin_bottom: float = 25 * mm
    font_size_range: tuple = (10, 14)
    line_spacing_range: tuple = (1.2, 1.8)
    paragraph_spacing_range: tuple = (8, 16)
    columns: int = 1
    add_noise: bool = False
    rotation_range: tuple = (-1, 1)  # degrees


@dataclass
class GeneratedSample:
    """A single training sample."""
    image: Image.Image
    marked_image: Optional[Image.Image] = None
    ground_truth_text: str = ""
    markers: dict = field(default_factory=dict)
    language: str = "mixed"  # hebrew, english, mixed
    page_info: dict = field(default_factory=dict)


class TextGenerator:
    """Generates synthetic text content."""

    def __init__(self, hebrew_texts: list[str] = None, english_texts: list[str] = None):
        self.hebrew_texts = hebrew_texts or HEBREW_SAMPLES
        self.english_texts = english_texts or ENGLISH_SAMPLES
        self.hebrew_paragraphs = HEBREW_PARAGRAPHS
        self.english_paragraphs = ENGLISH_PARAGRAPHS

    def generate_hebrew_sentence(self) -> str:
        """Generate a random Hebrew sentence."""
        return random.choice(self.hebrew_texts)

    def generate_english_sentence(self) -> str:
        """Generate a random English sentence."""
        return random.choice(self.english_texts)

    def generate_hebrew_paragraph(self) -> str:
        """Generate a Hebrew paragraph."""
        return random.choice(self.hebrew_paragraphs)

    def generate_english_paragraph(self) -> str:
        """Generate an English paragraph."""
        return random.choice(self.english_paragraphs)

    def generate_mixed_content(self, num_paragraphs: int = 3) -> list[tuple[str, str]]:
        """Generate mixed Hebrew/English content.

        Returns list of (text, language) tuples.
        """
        content = []
        for _ in range(num_paragraphs):
            if random.random() < 0.5:
                content.append((self.generate_hebrew_paragraph(), "hebrew"))
            else:
                content.append((self.generate_english_paragraph(), "english"))
        return content

    def generate_document(self,
                          language: Literal["hebrew", "english", "mixed"] = "mixed",
                          num_paragraphs: int = 4) -> list[tuple[str, str]]:
        """Generate a full document with specified language distribution."""
        if language == "hebrew":
            return [(self.generate_hebrew_paragraph(), "hebrew") for _ in range(num_paragraphs)]
        elif language == "english":
            return [(self.generate_english_paragraph(), "english") for _ in range(num_paragraphs)]
        else:
            return self.generate_mixed_content(num_paragraphs)


class PDFGenerator:
    """Generates synthetic PDF pages."""

    def __init__(self, style: PDFStyle = None):
        self.style = style or PDFStyle()
        self.text_gen = TextGenerator()
        self._register_fonts()

    def _register_fonts(self):
        """Register fonts for Hebrew/English support."""
        # Try to find system fonts that support Hebrew
        # On macOS, Arial Hebrew is usually available
        # On Linux/Windows, other fonts may be needed
        hebrew_fonts = [
            "/System/Library/Fonts/ArialHB.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/Arial Hebrew.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]

        self.hebrew_font = "Helvetica"  # Fallback
        for font_path in hebrew_fonts:
            if Path(font_path).exists():
                try:
                    pdfmetrics.registerFont(TTFont("HebrewFont", font_path))
                    self.hebrew_font = "HebrewFont"
                    break
                except Exception:
                    continue

    def create_pdf_page(self,
                        content: list[tuple[str, str]],
                        style: PDFStyle = None) -> BytesIO:
        """Create a PDF page with given content.

        Args:
            content: List of (text, language) tuples
            style: Optional style override

        Returns:
            BytesIO containing PDF data
        """
        style = style or self.style
        buffer = BytesIO()

        c = canvas.Canvas(buffer, pagesize=style.page_size)
        width, height = style.page_size

        # Calculate content area
        content_width = width - style.margin_left - style.margin_right
        y_position = height - style.margin_top

        font_size = random.uniform(*style.font_size_range)
        line_spacing = random.uniform(*style.line_spacing_range)

        for text, lang in content:
            is_hebrew = (lang == "hebrew")

            # Set font based on language
            if is_hebrew:
                c.setFont(self.hebrew_font, font_size)
            else:
                c.setFont("Helvetica", font_size)

            # Simple text wrapping
            words = text.split()
            current_line = []
            lines_to_draw = []

            for word in words:
                current_line.append(word)
                test_line = " ".join(current_line)

                # Check if line is too long (approximate)
                if len(test_line) * font_size * 0.5 > content_width:
                    # Store previous line
                    if len(current_line) > 1:
                        line_text = " ".join(current_line[:-1])
                        lines_to_draw.append(line_text)
                        current_line = [word]

            # Add remaining text
            if current_line:
                line_text = " ".join(current_line)
                lines_to_draw.append(line_text)

            # Draw lines
            for line_text in lines_to_draw:
                if is_hebrew:
                    # Convert to visual order for correct RTL display
                    visual_text = to_visual_hebrew(line_text)
                    # Right-align Hebrew text
                    text_width = c.stringWidth(visual_text, self.hebrew_font, font_size)
                    x_position = width - style.margin_right - text_width
                    c.drawString(x_position, y_position, visual_text)
                else:
                    # Left-align English text
                    c.drawString(style.margin_left, y_position, line_text)

                y_position -= font_size * line_spacing

            # Paragraph spacing
            y_position -= random.uniform(*style.paragraph_spacing_range)

            # Check if we need a new page
            if y_position < style.margin_bottom:
                break

        c.save()
        buffer.seek(0)
        return buffer

    def pdf_to_image(self, pdf_buffer: BytesIO, dpi: int = 150) -> Image.Image:
        """Convert PDF page to PIL Image.

        Args:
            pdf_buffer: BytesIO containing PDF
            dpi: Resolution for rendering

        Returns:
            PIL Image of the PDF page
        """
        pdf_buffer.seek(0)
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
        page = doc[0]

        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
        return img


class HebMarkDataGenerator:
    """Main class for generating HebMark training data."""

    def __init__(self,
                 output_dir: str | Path,
                 tokenizer_name: str = "utf8"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "marked_images").mkdir(exist_ok=True)
        (self.output_dir / "mappings").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        self.pdf_gen = PDFGenerator()
        self.text_gen = TextGenerator()
        self.tokenizer_name = tokenizer_name

    def generate_sample(self,
                        language: Literal["hebrew", "english", "mixed"] = "mixed",
                        num_paragraphs: int = 4,
                        apply_hebmark: bool = True,
                        dpi: int = 150) -> GeneratedSample:
        """Generate a single training sample.

        Args:
            language: Content language distribution
            num_paragraphs: Number of paragraphs
            apply_hebmark: Whether to apply HebMark markers
            dpi: Image resolution

        Returns:
            GeneratedSample with image and metadata
        """
        # Generate content
        content = self.text_gen.generate_document(language, num_paragraphs)
        ground_truth = "\n\n".join(text for text, _ in content)

        # Create PDF
        pdf_buffer = self.pdf_gen.create_pdf_page(content)

        # Get original image
        original_image = self.pdf_gen.pdf_to_image(pdf_buffer, dpi)

        sample = GeneratedSample(
            image=original_image,
            ground_truth_text=ground_truth,
            language=language,
            page_info={"num_paragraphs": num_paragraphs, "dpi": dpi}
        )

        if apply_hebmark and language in ("hebrew", "mixed"):
            # Apply HebMark
            marked_image, markers = self._apply_hebmark(pdf_buffer, dpi)
            sample.marked_image = marked_image
            sample.markers = markers

        return sample

    def _apply_hebmark(self, pdf_buffer: BytesIO, dpi: int) -> tuple[Image.Image, dict]:
        """Apply HebMark markers to PDF.

        Returns:
            Tuple of (marked_image, marker_mapping)
        """
        pdf_buffer.seek(0)
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")

        # Initialize encoder and renderer
        tokenizer = get_tokenizer(self.tokenizer_name)
        encoder = HebMarkEncoder(tokenizer=tokenizer)
        renderer = HebMarkRenderer(encoder)

        page = doc[0]

        # Extract and mark Hebrew words
        hebrew_words = extract_hebrew_words_with_boxes(page)
        renderer.replace_hebrew_with_markers(page, hebrew_words, page_num=0)

        # Render marked page
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        marked_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Get marker mapping
        markers = encoder.get_mapping()

        doc.close()
        return marked_image, markers

    def generate_dataset(self,
                         num_samples: int = 100,
                         language_distribution: dict = None,
                         save: bool = True) -> list[GeneratedSample]:
        """Generate a full dataset.

        Args:
            num_samples: Total number of samples
            language_distribution: Dict with language percentages
                e.g., {"hebrew": 0.4, "english": 0.4, "mixed": 0.2}
            save: Whether to save samples to disk

        Returns:
            List of GeneratedSample objects
        """
        if language_distribution is None:
            language_distribution = {"hebrew": 0.4, "english": 0.4, "mixed": 0.2}

        samples = []
        metadata_list = []

        # Calculate samples per language
        lang_counts = {
            lang: int(num_samples * pct)
            for lang, pct in language_distribution.items()
        }

        # Adjust for rounding
        total = sum(lang_counts.values())
        if total < num_samples:
            lang_counts["mixed"] += num_samples - total

        sample_idx = 0
        for language, count in lang_counts.items():
            for i in range(count):
                print(f"Generating sample {sample_idx + 1}/{num_samples} ({language})...")

                sample = self.generate_sample(
                    language=language,
                    num_paragraphs=random.randint(3, 6)
                )
                samples.append(sample)

                if save:
                    self._save_sample(sample, sample_idx)

                metadata_list.append({
                    "index": sample_idx,
                    "language": language,
                    "has_markers": sample.marked_image is not None,
                    "num_markers": len(sample.markers),
                    "text_length": len(sample.ground_truth_text)
                })

                sample_idx += 1

        # Save dataset metadata
        if save:
            metadata_path = self.output_dir / "dataset_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_samples": num_samples,
                    "language_distribution": language_distribution,
                    "samples": metadata_list
                }, f, indent=2)
            print(f"\nDataset saved to: {self.output_dir}")
            print(f"Metadata: {metadata_path}")

        return samples

    def _save_sample(self, sample: GeneratedSample, idx: int):
        """Save a single sample to disk."""
        # Save original image
        img_path = self.output_dir / "images" / f"sample_{idx:06d}.png"
        sample.image.save(img_path, "PNG")

        # Save marked image if available
        if sample.marked_image is not None:
            marked_path = self.output_dir / "marked_images" / f"sample_{idx:06d}_marked.png"
            sample.marked_image.save(marked_path, "PNG")

        # Save markers mapping
        if sample.markers:
            mapping_path = self.output_dir / "mappings" / f"sample_{idx:06d}_mapping.json"
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(sample.markers, f, ensure_ascii=False, indent=2)

        # Save metadata
        meta_path = self.output_dir / "metadata" / f"sample_{idx:06d}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "language": sample.language,
                "ground_truth": sample.ground_truth_text,
                "has_marked_image": sample.marked_image is not None,
                "num_markers": len(sample.markers),
                "page_info": sample.page_info
            }, f, ensure_ascii=False, indent=2)


def load_text_sources(sources_dir: Path | str) -> tuple[list[str], list[str]]:
    """Load text sources from files.

    Args:
        sources_dir: Directory containing hebrew.txt and english.txt

    Returns:
        Tuple of (hebrew_texts, english_texts)
    """
    sources_dir = Path(sources_dir)

    hebrew_texts = HEBREW_SAMPLES.copy()
    english_texts = ENGLISH_SAMPLES.copy()

    hebrew_file = sources_dir / "hebrew.txt"
    if hebrew_file.exists():
        with open(hebrew_file, "r", encoding="utf-8") as f:
            hebrew_texts.extend([line.strip() for line in f if line.strip()])

    english_file = sources_dir / "english.txt"
    if english_file.exists():
        with open(english_file, "r", encoding="utf-8") as f:
            english_texts.extend([line.strip() for line in f if line.strip()])

    return hebrew_texts, english_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic HebMark training data")
    parser.add_argument("--output", "-o", default="./training_data", help="Output directory")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples")
    parser.add_argument("--hebrew-pct", type=float, default=0.4, help="Hebrew percentage")
    parser.add_argument("--english-pct", type=float, default=0.4, help="English percentage")
    parser.add_argument("--mixed-pct", type=float, default=0.2, help="Mixed percentage")

    args = parser.parse_args()

    generator = HebMarkDataGenerator(args.output)

    generator.generate_dataset(
        num_samples=args.num_samples,
        language_distribution={
            "hebrew": args.hebrew_pct,
            "english": args.english_pct,
            "mixed": args.mixed_pct
        }
    )
