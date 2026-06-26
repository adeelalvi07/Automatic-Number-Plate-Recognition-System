# ============================================================
# FILE: utils/text_cleaner.py
# PURPOSE: Clean and validate OCR output for Pakistani plates
# EXPLANATION:
#   OCR makes common mistakes: 0 vs O, 1 vs I, 5 vs S etc.
#   Pakistani plate formats:
#     Old format : ABC-1234   (city code + numbers)
#     New format : ABC-12-1234 (city + series + numbers)
#     Govt format: G-1234
#   This module corrects common errors and validates the format.
# ============================================================

import re
import logging

logger = logging.getLogger(__name__)


# ── Common OCR character confusions ───────────────────────
# Left = what OCR reads | Right = what it probably means
# These are the most common mistakes on license plates.

CHAR_CORRECTIONS = {
    # Digit context errors
    "O": "0",   # letter O → digit 0 (in numeric section)
    "I": "1",   # letter I → digit 1
    "l": "1",   # lowercase L → 1
    "Z": "2",   # Z → 2 (sometimes)
    "S": "5",   # S → 5 (sometimes)
    "B": "8",   # B → 8 (sometimes)
    "G": "6",   # G → 6 (sometimes)
}

# Pakistani city codes (most common)
PAKISTAN_CITY_CODES = {
    "LHR", "LHE",           # Lahore
    "KHI", "KRH",           # Karachi
    "ISB", "ISL",           # Islamabad
    "RWP", "RWL",           # Rawalpindi
    "PES", "PSH",           # Peshawar
    "QTA", "QUT",           # Quetta
    "FSD", "FSB",           # Faisalabad
    "MUL",                  # Multan
    "GRW",                  # Gujranwala
    "SRG",                  # Sargodha
    "HYD",                  # Hyderabad
    "SKT",                  # Sialkot
    "AJK",                  # Azad Kashmir
    "LEA",                  # Islamabad private
    "G",                    # Government
}


def clean_plate_text(raw_text: str) -> str:
    """
    Master cleaning function.
    Takes raw OCR output, returns the best-guess cleaned plate.

    Steps:
      1. Uppercase + strip
      2. Remove junk characters
      3. Remove extra spaces
      4. Apply format-aware corrections
      5. Insert hyphen if missing
    """
    if not raw_text:
        return ""

    # Step 1: Uppercase and strip whitespace
    text = raw_text.strip().upper()

    # Step 2: Remove characters that never appear on plates
    # Keep: A-Z, 0-9, hyphen, space
    text = re.sub(r"[^A-Z0-9\- ]", "", text)

    # Step 3: Collapse multiple spaces/hyphens
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("- ")

    # Step 4: Apply format-aware correction
    text = apply_format_correction(text)

    # Step 5: Normalize spacing around hyphens
    text = re.sub(r"\s*-\s*", "-", text)

    return text.strip()


def apply_format_correction(text: str) -> str:
    """
    Apply smart correction based on Pakistani plate formats.

    Pakistani plates:
      [LETTERS]-[DIGITS]         e.g. LHR-1234
      [LETTERS]-[DIGITS]-[DIGITS] e.g. LHR-12-1234
      G-[DIGITS]                 Government plates

    Logic:
      - Letter section: digits that look like letters are kept as letters
      - Digit section: letters that look like digits get converted
    """
    # Try to split into letter and digit sections
    # Pattern: starts with 1-4 uppercase letters, then digits
    parts = re.split(r"[\s\-]+", text)

    if not parts:
        return text

    corrected_parts = []

    for i, part in enumerate(parts):
        if i == 0:
            # First part = city code (should be all letters)
            corrected_parts.append(_clean_letter_section(part))
        else:
            # Remaining parts = numbers
            corrected_parts.append(_clean_digit_section(part))

    # Rejoin with hyphens
    result = "-".join(p for p in corrected_parts if p)
    return result


def _clean_letter_section(text: str) -> str:
    """
    Clean a section that should be letters (city code).
    Convert digit-that-looks-like-letter back to letters.
    e.g. "LHR" stays "LHR", "L4R" stays (4 is probably 4 not H)
    """
    # Reverse corrections: 0→O for letter sections
    reverse = {"0": "O", "1": "I", "5": "S", "8": "B"}
    result  = ""
    for ch in text:
        result += reverse.get(ch, ch)
    return result


def _clean_digit_section(text: str) -> str:
    """
    Clean a section that should be digits.
    Convert letters that look like digits.
    e.g. "O1234" → "01234", "I234" → "1234"
    """
    result = ""
    for ch in text:
        result += CHAR_CORRECTIONS.get(ch, ch)
    return result


def is_valid_pakistani_plate(text: str) -> bool:
    """
    Check if the cleaned text matches known Pakistani plate formats.

    Valid formats:
      ABC-1234        3-letter city code + 4 digits
      AB-1234         2-letter code + 4 digits
      ABC-12-1234     city + 2-digit series + 4 digits
      G-1234          Government
      G-12-1234       Government (new)
      LEA-1234        Islamabad special
    """
    if not text or len(text) < 4:
        return False

    patterns = [
        r"^[A-Z]{2,4}-\d{3,5}$",          # LHR-1234
        r"^[A-Z]{2,4}-\d{1,3}-\d{3,5}$",  # LHR-12-1234
        r"^G-\d{3,5}$",                    # G-1234 (Govt)
        r"^G-\d{1,3}-\d{3,5}$",           # G-12-1234
    ]

    for pattern in patterns:
        if re.match(pattern, text):
            return True

    return False


def format_plate_display(text: str) -> str:
    """
    Format plate text for display (uppercase, with hyphen).
    Returns the text as-is if already clean.
    """
    return text.upper().strip() if text else "UNKNOWN"


def deduplicate_text(texts: list[str]) -> str:
    """
    Given multiple OCR readings of the same plate (from different
    preprocessed versions), pick the most common one.
    Tie-break: prefer the one that matches valid format.
    """
    if not texts:
        return ""

    # Filter empty
    texts = [t for t in texts if t]
    if not texts:
        return ""

    # Count occurrences
    from collections import Counter
    counts = Counter(texts)

    # Prefer valid plates
    valid = [t for t in counts if is_valid_pakistani_plate(t)]
    if valid:
        return max(valid, key=lambda t: counts[t])

    # Otherwise return most common
    return counts.most_common(1)[0][0]