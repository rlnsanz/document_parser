from typing import Any, Dict, List, Optional, Set
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import safe_join, secure_filename
from werkzeug.exceptions import NotFound
import os
import flordb as flor
import warnings
import mimetypes
import math
import sys
import re
from functools import lru_cache

from wordfreq import zipf_frequency, top_n_list


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

from .constants import DOC_DIR

app = Flask(__name__)

mimetypes.add_type("text/javascript", ".mjs")

pdf_names = []
image_names = []
feat_names = [config.skip_ocr, config.page_text, config.page_color]
memoized_pdfs = None
memoized_images = None


def get_colors():
    # TODO: this method may also be called by apply_split
    df = flor.dataframe(config.first_page, config.page_color)
    if not df.empty:
        df = df[df["document_value"] == pdf_names[-1]]
        if not df.empty:
            if df[config.page_color].isna().all():
                df = flor.utils.latest(df)
                df = df.sort_values(by=["tstamp", "page"])
                return (df[config.first_page].astype(int).cumsum() - 1).tolist()
            else:
                df = flor.utils.latest(df[df.page_color.notna()])
                df = df.sort_values(by=["tstamp", "page"])
                return df[config.page_color].astype(int).tolist()


_natural_splitter = re.compile(r"(\d+)")


def _natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _natural_splitter.split(value)
        if part
    ]


@app.route("/")
def index():
    global memoized_pdfs, feat_names, memoized_images

    pdf_files = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(DOC_DIR) if f.endswith(".pdf")],
        key=_natural_key,
    )

    if pdf_files:
        if memoized_pdfs is None:
            memoized_pdfs = flor.dataframe(*feat_names)

        # Resize each image and create a list of tuples (pdf, image_path)
        pdf_previews = []
        for doc_name in flor.loop("document", pdf_files):
            image_path = os.path.join(DOC_DIR, doc_name, "preview.png")
            assert os.path.exists(image_path)
            # Only include the part of the image_path that comes after 'app/static/private/imgs'
            relative_image_path = os.path.relpath(image_path, start="app/static")
            print("relative image path", relative_image_path)
            pdf_previews.append((doc_name + ".pdf", relative_image_path))

    image_files = sorted(
        [f for f in os.listdir(DOC_DIR) if f.endswith((".png", ".jpg", ".jpeg"))],
        key=_natural_key,
    )

    image_previews = []
    if image_files:
        if memoized_images is None:
            memoized_images = flor.utils.latest(flor.dataframe("image-text"))

        # Resize each image and create a list of tuples (pdf, image_path)
        for image_name in flor.loop("image", image_files):
            base, ext = os.path.splitext(image_name)
            image_path = os.path.join(DOC_DIR, base, "preview.png")
            assert os.path.exists(image_path)
            # Only include the part of the image_path that comes after 'app/static/private/imgs'
            relative_image_path = os.path.relpath(image_path, start="app/static")
            print("relative image path", relative_image_path)
            image_previews.append((image_name, relative_image_path))

    # Render the template with the PDF previews
    return render_template(
        "index.html", pdf_previews=pdf_previews, image_previews=image_previews
    )


@app.route("/view-pdf")
def view_pdf():
    pdf_name = request.args.get("name")
    if not pdf_name:
        return "No file specified.", 400

    pdf_name = os.path.basename(pdf_name)
    try:
        pdf_path = safe_join(DOC_DIR, pdf_name)
    except NotFound:
        return "File not found.", 404

    if not os.path.isfile(pdf_path):
        return "File not found.", 404

    pdf_names.append(pdf_name)
    return render_template("label_pdf.html", pdf_name=pdf_name, colors=get_colors())


@app.route("/view-image")
def view_image():
    image_name = request.args.get("name")
    if not image_name:
        return "No file specified.", 400

    image_name = os.path.basename(image_name)
    if not memoized_images.empty:
        text = memoized_images["image-text"][
            memoized_images["image_value"] == image_name
        ].values[0]
    else:
        text = ""

    try:
        image_path = safe_join(DOC_DIR, image_name)
    except NotFound:
        return "File not found.", 404

    if os.path.isfile(image_path):
        image_names.append(image_name)
        return render_template("label_image.html", image_name=image_name, text=text)
    return "File not found.", 404


@app.route("/save_colors", methods=["POST"])
def save_colors():
    global memoized_pdfs
    j = request.get_json()
    colors = j.get("colors", [])
    pages = j.get("metadata", [])
    # Process the colors here...
    pdf_name = pdf_names.pop()
    pdf_names.clear()
    with flor.iteration("document", None, pdf_name):
        for i in flor.loop("page", range(len(colors))):
            flor.log(config.page_color, colors[i])
            # Use empty string if page text key is missing
            # Handle cases where 'data' or the specific text key might be missing
            # check if txt-page-{i+1} exists or ocr-page-{i+1}
            if f"txt-page-{i+1}" not in pages[i].get("data", {}):
                flor.log(
                    config.page_text,
                    pages[i].get("data", {}).get(f"ocr-page-{i+1}", ""),
                )
            else:
                flor.log(
                    config.page_text,
                    pages[i].get("data", {}).get(f"txt-page-{i+1}", ""),
                )
    flor.commit()
    memoized_pdfs = None
    return jsonify({"message": "Colors and Text saved successfully."}), 200


@app.route("/metadata-for-page/<int:page_num>")
def metadata_for_page(page_num: int):
    # if page_num == 0:
    #     # refresh
    #     memoized_pdfs = flor.utils.latest(flor.dataframe(*feat_names))
    assert memoized_pdfs is not None

    record = flor.utils.latest(
        memoized_pdfs[
            (memoized_pdfs["document_value"] == pdf_names[-1])
            & (memoized_pdfs["page"] == page_num + 1)
        ]
    )
    if record.empty:
        warnings.warn(f"No record found for page {page_num} of {pdf_names[-1]}")
        return jsonify([{f"txt-page-{page_num+1}": ""}])

    skip_ocr = record[config.skip_ocr].values[0]

    if (
        skip_ocr == True
        or (isinstance(skip_ocr, str) and skip_ocr.lower() == "true")
        or (isinstance(skip_ocr, float) and math.isnan(skip_ocr))
    ):
        # if page_color is 0 or NaN, return reflowed text

        if record[config.page_color].values[0] == 0 or (
            isinstance(record[config.page_color].values[0], float)
            and math.isnan(record[config.page_color].values[0])
        ):
            display_text = reflow_ocr_text(record[config.page_text].values[0])
        else:
            display_text = record[config.page_text].values[0]
        return jsonify([{f"txt-page-{page_num+1}": display_text}])
    else:
        return jsonify([{f"ocr-page-{page_num+1}": record[config.page_text].values[0]}])


# --- Constants and Helpers --------------------------------------------------

# Sets of words to conservatively prevent incorrect line joining.
# e.g., don't join "is" and "a" to form "isa".
_JOIN_BLOCK_PREV = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
    "per",
}
_JOIN_BLOCK_NEXT = {
    "a",
    "an",
    "and",
    "the",
    "that",
    "this",
    "these",
    "those",
    "then",
    "there",
    "their",
    "they",
    "with",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "between",
    "within",
    "without",
    "another",
    "any",
    "all",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "can",
    "may",
    "will",
    "shall",
    "must",
    "should",
    "could",
    "would",
    "not",
}
# Allow joining if the next line looks like a common suffix.
_ALLOW_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ness",
    "ity",
    "bility",
    "ability",
    "tity",
    "gether",
    "sional",
    "ker",
    "ation",
)
_WORD_SPLIT_PATTERN = re.compile(r"\b([A-Za-z]{2,})\b(\s+)\b([A-Za-z]{2,})\b")

_COMMON_WORDS: Optional[Set[str]] = None


if zipf_frequency is not None:

    @lru_cache(maxsize=8192)
    def _zipf(word: str) -> float:
        return zipf_frequency(word, "en")

else:

    def _zipf(word: str) -> float:
        return 0.0


def _is_common_word(word: str) -> bool:
    global _COMMON_WORDS
    if top_n_list is None:
        return False
    if _COMMON_WORDS is None:
        _COMMON_WORDS = set(top_n_list("en", 50000))
    return word.lower() in _COMMON_WORDS


def _is_all_caps_heading(s: str) -> bool:
    """Checks if a line is likely a heading (e.g., 'IMPORTANT NOTICE')."""
    s = s.strip()
    if len(s) < 4:
        return False
    # Check for 2+ words, all uppercase, with simple punctuation.
    words = [w for w in s.split() if re.search(r"[A-Z0-9]", w)]
    return (
        len(words) >= 2
        and all(w.upper() == w for w in words)
        and bool(re.fullmatch(r"[A-Z0-9 ,.'\"&:-]+", s))
    )


def _is_bullet_start(s: str) -> bool:
    """Checks if a line starts with a bullet point character."""
    # Allow common bullet characters (• · ● ▪ ‣ ◦ ∙ * -) and accept zero or more spaces
    # after the bullet so we catch "•Text" as well as "• Text".
    return bool(re.match(r"^\s*[•·●▪‣◦∙*\-]\s*", s))


# --- Core Logic -------------------------------------------------------------


def _process_text_blocks(lines: list[str]):
    """
    Yields paragraphs, headings, or bullet items, merging bullet-only lines with their content.
    """
    buffer: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        is_heading = _is_all_caps_heading(line)
        is_bullet = _is_bullet_start(line)
        is_break = not line or is_heading or is_bullet

        if is_break:
            if buffer:
                yield " ".join(buffer)
                buffer = []
            if not line:
                i += 1
                continue
            if is_heading:
                yield line
                i += 1
                continue
            # Canonicalize bullet
            bullet_line = re.sub(r"^\s*[•·●▪‣◦∙*\-]\s*", "- ", line)
            # If bullet is just "- ", merge with next non-break line
            if bullet_line.strip() == "-":
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        not next_line
                        or _is_all_caps_heading(next_line)
                        or _is_bullet_start(next_line)
                    ):
                        break
                    bullet_line = "- " + next_line
                    i = j  # skip merged line
                    break
            yield bullet_line
            i += 1
            continue

        if not buffer:
            buffer.append(line)
        elif buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line
        else:
            prev_word = buffer[-1].split()[-1].lower()
            next_word = line.split()[0].lower()
            is_blocked = (
                prev_word in _JOIN_BLOCK_PREV
                and not next_word.startswith(_ALLOW_SUFFIXES)
            ) or next_word in _JOIN_BLOCK_NEXT

            if is_blocked:
                buffer.append(line)
            else:
                buffer[-1] += " " + line  # Instead of buffer[-1] += line

        i += 1

    if buffer:
        yield " ".join(buffer)


def _merge_dictionary_splits(text: str) -> str:
    if zipf_frequency is None:
        return text

    def should_merge(left: str, right: str) -> bool:
        joined = f"{left}{right}"
        freq_joined = _zipf(joined.lower())
        freq_left = _zipf(left.lower())
        freq_right = _zipf(right.lower())
        max_parts = max(freq_left, freq_right)
        min_parts = min(freq_left, freq_right)

        if freq_joined >= 5.0:
            return True
        if freq_joined >= 4.0 and max_parts < 3.6:
            return True
        if freq_joined - max_parts >= 0.75 and freq_joined >= 3.5:
            return True
        if len(joined) >= 7 and freq_joined >= 3.6 and min_parts < 2.8:
            return True
        if (
            _is_common_word(joined)
            and not _is_common_word(left)
            and not _is_common_word(right)
        ):
            return True
        if right.lower().startswith(_ALLOW_SUFFIXES) and freq_joined >= 3.5:
            return True
        return False

    def replacer(match: re.Match) -> str:
        gap = match.group(2)
        if "\n\n" in gap:
            return match.group(0)
        left, right = match.group(1), match.group(3)
        return left + right if should_merge(left, right) else match.group(0)

    for _ in range(3):
        new_text = _WORD_SPLIT_PATTERN.sub(replacer, text)
        if new_text == text:
            break
        text = new_text
    return text


def _join_two_char_splits(text: str) -> str:
    """
    Join patterns like "1 0" -> "10" and "A n" -> "An".
    Conservative: joins digit-digit pairs always; joins letter-letter pairs
    when the joined form looks common/high-frequency or the first char is uppercase.
    Multiple passes to collapse sequences like "1 0 0" -> "100".
    """
    pattern = re.compile(r"\b([A-Za-z0-9])\s+([A-Za-z0-9])\b")

    def repl(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        # join digits always
        if a.isdigit() and b.isdigit():
            return a + b
        # join letters conservatively
        if a.isalpha() and b.isalpha():
            joined = a + b
            # join when first char is uppercase (likely OCR split) or joined is common/high freq
            score = _zipf(joined.lower()) if zipf_frequency is not None else 0.0
            if a.isupper() or _is_common_word(joined) or score >= 4.0:
                # preserve sensible casing: if A n -> An (b.lower()), otherwise keep original
                if a.isupper() and b.islower():
                    return a + b.lower()
                return joined
        return m.group(0)

    for _ in range(3):
        new = pattern.sub(repl, text)
        if new == text:
            break
        text = new
    return text


def _post_process(text: str) -> str:
    """Applies final regex cleanup for common OCR artifacts."""

    # --- Fix incorrectly joined common words ---
    _SPLIT_WORDS = {
        "asa": "as a",
        "ofa": "of a",
        "ina": "in a",
        "tothe": "to the",
        "isa": "is a",
        "wasa": "was a",
        "andthe": "and the",
        "inthe": "in the",
        "ofthe": "of the",
        "bea": "be a",
        "ora": "or a",
        "itis": "it is",
        "ona": "on a",
    }
    for word, replacement in _SPLIT_WORDS.items():
        # Use \b for word boundaries to avoid replacing parts of other words
        pattern = f"\\b{word}\\b"
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Fix common acronyms and remove spaces in s p a c e d o u t words
    text = re.sub(r"\b([A-Z])\.\s*([A-Z])\.\b", r"\1.\2.", text)

    # New join-pass for two-character splits (digits and common letter pairs)
    text = _join_two_char_splits(text)

    text = re.sub(
        r"\b([a-zA-Z]\s+){2,}[a-zA-Z]\b",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )
    # Standardize spacing around punctuation and remaining hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?\"'])", r"\1", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()


def _merge_dictionary_splits(text: str) -> str:
    if zipf_frequency is None:
        return text

    def should_merge(left: str, right: str) -> bool:
        joined = f"{left}{right}"
        freq_joined = _zipf(joined.lower())
        freq_left = _zipf(left.lower())
        freq_right = _zipf(right.lower())
        max_parts = max(freq_left, freq_right)
        min_parts = min(freq_left, freq_right)

        if freq_joined >= 5.0:
            return True
        if freq_joined >= 4.0 and max_parts < 3.6:
            return True
        if freq_joined - max_parts >= 0.75 and freq_joined >= 3.5:
            return True
        if len(joined) >= 7 and freq_joined >= 3.6 and min_parts < 2.8:
            return True
        if (
            _is_common_word(joined)
            and not _is_common_word(left)
            and not _is_common_word(right)
        ):
            return True
        if right.lower().startswith(_ALLOW_SUFFIXES) and freq_joined >= 3.5:
            return True
        return False

    def replacer(match: re.Match) -> str:
        gap = match.group(2)
        if "\n\n" in gap:
            return match.group(0)
        left, right = match.group(1), match.group(3)
        return left + right if should_merge(left, right) else match.group(0)

    for _ in range(3):
        new_text = _WORD_SPLIT_PATTERN.sub(replacer, text)
        if new_text == text:
            break
        text = new_text
    return text


# --- Main Function ----------------------------------------------------------


def reflow_ocr_text(text: str) -> str:
    """
    Reflows OCR text by intelligently joining lines while preserving
    paragraphs, headings, and bullet points.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    processed_blocks = list(_process_text_blocks(lines))
    processed = "\n\n".join(processed_blocks)
    processed = _merge_dictionary_splits(processed)
    return _post_process(processed)


if __name__ == "__main__":
    app.run(debug=True)
