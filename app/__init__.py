from typing import Any, Dict, List
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

from .constants import DOC_DIR

app = Flask(__name__)

mimetypes.add_type("text/javascript", ".mjs")

pdf_names = []
image_names = []
feat_names = [config.skip_ocr, config.page_text]
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
        return jsonify(
            [
                {
                    f"txt-page-{page_num+1}": reflow_ocr_text_conservative(
                        record[config.page_text].values[0]
                    )
                }
            ]
        )
    else:
        return jsonify([{f"ocr-page-{page_num+1}": record[config.page_text].values[0]}])


COMMON_SUBS = {
    r"\b([A-Z])\.\s*([A-Z])\.\b": r"\1.\2.",  # handles U. S., E. U., etc.
}
SPACED_WORD = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")


def _wide_clean(text: str) -> str:
    for pattern, repl in COMMON_SUBS.items():
        text = re.sub(pattern, repl, text)
    return SPACED_WORD.sub(lambda m: m.group(0).replace(" ", ""), text)


def reflow_ocr_text_conservative(text: str) -> str:
    """
    Reflow OCR text conservatively:
      - Preserve blank-line paragraph breaks
      - Treat ALL-CAPS lines (2+ words) as headings on their own lines
      - Keep bullets on separate lines (•, ·, ●, -, *)
      - Join other line wraps; fix hyphenated joins and common OCR spaces
    """
    # Normalize newlines
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def is_all_caps_heading(s: str) -> bool:
        s = s.strip()
        if len(s) < 4:
            return False
        # Only letters, digits, spaces, and simple punct allowed
        if not re.fullmatch(r"[A-Z0-9 ,.'\"&:-]+", s):
            return False
        words = [w for w in s.split() if re.search(r"[A-Z0-9]", w)]
        return len(words) >= 2 and all(w.upper() == w for w in words)

    def is_bullet_start(s: str) -> bool:
        return bool(re.match(r"\s*(?:[•·●]|[-*])\s*$", s)) or bool(
            re.match(r"\s*(?:[•·●]|[-*])\s+\S", s)
        )

    paras = []
    buf = []  # current paragraph word buffer
    in_list = False  # whether we're currently building a bullet list

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Paragraph break
        if line == "":
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            in_list = False
            i += 1
            continue

        # ALL-CAPS heading on its own paragraph
        if is_all_caps_heading(line):
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            paras.append(line)  # keep as-is
            in_list = False
            i += 1
            continue

        # Bullets (allow cases where bullet symbol is on its own line)
        if is_bullet_start(line):
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            # Normalize to "- "
            if re.match(r"\s*(?:[•·●]|[-*])\s*$", line) and i + 1 < len(lines):
                # bullet mark alone on a line; consume next content line(s)
                j = i + 1
                # accumulate until next blank/heading/bullet
                item_parts = []
                while j < len(lines):
                    nxt = lines[j].strip()
                    if nxt == "" or is_all_caps_heading(nxt) or is_bullet_start(nxt):
                        break
                    item_parts.append(nxt)
                    j += 1
                paras.append("- " + " ".join(item_parts).strip())
                i = j
                in_list = True
                continue
            else:
                # bullet with inline text
                # normalize bullet prefix to "- "
                line = re.sub(r"^\s*(?:[•·●]|[-*])\s*", "- ", line)
                paras.append(line)
                in_list = True
                i += 1
                continue

        # Normal text: join with previous (handle hyphenation)
        if buf:
            if buf[-1].endswith("-"):
                buf[-1] = buf[-1][:-1]  # remove hyphen, no space
                buf.append(line)
            else:
                buf.append(line)
        else:
            buf.append(line)
        i += 1

    if buf:
        paras.append(" ".join(buf).strip())

    # Post-clean: collapse spaces, fix punctuation spacing, common OCR artifacts
    out = "\n\n".join(paras)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    # out = re.sub(r"\bU\.\s*S\.", "U.S.", out)
    # out = re.sub(r"\bi\s+n\b", "in", out)  # "i n" -> "in"
    out = re.sub(r"(\w)-\s+(\w)", r"\1\2", out)  # leftover hyphen-wraps
    out = re.sub(
        r"\b(?:[a-z]\s+){1,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), out
    )
    out = re.sub(r"\s{3,}", " ", out).strip()
    out = _wide_clean(out)

    return out


if __name__ == "__main__":
    app.run(debug=True)
