from typing import Any, Dict, List
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import flordb as flor
import warnings
import mimetypes
import math
import sys

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


@app.route("/")
def index():
    global memoized_pdfs, feat_names, memoized_images

    pdf_files = [
        os.path.splitext(f)[0] for f in os.listdir(DOC_DIR) if f.endswith(".pdf")
    ]

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

    image_files = [
        f for f in os.listdir(DOC_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

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
    # TODO: Display the PNG not the PDF. Easier overlay.
    pdf_name = request.args.get("name")
    if not pdf_name:
        return "No file specified.", 400

    pdf_name = secure_filename(pdf_name)
    pdf_names.append(pdf_name)

    pdf_path = os.path.join(DOC_DIR, pdf_name)

    if os.path.isfile(pdf_path):
        return render_template("label_pdf.html", pdf_name=pdf_name, colors=get_colors())
    else:
        return "File not found.", 404


@app.route("/view-image")
def view_image():
    image_name = request.args.get("name")
    if not image_name:
        return "No file specified.", 400

    if not memoized_images.empty:
        text = memoized_images["image-text"][
            memoized_images["image_value"] == image_name
        ].values[0]
    else:
        text = ""

    image_name = secure_filename(image_name)
    image_names.append(image_name)

    image_path = os.path.join(DOC_DIR, image_name)

    if os.path.isfile(image_path):
        return render_template("label_image.html", image_name=image_name, text=text)


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
    print("skip ocr::", skip_ocr, type(skip_ocr))

    if (
        skip_ocr == True
        or (isinstance(skip_ocr, str) and skip_ocr.lower() == "true")
        or (isinstance(skip_ocr, float) and math.isnan(skip_ocr))
    ):
        return jsonify([{f"txt-page-{page_num+1}": record[config.page_text].values[0]}])
    else:
        return jsonify([{f"ocr-page-{page_num+1}": record[config.page_text].values[0]}])


if __name__ == "__main__":
    app.run(debug=True)
