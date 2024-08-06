import os
import re
from app.constants import DOC_DIR

import flor


def merge_text_lattice(pdf_name, page_num, txt_page_numbers, ocr_page_numbers):
    """***********************************
    Look at how flor.log is used for featurization
    ***********************************"""
    metadata = []

    metadata.append({"pdf_name": pdf_name})
    # Construct path to the text file
    txt_path = os.path.join(DOC_DIR, pdf_name, "txt")
    txt_name = os.path.join(txt_path, f"page_{page_num}.txt")
    last_page = len(os.listdir(txt_path))

    # Analyze the text on the page
    headings, page_numbers, txt_text = analyze_text(txt_name)
    # Add the results to the metadata dictionary

    metadata.append({"txt-headings": flor.log("txt-headings", headings)})
    txt_page_numbers[page_num] = [int(each) for each in page_numbers]
    metadata.append(
        {
            "txt-page_numbers": flor.log(
                "txt-page_numbers",
                estimate_page_num(
                    page_num,
                    last_page,
                    txt_page_numbers[page_num],
                    ([] if page_num == 0 else txt_page_numbers[page_num - 1]),
                ),
            )
        }
    )

    # Construct path to the OCR file
    ocr_path = os.path.join(DOC_DIR, pdf_name, "ocr")
    ocr_name = os.path.join(ocr_path, f"page_{page_num}.txt")
    # Analyze the ocr on the page
    headings, page_numbers, ocr_text = analyze_text(ocr_name)
    # Add the results to the metadata dictionary
    metadata.append({"ocr-headings": flor.log("ocr-headings", headings)})
    ocr_page_numbers[page_num] = [int(each) for each in page_numbers]
    metadata.append(
        {
            "ocr-page_numbers": flor.log(
                "ocr-page_numbers",
                estimate_page_num(
                    page_num,
                    last_page,
                    ocr_page_numbers[page_num],
                    ([] if page_num == 0 else ocr_page_numbers[page_num - 1]),
                ),
            )
        }
    )

    if check_for_invalid_char_in_file(txt_name) or (
        len(txt_text) < len(ocr_text) // 2
        or len(txt_text.strip()) < len(txt_text) * 3 // 4
    ):
        flor.log("merge-source", "ocr")
        metadata.append({"ocr-text": flor.log("merged-text", ocr_text)})
    else:
        flor.log("merge-source", "txt")
        metadata.append({"txt-text": flor.log("merged-text", txt_text)})

    metadata.clear()


def estimate_page_num(page_num, final_page, page_numbers, prev_page_numbers):
    res = [each for each in page_numbers if each == page_num + 1]
    intersecting_page_numbers = set(page_numbers) & set(
        [int(each) + 1 for each in prev_page_numbers]
    )
    for n in sorted([n for n in intersecting_page_numbers]):
        res.append(n)
    return list(set([n for n in res if n <= final_page]))


def check_for_invalid_char_in_file(filename, invalid_char="�"):
    """Opens a file and checks for the presence of a specified invalid character.

    Args:
        filename (str): The name of the file to check.
        invalid_char (str): The character to search for. Defaults to '�'.

    Returns:
        bool: True if the invalid character is found, False otherwise.
    """

    with open(filename, "r") as f:
        for line in f:
            if invalid_char in line:
                return True
    return False


def analyze_text(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Example pattern for detecting headings (e.g., all caps)
    headings = re.findall(r"^[A-Z\s]+$", text, re.MULTILINE)

    # Example pattern for detecting page numbers
    page_numbers = re.findall(r"\b\d+\b", text)  # Simplistic; needs refinement

    # Other analysis can be added here

    return headings, page_numbers, text


if __name__ == "__main__":
    pdf_files = [
        os.path.splitext(f)[0] for f in os.listdir(DOC_DIR) if f.endswith(".pdf")
    ]

    for doc_name in flor.loop("document", pdf_files):
        txt_page_numbers = {}
        ocr_page_numbers = {}
        for page in flor.loop(
            "page", range(len(os.listdir(os.path.join(DOC_DIR, doc_name, "images"))))
        ):
            merge_text_lattice(doc_name, page, txt_page_numbers, ocr_page_numbers)

    image_files = [
        each for each in os.listdir(DOC_DIR) if each.endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_file in flor.loop("image", image_files):
        base, ext = os.path.splitext(image_file)
        image_path = os.path.join(DOC_DIR, base)

        original_path = os.path.join(image_path, image_file)
        txt_path = os.path.join(image_path, "ocr.txt")

        _, _, text = analyze_text(txt_path)
        flor.log("merge-source", "ocr")
        flor.log("merged-text", text)
