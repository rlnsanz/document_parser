import os
import re
from app.constants import DOC_DIR

import flor


def get_headings(text):
    return re.findall(r"^[A-Z\s]+$", text, re.MULTILINE)


def get_page_numbers(text):
    return re.findall(r"\b\d+\b", text)  # Simplistic; needs refinement


if __name__ == "__main__":
    page_text = flor.dataframe("page_text")

    # iterate over the rows of page_text
    for document in flor.loop(
        "document", page_text["document_value"].drop_duplicates().values
    ):
        doc_pages = flor.utils.latest(
            page_text[page_text["document_value"] == document]
        )
        doc_pages = doc_pages.sort_values(by="page", ascending=True)

        features = []

        for i, row in flor.loop("page", doc_pages.iterrows()):
            text = row["page_text"]
            flor.log("f_headings", get_headings(text))
            flor.log("f_page_numbers", get_page_numbers(text))
