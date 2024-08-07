import os
from app import DOC_DIR, config
import flor


documents = [
    os.path.splitext(each)[0] for each in os.listdir(DOC_DIR) if each.endswith(".pdf")
]
for doc in documents:
    # Assert doc is directory
    assert os.path.isdir(
        os.path.join(DOC_DIR, doc)
    ), f"{doc} has not been parsed. Check your `make` call."


# 0-indexed first pages of each document
first_page = {
    "mueller_report": [0, 207, 394],
    # v1 :: TOC + 7
    # "mueller_report_V1": [0, (11 + 7), (14 + 7), (36 + 7), (66 + 7), (174 + 7)],
    # v2 :: TOC + 4
    # "mueller_report_V2": [0, (9 + 4), (15 + 4), (159 + 4), (182 + 4)],
    "Presidential_Immunity": [0, 51, 60, 67, 97],  # TODO: split on syllabus?
}


if __name__ == "__main__":
    for doc in flor.loop("document", documents):
        IMGS_DIR = os.path.join(DOC_DIR, doc, "images")
        for i in flor.loop("page", range(len(os.listdir(IMGS_DIR)))):
            page_path = f"page_{i}.png"
            if doc in first_page:
                flor.log(config.first_page, 1 if i in first_page[doc] else 0)
            else:
                flor.log(config.first_page, 1 if i == 0 else 0)
