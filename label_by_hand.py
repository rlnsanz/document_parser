import os
from app import DOC_DIR, config
import flordb as flor


documents = [each for each in os.listdir(DOC_DIR) if each.endswith(".pdf")]
for doc in documents:
    # Assert doc is directory
    base, ext = os.path.splitext(doc)
    assert os.path.isdir(
        os.path.join(DOC_DIR, base)
    ), f"{doc} has not been parsed. Check your `make` call."

first_page = {}


if __name__ == "__main__":
    first_page = flor.dataframe("first_page")
    if first_page.empty:
        before_seen = []
    else:
        before_seen = first_page["document_value"].unique()
    for doc in flor.loop("document", documents):
        if doc in before_seen:
            continue
        base, ext = os.path.splitext(doc)
        IMGS_DIR = os.path.join(DOC_DIR, base, "images")
        for i in flor.loop("page", range(len(os.listdir(IMGS_DIR)))):
            page_path = f"page_{i}.png"
            if doc in first_page:
                flor.log(config.first_page, 1 if i in first_page[doc] else 0)
            else:
                flor.log(config.first_page, 1 if i == 0 else 0)
