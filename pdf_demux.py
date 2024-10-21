import fitz
import os
from PIL import Image

from app.constants import DOC_DIR
import io
import time

import flor
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(
    det_arch="linknet_resnet50", reco_arch="master", pretrained=True
).to("cuda")

pdf_files = [each for each in os.listdir(DOC_DIR) if each.endswith(".pdf")]
for pdf_file in flor.loop("document", pdf_files):
    pdf_path = os.path.join(DOC_DIR, pdf_file)
    base, ext = os.path.splitext(pdf_path)
    img_path = os.path.join(base, "images")
    os.makedirs(img_path, exist_ok=True)
    doc = fitz.open(pdf_path)
    doctr_doc = DocumentFile.from_pdf(pdf_path)
    for page_num in flor.loop("page", range(doc.page_count)):
        page = doc.load_page(page_num)
        # Extract text and save as TXT
        flor.log("plain_text", page.get_text())

        pix = page.get_pixmap()
        output_image = os.path.join(img_path, f"page_{page_num}.png")
        pix.save(output_image)

        img_bytes = io.BytesIO(pix.tobytes("png"))
        img = Image.open(img_bytes)

        if page_num == 0:
            # Save the first page as the preview
            preview_path = os.path.join(base, "preview.png")
            img = img.resize((300, 400), Image.LANCZOS)
            img.save(preview_path)

        # Extract text with doctr
        result = model(doctr_doc[page_num : page_num + 1])
        flor.log("ocr_text", result.render())

    doc.close()

print("De-multiplexing Done!")
