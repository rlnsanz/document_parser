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


def resize_image(image_path, max_size=(300, 300)):
    # Open an image file
    with Image.open(image_path) as img:
        # Get original dimensions
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        # Calculate new dimensions
        if aspect_ratio > 1:
            # Landscape orientation
            new_width = min(max_size[0], original_width)
            new_height = int(new_width / aspect_ratio)
        else:
            # Portrait orientation or square
            new_height = min(max_size[1], original_height)
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        return img.resize((new_width, new_height), Image.LANCZOS)


IMG_EX_T = (".png", ".jpg", ".jpeg")

if __name__ == "__main__":

    model = ocr_predictor(
        det_arch="linknet_resnet50", reco_arch="master", pretrained=True
    ).to("cuda")

    pdf_files = [each for each in os.listdir(DOC_DIR) if each.endswith(".pdf")]
    image_files = [each for each in os.listdir(DOC_DIR) if each.endswith(IMG_EX_T)]
    for doc_file in flor.loop("document", pdf_files + image_files):
        doc_path = os.path.join(DOC_DIR, doc_file)
        base, ext = os.path.splitext(doc_path)

        if ext in IMG_EX_T:
            img_path = doc_path
            doctr_doc = DocumentFile.from_images(img_path)
            result = model(doctr_doc)
            flor.log("img_ocr", result.render())
            continue

        pdf_path = doc_path

        # Create a directory for the document
        images = os.path.join(base, "images")
        os.makedirs(images, exist_ok=True)

        # Load the PDF
        doc = fitz.open(pdf_path)
        doctr_doc = DocumentFile.from_pdf(pdf_path)
        for page_num in flor.loop("page", range(doc.page_count)):
            page = doc.load_page(page_num)
            # Extract text and save as TXT
            flor.log("page_text", page.get_text())

            # Save page PNG
            pix = page.get_pixmap()
            output_image = os.path.join(images, f"page_{page_num}.png")
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
            flor.log("page_ocr", result.render())

        doc.close()

    print("De-multiplexing Done!")
