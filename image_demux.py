import os
import shutil
from PIL import Image
import pytesseract
from multiprocessing import Pool

from app.constants import DOC_DIR
from tqdm import tqdm
import io
import time


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
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the image back to the same path
        img.save(image_path)


def process_page(image_path):
    base, ext = os.path.splitext(image_path)

    img = Image.open(image_path)

    # Save the first page as the preview
    preview_path = os.path.join(base, "preview.png")

    img.save(preview_path)
    resize_image(preview_path)

    # Extract text with pytesseract
    extracted_text = pytesseract.image_to_string(img)

    # Save the extracted text
    ocr_file_path = os.path.join(base, "ocr.txt")
    with open(ocr_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)


def process_pdf(image_path, all_args):
    # set img_path, txt_path, and ocr_path
    base, ext = os.path.splitext(image_path)
    os.makedirs(base, exist_ok=True)

    # Copy the image to the base directory
    shutil.copy(image_path, base)

    all_args.append((image_path,))


if __name__ == "__main__":
    all_args = []

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 4:
        max_workers = 1
    else:
        max_workers = 4

    image_files = [
        each for each in os.listdir(DOC_DIR) if each.endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_file in tqdm(image_files):
        image_path = os.path.join(DOC_DIR, image_file)
        process_pdf(image_path, all_args)

    # Create a pool of workers and distribute the tasks
    print(f"Parallel processing over {max_workers} cores ...")
    start_time = time.time()
    with Pool(max_workers) as pool:
        pool.starmap(process_page, all_args)
    end_time = time.time()
    print(f"Parallel processing took {end_time - start_time} seconds")
    print("Done!")
