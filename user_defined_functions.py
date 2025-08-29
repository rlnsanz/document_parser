import re


def get_headings(text):
    return re.findall(r"^[A-Z\s]+$", text, re.MULTILINE)


def get_page_numbers(text):
    return re.findall(r"\b\d+\b", text)  # Simplistic; needs refinement
