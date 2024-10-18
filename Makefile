.PHONY: run install clean apply_split train install_tesseract

												 
#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#                                                 


# Set the FLASK_APP environment variable
export FLASK_APP=run.py
export FLASK_ENV=developmentß

# Absolute path to the virtual environment's Python interpreter
VENV_PATH := $(abspath .venv)
PYTHON := $(VENV_PATH)/bin/python
UNAME_S := $(shell uname -s)

# Create a virtual environment
.venv:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv

# Tesseract installation depending on the OS
install_tesseract:
	@echo "Installing Tesseract OCR..."
ifeq ($(UNAME_S),Linux)
	sudo apt-get update
	sudo apt-get install tesseract-ocr
endif
ifeq ($(UNAME_S),Darwin)
	brew install tesseract
endif
ifeq ($(UNAME_S),Windows_NT)
	@echo "Please install Tesseract OCR manually from https://github.com/UB-Mannheim/tesseract/wiki"
endif

# Install dependencies from requirements.txt
install: install_tesseract .venv requirements.txt 
	@echo "Installing dependencies..."
	$(VENV_PATH)/bin/pip install -r requirements.txt

doc_links:
	@if [ -L app/static/private ]; then \
		echo "Softlink already exists."; \
	else \
		echo "Creating softlink to PDF directory..."; \
		ln -sf $(realpath private) app/static/private; \
	fi
	@touch doc_links

process_pdfs: doc_links pdf_demux.py 
	@echo "Processing PDF files..."
	$(PYTHON) pdf_demux.py
	@touch process_pdfs

process_images: doc_links image_demux.py
	@echo "Processing Image files..."
	$(PYTHON) image_demux.py
	@touch process_images

featurize: process_pdfs process_images featurize.py
	@echo "Featurizing Data..."
	$(PYTHON) featurize.py
	@touch featurize

# model.pth: export_ckpt.py
# 	@echo "Generating model..."
# 	$(PYTHON) export_ckpt.py

# infer: model.pth infer.py
# 	@echo "Inferencing..."
# 	$(PYTHON) infer.py
# 	@touch infer

first_pages: label_by_hand.py featurize
	@echo "Labeling by hand"
	$(PYTHON) label_by_hand.py
	@touch first_pages

# train: featurize first_pages train.py
# 	@echo "Training..."
# 	$(PYTHON) train.py

# apply_split: split.py clean
# 	@echo "Applying split..."
# 	$(PYTHON) split.py

# Run the Flask development server
# run_infer: featurize infer
# 	@echo "Starting Flask development server..."
# 	# @flask run --port 5000
# 	$(PYTHON) run.py

run: first_pages run.py
	@echo "Starting Flask development server..."
	$(PYTHON) run.py


# Clean up pyc files and __pycache__ directories
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -f infer
	@rm -f process_pdfs
	@rm -f first_pages
	@rm -f featurize
	@rm -f doc_links
	@rm -f process_images

