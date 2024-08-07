.PHONY: run install clean apply_split train install_tesseract

												 
#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#                                                 


# Set the FLASK_APP environment variable
export FLASK_APP=run.py
export FLASK_ENV=developmentß

UNAME_S := $(shell uname -s)
GIT_COMMIT := $(shell git rev-parse HEAD | cut -c 1-6)
PDFS := $(wildcard app/static/private/pdfs/*.pdf)

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
	@python pdf_demux.py
	@touch process_pdfs

process_images: doc_links image_demux.py
	@echo "Processing Image files..."
	@python image_demux.py
	@touch process_images

featurize: process_pdfs process_images featurize.py
	@echo "Featurizing Data..."
	@python featurize.py
	@touch featurize

# model.pth: export_ckpt.py
# 	@echo "Generating model..."
# 	@python export_ckpt.py

# infer: model.pth infer.py
# 	@echo "Inferencing..."
# 	@python infer.py
# 	@touch infer

first_pages: label_by_hand.py featurize
	@echo "Labeling by hand"
	@python label_by_hand.py
	@touch first_pages

# train: featurize first_pages train.py
# 	@echo "Training..."
# 	@python train.py

# apply_split: split.py clean
# 	@echo "Applying split..."
# 	@python split.py

# Run the Flask development server
# run_infer: featurize infer
# 	@echo "Starting Flask development server..."
# 	# @flask run --port 5000
# 	@python run.py

run: first_pages run.py
	@echo "Starting Flask development server..."
	@python run.py

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
install: install_tesseract requirements.txt
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
 
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

