.PHONY: run install clean apply_split install_tesseract all

												 
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
PIP := $(VENV_PATH)/bin/pip
UNAME_S := $(shell uname -s)

all: run

# Create a virtual environment
.venv:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv

# Install dependencies from requirements.txt
install: .venv requirements.txt 
	@echo "Installing dependencies..."
	$(VENV_PATH)/bin/pip install -r requirements.txt

# TODO Add target for download
# private: download.py
# 	@echo "Downloading PDF files..."
# 	$(PYTHON) download.py

# download: private

app/static/private:
	@echo "Creating softlink to PDF directory..."
	@ln -sf $(realpath private) app/static/private

process_all: app/static/private doc_demux.py 
	@echo "Processing PDF files..."
	$(PYTHON) doc_demux.py
	@touch process_all

process_images: app/static/private image_demux.py
	@echo "Processing Image files..."
	$(PYTHON) image_demux.py
	@touch process_images


model.pth: export_ckpt.py
	@echo "Generating model..."
	$(PYTHON) export_ckpt.py

first_pages: label_by_hand.py process_all
	@echo "Labeling by hand"
	$(PYTHON) label_by_hand.py
	@touch first_pages

.PHONY: train_llmv3
train_llmv3: train/layoutlmv3.py
	@echo "Training LayoutLMV3..."
	$(PYTHON) train/layoutlmv3.py

.PHONY: train_fp
train_fp: first_pages train/first_page_clf.py
	@echo "Training First Page Classifier..."
	$(PYTHON) train/first_page_clf.py

apply_split: split.py clean
	@echo "Applying split..."
	$(PYTHON) split.py

run: first_pages run.py
	@echo "Starting Flask development server..."
	$(PYTHON) run.py

# Clean up pyc files and __pycache__ directories
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -f process_all
	@rm -f first_pages
	@rm -f doc_links
	@rm -rf private/*/

