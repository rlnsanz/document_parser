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

featurize: process_all featurize.py
	@echo "Featurizing Data..."
	$(PYTHON) featurize.py
	@touch featurize

model.pth: export_ckpt.py
	@echo "Generating model..."
	$(PYTHON) export_ckpt.py

infer: model.pth infer.py
	@echo "Inferencing..."
	$(PYTHON) infer.py
	@touch infer

first_pages: label_by_hand.py featurize
	@echo "Labeling by hand"
	$(PYTHON) label_by_hand.py
	@touch first_pages

train: featurize first_pages train.py
	@echo "Training..."
	$(PYTHON) train.py

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
	@rm -f infer
	@rm -f process_all
	@rm -f first_pages
	@rm -f featurize
	@rm -f doc_links

