# Document Parser
This project presents a Flask-based web application with a focus on user interface and optional AI integration. The primary command, `make run`, initiates the web server and provides access to the core functionalities. Advanced users can optionally enhance the application by training a model or updating it with the best version.

## Running the Web Application

### Prerequisites
- Python 3.x
- Flask
- Other dependencies in `requirements.txt`

### Quick Start
To quickly start the web application:
```bash
git clone git@github.com:rlnsanz/document_parser.git
cd document_parser
make install
make run
```
This command sets up the environment and launches the Flask web server, ready for use.

## Storing PDFs for Processing

For privacy and organization, this application processes PDFs and PNGs stored in a specific directory: `private/`. This directory should be created at the root of this repository (same dir that contains the Makefile), and it will be excluded from version control via `.gitignore` to ensure privacy and data security.


## Contributing
Contributions are welcome. Please use standard fork-and-pull request workflow for any contributions.

## License
This project is licensed under the Apache License, Version 2.0 