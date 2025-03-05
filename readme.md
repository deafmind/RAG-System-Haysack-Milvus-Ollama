# AI Agent RAG System

## Description
This project implements a Retrieval-Augmented Generation (RAG) system where an AI agent answers questions based on the content of a provided PDF. The project utilizes Haystack and MilvusDB for document processing and vector storage.

<!-- ## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Licence](#licence)
- [Features](#features)
- [Challenges](#challenges) -->

## Installation
Follow these steps to set up the project:
1. Install the **uv** package manager:
    ```bash
    pip install uv # if uv package manager not installed on you system.
    git clone https://github.com/deafmind/RAG-System-Haysack-Milvus-Ollama.git
    ```
2. Install dependencies using **uv**:
    ```bash
    uv init
    uv venv .venv
    uv add farm-haystack milvus haystack-integrations
    uv sync
    ```

## Usage
### 1.Indexing the **PDF** File:
  - Run the `indexing_pipes.py` script:
      ```bash
      python indexing_pipes.py
      ```
  - Provide the path to your PDF file when prompted.
  - The script will chunk the file, embed the content into vectors, and save them into the Milvus database.
### 2.Generating Answers:
  - Run the `rag_pipes.py` script to query the system:
      ```bash
      python rag_pipes.py
      ```
  - The system will generate answers based on the content of the indexed document.

**Notes:**
- The system will read the data from the `data` folder.
- Update the `path` variable in the `indexing_pipes.py` and `main.py` files to match the name of your PDF file.
## Licence
This project is licensed under the MIT License.