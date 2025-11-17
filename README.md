# PDF Processing and Document Analysis API with FastAPI

This application provides an API to process PDF files, extract text, perform OCR, calculate a legibility score, and classify documents using an LLM.
Deployed on Heroku - https://ai-powered-doc-classifier-59728d598ae1.herokuapp.com/

## Technologies Used

- Python
- FastAPI
- PyMuPDF (fitz)
- Pytesseract
- Pillow (for OCR fallback)
- Ollama (for LLM inference)

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   Tesseract OCR engine. You can download it from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract). Make sure to add the Tesseract executable to your system's PATH.

2.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the required LLM models:**
    You need to download the `llama3.1:8b-instruct` model (and optionally `mistral:7b-instruct` for verification) for Ollama:
    ```bash
    ollama pull llama3.1:8b-instruct
    ollama pull mistral:7b-instruct
    ```

Open your browser and navigate to `http://127.0.0.1:8001/docs`. You will see the Swagger UI, which allows you to interact with the API.

## API Endpoint

### `POST /classify/`

-   **Description:** Upload a PDF file to be processed and classified.
-   **Request:**
    -   `file`: The PDF file to be uploaded.
-   **Response:** A JSON object containing the processing results, including:
    -   `filename`: The name of the uploaded file.
    -   `page_count`: The total number of pages in the PDF.
    -   `average_legibility_score`: The average legibility score across all pages.
    -   `total_images`: The total number of images found in the PDF.
    -   `pages`: A list of results for each page, including extracted text, OCR text, legibility score, image count, and bounding boxes for found matches.
    -   `document_classification`: The LLM-based classification result for the entire document.
