import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from typing import List
import io
from llm_classifier import classify_document_bytes # Import the new LLM classifier

app = FastAPI()

def get_pixmap(doc, page, zoom=2):
    """Rasterize a page for OCR."""
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return pix

def ocr_page(pix):
    """Perform OCR on a pixmap."""
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:  # RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    custom_config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    return ocr_data

def calculate_legibility(ocr_data):
    """Calculate the mean confidence of OCR."""
    confidences = [int(c) for c in ocr_data['conf'] if int(c) > -1]
    if not confidences:
        return 0
    return sum(confidences) / len(confidences)

def variance_of_laplacian(image):
    """Compute the Laplacian variance to score blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def ocr_image_from_bytes(img_bytes):
    """Perform OCR on image bytes."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text.strip()

@app.post("/process_doc/")
async def process_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
    
    page_count = len(doc)
    results = []
    total_legibility = 0
    total_images = 0
    
    result_bytes_classifier = classify_document_bytes(
        contents,
        model="mistral:7b-instruct",
        verify=True,
        ocr=False,                  # set True if you expect scans
        stop_early=True,
        ollama_url="http://localhost:11434/api/generate",
        filename=file.filename,
    )

    for i in range(page_count):
        page = doc.load_page(i)
        
        # 1. Extract text
        page_text = page.get_text("text")
        
        # 2. Calculate legibility for all pages and OCR sparse ones
        ocr_text = ""
        
        pix = get_pixmap(doc, page)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        blur_score = variance_of_laplacian(img_np)

        if len(page_text.strip()) < 100:  # Threshold for 'sparse'
            ocr_data = ocr_page(pix)
            ocr_text = " ".join(ocr_data['text']).strip()
            ocr_confidence = calculate_legibility(ocr_data)
            legibility_score = float((ocr_confidence + blur_score) / 2) # Convert to standard float
        else:
            legibility_score = float(blur_score) # Convert to standard float
        
        # 3. Extract images and perform OCR on them
        page_images_list = page.get_images(full=True)
        total_images += len(page_images_list)
        image_texts = []
        for img_index, img_info in enumerate(page_images_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_text = ocr_image_from_bytes(image_bytes)
            if img_text:
                image_texts.append(img_text)

        # 4. Bounding boxes for text
        matches = page.search_for("text") # Simple search for "text"
        bboxes = [list(m) for m in matches]

        results.append({
            "page_number": i + 1,
            "extracted_text": page_text,
            "ocr_text_from_page": ocr_text,
            "text_from_images": image_texts,
            "legibility_score": legibility_score,
            "image_count": len(page_images_list),
            "found_matches_bboxes": bboxes
        })
        total_legibility += legibility_score

    avg_legibility = float(total_legibility / page_count) if page_count > 0 else 0.0 # Convert to standard float

    return {
        "filename": file.filename,
        "page_count": page_count,
        "average_legibility_score": avg_legibility,
        "total_images": total_images,
        "pages": results,
        "document_classification": result_bytes_classifier
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
