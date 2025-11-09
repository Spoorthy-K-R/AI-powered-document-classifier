#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import fitz  # PyMuPDF
import requests

# ---------------- Configuration ----------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
VERIFIER_MODEL = "llama3.1:8b-instruct"
PRIMARY_MODEL = "mistral:7b-instruct"  # used only if verify=True

MIN_TEXT_CHARS = 40

SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b")
CC_RE  = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

HARD_PII_KEYS = {"ssn", "credit_card", "bank_account", "passport_or_dl"}

# ------------- Optional OCR ------------------

try:
    from PIL import Image
    import pytesseract
    import io
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


def page_text_with_optional_ocr(pdf_page, enable_ocr: bool = False) -> str:
    text = pdf_page.get_text("text") or ""
    if (not enable_ocr) or (len(text.strip()) >= MIN_TEXT_CHARS) or (not OCR_AVAILABLE):
        return text
    pix = pdf_page.get_pixmap(dpi=220, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_text = pytesseract.image_to_string(img)
    return (text + "\n" + ocr_text).strip()

def analyze_pdf_bytes(
    pdf_bytes: bytes,
    model: str,
    enable_ocr: bool,
    stop_early: bool,
    ollama_url: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_results: List[dict] = []

    for i in range(len(doc)):
        page = doc[i]
        text = page_text_with_optional_ocr(page, enable_ocr=enable_ocr) or ""
        if not text.strip():
            continue

        prompt = PAGE_PROMPT.format(page_no=i + 1, page_text=text[:8000])
        try:
            out = call_ollama(prompt, model=model, ollama_url=ollama_url, timeout=120)
        except Exception as e:
            out = {
                "page": i + 1,
                "pii": {"ssn": [], "credit_card": [], "bank_account": [], "passport_or_dl": [],
                        "email": [], "phone": [], "address": [], "person_names": []},
                "unsafe": {
                    "child_sexual_content": False,
                    "hate_or_identity_attack": False,
                    "violent_or_threatening": False,
                    "criminal_activity": False,
                    "cyber_threat_or_instructions": False,
                    "exploitative_content": False,
                    "political_news": False
                },
                "internal_business_only": {"is_internal": False, "evidence": []},
                "notes": f"llm_error: {e}"
            }
        page_results.append(out)

        if stop_early:
            tmp = aggregate_document(page_results)
            if any(lbl in tmp["final_labels"] for lbl in ("Unsafe", "Highly Sensitive")):
                break

    res = aggregate_document(page_results)
    if filename:
        res["filename"] = filename
    return res

def classify_document_bytes(
    contents: bytes,
    model: str = PRIMARY_MODEL,
    verify: bool = False,
    ocr: bool = False,
    stop_early: bool = False,
    ollama_url: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    ollama_url = ollama_url or DEFAULT_OLLAMA_URL
    t0 = time.time()
    res = analyze_pdf_bytes(
        pdf_bytes=contents,
        model=model,
        enable_ocr=ocr,
        stop_early=stop_early,
        ollama_url=ollama_url,
        filename=filename,
    )
    if verify:
        audit = verify_with_second_model(res, ollama_url=ollama_url)
        res["verifier"] = audit
        if not audit.get("agree", True):
            res["final_labels"] = list(dict.fromkeys(audit.get("suggested_final_labels", []) + ["HITL"]))
    res["elapsed_sec"] = round(time.time() - t0, 2)
    return res

def luhn_ok(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not digits:
        return False
    checksum, dbl = 0, False
    for d in reversed(digits):
        checksum += (d * 2 - 9) if dbl and d > 4 else (d * 2 if dbl else d)
        dbl = not dbl
    return checksum % 10 == 0 and 13 <= len(digits) <= 19


# ---------------- LLM Calls -------------------

def _extract_json(text: str) -> dict:
    m = re.search(r"\{(?:[^{}]|(?R))*\}\s*$", text, re.S)
    if not m:
        m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(m.group(0))


def call_ollama(prompt: str,
                model: str,
                ollama_url: str,
                timeout: int = 120,
                retries: int = 2) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": 0, "num_ctx": 6144},
        "stream": False
    }
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(ollama_url, json=payload, timeout=timeout)
            r.raise_for_status()
            resp = r.json().get("response", "").strip()
            return _extract_json(resp)
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"Ollama call failed: {last_err}")


# ---------------- Prompt ---------------------

PAGE_PROMPT = """\
You are a careful compliance analyst. Follow these rules and output STRICT JSON only.

TASK: Analyze ONE page of a document for policy signals and return JSON matching this schema exactly:

{{
  "page": {page_no},
  "pii": {{
    "ssn": [], "credit_card": [], "bank_account": [], "passport_or_dl": [],
    "email": [], "phone": [], "address": [], "person_names": []
  }},
  "unsafe": {{
    "child_sexual_content": false,
    "hate_or_identity_attack": false,
    "violent_or_threatening": false,
    "criminal_activity": false,
    "cyber_threat_or_instructions": false,
    "exploitative_content": false,
    "political_news": false
  }},
  "internal_business_only": {{
    "is_internal": false,
    "evidence": []
  }},
  "notes": ""
}}

Labeling hints:
- "HARD PII": ssn, credit_card, bank_account, passport_or_dl. These require explicit patterns or fields.
- "SOFT PII": email, phone, address, person_names. Soft PII alone does not mean highly sensitive.
- unsafe.child_sexual_content is true ONLY if sexual terms occur near minors/children/teens.
- cyber_threat_or_instructions is true if the page provides step-by-step exploit/malware instructions; news/advisories are not.
- internal_business_only.is_internal is true for internal memos, SOPs, training docs, revision logs, internal schedules, or non-public operations.
- If uncertain, prefer false. Keep lists short; include small snippets in pii arrays as dicts: {{ "text": "...", "start": 0, "end": 0 }}.

PAGE_TEXT:
\"\"\"{page_text}\"\"\"
"""


# ------------- Aggregation & post-validation --

def post_validate_hard_pii(decisions: List[dict]) -> List[dict]:
    fixed = []
    for dec in decisions:
        d = json.loads(json.dumps(dec))  # deep copy
        if "pii" in d and isinstance(d["pii"].get("ssn"), list):
            d["pii"]["ssn"] = [p for p in d["pii"]["ssn"] if SSN_RE.search(p.get("text", ""))]
        if "pii" in d and isinstance(d["pii"].get("credit_card"), list):
            cc_new = []
            for p in d["pii"]["credit_card"]:
                t = p.get("text", "")
                if CC_RE.search(t) and luhn_ok(t):
                    cc_new.append(p)
            d["pii"]["credit_card"] = cc_new
        fixed.append(d)
    return fixed


def aggregate_document(decisions: List[dict]) -> Dict[str, Any]:
    decisions = post_validate_hard_pii(decisions)

    def has_any_unsafe(d: dict) -> bool:
        return any(bool(v) for v in d.get("unsafe", {}).values())

    def has_any_hard_pii(d: dict) -> bool:
        pii = d.get("pii", {})
        return any(pii.get(k) for k in HARD_PII_KEYS)

    has_unsafe = any(has_any_unsafe(d) for d in decisions)
    has_hard   = any(has_any_hard_pii(d) for d in decisions)
    has_internal = any(d.get("internal_business_only", {}).get("is_internal", False) for d in decisions)

    labels: List[str] = []
    if has_unsafe:
        labels.append("Unsafe")
    if has_hard:
        labels.append("Highly Sensitive")
    elif has_internal:
        labels.append("Confidential")
    else:
        labels.append("Public")

    confidence = 0.50
    if "Unsafe" in labels:
        confidence += 0.25
    if "Highly Sensitive" in labels or "Confidential" in labels:
        confidence += 0.15

    return {
        "final_labels": labels,
        "confidence": round(min(confidence, 0.99), 2),
        "pages_analyzed": len(decisions),
        "evidence": decisions
    }


def verify_with_second_model(doc_json: Dict[str, Any],
                             ollama_url: str) -> Dict[str, Any]:
    prompt = f"""
You are auditing a classification. Respond with STRICT JSON only.

Given this page-level JSON, decide if the final_labels follow the rules:
- If any page has hard PII (ssn, credit_card, bank_account, passport_or_dl) -> include "Highly Sensitive".
- If any page unsafe.* is true -> include "Unsafe".
- If only internal_business_only true and no hard PII -> "Confidential".
- Else -> "Public".

Return:
{{
  "agree": true/false,
  "suggested_final_labels": ["..."],
  "notes": "one short sentence"
}}

INPUT:
{json.dumps(doc_json, ensure_ascii=False)}
"""
    try:
        return call_ollama(prompt, model=VERIFIER_MODEL, ollama_url=ollama_url, timeout=120)
    except Exception as e:
        return {
            "agree": True,
            "suggested_final_labels": doc_json.get("final_labels", []),
            "notes": f"verify_error: {e}"
        }


# ---------------- Core analysis ----------------

def analyze_pdf(pdf_path: Path,
                model: str,
                enable_ocr: bool,
                stop_early: bool,
                ollama_url: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    page_results: List[dict] = []

    for i in range(len(doc)):
        page = doc[i]
        text = page_text_with_optional_ocr(page, enable_ocr=enable_ocr) or ""
        if not text.strip():
            continue

        prompt = PAGE_PROMPT.format(page_no=i + 1, page_text=text[:8000])
        try:
            out = call_ollama(prompt, model=model, ollama_url=ollama_url, timeout=120)
        except Exception as e:
            out = {
                "page": i + 1,
                "pii": {"ssn": [], "credit_card": [], "bank_account": [], "passport_or_dl": [],
                        "email": [], "phone": [], "address": [], "person_names": []},
                "unsafe": {
                    "child_sexual_content": False,
                    "hate_or_identity_attack": False,
                    "violent_or_threatening": False,
                    "criminal_activity": False,
                    "cyber_threat_or_instructions": False,
                    "exploitative_content": False,
                    "political_news": False
                },
                "internal_business_only": {"is_internal": False, "evidence": []},
                "notes": f"llm_error: {e}"
            }
        page_results.append(out)

        # Optional speed-up: stop after first strong label
        if stop_early:
            tmp_doc = aggregate_document(page_results)
            if any(lbl in tmp_doc["final_labels"] for lbl in ("Unsafe", "Highly Sensitive")):
                break

    return aggregate_document(page_results)


# ---------------- Public callable ----------------

def classify_documents(paths: List[Path],
                       model: str = PRIMARY_MODEL,
                       verify: bool = False,
                       ocr: bool = False,
                       stop_early: bool = False,
                       write_files: bool = False,
                       ollama_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Classify one or more PDFs.

    Args:
        paths: list of Path objects (files or directories). Directories are searched recursively for *.pdf
        model: Ollama model name (default: llama3.1:8b-instruct)
        verify: if True, cross-verify with a second model (mistral:7b-instruct)
        ocr: enable OCR fallback for image/scanned pages
        stop_early: stop after first page that yields 'Unsafe' or 'Highly Sensitive'
        write_files: if True, also write <pdf>.classification.json files next to each PDF
        ollama_url: override Ollama endpoint; default is http://localhost:11434/api/generate

    Returns:
        List of dicts: [{ "path": Path, "result": <classification_json> }, ...]
    """
    ollama_url = ollama_url or DEFAULT_OLLAMA_URL
    outputs: List[Dict[str, Any]] = []

    # Expand inputs to a flat list of PDFs
    expanded: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            expanded.extend(sorted(p.rglob("*.pdf")))
        else:
            expanded.append(p)

    for pdf in expanded:
        t0 = time.time()
        res = analyze_pdf(pdf, model=model, enable_ocr=ocr, stop_early=stop_early, ollama_url=ollama_url)
        if verify:
            audit = verify_with_second_model(res, ollama_url=ollama_url)
            res["verifier"] = audit
            if not audit.get("agree", True):
                res["final_labels"] = list(dict.fromkeys(audit.get("suggested_final_labels", []) + ["HITL"]))
        res["elapsed_sec"] = round(time.time() - t0, 2)

        if write_files:
            out_path = pdf.with_suffix(".classification.json")
            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))

        outputs.append({"path": pdf, "result": res})

    return outputs


# ---------------- Optional CLI ----------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only document classifier (callable + CLI)")
    ap.add_argument("inputs", nargs="+", help="PDF files or folders")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--stop-early", action="store_true")
    ap.add_argument("--write-files", action="store_true", help="Also write <pdf>.classification.json files")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                    help="Ollama endpoint (default http://localhost:11434/api/generate)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    paths = [Path(s) for s in args.inputs]
    results = classify_documents(
        paths=paths,
        model=args.model,
        verify=args.verify,
        ocr=args.ocr,
        stop_early=args.stop_early,
        write_files=args.write_files,
        ollama_url=args.ollama_url,
    )
    for item in results:
        pdf = item["path"]
        res = item["result"]
        print(f"→ {pdf} | labels={res['final_labels']} | time={res['elapsed_sec']}s")