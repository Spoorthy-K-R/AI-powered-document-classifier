import os, io, re, json, time, logging, uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, render_template_string, g
import requests

# ---- File parsers
import fitz  # PyMuPDF (PDF)
from PIL import Image
try:
    import pytesseract  # OCR
except Exception:
    pytesseract = None

try:
    import docx  # python-docx for .docx
except Exception:
    docx = None

# ==============================
# Config (override via env vars)
# ==============================
OPENROUTER_BASE   = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_APIKEY = os.getenv("OPENROUTER_API_KEY", "")
# Use free-tier models by default (can be overridden)
PRIMARY_MODEL     = os.getenv("PRIMARY_MODEL", "mistralai/mistral-7b-instruct")
VERIFIER_MODEL    = os.getenv("VERIFIER_MODEL", "meta-llama/llama-3.1-8b-instruct")

ENABLE_OCR       = os.getenv("ENABLE_OCR", "false").lower() == "true"
STOP_EARLY       = os.getenv("STOP_EARLY", "true").lower() == "true"
MAX_PAGE_CHARS   = int(os.getenv("MAX_PAGE_CHARS", "8000"))
MIN_TEXT_CHARS   = int(os.getenv("MIN_TEXT_CHARS", "40"))
MAX_PAGES        = int(os.getenv("MAX_PAGES", "0"))  # 0 = no cap
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()
# Where to persist HITL feedback (newline-delimited JSON)
HITL_LOG = os.getenv("HITL_LOG", "hitl_feedback.jsonl")

# Optional headers OpenRouter recommends (not required)
OPENROUTER_SITE   = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP    = os.getenv("OPENROUTER_APP_NAME", "AI-Powered Document Classifier")

# ==============================
# Logging / request-scoped debug
# ==============================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("classifier")

def _log_stage(event: str, **kv):
    rec = {"req": getattr(g, "req_id", "-"), "event": event, **kv}
    if hasattr(g, "stage_logs"):
        g.stage_logs.append(rec)
    logger.info(json.dumps(rec))

@contextmanager
def stage(name: str, **meta):
    t0 = time.perf_counter()
    _log_stage(f"{name}:start", **meta)
    try:
        yield
    finally:
        dt = round((time.perf_counter() - t0) * 1000.0, 1)
        _log_stage(f"{name}:end", ms=dt, **meta)

# ==============================
# Utilities
# ==============================
SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b")
CC_RE  = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

def luhn_ok(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not digits: return False
    checksum, dbl = 0, False
    for d in reversed(digits):
        checksum += (d*2 - 9) if dbl and d > 4 else (d*2 if dbl else d)
        dbl = not dbl
    return checksum % 10 == 0 and 13 <= len(digits) <= 19

def _extract_json(response_text: str) -> dict:
    """Try parse; if model added chatter, take last {...} block."""
    txt = response_text.strip()
    try:
        return json.loads(txt)
    except Exception:
        start, end = txt.rfind("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise ValueError("No JSON object found in LLM response")

def call_openrouter(prompt: str, model: str, timeout: int = 120) -> dict:
    """
    Robust OpenRouter caller:
      1) Try /chat/completions (OpenAI-compatible).
      2) On 404/405, fallback to /completions.
      3) Always print error text when non-200.
    """
    base = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # Shared headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but useful for OpenRouter analytics/rate-limits attribution
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "") or "http://localhost",
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI-Powered Document Classifier"),
        "Accept": "application/json",
    }

    # ---------- Try Chat Completions first ----------
    chat_url = f"{base}/chat/completions"
    chat_payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a careful compliance analyst. Reply with STRICT JSON only."},
            {"role": "user", "content": prompt}
        ],
    }

    t0 = time.perf_counter()
    r = requests.post(chat_url, headers=headers, json=chat_payload, timeout=timeout)
    dt = round((time.perf_counter() - t0) * 1000.0, 1)
    _log_stage("openrouter:chat", url=chat_url, model=model, status=r.status_code, ms=dt, prompt_chars=len(prompt))

    # If OK, parse and return
    if r.ok:
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _extract_json(content)

    # If clearly "wrong endpoint" for this model or server: fallback to /completions
    if r.status_code in (404, 405):
        # ---------- Fallback: legacy /completions ----------
        comp_url = f"{base}/completions"
        comp_payload = {
            "model": model,
            "prompt": (
                "You are a careful compliance analyst. Reply with STRICT JSON only.\n\n"
                + prompt
            ),
            "temperature": 0,
        }
        t1 = time.perf_counter()
        r2 = requests.post(comp_url, headers=headers, json=comp_payload, timeout=timeout)
        dt2 = round((time.perf_counter() - t1) * 1000.0, 1)
        _log_stage("openrouter:completions", url=comp_url, model=model, status=r2.status_code, ms=dt2)

        if r2.ok:
            data2 = r2.json()
            text = (data2.get("choices") or [{}])[0].get("text", "")
            return _extract_json(text)

        # Show server message to help diagnose (model slug, etc.)
        try:
            err_text = r2.text
        except Exception:
            err_text = "<no body>"
        raise requests.HTTPError(f"OpenRouter /completions error {r2.status_code}: {err_text}")

    # For other non-OK cases, raise with body so you can see what's wrong
    try:
        err_text = r.text
    except Exception:
        err_text = "<no body>"
    raise requests.HTTPError(f"OpenRouter /chat/completions error {r.status_code}: {err_text}")

# ==============================
# Prompt (page-level)
# ==============================
PAGE_PROMPT = """\
You are a careful compliance analyst. Output STRICT JSON only.

Analyze ONE page for policy signals. Use this exact schema:

{
  "page": %(page_no)d,
  "images_on_page": %(images_on_page)d,
  "pii": {
    "ssn": [], "credit_card": [], "bank_account": [], "passport_or_dl": [],
    "email": [], "phone": [], "address": [], "person_names": []
  },
  "pii_summary": {
    "has_pii": false,
    "kinds": [],
    "counts": {
      "ssn": 0, "credit_card": 0, "bank_account": 0, "passport_or_dl": 0,
      "email": 0, "phone": 0, "address": 0, "person_names": 0
    }
  },
  "unsafe": {
    "child_sexual_content": false,
    "hate_or_identity_attack": false,
    "violent_or_threatening": false,
    "criminal_activity": false,
    "cyber_threat_or_instructions": false,
    "exploitative_content": false,
    "political_news": false
  },
  "internal_business_only": { "is_internal": false, "evidence": [] },
  "notes": ""
}

Rules:
- HARD PII: ssn, credit_card, bank_account, passport_or_dl. Return small text snippets and offsets if visible.
- SOFT PII: email/phone/address/person_names. Presence of SOFT PII alone does NOT make the page internal.
- Treat obvious public-facing contact details as PUBLIC:
  • Company office addresses, headquarters locations, front-desk/office phone numbers.  
  • Business emails/phones for sales/support/press (e.g., info@, sales@, support@, press@, careers@, or named employees at the company domain like jane.doe@company.com when shown as a contact on a brochure, flyer, website snippet, job ad, or similar).  
- Marketing or services material is PUBLIC, not internal: brochures, product/service descriptions, one-pagers, datasheets, press releases, public event flyers, “About Us / Contact Us” sections.
- Mark internal_business_only.is_internal = true ONLY for non-public operational content: SOPs, internal memos, roadmaps, non-public schedules, draft contracts, pricing not meant for public, internal org charts, employee directories with private/home contact info, etc.
- child_sexual_content true ONLY if sexual terms appear near minors/children/teens.
- cyber_threat_or_instructions true if step-by-step exploit/malware guidance (news/advisory alone is false).
- If uncertain, prefer false. Keep arrays short. Use 'notes' to explain your decision briefly.

PAGE_TEXT:
\"\"\"%(page_text)s\"\"\"
"""

# ==============================
# Extraction
# ==============================
@dataclass
class PageItem:
    page_index: int
    text: str
    images_on_page: int

@dataclass
class ExtractedDoc:
    kind: str                  # pdf|image|docx|unknown
    pages: List[PageItem]
    page_count: int
    image_count: int

def ocr_image(pil_image: Image.Image) -> str:
    if pytesseract is None:
        return ""
    return pytesseract.image_to_string(pil_image)

def extract_from_pdf(pdf_bytes: bytes) -> ExtractedDoc:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[PageItem] = []
    total_images = 0
    page_limit = min(len(doc), MAX_PAGES or len(doc))
    for i in range(page_limit):
        page = doc[i]
        text = page.get_text("text") or ""
        img_list = page.get_images(full=True) or []
        total_images += len(img_list)
        if ENABLE_OCR and len(text.strip()) < MIN_TEXT_CHARS and img_list:
            pix = page.get_pixmap(dpi=220, alpha=False)
            pil = Image.open(io.BytesIO(pix.tobytes("png")))
            text = (text + "\n" + ocr_image(pil)).strip()
        pages.append(PageItem(i+1, text, len(img_list)))
    return ExtractedDoc("pdf", pages, page_limit, total_images)

def extract_from_image(img_bytes: bytes) -> ExtractedDoc:
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = ocr_image(pil) if (ENABLE_OCR or pytesseract) else ""
    return ExtractedDoc("image", [PageItem(1, text, 1)], 1, 1)

def extract_from_docx(docx_bytes: bytes) -> ExtractedDoc:
    if docx is None:
        return ExtractedDoc("docx", [PageItem(1, "", 0)], 1, 0)
    f = io.BytesIO(docx_bytes)
    d = docx.Document(f)
    text = "\n".join(p.text for p in d.paragraphs if p.text)
    img_cnt = sum(1 for r in d.part.rels.values() if 'image' in r.reltype or 'image' in r.target_ref)
    return ExtractedDoc("docx", [PageItem(1, text, img_cnt)], 1, img_cnt)

def sniff_and_extract(filename: str, data: bytes) -> ExtractedDoc:
    with stage("sniff_and_extract", filename=filename, size=len(data)):
        name = (filename or "").lower()
        if name.endswith(".pdf"):
            doc = extract_from_pdf(data)
        elif name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
            doc = extract_from_image(data)
        elif name.endswith(".docx"):
            doc = extract_from_docx(data)
        else:
            try:
                doc = extract_from_pdf(data)
            except Exception:
                try:
                    doc = extract_from_image(data)
                except Exception:
                    text = data.decode("utf-8", errors="ignore")
                    doc = ExtractedDoc("unknown", [PageItem(1, text, 0)], 1, 0)
        _log_stage("sniff_and_extract:result", kind=doc.kind, pages=doc.page_count, images=doc.image_count)
        return doc

# ==============================
# LLM page analysis + aggregation
# ==============================
HARD_PII_KEYS = {"ssn","credit_card","bank_account","passport_or_dl"}

def post_validate(decisions: List[dict]) -> List[dict]:
    fixed = []
    for d in decisions:
        dd = json.loads(json.dumps(d))
        if "pii" in dd and isinstance(dd["pii"].get("ssn"), list):
            dd["pii"]["ssn"] = [p for p in dd["pii"]["ssn"] if SSN_RE.search(str(p))]
        if "pii" in dd and isinstance(dd["pii"].get("credit_card"), list):
            cc_new=[]
            for p in dd["pii"]["credit_card"]:
                s = p.get("text", str(p))
                if CC_RE.search(s) and luhn_ok(s):
                    cc_new.append(p)
            dd["pii"]["credit_card"]=cc_new
        fixed.append(dd)
    return fixed

def aggregate(decisions: List[dict]) -> Dict[str, Any]:
    decisions = post_validate(decisions)

    def any_unsafe(docs): 
        return any(any(bool(v) for v in d.get("unsafe", {}).values()) for d in docs)

    def any_hard_pii(docs):
        for d in docs:
            pii = d.get("pii", {}) or {}
            if any(pii.get(k) for k in HARD_PII_KEYS):
                return True
        return False

    def any_internal(docs):
        return any(d.get("internal_business_only", {}).get("is_internal", False) for d in docs)

    # Single-label precedence
    if any_unsafe(decisions):
        final = "Unsafe"
        confidence = 0.90
    elif any_hard_pii(decisions):
        final = "Highly Sensitive"
        confidence = 0.80
    elif any_internal(decisions):
        final = "Confidential"
        confidence = 0.70
    else:
        final = "Public"
        confidence = 0.60

    return {
        "final_labels": [final],          
        "confidence": round(min(confidence, 0.99), 2),
        "evidence": decisions
    }

def _fallback_page_obj(p: PageItem) -> dict:
    return {
        "page": p.page_index,
        "images_on_page": p.images_on_page,
        "pii": {"ssn": [], "credit_card": [], "bank_account": [], "passport_or_dl": [],
                "email": [], "phone": [], "address": [], "person_names": []},
        "unsafe": {"child_sexual_content": False,"hate_or_identity_attack": False,"violent_or_threatening": False,
                   "criminal_activity": False,"cyber_threat_or_instructions": False,"exploitative_content": False,
                   "political_news": False},
        "internal_business_only": {"is_internal": False, "evidence": []},
        "notes": "fallback"
    }

def redact_for_client(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with all PII values removed, leaving only counts."""
    red = json.loads(json.dumps(result))  # deep copy

    # Strip PII values to counts
    for p in red.get("evidence", []) or []:
        pii = p.get("pii", {})
        if isinstance(pii, dict):
            for k, vals in list(pii.items()):
                count = len(vals) if isinstance(vals, list) else (vals.get("count", 0) if isinstance(vals, dict) else 0)
                pii[k] = {"count": int(count)}
        # (optional) trim notes if you worry notes might echo snippets
        if isinstance(p.get("notes"), str):
            p["notes"] = p["notes"][:400]

    # Remove any text snippets in citations (if you add them later)
    for c in red.get("citations", []) or []:
        c.pop("snippet", None)

    # Never ship debug traces/prompts
    red.pop("debug", None)
    return red

def analyze_pages(doc: ExtractedDoc) -> Dict[str, Any]:
    page_results=[]
    with stage("analyze_pages", pages=doc.page_count, kind=doc.kind):
        for p in doc.pages:
            page_text = (p.text or "").strip()[:MAX_PAGE_CHARS]
            if not page_text and doc.kind=="image" and not ENABLE_OCR:
                page_text=""
            prompt = PAGE_PROMPT % {"page_no": p.page_index, "images_on_page": p.images_on_page, "page_text": page_text}
            with stage("page_llm", page=p.page_index, chars=len(page_text), imgs=p.images_on_page, model=PRIMARY_MODEL):
                try:
                    out = call_openrouter(prompt, model=PRIMARY_MODEL)
                except Exception as e:
                    _log_stage("page_llm:error", page=p.page_index, err=str(e))
                    out = _fallback_page_obj(p)
            page_results.append(out)

            if STOP_EARLY:
                tmp = aggregate(page_results)
                _log_stage("stop_early:check", labels=tmp["final_labels"])
                if any(lbl in tmp["final_labels"] for lbl in ("Unsafe","Highly Sensitive")):
                    _log_stage("stop_early:triggered", page=p.page_index)
                    break

    summary = aggregate(page_results)
    _log_stage("aggregate:result", labels=summary["final_labels"], confidence=summary["confidence"])
    summary["page_count"]  = doc.page_count
    summary["image_count"] = doc.image_count
    return summary

def verify_with_second_llm(doc_json: Dict[str,Any]) -> Dict[str,Any]:
    prompt = f"""
You are auditing a classification. Output STRICT JSON.

Pick exactly ONE final label for the WHOLE document using this precedence:
1) "Unsafe" if ANY page has an unsafe flag true.
2) Else "Highly Sensitive" if ANY page has hard PII (ssn, credit_card, bank_account, passport_or_dl).
3) Else "Confidential" if ANY page is internal_business_only.is_internal = true.
4) Else "Public".

CRITICAL:
- Return exactly ONE label (not per-page), in a one-element array.
- Allowed values: "Public", "Confidential", "Highly Sensitive", "Unsafe".

Return:
{{
  "agree": true/false,
  "suggested_final_labels": ["Public" | "Confidential" | "Highly Sensitive" | "Unsafe"],
  "notes": "one short sentence"
}}

INPUT:
{json.dumps(doc_json, ensure_ascii=False)}
"""
    with stage("verify_llm", model=VERIFIER_MODEL):
        try:
            return call_openrouter(prompt, model=VERIFIER_MODEL)
        except Exception as e:
            _log_stage("verify_llm:error", err=str(e))
            return {
                "agree": True,
                "suggested_final_labels": doc_json.get("final_labels", []),
                "notes": f"verify_error: {e}"
            }

# ==============================
# Flask app
# ==============================
app = Flask(__name__)

UPLOAD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>AI-Powered Regulatory Document Classifier </title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .badge { @apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium; }
    .badge-public { @apply bg-green-100 text-green-800; }
    .badge-confidential { @apply bg-amber-100 text-amber-800; }
    .badge-sensitive { @apply bg-red-100 text-red-800; }
    .badge-unsafe { @apply bg-rose-100 text-rose-800; }
    .badge-hitl { @apply bg-indigo-100 text-indigo-800; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .truncate-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  </style>
</head>
<body class="bg-slate-50">
  <div class="max-w-6xl mx-auto p-6">
    <header class="mb-6">
      <h1 class="text-2xl font-semibold text-slate-900">AI-Powered Document Classifier</h1>
      <p class="text-slate-600">Upload any document (PDF, DOCX, images). Two LLMs cross-verify results and produce audit-grade citations.</p>
      <p class="text-xs text-slate-500 mt-1">Models: <span class="font-medium">PRIMARY={{primary}}</span> • <span class="font-medium">VERIFIER={{verifier}}</span> • Provider: <span class="font-medium">OpenRouter</span></p>
    </header>

    <!-- Upload Card -->
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
      <form id="uploadForm" class="space-y-4">
        <div id="drop" class="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center cursor-pointer hover:bg-slate-50">
          <p class="text-slate-700 font-medium">Drag & drop your files here</p>
          <p class="text-slate-500 text-sm">or click to choose</p>
          <input class="hidden" id="files" name="files" type="file" multiple />
        </div>
        <div id="fileList" class="text-sm text-slate-600 mt-2"></div>

        <div class="flex flex-wrap items-center gap-4">
          <label class="inline-flex items-center gap-2">
            <input type="checkbox" name="verify" checked class="rounded border-slate-300">
            <span class="text-sm text-slate-700">Dual-LLM verify</span>
          </label>
          <label class="inline-flex items-center gap-2">
            <input type="checkbox" name="ocr" {{ocr_checked}} class="rounded border-slate-300">
            <span class="text-sm text-slate-700">OCR (for scans/images)</span>
          </label>
          <button id="submitBtn" type="submit" class="ml-auto inline-flex items-center gap-2 bg-slate-900 text-white px-4 py-2 rounded-lg hover:bg-black disabled:opacity-50">
            <svg id="spinner" class="hidden animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4A4 4 0 008 12H4z"/></svg>
            <span>Classify</span>
          </button>
        </div>
      </form>
    </section>

    <!-- Results -->
    <section id="resultWrap" class="mt-6 hidden">
      <div class="p-2 text-xs text-slate-500" id="batchMeta"></div>
      <div id="resultsList" class="space-y-6"></div>
    </section>
  </div>

<script>
const drop = document.getElementById('drop');
const filesInput = document.getElementById('files');
const spinner = document.getElementById('spinner');
const form = document.getElementById('uploadForm');
const resultWrap = document.getElementById('resultWrap');
const resultsList = document.getElementById('resultsList');
const fileListEl = document.getElementById('fileList');

let selectedFiles = [];

drop.addEventListener('click', () => filesInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('bg-slate-50'); });
drop.addEventListener('dragleave', e => { drop.classList.remove('bg-slate-50'); });
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('bg-slate-50');
  addFiles(e.dataTransfer && e.dataTransfer.files);
});

filesInput.addEventListener('change', e => {
  addFiles(e.target.files);
  filesInput.value = ''; // allow re-pick without wiping list
});

function addFiles(fileList){
  if (!fileList) return;
  for (const f of Array.from(fileList)){
    const key = f.name + '|' + f.size + '|' + f.lastModified;
    if (!selectedFiles.some(sf => (sf.name + '|' + sf.size + '|' + sf.lastModified) === key)){
      selectedFiles.push(f);
    }
  }
  renderFileList();
}

function prettySize(bytes){
  const kb = bytes/1024;
  if (kb < 1024) return kb.toFixed(1) + ' KB';
  return (kb/1024).toFixed(2) + ' MB';
}

function renderFileList(){
  if (!selectedFiles.length){ fileListEl.textContent=''; return; }
  fileListEl.innerHTML = `
    <div class="flex items-center justify-between">
      <div class="font-medium">${selectedFiles.length} file(s) queued</div>
      <button type="button" id="clearList" class="text-xs underline">Clear</button>
    </div>
    ${selectedFiles.map(f => `<div class="text-slate-600">${f.name} • ${prettySize(f.size)}</div>`).join('')}
  `;
  document.getElementById('clearList').addEventListener('click', () => { selectedFiles = []; renderFileList(); });
}

function badge(label){
  const L = (label||'').toLowerCase();
  let klass = 'badge-public';
  if (L.includes('confidential')) klass = 'badge-confidential';
  if (L.includes('sensitive')) klass = 'badge-sensitive';
  if (L.includes('unsafe')) klass = 'badge-unsafe';
  if (L.includes('hitl')) klass = 'badge-hitl';
  return `<span class="badge ${klass}">${label}</span>`;
}

function renderCitations(arr){
  if (!arr || !arr.length) return '<div class="text-sm text-slate-500">None.</div>';
  return arr.map(c => `
    <div class="border rounded p-2 text-sm">
      <div class="flex gap-2 items-center">
        <span class="font-medium text-slate-800">${c.label || ''}</span>
        ${c.page ? `<span class="text-slate-500">Page ${c.page}</span>` : ''}
      </div>
      <div class="text-slate-700 mt-1">${c.reason || ''}</div>
      ${c.snippet ? `<div class="mono text-xs bg-slate-50 border rounded p-2 mt-1">${String(c.snippet).slice(0,240)}</div>`:''}
    </div>
  `).join('');
}

function renderEvidence(pagesArr){
  if (!pagesArr || !pagesArr.length) return '<div class="text-sm text-slate-500">No page evidence.</div>';
  return pagesArr.map(p => {
    const piiList = Object.entries(p.pii || {}).map(([k,vals]) => {
      let count = 0;
      if (Array.isArray(vals)) count = vals.length;
      else if (vals && typeof vals.count === 'number') count = vals.count;
      return count > 0
        ? `<div><span class="text-xs font-medium text-slate-600">${k}</span>: <span class="text-xs text-slate-700">${count} found</span></div>`
        : '';
    }).join('');

    const unsafeFlags = Object.entries(p.unsafe || {})
      .filter(([_,v]) => !!v)
      .map(([k]) => `<span class="badge badge-unsafe">${k}</span>`).join(' ');

    const internal = (p.internal_business_only && p.internal_business_only.is_internal)
      ? '<span class="badge badge-confidential">internal</span>' : '';

    return `
      <details class="border rounded-lg p-3">
        <summary class="cursor-pointer flex items-center justify-between">
          <div class="font-medium text-slate-800">Page ${p.page}</div>
          <div class="flex gap-2">${unsafeFlags}${internal}</div>
        </summary>
        <div class="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div class="text-xs text-slate-500 mb-1">PII (counts only)</div>
            <div class="space-y-1">${piiList || '<span class="text-xs text-slate-400">none</span>'}</div>
          </div>
          <div>
            <div class="text-xs text-slate-500 mb-1">Notes</div>
            <pre class="mono text-xs bg-slate-50 border rounded p-2 max-h-40 overflow-auto">${(p.notes||'').slice(0,2000) || '—'}</pre>
          </div>
        </div>
      </details>
    `;
  }).join('');
}

function cardTemplate(item, idx){
  const labels = (item.final_labels || []).map(badge).join(' ');
  const conf = item.confidence != null ? (item.confidence*100).toFixed(0)+'%' : '—';
  const verifier = item.verifier 
    ? `${item.verifier.agree ? '✅' : '⚠️'} <span class="mono">${JSON.stringify(item.verifier.suggested_final_labels||[])}</span><div class="text-slate-500">${item.verifier.notes || ''}</div>`
    : '—';

  return `
  <div class="bg-white rounded-xl shadow-sm border border-slate-200" data-idx="${idx}">
    <div class="p-5 border-b border-slate-200 flex items-center gap-3">
      <div class="text-sm text-slate-600">#${idx+1}</div>
      <h2 class="text-lg font-semibold text-slate-900 truncate">${item.filename || '—'}</h2>
      <div class="ml-auto flex gap-2 items-center">
        <div class="text-sm">${labels}</div>
        <div class="text-xs text-slate-500">Pages ${item.page_count ?? '–'} • Images ${item.image_count ?? '–'} • Conf ${conf}</div>
      </div>
    </div>

    <div class="p-5 grid grid-cols-1 xl:grid-cols-3 gap-5">
      <div class="col-span-1 space-y-3">
        <div class="p-4 rounded-lg border bg-slate-50">
          <div class="text-xs text-slate-500">Doc kind</div>
          <div class="font-medium text-slate-800">${item.doc_kind || '—'}</div>
        </div>

        <div class="p-4 rounded-lg border">
          <div class="text-sm font-medium text-slate-800">Verifier</div>
          <div class="text-sm text-slate-700 mt-2">${verifier}</div>
        </div>

        <div class="p-4 rounded-lg border">
          <div class="text-sm font-medium text-slate-800 mb-2">Human Review</div>
          <div class="flex items-center gap-2">
            <button class="px-3 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 js-approve">Approve</button>
            <button class="px-3 py-2 rounded-lg bg-rose-600 text-white hover:bg-rose-700 js-reject">Reject</button>
          </div>

          <div class="mt-3 hidden js-override">
            <label class="text-xs text-slate-600">Override labels</label>
            <div class="mt-2 grid grid-cols-2 gap-2">
              <label class="inline-flex items-center gap-2 text-sm"><input type="checkbox" class="ov" value="Public">Public</label>
              <label class="inline-flex items-center gap-2 text-sm"><input type="checkbox" class="ov" value="Confidential">Confidential</label>
              <label class="inline-flex items-center gap-2 text-sm"><input type="checkbox" class="ov" value="Highly Sensitive">Highly Sensitive</label>
              <label class="inline-flex items-center gap-2 text-sm"><input type="checkbox" class="ov" value="Unsafe">Unsafe</label>
            </div>
            <textarea placeholder="Add reviewer notes…" class="mt-3 w-full border rounded-lg p-2 text-sm js-notes" rows="3"></textarea>
            <button class="mt-2 px-3 py-2 rounded-lg border hover:bg-slate-50 js-submit-override">Submit override</button>
          </div>
          <div class="text-xs text-slate-500 mt-2 js-hitlmsg"></div>
        </div>

        <div class="flex items-center gap-3">
          <button class="inline-flex items-center gap-2 px-3 py-2 rounded-lg border hover:bg-slate-50 js-dl">Download JSON</button>
          <button class="inline-flex items-center gap-2 px-3 py-2 rounded-lg border hover:bg-slate-50 js-toggle-raw">Toggle Raw JSON</button>
        </div>
      </div>

      <div class="col-span-2">

        <h3 class="text-sm font-semibold text-slate-800 mt-5 mb-2">Citations and Evidence (page-level)</h3>
        <div class="space-y-2 js-evidence">${renderEvidence(item.evidence || [])}</div>

        <pre class="mono text-xs bg-slate-900 text-slate-100 p-3 rounded-lg mt-4 hidden overflow-auto js-raw">${JSON.stringify(item, null, 2)}</pre>
      </div>
    </div>
  </div>`;
}

function attachCardHandlers(card, item){
  const hitlMsg = card.querySelector('.js-hitlmsg');

  card.querySelector('.js-dl').addEventListener('click', () => {
    const el = document.createElement('a');
    el.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(item || {}, null, 2));
    el.download = (item.filename || 'classification') + '.json';
    el.click();
  });
  card.querySelector('.js-toggle-raw').addEventListener('click', () => {
    card.querySelector('.js-raw').classList.toggle('hidden');
  });

  card.querySelector('.js-approve').addEventListener('click', async () => {
    hitlMsg.textContent = 'Saving approval...';
    const payload = {
      decision: 'approved',
      req_id: item.req_id,
      filename: item.filename,
      model_labels: item.final_labels,
      verifier: item.verifier || null,
      citations: item.citations || [],
      notes: ''
    };
    const res = await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json();
    hitlMsg.textContent = data.ok ? 'Approved ✅' : ('Error: ' + (data.error||''));
  });

  const ovBox = card.querySelector('.js-override');
  card.querySelector('.js-reject').addEventListener('click', () => {
    ovBox.classList.remove('hidden');
  });
  card.querySelector('.js-submit-override').addEventListener('click', async () => {
    const checked = Array.from(card.querySelectorAll('.ov:checked')).map(x=>x.value);
    const notes = card.querySelector('.js-notes').value || '';
    if (!checked.length) { alert('Pick at least one override label.'); return; }
    hitlMsg.textContent = 'Submitting override...';
    const payload = {
      decision: 'rejected',
      req_id: item.req_id,
      filename: item.filename,
      model_labels: item.final_labels,
      override_labels: checked,
      verifier: item.verifier || null,
      citations: item.citations || [],
      notes
    };
    const res = await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json();
    if (data.ok) {
      item.final_labels = checked;
      card.querySelector('.js-raw').textContent = JSON.stringify(item, null, 2);
      hitlMsg.textContent = 'Override saved ✅';
    } else {
      hitlMsg.textContent = 'Error: ' + (data.error || '');
    }
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFiles.length) { alert('Add at least one file.'); return; }

  spinner.classList.remove('hidden');

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f, f.name));

  const verify = form.querySelector('input[name="verify"]').checked;
  const ocr = form.querySelector('input[name="ocr"]').checked;
  fd.set('verify', verify ? 'on' : '');
  fd.set('ocr', ocr ? 'on' : '');

  try {
    const res = await fetch('/classify', { method:'POST', body: fd });
    const data = await res.json();

    const batchInfo = `Batch ${data.batch_req_id || ''} • ${data.results?.length || 0} file(s) • Elapsed: ${data.elapsed_sec ?? '–'}s`;
    document.getElementById('batchMeta').textContent = batchInfo;

    resultsList.innerHTML = '';
    (data.results || []).forEach((item, idx) => {
      const html = cardTemplate(item, idx);
      const wrap = document.createElement('div');
      wrap.innerHTML = html.trim();
      const card = wrap.firstChild;
      resultsList.appendChild(card);
      attachCardHandlers(card, item);
    });
    resultWrap.classList.remove('hidden');
  } catch (err) {
    alert('Failed: ' + err);
  } finally {
    spinner.classList.add('hidden');
  }
});
</script>
</body>
</html>
"""

@app.before_request
def _start_request():
    g.req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.stage_logs = []
    g.t0 = time.perf_counter()
    _log_stage("request:start", path=request.path, method=request.method)

@app.after_request
def _after(resp):
    try:
        resp.headers["X-Request-ID"] = g.req_id
    except Exception:
        pass
    _log_stage("request:end", ms=round((time.perf_counter()-g.t0)*1000.0,1), status=resp.status_code)
    return resp

@app.get("/")
def index():
    return render_template_string(
        UPLOAD_HTML,
        primary=PRIMARY_MODEL,
        verifier=VERIFIER_MODEL,
        ocr_checked=("checked" if ENABLE_OCR else "")
    )

@app.post("/classify")
def classify():
    if not OPENROUTER_APIKEY:
        return jsonify({"ok": False, "error": "Missing OPENROUTER_API_KEY"}), 400

    with stage("classify"):
        files = request.files.getlist("files")
        if not files:
            fone = request.files.get("file")
            if fone:
                files = [fone]

        if not files:
            return jsonify({"ok": False, "error": "no files"}), 400

        verify = bool(request.form.get("verify"))
        # Per-request OCR override
        global ENABLE_OCR
        req_ocr = bool(request.form.get("ocr"))
        if req_ocr and not ENABLE_OCR:
            ENABLE_OCR = True

        batch_results = []
        for idx, f in enumerate(files, start=1):
            per_req_id = f"{g.req_id}-{idx}"
            g.stage_logs = []  # keep per-file debug compact
            try:
                data = f.read()

                with stage("extract_total", filename=f.filename, size=len(data)):
                    extracted = sniff_and_extract(f.filename, data)

                with stage("analyze_total"):
                    result = analyze_pages(extracted)

                if verify:
                  with stage("verify_total"):
                      audit = verify_with_second_llm(result)

                      # Expect exactly one label from verifier; fall back to model's if missing.
                      suggested_list = audit.get("suggested_final_labels") or []
                      suggested = suggested_list[0] if suggested_list else None

                      actual_list = result.get("final_labels") or []
                      actual = actual_list[0] if actual_list else None

                      agree = (suggested is None) or (suggested == actual)
                      if not agree and suggested:
                          audit["agree"] = False
                          audit.setdefault("notes", "")
                          if "auto-detected mismatch between models" not in audit["notes"]:
                              audit["notes"] += (" | " if audit["notes"] else "") + "auto-detected mismatch between models"

                          # Adopt verifier's single overall label; optionally tag HITL for visibility.
                          result["final_labels"] = [suggested, "HITL"]
                      else:
                          # Ensure we only ever keep a single overall label (no duplicates)
                          result["final_labels"] = [actual or suggested or "Public"]

                      result["verifier"] = audit

                # ... inside the for-file loop, after building `result`
                public_result = redact_for_client(result)
                # Ensure the public copy still carries the meta you need
                public_result["req_id"]      = per_req_id
                public_result["filename"]    = f.filename
                public_result["doc_kind"]    = extracted.kind
                public_result["page_count"]  = extracted.page_count
                public_result["image_count"] = extracted.image_count
                public_result["debug"]        = g.stage_logs

                batch_results.append(public_result)
                
                
            except Exception as e:
                _log_stage("file:error", filename=getattr(f, "filename", "?"), err=str(e))
                batch_results.append({
                    "req_id": per_req_id,
                    "filename": getattr(f, "filename", "?"),
                    "error": str(e),
                    "final_labels": ["HITL"],
                    "confidence": 0.0,
                    "evidence": []
                })

        resp = {
            "ok": True,
            "batch_req_id": g.req_id,
            "elapsed_sec": round(time.perf_counter()-g.t0, 2),
            "results": batch_results
        }
        return jsonify(resp)

@app.get("/health")
def health():
    with stage("health"):
        try:
            base = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                return jsonify({"ok": False, "error": "Missing OPENROUTER_API_KEY"})

            url = f"{base}/models"
            headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            r = requests.get(url, headers=headers, timeout=5)

            if not r.ok:
                return jsonify({"ok": False, "provider": "openrouter", "status": r.status_code, "body": r.text})

            data = r.json()
            # Optional: verify the configured models exist
            have_primary  = any(m.get("id") == PRIMARY_MODEL for m in data.get("data", []))
            have_verifier = any(m.get("id") == VERIFIER_MODEL for m in data.get("data", []))
            return jsonify({
                "ok": True,
                "provider": "openrouter",
                "models_listed": len(data.get("data", [])),
                "primary_model_ok": have_primary,
                "verifier_model_ok": have_verifier
            })
        except Exception as e:
            return jsonify({"ok": False, "provider": "openrouter", "error": str(e)})

@app.get("/ping")
def ping():
    return jsonify({"ok": True})

@app.post("/feedback")
def feedback():
    try:
        data = request.get_json(force=True, silent=False)
        entry = {
            "ts": time.time(),
            "req_id": data.get("req_id"),
            "filename": data.get("filename"),
            "decision": data.get("decision"),  # "approved" | "rejected"
            "model_labels": data.get("model_labels"),
            "override_labels": data.get("override_labels"),
            "verifier": data.get("verifier"),
            "citations": data.get("citations"),
            "notes": data.get("notes", "")
        }
        if entry["decision"] not in ("approved", "rejected"):
            return jsonify({"ok": False, "error": "invalid decision"}), 400

        with open(HITL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        _log_stage("hitl:feedback", decision=entry["decision"], labels=(entry.get("override_labels") or entry.get("model_labels")))
        return jsonify({"ok": True})
    except Exception as e:
        _log_stage("hitl:error", err=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Quick guardrail so you see a helpful message if key is missing
    if not OPENROUTER_APIKEY:
        print("WARNING: OPENROUTER_API_KEY is not set. Set it before using /classify.")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

HITL_LOG = os.getenv("HITL_LOG", "hitl_feedback.jsonl")