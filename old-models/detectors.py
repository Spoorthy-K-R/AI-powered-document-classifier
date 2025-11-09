import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy
from detoxify import Detoxify
from unsafe_ensemble import analyze_unsafe, load_models

# Regex for SSN
SSN_REGEX = re.compile(r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b')
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b') # Basic credit card format

# Heuristics for document classification
CONFIDENTIAL_TERMS = [
    "internal memo", "confidential", "internal use", "do not distribute", "proprietary", "sensitive information",
    "internal communications", "business documents", "customer details", "non-public operational content"
]
PUBLIC_TERMS = [
    "Brochure",
    "Contact us", "product features", "marketing materials", "product brochures", "public website content", "generic images"
]

HARD_ENTITIES = [
    "US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "IBAN_CODE", "US_BANK_NUMBER", "IP_ADDRESS",
    "US_PASSPORT", "US_DRIVER_LICENSE"
]

# Load models once
analyzer = AnalyzerEngine()
spacy_nlp = spacy.load("en_core_web_sm")
detoxify_model = Detoxify('original')

def luhn_filter(card_number):
    """Validate credit card number using the Luhn algorithm."""
    digits = [int(d) for d in str(card_number) if d.isdigit()]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for digit in even_digits:
        total += sum(divmod(digit * 2, 10))
    return total % 10 == 0

def analyze_text(text):
    """
    Analyze text to detect PII, classify the document, and check for unsafe content.
    """
    # PII Detection
    presidio_results = analyzer.analyze(text=text, language="en", entities=HARD_ENTITIES)
    
    ssn_matches = [match.group(0) for match in SSN_REGEX.finditer(text)]
    
    credit_card_matches = []
    for match in CREDIT_CARD_REGEX.finditer(text):
        card_number = match.group(0).replace('-', '').replace(' ', '')
        if luhn_filter(card_number):
            credit_card_matches.append(match.group(0))
    
    # spaCy NER for names and addresses
    doc = spacy_nlp(text)
    spacy_entities = {ent.label_: ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "LOC", "ORG"]} # Added ORG for organizations

    # Unsafe Content Detection
    unsafe_scores_raw = detoxify_model.predict(text)
    unsafe_scores = {k: float(v) for k, v in unsafe_scores_raw.items()} # Convert all scores to standard floats
    # is_unsafe = any(score > 0.7 for score in unsafe_scores.values())
    models = load_models(device=-1)
    is_unsafe = analyze_unsafe(text, models)

    # Document Classification
    doc_class = "Public" # Default classification
    if any(term in text for term in CONFIDENTIAL_TERMS):
        doc_class = "Confidential"
    
    # Document Classification (ordered by priority)
    doc_class = "Public" # Default classification

    # 1. Unsafe Content
    if is_unsafe:
        doc_class = "Unsafe Content"
    else:
        # 2. Sensitive/Highly Sensitive
        pii_found = bool(presidio_results or ssn_matches or credit_card_matches)
        if pii_found:
            doc_class = "Highly Sensitive"
        else:
            # 3. Confidential
            confidential_heuristic_found = any(term in text.lower() for term in CONFIDENTIAL_TERMS)
            customer_details_found = any(label in spacy_entities for label in ["PERSON", "GPE", "LOC", "ORG"])
            
            if confidential_heuristic_found :
                doc_class = "Confidential"
            # 4. Public (already default if no other conditions met)

    return {
        "classification": doc_class,
        "is_unsafe": is_unsafe,
        "unsafe_scores": unsafe_scores,
        "pii": {
            "presidio": [result.to_dict() for result in presidio_results],
            "ssn_regex": ssn_matches,
            "credit_card_regex": credit_card_matches,
            "spacy_ner": spacy_entities # Include spaCy entities in PII for context
        }
    }