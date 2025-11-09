import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# 1) Detoxify
from detoxify import Detoxify

# 2) HF zero-shot
from transformers import pipeline


# -------------------------
# Configuration & Lexicons
# -------------------------
MINOR_TERMS = {
    "minor", "underage", "child", "children", "kid", "kids", "teen", "teenager",
    "17-year-old", "16-year-old", "15-year-old", "high school", "middle school"
}
SEX_TERMS = {
    "sex", "sexual", "explicit", "nude", "naked", "porn", "pornographic",
    "intimate", "aroused", "erotic"
}

CYBER_THREAT_TERMS = {
    "exploit", "payload", "shellcode", "c2", "command and control", "ransomware",
    "malware", "phishing kit", "ddos", "botnet", "keylogger", "backdoor",
    "zero-day", "0-day", "sql injection", "xss", "cve-", "privilege escalation",
    "bypass", "credential stuffing", "bruteforce", "mitm"
}

INSTRUCTIONAL_CUES = {
    "how to", "step by step", "steps:", "walkthrough", "tutorial", "guide",
    "run the following", "copy and paste", "execute", "use this command", "compile",
    "install this", "build from source"
}

# Zero-shot topics relevant to your policy (expand as needed)
ZS_LABELS = [
    "violent content", "criminal activity", "exploitative content",
    "cyber threat", "weapons", "political news"
]

# Thresholds (tune for your corpus)
THR = {
    "toxicity": 0.85,
    "identity_attack": 0.60,
    "threat": 0.60,
    "zero_shot_high": 0.75,   # strong topic match
    "zero_shot_medium": 0.60, # medium topic match (for HITL)
    "child_window_chars": 80
}


# -------------------------
# Utilities
# -------------------------
def _any_kw_present(text_l: str, keywords: set) -> bool:
    return any(k in text_l for k in keywords)

def _find_kw_spans(text_l: str, keywords: set) -> List[Dict[str, Any]]:
    hits = []
    for k in keywords:
        start = 0
        while True:
            i = text_l.find(k, start)
            if i == -1: break
            hits.append({"term": k, "start": i, "end": i + len(k)})
            start = i + len(k)
    return hits

def _child_safety_flag(text: str) -> Dict[str, Any]:
    """
    Flags if a sexual term appears near a minor-related term within a small window.
    Always treat as Unsafe + HITL.
    """
    low = text.lower()
    reasons = []
    for m_hit in _find_kw_spans(low, MINOR_TERMS):
        s = max(0, m_hit["start"] - THR["child_window_chars"])
        e = min(len(low), m_hit["end"] + THR["child_window_chars"])
        window = low[s:e]
        sex_hits = _find_kw_spans(window, SEX_TERMS)
        if sex_hits:
            reasons.append({
                "type": "child_safety",
                "minor_term": m_hit["term"],
                "window": (s, e),
                "sex_terms": [h["term"] for h in sex_hits]
            })
    return {"flag": bool(reasons), "reasons": reasons}

def _is_instructional(text_l: str) -> bool:
    return _any_kw_present(text_l, INSTRUCTIONAL_CUES)

def _cyber_threat_flag(text_l: str) -> Dict[str, Any]:
    """
    Flags likely cyber-threat instructional content:
    - presence of threat keywords
    - AND instructional cues
    """
    threat_hits = _find_kw_spans(text_l, CYBER_THREAT_TERMS)
    instruct = _is_instructional(text_l)
    return {
        "flag": bool(threat_hits and instruct),
        "reasons": {
            "keywords": [h["term"] for h in threat_hits],
            "instructional": instruct
        }
    }


# -------------------------
# Models wrapper
# -------------------------
@dataclass
class UnsafeModels:
    detox: Any
    zshot: Any

def load_models(device: Optional[int] = -1) -> UnsafeModels:
    """
    device: -1 for CPU, or GPU id if available.
    """
    detox = Detoxify('original')  # text toxicity model
    zshot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    return UnsafeModels(detox=detox, zshot=zshot)


# -------------------------
# Main analysis
# -------------------------
@dataclass
class UnsafeResult:
    unsafe: bool
    hitl_recommended: bool
    reasons: List[str]
    scores: Dict[str, float]
    topic_scores: Dict[str, float]
    tags: List[str]

def analyze_unsafe(text: str, models: UnsafeModels) -> UnsafeResult:
    """
    Run ensemble checks on `text` and return a structured decision.
    """
    reasons = []
    tags = []
    scores: Dict[str, float] = {}
    topic_scores: Dict[str, float] = {}

    text_l = text.lower().strip()

    # 1) Detoxify
    try:
        tox = models.detox.predict(text)
        # tox keys: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
        scores.update(tox)
        if tox.get("identity_attack", 0) > THR["identity_attack"]:
            reasons.append("identity_attack high")
        if tox.get("threat", 0) > THR["threat"]:
            reasons.append("threat high")
        if tox.get("toxicity", 0) > THR["toxicity"]:
            reasons.append("toxicity very high")
    except Exception as e:
        reasons.append(f"detoxify_error: {e}")

    # 2) Child-safety heuristic (always HITL if true)
    cs = _child_safety_flag(text)
    if cs["flag"]:
        reasons.append("child-safety cues (sexual terms near minors)")
        tags.append("child_safety")

    # 3) Zero-shot topics (contextual tags)
    #    Note: political news is a tag, not unsafe by itself
    try:
        zres = models.zshot(text, candidate_labels=ZS_LABELS, multi_label=True)
        # zres["labels"], zres["scores"] aligned
        topic_scores = dict(zip(zres["labels"], zres["scores"]))
        # Set tags for high/medium scores
        for lab, sc in topic_scores.items():
            if sc >= THR["zero_shot_high"]:
                tags.append(f"topic:{lab}")
            elif sc >= THR["zero_shot_medium"]:
                tags.append(f"topic_maybe:{lab}")
    except Exception as e:
        reasons.append(f"zero_shot_error: {e}")

    # 4) Cyber-threat instructional heuristic
    ct = _cyber_threat_flag(text_l)
    if ct["flag"]:
        reasons.append("cyber-threat instructional (keywords + how-to cues)")
        tags.append("cyber_threat_instructional")

    # Final decision
    # Unsafe if: any strong Detoxify signal OR child-safety OR cyber-threat how-to
    detox_unsafe = any([
        scores.get("identity_attack", 0) > THR["identity_attack"],
        scores.get("threat", 0) > THR["threat"],
        scores.get("toxicity", 0) > THR["toxicity"],
    ])
    child_unsafe = cs["flag"]
    cyber_unsafe = ct["flag"]

    unsafe = detox_unsafe or child_unsafe or cyber_unsafe
    # HITL if any unsafe flag OR medium/high risky topics
    hitl_recommended = unsafe or any(
        t.startswith("topic:violent content") or
        t.startswith("topic:criminal activity") or
        t.startswith("topic:exploitative content") or
        t.startswith("topic_maybe:violent content") or
        t.startswith("topic_maybe:criminal activity") or
        t.startswith("topic_maybe:exploitative content") or
        t.startswith("topic:cyber threat") or
        t.startswith("topic_maybe:cyber threat")
        for t in tags
    )

    return UnsafeResult(
        unsafe=unsafe,
        hitl_recommended=hitl_recommended,
        reasons=reasons if reasons else ["no strong unsafe signals"],
        scores=scores,
        topic_scores=topic_scores,
        tags=sorted(set(tags))
    )


# -------------------------
# Optional: simple demo
# -------------------------
if __name__ == "__main__":
    models = load_models(device=-1)
    sample = """
    This tutorial shows how to run the exploit step by step. Copy and paste the payload
    to gain a reverse shell. CVE-2024-XXXX. Not for kids.
    """
    res = analyze_unsafe(sample, models)
    print(asdict(res))