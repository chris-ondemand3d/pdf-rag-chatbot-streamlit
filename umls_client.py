import os, re, json, logging
import numpy as np
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

UMLS_API_KEY = os.getenv("UMLS_API_KEY", "")
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
UMLS_AUTH = "https://utslogin.nlm.nih.gov/cas/v1/api-key"

# Semantic types relevant to dental/medical/implant context
RELEVANT_STYS = {
    "T023": "Body Part",
    "T024": "Tissue",
    "T025": "Cell",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T042": "Organ or Tissue Function",
    "T046": "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T061": "Procedure",
    "T074": "Medical Device",
    "T109": "Organic Chemical",
    "T121": "Pharmacologic Substance",
    "T122": "Biomedical/Dental Material",
    "T184": "Sign or Symptom",
    "T191": "Neoplastic Process",
}


def get_tgt() -> str:
    """Obtain a Ticket Granting Ticket from UMLS CAS."""
    r = requests.post(UMLS_AUTH, data={"apikey": UMLS_API_KEY}, timeout=15)
    r.raise_for_status()
    m = re.search(r'action="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("UMLS TGT not found in auth response")
    return m.group(1)


def _service_ticket(tgt_url: str) -> str:
    """Get a one-time Service Ticket from an active TGT."""
    r = requests.post(tgt_url, data={"service": "http://umlsks.nlm.nih.gov"}, timeout=10)
    r.raise_for_status()
    return r.text.strip()


def _get(path: str, tgt_url: str, **params) -> dict:
    params["ticket"] = _service_ticket(tgt_url)
    r = requests.get(f"{UMLS_BASE}{path}", params=params, timeout=15)
    return r.json() if r.status_code == 200 else {}


def search_umls(term: str, tgt_url: str, page_size: int = 10) -> list[dict]:
    """Search UMLS concept index for a term."""
    data = _get("/search/current", tgt_url,
                string=term, pageSize=page_size, returnIdType="concept")
    results = data.get("result", {}).get("results", [])
    return [r for r in results if r.get("ui") != "NONE"]


def get_semantic_types(cui: str, tgt_url: str) -> list[str]:
    """Return list of semantic type abbreviations for a CUI."""
    data = _get(f"/content/current/CUI/{cui}/semanticTypes", tgt_url)
    return [t.get("abbreviation", "") for t in data.get("result", [])]


def get_snomed_atoms(cui: str, tgt_url: str) -> list[dict]:
    """Return SNOMED-CT atoms (code + preferred name) for a CUI."""
    data = _get(f"/content/current/CUI/{cui}/atoms", tgt_url,
                sabs="SNOMEDCT_US", pageSize=5)
    return [{"name": a.get("name"), "code": a.get("code")}
            for a in data.get("result", [])]


def _cosine(a: list, b: list) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def map_entity(term: str, tgt_url: str, embeddings,
               threshold: float = 0.72) -> dict:
    """
    Search UMLS for `term`, rank candidates by embedding cosine similarity,
    filter by relevant semantic types, and return the best match.
    """
    base = {"term": term, "status": "new", "cui": None, "name": None,
            "similarity": None, "semantic_types": [], "snomed": []}

    candidates = search_umls(term, tgt_url)
    if not candidates:
        return base

    term_vec = embeddings.embed_query(term)
    names = [c.get("name", "") for c in candidates]
    name_vecs = embeddings.embed_documents(names)

    best, best_sim = None, -1.0
    for i, c in enumerate(candidates):
        sim = _cosine(term_vec, name_vecs[i])
        if sim <= best_sim:
            continue
        cui = c.get("ui")
        stys = get_semantic_types(cui, tgt_url)
        relevant = [RELEVANT_STYS[t] for t in stys if t in RELEVANT_STYS]
        if not relevant:
            continue
        best_sim = sim
        best = {
            "term": term,
            "status": "mapped" if sim >= threshold else "partial",
            "cui": cui,
            "name": c.get("name"),
            "similarity": round(sim, 3),
            "semantic_types": relevant,
            "snomed": get_snomed_atoms(cui, tgt_url),
        }

    return best or base


def suggest_snomed_parent(term: str, llm) -> dict:
    """Ask the LLM which existing SNOMED-CT concept should be the parent of `term`."""
    prompt = ChatPromptTemplate.from_template(
        'You are a biomedical ontology expert.\n'
        'The term "{term}" does not exist in SNOMED-CT.\n'
        'Suggest the most appropriate existing SNOMED-CT concept as its PARENT node.\n'
        'Respond ONLY with a JSON object (no markdown):\n'
        '{{"parent_name": "...", "parent_id": "... or null", "rationale": "one sentence"}}'
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"term": term})
    clean = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        return {"parent_name": clean, "parent_id": None, "rationale": ""}


def map_entities_to_umls(terms: list[str], progress_cb=None) -> list[dict]:
    """
    Map each term to UMLS/SNOMED-CT.
    For unmatched terms, suggests a SNOMED-CT parent via LLM.
    `progress_cb(i, total, term)` is called before each term is processed.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-3-flash-preview")
    tgt_url = get_tgt()

    results = []
    for i, term in enumerate(terms):
        if progress_cb:
            progress_cb(i, len(terms), term)
        logging.info(f"UMLS mapping [{i+1}/{len(terms)}]: {term}")
        r = map_entity(term, tgt_url, embeddings)
        if r["status"] == "new":
            r["suggested_parent"] = suggest_snomed_parent(term, llm)
        results.append(r)

    return results


def parse_entity_terms(entities_text: str) -> list[str]:
    """
    Extract individual term strings from the grouped bullet-list output
    produced by get_document_info().
    """
    terms = []
    for line in entities_text.splitlines():
        line = line.strip()
        # Match bullet markers: -, •, *, ·
        if line and line[0] in "-•*·" and len(line) > 2:
            term = line[1:].strip()
        else:
            continue
        # Drop trailing parenthetical, colon, or dash explanations
        for sep in ("(", ":", " -", " –"):
            if sep in term:
                term = term[:term.index(sep)].strip()
        term = term.strip("*_ ")
        if term:
            terms.append(term)
    return list(dict.fromkeys(terms))  # deduplicate preserving order
