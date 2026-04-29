from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import io
import tarfile
import PyPDF2
import requests
from bs4 import BeautifulSoup

from django.conf import settings
from django.db import transaction

from DSapp.models import Article

try:
    from google import genai
except Exception:
    genai = None


NCBI_TOOL = os.getenv("NCBI_TOOL", "kroma")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@domain.edu")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

DEFAULT_DB_ALIAS = "dsai"
DEFAULT_SEARCH_TERM = '"Dravet Syndrome" OR "Severe Myoclonic Epilepsy of Infancy" OR SCN1A'
DEFAULT_RETMAX = 200
REQUEST_TIMEOUT = 60
SYNC_DELAY_SECONDS = 0.34

PDF_SUBDIR = "pdfs"
XML_SUBDIR = "fulltext_xml"

AXIS_OPTIONS = [
    "Seizures",
    "Genetics",
    "Development",
    "Pharmacology",
    "Comorbidities",
    "Behavior",
    "SUDEP",
    "Drug Responsiveness",
    "Electrophysiology",
]

DS_TERMS = [
    "dravet syndrome",
    "severe myoclonic epilepsy of infancy",
    "smei",
    "scn1a-related epileptic encephalopathy",
    "dravet",
]

DS_ABBREVIATIONS = [
    " ds ",
    " ds,",
    " ds.",
    " ds;",
    " ds:",
    "(ds)",
    "[ds]",
]

DS_PATIENT_PATTERNS = [
    "patients with dravet syndrome",
    "children with dravet syndrome",
    "dravet syndrome patients",
    "dravet cohort",
    "dravet group",
    "dravet mouse model",
    "dravet zebrafish model",
    "scn1a dravet",
    "in dravet syndrome",
    "with dravet syndrome",
]

DS_CONTEXT_NEGATIVE_PATTERNS = [
    "such as dravet syndrome",
    "including dravet syndrome",
    "for example dravet syndrome",
    "for example, dravet syndrome",
    "e.g. dravet syndrome",
    "e.g., dravet syndrome",
    "other epilepsies such as dravet syndrome",
    "conditions such as dravet syndrome",
]

SECTION_WEIGHT_MAP = {
    "title": 6.0,
    "abstract": 4.0,
    "methods": 4.5,
    "materials and methods": 4.5,
    "patients and methods": 4.5,
    "results": 4.5,
    "case report": 4.2,
    "case presentation": 4.2,
    "discussion": 2.2,
    "conclusion": 2.2,
    "supplementary": 1.0,
    "background": 1.0,
    "introduction": 0.75,
    "default": 1.5,
}

AXIS_RULES: Dict[str, List[str]] = {
    "Seizures": [
        "seizure", "status epilepticus", "convulsion", "myoclonic", "tonic-clonic",
        "semiology", "febrile seizure", "epileptic spasm",
    ],
    "Genetics": [
        "scn1a", "gene", "genetic", "genotype", "variant", "mutation", "allele",
        "pathogenic", "de novo",
    ],
    "Development": [
        "development", "developmental", "cognitive", "intellectual", "language",
        "learning", "adaptive", "milestone", "neurodevelopment",
    ],
    "Pharmacology": [
        "drug", "therapy", "treatment", "medication", "antiseizure", "anticonvulsant",
        "cannabidiol", "fenfluramine", "stiripentol", "valproate", "clobazam",
    ],
    "Comorbidities": [
        "comorbidity", "sleep", "gait", "ataxia", "gastrointestinal", "autonomic",
        "infection", "pain", "orthopedic", "endocrine",
    ],
    "Behavior": [
        "behavior", "behaviour", "autism", "attention", "adhd", "psychiatric",
        "anxiety", "mood", "aggression", "social",
    ],
    "SUDEP": ["sudep", "sudden unexpected death", "mortality", "death", "fatal"],
    "Drug Responsiveness": [
        "response", "responsive", "resistance", "refractory", "efficacy", "effectiveness",
        "outcome", "retention", "adverse event", "tolerability",
    ],
    "Electrophysiology": [
        "eeg", "electroencephal", "electrophysiolog", "spike", "wave", "recording",
        "channel", "patch clamp", "neuronal excitability",
    ],
}

ARTICLE_TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Systematic Review", ["systematic review", "systematic literature review"]),
    ("Meta-Analysis", ["meta-analysis", "metaanalysis"]),
    ("Review", ["review", "narrative review"]),
    ("Case report", ["case report"]),
    ("Case series", ["case series"]),
    ("Editorial", ["editorial"]),
    ("Letter", ["letter to the editor", "letter"]),
    ("Commentary", ["commentary", "perspective"]),
    ("Opinion", ["opinion"]),
    ("Conference Abstract", ["conference abstract", "meeting abstract", "poster abstract"]),
    ("Original", ["clinical trial", "cohort", "case-control", "cross-sectional", "observational"]),
]

TYPE_RULES: Dict[str, List[str]] = {
    "Review": ["review", "narrative review"],
    "Systematic Review": ["systematic review"],
    "Meta-Analysis": ["meta-analysis", "metaanalysis"],
    "Case report": ["case report"],
    "Case series": ["case series"],
    "Editorial": ["editorial"],
    "Commentary": ["commentary", "perspective"],
    "Letter": ["letter to the editor", "letter"],
    "Opinion": ["opinion"],
    "Conference Abstract": ["conference abstract", "meeting abstract", "poster abstract"],
    "Original": [
        "methods", "results", "participants", "patients and methods",
        "we enrolled", "we analyzed", "we analysed", "study cohort",
        "retrospective", "prospective", "experiment", "mouse model",
        "clinical trial", "observational", "case-control", "cross-sectional",
    ],
}

ORGANISM_RULES: Dict[str, List[str]] = {
    "Zebrafish": ["zebrafish", "danio rerio", "larvae", "embryo", "embryos"],
    "Mouse": ["mouse", "mice", "murine", "mus musculus", "knock-in", "knockout", "scn1a+/−", "scn1a+/-"],
    "Rat": ["rat", "rats", "rattus"],
    "Drosophila": ["drosophila", "fruit fly"],
    "Primate": ["primate", "macaque", "monkey"],
    "Human": [
        "patient", "patients", "child", "children", "adult", "adults",
        "human", "humans", "cohort", "clinical", "retrospective", "prospective",
        "case report", "case series", "participant", "participants",
    ],
}

ORGANISM_STRONG_CONTEXT: Dict[str, List[str]] = {
    "Human": [
        "patients with", "children with", "adults with", "study cohort", "we enrolled",
        "we included", "medical records", "retrospective review", "prospective study",
        "case report", "case series", "participant", "participants",
    ],
    "Mouse": [
        "mouse model", "mice were", "mice received", "murine model", "scn1a mice",
        "knock-in mouse", "knockout mouse", "mutant mice",
    ],
    "Rat": [
        "rats were", "rat model", "rattus",
    ],
    "Zebrafish": [
        "zebrafish model", "zebrafish larvae", "larvae were", "danio rerio",
    ],
    "Drosophila": [
        "drosophila model", "flies were", "fruit fly model",
    ],
    "Primate": [
        "macaque", "monkey model", "primate model",
    ],
}

ORGANISM_NEGATIVE_CONTEXT = [
    "previous studies in",
    "prior studies in",
    "reported in mice",
    "reported in zebrafish",
    "reported in rats",
    "reported in patients",
    "animal models such as",
    "models such as",
]

ENABLE_GEMINI_CLASSIFIER = os.getenv("KROMA_ENABLE_GEMINI_CLASSIFIER", "false").lower() == "true"
GEMINI_CLASSIFIER_MODEL = os.getenv("KROMA_GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash")
CLASSIFIER_LOW_CONFIDENCE_THRESHOLD = 0.60
FULLTEXT_SNIPPET_LIMIT = 12000


class PMCPipelineError(Exception):
    pass


@dataclass
class IngestResult:
    pmcid: int
    created: bool
    updated: bool
    skipped_duplicate: bool
    has_pdf: bool
    has_xml: bool
    article_type: str
    ds: Optional[str]
    organism: str
    axis: str


def _ncbi_params(extra: Optional[dict] = None) -> dict:
    params = {"tool": NCBI_TOOL, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    if extra:
        params.update(extra)
    return params


def _safe_int(value: Optional[str], default: int = 0) -> int:
    text = (value or "").strip()
    return int(text) if text.isdigit() else default


def _clean_text(value: str, max_len: Optional[int] = None) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    if max_len:
        return value[:max_len]
    return value


def _itertext_clean(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return _clean_text("".join(el.itertext()))


def _relpath_under_media(abs_path: Path) -> str:
    return os.path.relpath(abs_path, settings.MEDIA_ROOT).replace("\\", "/")


def _media_path(subdir: str, filename: str) -> Path:
    outdir = Path(settings.MEDIA_ROOT) / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / filename


def _looks_like_pdf(response: requests.Response) -> bool:
    ctype = (response.headers.get("Content-Type") or "").lower()
    return "application/pdf" in ctype or response.content[:4] == b"%PDF"


def _request_xml(url: str, params: dict) -> ET.Element:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    try:
        return ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise PMCPipelineError(f"Could not parse XML from {url}: {exc}") from exc


def search_pmcids(
    term: str,
    mindate: str,
    maxdate: str,
    retmax: int = DEFAULT_RETMAX,
    db: str = "pmc",
) -> List[int]:
    params = _ncbi_params({
        "db": db,
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "mindate": mindate,
        "maxdate": maxdate,
        "datetype": "pdat",
        "sort": "pub date",
    })
    response = requests.get(ESEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    ids = payload.get("esearchresult", {}).get("idlist", [])
    return [_safe_int(x) for x in ids if str(x).isdigit()]


def article_exists(pmcid: int, db_alias: str = DEFAULT_DB_ALIAS) -> bool:
    return Article.objects.using(db_alias).filter(pmcid=pmcid).exists()


def fetch_article_xml_root(pmcid: int) -> ET.Element:
    params = _ncbi_params({"db": "pmc", "id": str(pmcid), "retmode": "xml"})
    return _request_xml(EFETCH_URL, params)


def _extract_section_title(sec: ET.Element) -> str:
    for path in ["./title", ".//title"]:
        el = sec.find(path)
        if el is not None:
            title = _itertext_clean(el).lower()
            if title:
                return title
    return ""


def _normalize_section_label(label: str) -> str:
    text = (label or "").strip().lower()
    if not text:
        return "default"

    mappings = [
        ("materials and methods", "materials and methods"),
        ("patients and methods", "patients and methods"),
        ("methods", "methods"),
        ("method", "methods"),
        ("results", "results"),
        ("result", "results"),
        ("discussion", "discussion"),
        ("conclusion", "conclusion"),
        ("background", "background"),
        ("introduction", "introduction"),
        ("case report", "case report"),
        ("case presentation", "case presentation"),
        ("supplementary", "supplementary"),
    ]
    for needle, label_norm in mappings:
        if needle in text:
            return label_norm
    return "default"


def _extract_weighted_sections(root: ET.Element) -> List[Tuple[str, str, float]]:
    sections: List[Tuple[str, str, float]] = []

    title = _itertext_clean(root.find(".//article-title"))
    if title:
        sections.append(("title", title, SECTION_WEIGHT_MAP["title"]))

    abstract_nodes = root.findall(".//abstract")
    abstract_text = _clean_text(" ".join(_itertext_clean(el) for el in abstract_nodes if _itertext_clean(el)))
    if abstract_text:
        sections.append(("abstract", abstract_text, SECTION_WEIGHT_MAP["abstract"]))

    body = root.find(".//body")
    if body is None:
        return sections

    sec_nodes = body.findall("./sec")
    if not sec_nodes:
        body_text = _itertext_clean(body)
        if body_text:
            sections.append(("default", body_text, SECTION_WEIGHT_MAP["default"]))
        return sections

    for sec in sec_nodes:
        section_title = _extract_section_title(sec)
        section_label = _normalize_section_label(section_title)
        section_text = _itertext_clean(sec)
        if section_text:
            sections.append((section_label, section_text, SECTION_WEIGHT_MAP.get(section_label, SECTION_WEIGHT_MAP["default"])))

    return sections


def fetch_article_metadata(pmcid: int) -> dict:
    root = fetch_article_xml_root(pmcid)

    title = _itertext_clean(root.find(".//article-title"))
    abstract_parts = [_itertext_clean(el) for el in root.findall(".//abstract")]
    abstract = _clean_text(" ".join(p for p in abstract_parts if p), max_len=100)
    journal = _itertext_clean(root.find(".//journal-title"))

    authors_list: List[str] = []
    for contrib in root.findall('.//contrib[@contrib-type="author"]'):
        surname = _itertext_clean(contrib.find(".//surname"))
        given = _itertext_clean(contrib.find(".//given-names"))
        full_name = _clean_text(" ".join(p for p in [surname, given] if p), max_len=100)
        if full_name:
            authors_list.append(full_name)

    article_ids = {}
    for aid in root.findall(".//article-id"):
        key = (aid.attrib.get("pub-id-type") or "").lower()
        val = _clean_text(aid.text or "")
        if key and val:
            article_ids[key] = val

    pub_date = date(1900, 1, 1)
    for pub in root.findall(".//pub-date"):
        ptype = (pub.attrib.get("pub-type") or "").lower()
        if ptype in {"epub", "ppub", "collection", "pmc-release"}:
            year = _safe_int(_itertext_clean(pub.find("year")), default=1900)
            month = _safe_int(_itertext_clean(pub.find("month")), default=1)
            day = _safe_int(_itertext_clean(pub.find("day")), default=1)
            month = max(1, min(month, 12))
            day = max(1, min(day, 28))
            pub_date = date(year, month, day)
            break

    publication_types = [
        _clean_text("".join(el.itertext())).lower()
        for el in root.findall(".//article-categories//subj-group//subject")
    ]
    article_type_attr = (root.attrib.get("article-type") or "").lower()

    body = root.find(".//body")
    body_text = _itertext_clean(body)
    section_text_map = _extract_weighted_sections(root)

    metadata = {
        "pmcid": pmcid,
        "pmid": _safe_int(article_ids.get("pmid")),
        "title": _clean_text(title, max_len=200),
        "authors": _clean_text(", ".join(authors_list), max_len=100),
        "first_author": _clean_text(authors_list[0] if authors_list else "", max_len=100),
        "journal": _clean_text(journal, max_len=100),
        "year": pub_date.year,
        "date": pub_date,
        "doi": _clean_text(article_ids.get("doi", ""), max_len=100),
        "abstract": abstract,
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/",
        "article_type_attr": article_type_attr,
        "publication_types": publication_types,
        "body_text": body_text,
        "section_text_map": section_text_map,
    }
    return metadata


def download_fulltext_xml(pmcid: int) -> str:
    return download_fulltext_xml_with_fallback(pmcid)

def get_pdf_url_from_oa_service(pmcid: int) -> Optional[str]:
    response = requests.get(
        PMC_OA_URL,
        params={"id": f"PMC{pmcid}"},
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 (email: {NCBI_EMAIL})"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    for link in root.iter("link"):
        if (link.attrib.get("format") or "").lower() == "pdf":
            href = link.attrib.get("href")
            if not href:
                continue
            if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
                href = href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
            return href
    return None


def get_pdf_url_from_article_page(pmcid: int) -> Optional[str]:
    """
    Scrape the PMC article webpage for a PDF link.

    This is needed because some PMC articles have a free PDF on the site
    even when the OA service does not provide a usable PDF URL.
    """
    article_url = get_pmc_article_page_url(pmcid)

    response = requests.get(
        article_url,
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 (email: {NCBI_EMAIL})"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Look for links that contain '/pdf/' or end with '.pdf'
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if "/pdf/" in href or href.lower().endswith(".pdf"):
            if href.startswith("/"):
                return f"https://pmc.ncbi.nlm.nih.gov{href}"
            if href.startswith("http://") or href.startswith("https://"):
                return href

    return None


def get_oa_package_url_from_oa_service(pmcid: int) -> Optional[str]:
    """
    Returns the PMC OA package URL (usually .tar.gz) when available.
    This package often contains the full NXML/JATS article, which is better
    than EFetch when EFetch only returns abstract-level XML.
    """
    response = requests.get(
        PMC_OA_URL,
        params={"id": f"PMC{pmcid}"},
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 (email: {NCBI_EMAIL})"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)

    # Prefer tgz/tar.gz package if present
    for link in root.iter("link"):
        href = link.attrib.get("href") or ""
        if not href:
            continue

        if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            href = href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")

        lowered = href.lower()
        if lowered.endswith(".tar.gz") or lowered.endswith(".tgz"):
            return href

    return None


def get_pmc_article_page_url(pmcid: int) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"


def xml_contains_body_text(xml_bytes: bytes) -> bool:
    """
    Returns True only if the XML has a non-trivial <body> section.
    """
    try:
        root = ET.fromstring(xml_bytes)
        body = root.find(".//body")
        if body is None:
            return False
        body_text = " ".join("".join(body.itertext()).split())
        return len(body_text) >= 500
    except Exception:
        return False


def download_oa_fulltext_xml_from_package(pmcid: int) -> str:
    """
    Download the PMC OA package and extract the main .nxml/.xml file.
    Saves it under MEDIA_ROOT/fulltext_xml/{pmcid}.xml
    Returns the relative path under MEDIA_ROOT.
    """
    package_url = get_oa_package_url_from_oa_service(pmcid)
    if not package_url:
        raise FileNotFoundError(f"No OA package URL found for PMCID {pmcid}")

    response = requests.get(
        package_url,
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 (email: {NCBI_EMAIL})"},
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()

    xml_bytes = None

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        members = tar.getmembers()

        # Prefer .nxml first, then .xml
        candidate_names = [
            m.name for m in members
            if m.isfile() and (m.name.lower().endswith(".nxml") or m.name.lower().endswith(".xml"))
        ]

        if not candidate_names:
            raise FileNotFoundError(f"No XML/NXML found inside OA package for PMCID {pmcid}")

        preferred = None
        for name in candidate_names:
            lowered = name.lower()
            if lowered.endswith(".nxml"):
                preferred = name
                break
        if preferred is None:
            preferred = candidate_names[0]

        member = tar.getmember(preferred)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract XML/NXML from OA package for PMCID {pmcid}")

        xml_bytes = extracted.read()

    if not xml_bytes or not xml_contains_body_text(xml_bytes):
        raise ValueError(f"OA package XML for PMCID {pmcid} does not contain usable full body text")

    xml_path = _media_path(XML_SUBDIR, f"{pmcid}.xml")
    xml_path.write_bytes(xml_bytes)
    return _relpath_under_media(xml_path)


def download_fulltext_xml_with_fallback(pmcid: int) -> str:
    """
    Preferred strategy:
    1) Try PMC OA package full XML/NXML
    2) Fall back to EFetch XML
    3) If resulting XML is only abstract-like, keep it only as a weak fallback
       and rely on PDF for real full-text extraction
    """
    # Try OA package first
    try:
        rel_path = download_oa_fulltext_xml_from_package(pmcid)
        abs_path = Path(settings.MEDIA_ROOT) / rel_path
        if xml_has_usable_fulltext(abs_path):
            return rel_path
    except Exception:
        pass

    # Fall back to EFetch XML
    params = _ncbi_params({"db": "pmc", "id": str(pmcid), "retmode": "xml"})
    response = requests.get(EFETCH_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    xml_path = _media_path(XML_SUBDIR, f"{pmcid}.xml")
    xml_path.write_bytes(response.content)

    return _relpath_under_media(xml_path)


def extract_text_from_pdf(pdf_abs_path: str) -> str:
    text_parts = []
    with open(pdf_abs_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text_parts).strip()

def download_pdf_if_available(pmcid: int) -> str:
    """
    Try PDF download in this order:
    1) OA service PDF
    2) PMC article page PDF link
    """
    pdf_url = None

    # First try OA service
    try:
        pdf_url = get_pdf_url_from_oa_service(pmcid)
    except Exception:
        pdf_url = None

    # If OA service fails, scrape the actual PMC article page
    if not pdf_url:
        try:
            pdf_url = get_pdf_url_from_article_page(pmcid)
        except Exception:
            pdf_url = None

    if not pdf_url:
        return ""

    try:
        response = requests.get(
            pdf_url,
            headers={
                "User-Agent": f"{NCBI_TOOL}/1.0 (email: {NCBI_EMAIL})",
                "Accept": "application/pdf,*/*",
            },
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        if response.status_code == 404:
            return ""

        response.raise_for_status()

        if not _looks_like_pdf(response):
            return ""

        pdf_path = _media_path(PDF_SUBDIR, f"{pmcid}.pdf")
        pdf_path.write_bytes(response.content)
        return _relpath_under_media(pdf_path)

    except requests.RequestException:
        return ""


def abs_media_path(rel_path: str) -> str:
    rel_path = (rel_path or "").replace("\\", "/")
    return os.path.join(settings.MEDIA_ROOT, rel_path)


def jats_xml_to_text(xml_abs_path: str, body_only: bool = True) -> str:
    tree = ET.parse(xml_abs_path)
    root = tree.getroot()

    if body_only:
        body = root.find(".//body")
        if body is not None:
            return " ".join("".join(body.itertext()).split())

    return " ".join("".join(root.itertext()).split())


def xml_has_usable_fulltext(xml_path: Path) -> bool:
    """
    Return True only if the XML has substantial body text.
    Abstract-only XML should return False.
    """
    try:
        root = ET.parse(xml_path).getroot()
        body = root.find(".//body")
        if body is None:
            return False

        body_text = " ".join("".join(body.itertext()).split())
        return len(body_text) >= 1000
    except Exception:
        return False


def get_article_fulltext_text(article) -> str:
    xml_rel_path = getattr(article, "fulltext_path", "") or ""
    if not xml_rel_path:
        return ""

    xml_abs_path = abs_media_path(xml_rel_path)
    if not os.path.exists(xml_abs_path):
        return ""

    try:
        return jats_xml_to_text(xml_abs_path, body_only=True)
    except Exception:
        return ""


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _clip_text(text: str, limit: int = FULLTEXT_SNIPPET_LIMIT) -> str:
    text = text or ""
    return text[:limit]


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    text = _normalize_text(text)
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score


def _choose_best_scored_label(score_map: Dict[str, float], default: str = "") -> Tuple[str, float]:
    if not score_map:
        return default, 0.0

    best_label = default
    best_score = -1.0
    total = 0.0

    for label, score in score_map.items():
        total += max(score, 0.0)
        if score > best_score:
            best_label = label
            best_score = score

    if best_score <= 0:
        return default, 0.0

    confidence = best_score / max(total, best_score)
    confidence = round(min(max(confidence, 0.0), 1.0), 3)
    return best_label, confidence


def _contains_any(text: str, patterns: List[str]) -> bool:
    text = _normalize_text(text)
    return any(p in text for p in patterns)


def _count_ds_mentions(text: str) -> int:
    text = f" {_normalize_text(text)} "
    count = 0
    for term in DS_TERMS:
        count += text.count(term)
    if count > 0:
        for abbr in DS_ABBREVIATIONS:
            count += text.count(abbr)
    return count


def _weighted_section_score_for_terms(
    section_text_map: List[Tuple[str, str, float]],
    terms: List[str],
    section_bonus_patterns: Optional[List[str]] = None,
    negative_patterns: Optional[List[str]] = None,
) -> float:
    score = 0.0
    for section_label, section_text, weight in section_text_map:
        norm_text = _normalize_text(section_text)
        mention_hits = sum(norm_text.count(term) for term in terms)
        if mention_hits:
            score += min(mention_hits, 4) * weight
        if section_bonus_patterns and _contains_any(norm_text, section_bonus_patterns):
            score += weight * 1.75
        if negative_patterns and _contains_any(norm_text, negative_patterns):
            score -= weight * 1.1
        if section_label in {"methods", "materials and methods", "patients and methods", "results", "case report", "case presentation"}:
            if mention_hits:
                score += weight * 0.6
    return round(score, 3)


def _detect_ds_related(section_text_map: List[Tuple[str, str, float]], combined_text: str) -> Tuple[str, float, float]:
    combined_text = _normalize_text(combined_text)
    raw_mentions = _count_ds_mentions(combined_text)
    section_score = _weighted_section_score_for_terms(
        section_text_map,
        DS_TERMS,
        section_bonus_patterns=DS_PATIENT_PATTERNS,
        negative_patterns=DS_CONTEXT_NEGATIVE_PATTERNS,
    )

    title_text = next((text for label, text, _ in section_text_map if label == "title"), "")
    abstract_text = next((text for label, text, _ in section_text_map if label == "abstract"), "")
    methods_results_text = " ".join(
        text for label, text, _ in section_text_map
        if label in {"methods", "materials and methods", "patients and methods", "results", "case report", "case presentation"}
    )

    if _contains_any(title_text, DS_TERMS):
        section_score += 4.0
    if _count_ds_mentions(abstract_text) >= 2:
        section_score += 2.5
    elif _count_ds_mentions(abstract_text) == 1:
        section_score += 1.0
    if _count_ds_mentions(methods_results_text) >= 1:
        section_score += 3.0
    if _contains_any(methods_results_text, DS_PATIENT_PATTERNS):
        section_score += 2.5
    if raw_mentions >= 5:
        section_score += 1.2
    elif raw_mentions == 0:
        section_score -= 1.0

    if section_score >= 8.0:
        return "Yes", 0.95, section_score
    if section_score >= 5.0:
        return "Yes", 0.82, section_score
    if section_score >= 3.5 and _contains_any(title_text, DS_TERMS):
        return "Yes", 0.72, section_score
    return "No", 0.80 if section_score <= 1.5 else 0.62, section_score


def _detect_organism(section_text_map: List[Tuple[str, str, float]], ds_value: str) -> Tuple[str, float, float]:
    organism_scores: Dict[str, float] = defaultdict(float)

    for organism, keywords in ORGANISM_RULES.items():
        score = _weighted_section_score_for_terms(
            section_text_map,
            keywords,
            section_bonus_patterns=ORGANISM_STRONG_CONTEXT.get(organism, []),
            negative_patterns=ORGANISM_NEGATIVE_CONTEXT,
        )
        if score > 0:
            organism_scores[organism] += score

    # Penalize generic human mentions in non-DS articles.
    if ds_value != "Yes" and organism_scores.get("Human", 0) > 0:
        organism_scores["Human"] = max(0.0, organism_scores["Human"] - 2.0)

    label, confidence = _choose_best_scored_label(dict(organism_scores), default="")
    best_score = organism_scores.get(label, 0.0) if label else 0.0

    if best_score >= 7.0:
        return label, max(confidence, 0.90), best_score
    if best_score >= 4.0:
        return label, max(confidence, 0.78), best_score
    return "", 0.0, best_score


def _detect_axis(combined_text: str) -> Tuple[str, float]:
    score_map = {label: _count_keyword_hits(combined_text, kws) for label, kws in AXIS_RULES.items()}
    label, confidence = _choose_best_scored_label(score_map, default="")
    return label, confidence


def _detect_article_type(title_abstract_text: str, fulltext_text: str) -> Tuple[str, float]:
    combined = f"{title_abstract_text}\n{_clip_text(fulltext_text, 5000)}"
    score_map = {label: _count_keyword_hits(combined, kws) for label, kws in TYPE_RULES.items()}
    label, confidence = _choose_best_scored_label(score_map, default="Original")
    if not label:
        return "Original", 0.40
    return label, confidence


def classify_article_fields_from_xml(metadata: dict, fulltext_text: str) -> dict:
    title = metadata.get("title", "") or ""
    abstract = metadata.get("abstract", "") or ""
    combined_title_abstract = f"{title}\n{abstract}"
    combined_text = f"{title}\n{abstract}\n{_clip_text(fulltext_text)}"
    section_text_map = metadata.get("section_text_map") or [
        ("title", title, SECTION_WEIGHT_MAP["title"]),
        ("abstract", abstract, SECTION_WEIGHT_MAP["abstract"]),
        ("default", _clip_text(fulltext_text), SECTION_WEIGHT_MAP["default"]),
    ]

    ds_value, ds_conf, ds_score = _detect_ds_related(section_text_map, combined_text)
    org_value, org_conf, org_score = _detect_organism(section_text_map, ds_value)
    axis_value, axis_conf = _detect_axis(combined_text)
    type_value, type_conf = _detect_article_type(combined_title_abstract, fulltext_text)

    if ds_value != "Yes":
        if org_conf < 0.90:
            org_value = ""
        if axis_conf < 0.85:
            axis_value = ""

    return {
        "type": type_value or "Original",
        "ds": ds_value,
        "organism": org_value,
        "axis": axis_value,
        "confidence": {
            "type": type_conf,
            "ds": ds_conf,
            "organism": org_conf,
            "axis": axis_conf,
        },
        "scores": {
            "ds": ds_score,
            "organism": org_score,
        },
    }


def maybe_refine_with_gemini(metadata: dict, fulltext_text: str, current: dict) -> dict:
    if not ENABLE_GEMINI_CLASSIFIER:
        return current

    confidence = current.get("confidence", {})
    low_confidence = any(
        confidence.get(field, 0.0) < CLASSIFIER_LOW_CONFIDENCE_THRESHOLD
        for field in ["type", "ds", "organism", "axis"]
    )
    if not low_confidence:
        return current

    if genai is None:
        return current

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return current

    snippet = _clip_text(fulltext_text, FULLTEXT_SNIPPET_LIMIT)

    prompt = f"""
You are classifying a biomedical article into a fixed schema.

Return ONLY valid JSON with these exact keys:
"type", "ds", "organism", "axis"

Rules:
- "type" must be one of:
  ["Original", "Review", "Systematic Review", "Meta-Analysis", "Case report", "Case series", "Editorial", "Commentary", "Letter", "Opinion", "Conference Abstract"]
- "ds" must be "Yes" or "No"
- "organism" must be one of:
  ["Human", "Mouse", "Rat", "Zebrafish", "Primate", "Drosophila", ""]
- "axis" must be one of:
  ["Seizures", "Genetics", "Development", "Pharmacology", "Comorbidities", "Behavior", "SUDEP", "Drug Responsiveness", "Electrophysiology", ""]

Title:
{metadata.get("title", "")}

Abstract:
{metadata.get("abstract", "")}

Full text snippet:
{snippet}

Current rule-based classification:
{json.dumps({
    "type": current.get("type", ""),
    "ds": current.get("ds", ""),
    "organism": current.get("organism", ""),
    "axis": current.get("axis", ""),
}, ensure_ascii=False)}
""".strip()

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_CLASSIFIER_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        parsed = json.loads(text)

        out = dict(current)
        out["type"] = parsed.get("type", out.get("type", "Original")) or "Original"
        out["ds"] = parsed.get("ds", out.get("ds", "No")) or "No"
        out["organism"] = parsed.get("organism", out.get("organism", "")) or ""
        out["axis"] = parsed.get("axis", out.get("axis", "")) or ""

        out["confidence"] = {
            "type": max(current.get("confidence", {}).get("type", 0.0), 0.90),
            "ds": max(current.get("confidence", {}).get("ds", 0.0), 0.90),
            "organism": max(current.get("confidence", {}).get("organism", 0.0), 0.90),
            "axis": max(current.get("confidence", {}).get("axis", 0.0), 0.90),
        }
        return out

    except Exception:
        return current


def _infer_article_type(metadata: dict) -> str:
    candidates = " ".join([
        metadata.get("article_type_attr", ""),
        " ".join(metadata.get("publication_types", [])),
        metadata.get("title", ""),
        metadata.get("abstract", ""),
    ]).lower()

    if "research-article" in candidates or "original" in candidates:
        return "Original"

    for label, keywords in ARTICLE_TYPE_KEYWORDS:
        if any(keyword in candidates for keyword in keywords):
            return label
    return "Original"


def _infer_ds_related(metadata: dict) -> str:
    haystack = " ".join([
        metadata.get("title", ""),
        metadata.get("abstract", ""),
        metadata.get("body_text", "")[:5000],
    ]).lower()
    return "Yes" if any(term in haystack for term in DS_TERMS) else "No"


def _infer_organism(metadata: dict, ds_value: str) -> str:
    haystack = " ".join([
        metadata.get("title", ""),
        metadata.get("abstract", ""),
        metadata.get("body_text", "")[:6000],
    ]).lower()

    matches: List[str] = []
    for label, keywords in ORGANISM_RULES.items():
        if any(keyword in haystack for keyword in keywords):
            matches.append(label)

    if not matches and ds_value == "Yes":
        matches.append("Human")

    if not matches:
        return "Unknown"

    ordered = []
    seen = set()
    for item in matches:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ", ".join(ordered)[:200]


def _infer_axis(metadata: dict, ds_value: str) -> str:
    if ds_value != "Yes":
        return "Other"

    haystack = " ".join([
        metadata.get("title", ""),
        metadata.get("abstract", ""),
        metadata.get("body_text", "")[:8000],
    ]).lower()

    scores = {}
    for axis, keywords in AXIS_RULES.items():
        scores[axis] = sum(1 for keyword in keywords if keyword in haystack)

    best_axis = max(scores.items(), key=lambda kv: kv[1])
    return best_axis[0] if best_axis[1] > 0 else "Comorbidities"


def classify_article_fields(metadata: dict) -> dict:
    ds_value = _infer_ds_related(metadata)
    article_type = _infer_article_type(metadata)
    organism = _infer_organism(metadata, ds_value)
    axis = _infer_axis(metadata, ds_value)
    return {
        "type": article_type,
        "ds": ds_value,
        "organism": organism,
        "axis": axis,
    }


def ingest_article(
    pmcid: int,
    db_alias: str = DEFAULT_DB_ALIAS,
    overwrite_existing: bool = False,
    download_pdf: bool = True,
) -> IngestResult:
    existing = Article.objects.using(db_alias).filter(pmcid=pmcid).first()
    print(f"[PMC INGEST] pmcid={pmcid} existing_is_none={existing is None} overwrite_existing={overwrite_existing}")

    if existing is not None and overwrite_existing is False:
        existing_pdf_path = (getattr(existing, "pdf_path", "") or "").strip()
        existing_fulltext_path = (getattr(existing, "fulltext_path", "") or "").strip()

        return IngestResult(
            pmcid=pmcid,
            created=False,
            updated=False,
            skipped_duplicate=True,
            has_pdf=existing_pdf_path.lower().endswith(".pdf"),
            has_xml=bool(existing_fulltext_path),
            article_type=(getattr(existing, "type", "") or ""),
            ds=getattr(existing, "ds", None),
            organism=(getattr(existing, "organism", "") or ""),
            axis=(getattr(existing, "axis", "") or ""),
        )

    metadata = fetch_article_metadata(pmcid)

    xml_rel_path = download_fulltext_xml(pmcid)
    pdf_rel_path = download_pdf_if_available(pmcid) if download_pdf else ""

    fulltext_text = ""
    if xml_rel_path:
        try:
            xml_abs_path = abs_media_path(xml_rel_path)
            if os.path.exists(xml_abs_path):
                fulltext_text = jats_xml_to_text(xml_abs_path, body_only=True)
        except Exception:
            fulltext_text = ""

    inferred = classify_article_fields(metadata)
    refined = classify_article_fields_from_xml(metadata, fulltext_text)
    refined = maybe_refine_with_gemini(metadata, fulltext_text, refined)

    print(
        f"[CLASSIFY] pmcid={pmcid} "
        f"type={refined['type']} ds={refined['ds']} "
        f"organism={refined['organism']} axis={refined['axis']} "
        f"conf={refined['confidence']} scores={refined.get('scores', {})}"
    )

    defaults = {
        "pmid": metadata["pmid"],
        "title": metadata["title"],
        "authors": metadata["authors"],
        "first_author": metadata["first_author"],
        "journal": metadata["journal"],
        "year": metadata["year"],
        "date": metadata["date"],
        "doi": metadata["doi"],
        "organism": refined["organism"] or inferred.get("organism", ""),
        "url": metadata["url"],
        "type": refined["type"] or inferred.get("type", "Original"),
        "ds": refined["ds"] or inferred.get("ds", "No"),
        "pdf_path": pdf_rel_path or "",
        "fulltext_path": xml_rel_path or "",
        "fulltext_format": "xml" if xml_rel_path else "",
        "axis": refined["axis"] or inferred.get("axis", ""),
        "abstract": metadata["abstract"],
    }

    with transaction.atomic(using=db_alias):
        _, created = Article.objects.using(db_alias).update_or_create(
            pmcid=pmcid,
            defaults=defaults,
        )

    return IngestResult(
        pmcid=pmcid,
        created=created,
        updated=not created,
        skipped_duplicate=False,
        has_pdf=bool(pdf_rel_path),
        has_xml=bool(xml_rel_path),
        article_type=refined["type"],
        ds=refined["ds"],
        organism=refined["organism"],
        axis=refined["axis"],
    )


def ingest_search_results(
    term: str = DEFAULT_SEARCH_TERM,
    mindate: str = "2026/01/01",
    maxdate: str = "2026/12/31",
    retmax: int = DEFAULT_RETMAX,
    db_alias: str = DEFAULT_DB_ALIAS,
    overwrite_existing: bool = False,
    download_pdf: bool = True,
) -> dict:
    pmcids = search_pmcids(term=term, mindate=mindate, maxdate=maxdate, retmax=retmax, db="pmc")

    created = 0
    updated = 0
    skipped = 0
    errors: List[dict] = []
    ingested: List[dict] = []

    for pmcid in pmcids:
        try:
            result = ingest_article(
                pmcid=pmcid,
                db_alias=db_alias,
                overwrite_existing=overwrite_existing,
                download_pdf=download_pdf,
            )
            ingested.append({
                "pmcid": result.pmcid,
                "created": result.created,
                "updated": result.updated,
                "skipped_duplicate": result.skipped_duplicate,
                "has_pdf": result.has_pdf,
                "has_xml": result.has_xml,
                "type": result.article_type,
                "ds": result.ds,
                "organism": result.organism,
                "axis": result.axis,
            })
            if result.skipped_duplicate:
                skipped += 1
            elif result.created:
                created += 1
            else:
                updated += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({
                "pmcid": pmcid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        time.sleep(SYNC_DELAY_SECONDS)

    return {
        "query": {"term": term, "mindate": mindate, "maxdate": maxdate, "retmax": retmax},
        "counts": {
            "found": len(pmcids),
            "created": created,
            "updated": updated,
            "skipped_duplicates": skipped,
            "errors": len(errors),
        },
        "items": ingested,
        "errors": errors,
    }
