from __future__ import annotations

import csv
import json
import os
import re
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET

from django.conf import settings

from DSapp.kg_compact import build_compact_kg_files
from DSapp.models import Article

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
except Exception:
    genai = None
    GenerateContentConfig = None


KG_CSV_FILENAME = "kg_triples.csv"
REFERENCE_CSV_FILENAME = "kg_triples_references.csv"

GEMINI_TRIPLE_MODEL = os.getenv("KROMA_TRIPLE_MODEL", "gemini-2.5-pro")

# Preferred environment variable: DS_ONTOLOGY_CSV_PATH
# Backward-compatible environment variable: ONTOLOGY_CSV_PATH
# Default: MEDIA_ROOT/triples_ontology.csv
ONTOLOGY_CSV_PATH = os.getenv(
    "DS_ONTOLOGY_CSV_PATH",
    os.getenv(
        "ONTOLOGY_CSV_PATH",
        str(Path(settings.MEDIA_ROOT) / "triples_ontology.csv"),
    ),
)

MAX_GEMINI_OUTPUT_TOKENS = int(os.getenv("KROMA_TRIPLE_MAX_OUTPUT_TOKENS", "65536"))
MAX_SECTION_CHARS = int(os.getenv("KROMA_MAX_SECTION_CHARS", "18000"))
MAX_ONTOLOGY_CHARS = int(os.getenv("KROMA_MAX_ONTOLOGY_CHARS", "40000"))

SECTION_TRIPLE_TARGETS = {
    "Introduction": int(os.getenv("KROMA_INTRO_TRIPLE_TARGET", "25")),
    "Methods": int(os.getenv("KROMA_METHODS_TRIPLE_TARGET", "50")),
    "Results": int(os.getenv("KROMA_RESULTS_TRIPLE_TARGET", "80")),
    "Discussion": int(os.getenv("KROMA_DISCUSSION_TRIPLE_TARGET", "50")),
}

MIN_TOTAL_TRIPLES_WARNING = int(os.getenv("KROMA_MIN_TOTAL_TRIPLES_WARNING", "40"))

# Automated QA controls.
# By default, Introduction triples are kept only if they pass all other QA checks.
# Set KROMA_ACCEPT_INTRO_TRIPLES=false if you want to exclude Introduction-derived triples.
ACCEPT_INTRO_TRIPLES = os.getenv("KROMA_ACCEPT_INTRO_TRIPLES", "true").lower() == "true"

# If the same Subject-Predicate-Object is extracted from multiple sections in the
# same article, keep the most article-specific section.
SECTION_PRIORITY = {
    "Results": 4,
    "Methods": 3,
    "Discussion": 2,
    "Introduction": 1,
}

REJECTED_TRIPLES_CSV_FILENAME = "kg_triples_rejected.csv"

MAX_ENTITY_LABEL_CHARS = int(os.getenv("KROMA_MAX_ENTITY_LABEL_CHARS", "120"))
MAX_PREDICATE_LABEL_CHARS = int(os.getenv("KROMA_MAX_PREDICATE_LABEL_CHARS", "90"))
MAX_ENTITY_WORDS = int(os.getenv("KROMA_MAX_ENTITY_WORDS", "12"))
MAX_PREDICATE_WORDS = int(os.getenv("KROMA_MAX_PREDICATE_WORDS", "10"))

GENERIC_ENTITY_LABELS = {
    "study", "this study", "article", "paper", "authors", "researchers",
    "results", "method", "methods", "discussion", "introduction", "patient",
    "patients", "children", "subjects", "participants", "data", "analysis",
}

GENERIC_PREDICATE_LABELS = {
    "is", "are", "was", "were", "has", "have", "had", "shows", "show",
    "demonstrates", "demonstrated", "reports", "reported", "describes",
}

# Lowercase normalized predicate phrase -> canonical relation label.
PREDICATE_NORMALIZATION_MAP = {
    "treated with": "hasTreatment",
    "is treated with": "hasTreatment",
    "can be treated with": "hasTreatment",
    "used to treat": "hasTreatment",
    "therapy for": "hasTreatment",
    "has treatment": "hasTreatment",
    "preferred medication": "hasPreferredMedication",
    "has preferred medication": "hasPreferredMedication",
    "associated with": "isAssociatedWith",
    "is associated with": "isAssociatedWith",
    "association with": "isAssociatedWith",
    "linked to": "isAssociatedWith",
    "related to": "isAssociatedWith",
    "causes": "causes",
    "leads to": "causes",
    "results in": "causes",
    "contributes to": "contributesTo",
    "increases": "increases",
    "increased": "increases",
    "decreases": "decreases",
    "decreased": "decreases",
    "reduces": "reduces",
    "reduced": "reduces",
    "improves": "improves",
    "improved": "improves",
    "worsens": "worsens",
    "worsened": "worsens",
    "prevents": "prevents",
    "prevented": "prevents",
    "has phenotype": "hasPhenotype",
    "presents with": "hasPhenotype",
    "characterized by": "hasPhenotype",
    "has symptom": "hasSymptom",
    "has comorbidity": "hasComorbidity",
    "has mutation": "hasVariant",
    "has variant": "hasVariant",
    "involves": "involves",
    "affects": "affects",
    "modulates": "modulates",
    "targets": "targets",
    "measured by": "isMeasuredBy",
    "assessed by": "isAssessedBy",
    "evaluated by": "isAssessedBy",
    "has outcome": "hasOutcome",
    "has adverse event": "hasAdverseEvent",
    "has side effect": "hasAdverseEvent",
}

TEXT_ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def _media_path(filename: str) -> Path:
    return Path(settings.MEDIA_ROOT) / filename


def _normalize_text(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_section(value: str) -> str:
    value = _normalize_text(value).lower()
    mapping = {
        "introduction": "Introduction",
        "background": "Introduction",
        "intro": "Introduction",
        "overview": "Introduction",
        "methods": "Methods",
        "method": "Methods",
        "materials and methods": "Methods",
        "patients and methods": "Methods",
        "subjects and methods": "Methods",
        "experimental procedures": "Methods",
        "study design": "Methods",
        "results": "Results",
        "findings": "Results",
        "discussion": "Discussion",
        "conclusion": "Discussion",
        "conclusions": "Discussion",
        "discussion and conclusion": "Discussion",
        "discussion and conclusions": "Discussion",
    }
    return mapping.get(value, "")


def _normalize_triple_value(value: str) -> str:
    value = _normalize_text(value)
    value = value.strip("|")
    return value


def _required_text(article: Article, field_name: str) -> str:
    value = getattr(article, field_name, "") or ""
    return str(value).strip()


def read_text_with_fallback(path: str | Path, max_chars: Optional[int] = None) -> str:
    """
    Read a text file using common fallback encodings.

    This prevents UnicodeDecodeError when CSV/XML-like text files were saved
    from Excel, Windows, or external sources using cp1252/latin-1 rather than
    strict UTF-8.
    """
    path = Path(path)
    last_error: Optional[UnicodeDecodeError] = None

    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            text = path.read_text(encoding=encoding)
            return text[:max_chars] if max_chars is not None else text
        except UnicodeDecodeError as exc:
            last_error = exc

    # Final forgiving fallback. This should almost never be reached because
    # latin-1 can decode any byte sequence, but keep it for robustness.
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] if max_chars is not None else text


def open_csv_dict_reader_with_fallback(path: str | Path):
    """
    Return (file_handle, DictReader) with robust decoding.

    Important: do NOT just test the first few KB, because a decode error can
    occur later while csv.DictReader is iterating. We therefore try decoding
    the full file first. If no strict encoding succeeds, we fall back to
    utf-8 with replacement so the extraction job never crashes because an
    old CSV contains a non-UTF-8 character. Caller must close file_handle.
    """
    path = Path(path)

    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            path.read_text(encoding=encoding)  # test full-file decoding
            f = path.open("r", encoding=encoding, newline="")
            return f, csv.DictReader(f)
        except UnicodeDecodeError:
            continue

    # Final forgiving fallback.
    f = path.open("r", encoding="utf-8", errors="replace", newline="")
    return f, csv.DictReader(f)


def get_eligible_articles(
    db_alias: str = "dsai",
    pmcids: Optional[Sequence[int]] = None,
) -> List[Article]:
    """
    Select only articles that are ready for triple extraction:
    - Original study
    - DS-related
    - has downloaded XML full text
    """
    qs = (
        Article.objects.using(db_alias)
        .filter(type="Original", ds="Yes")
        .exclude(fulltext_path__isnull=True)
        .exclude(fulltext_path="")
    )

    if pmcids:
        qs = qs.filter(pmcid__in=list(pmcids))

    return list(qs.order_by("pmcid"))


def get_processed_pmcids() -> set[int]:
    ref_path = _media_path(REFERENCE_CSV_FILENAME)
    if not ref_path.exists():
        return set()

    seen: set[int] = set()
    f = None
    try:
        f, reader = open_csv_dict_reader_with_fallback(ref_path)
        for row in reader:
            pmcid = str(row.get("PMCID", "")).strip()
            if pmcid.isdigit():
                seen.add(int(pmcid))
    finally:
        if f is not None:
            f.close()
    return seen


def load_ontology_csv_text() -> str:
    if not ONTOLOGY_CSV_PATH:
        raise FileNotFoundError(
            "Ontology path is not configured. Either put triples_ontology.csv in MEDIA_ROOT "
            "or set DS_ONTOLOGY_CSV_PATH to the ontology CSV absolute path."
        )

    path = Path(ONTOLOGY_CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Ontology CSV not found at {path}")

    return read_text_with_fallback(path, max_chars=MAX_ONTOLOGY_CHARS)


def article_xml_path(article: Article) -> Path:
    """
    Return the saved JATS XML path for an article.

    Triple extraction requires XML full text. Older database rows may not have
    fulltext_path populated, so those rows should normally be filtered out by
    get_eligible_articles().
    """
    rel = _required_text(article, "fulltext_path").replace("\\", "/")

    if not rel:
        raise FileNotFoundError(
            f"Article {article.pmcid} has no fulltext_path. "
            "Run PMC sync with overwrite_existing=True to download XML first."
        )

    abs_path = Path(settings.MEDIA_ROOT) / rel

    if not abs_path.exists():
        raise FileNotFoundError(
            f"XML file not found for PMCID {article.pmcid}: {abs_path}. "
            "Run PMC sync with overwrite_existing=True to redownload XML."
        )

    return abs_path


def xml_sections_have_substantial_text(sections: Dict[str, str]) -> bool:
    total = sum(len((sections.get(k, "") or "").strip()) for k in ["Introduction", "Methods", "Results", "Discussion"])
    return total >= 1000


def _sec_title(sec: ET.Element) -> str:
    title = sec.find("./title")
    return _normalize_text("".join(title.itertext()) if title is not None else "")


def _sec_text(sec: ET.Element) -> str:
    text = " ".join("".join(sec.itertext()).split())
    return text[:MAX_SECTION_CHARS]


def extract_jats_sections(xml_path: Path) -> Dict[str, str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sections: Dict[str, List[str]] = {
        "Introduction": [],
        "Methods": [],
        "Results": [],
        "Discussion": [],
    }

    body = root.find(".//body")
    if body is None:
        return {k: "" for k in sections}

    def visit_sec(sec: ET.Element, inherited_label: str = ""):
        title = _sec_title(sec).lower()
        label = _normalize_section(title) or inherited_label
        text = _sec_text(sec)
        if label in sections and text:
            sections[label].append(text)
        for child in sec.findall("./sec"):
            visit_sec(child, label)

    top_secs = body.findall("./sec")
    for sec in top_secs:
        visit_sec(sec)

    if not any(sections.values()):
        body_text = " ".join("".join(body.itertext()).split())[:MAX_SECTION_CHARS]
        sections["Discussion"] = [body_text] if body_text else []

    return {label: "\n\n".join(parts)[:MAX_SECTION_CHARS] for label, parts in sections.items()}


def article_pdf_path(article: Article) -> Optional[Path]:
    rel = _required_text(article, "pdf_path").replace("\\", "/")
    if not rel:
        return None

    abs_path = Path(settings.MEDIA_ROOT) / rel
    if abs_path.exists():
        return abs_path
    return None


def extract_pdf_fallback_sections(pdf_path: Path) -> Dict[str, str]:
    """
    Fallback section map when XML full text is unavailable or unusable.
    Since PDFs usually do not preserve section structure reliably,
    we store the whole extracted PDF text under Discussion so the existing
    extraction pipeline can still operate.
    """
    import PyPDF2

    text_parts = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue

    full_text = " ".join(" ".join(text_parts).split())[:MAX_SECTION_CHARS]

    return {
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": full_text,
    }


def build_section_triple_extraction_prompt(
    article: Article,
    ontology_csv_text: str,
    section_label: str,
    section_text: str,
) -> str:
    """
    Build a high-recall, section-specific extraction prompt.

    The previous whole-article prompt often caused Gemini to summarize the
    article into only 1-2 triples. This prompt asks for granular extraction
    from one section at a time and sets an explicit target count.
    """
    title = _required_text(article, "title")
    abstract = _required_text(article, "abstract")
    article_type = _required_text(article, "type")
    organism = _required_text(article, "organism")
    axis = _required_text(article, "axis")
    pmcid = article.pmcid
    target_count = SECTION_TRIPLE_TARGETS.get(section_label, 50)

    return f"""You are extracting biomedical knowledge graph triples from ONE SECTION of a Dravet Syndrome article.

ARTICLE METADATA:
PMCID: {pmcid}
Title: {title}
Abstract: {abstract}
Article type: {article_type}
Article-level organism already determined outside this task: {organism}
Article-level axis already determined outside this task: {axis}

ONTOLOGY CSV:
{ontology_csv_text}

SECTION TO EXTRACT FROM:
{section_label}

TARGET:
Extract as many high-quality factual triples as possible from this section.
Aim for approximately {target_count} triples from this section if the text supports that many.
If the section contains fewer supported facts, return fewer.
If the section contains more than {target_count} meaningful facts, prioritize the most article-specific and clinically/mechanistically important facts.

IMPORTANT EXTRACTION RULES:
1. Do NOT summarize the section into only 1 or 2 triples.
2. Extract granular triples.
3. Split compound statements into multiple triples.
4. Prefer article-specific facts over generic background.
5. Extract facts about disease, phenotype, genotype, treatment, drug response, animal/cell model, electrophysiology, development, behavior, comorbidities, SUDEP, mechanisms, outcomes, and study design when explicitly stated.
6. For Methods, extract triples about study population, intervention, model, assay, measurement, and experimental design.
7. For Results, extract triples about observed findings, associations, response, outcomes, effects, and measured changes.
8. For Discussion, extract triples about article-supported interpretations, limitations, implications, and mechanistic conclusions.
9. For Introduction, extract only DS-relevant background that supports the study rationale. Do not extract broad unrelated background examples.
10. Do NOT output organism as a standalone triple merely because organism is already stored at article level. However, if the section explicitly states an article-specific model/cohort relationship, you may extract that relationship.
11. Do NOT invent facts.
12. Do NOT output duplicates within this section.
13. Do NOT use full sentences as subject, predicate, or object.
14. Use compact biomedical labels, preferably CamelCase or short normalized labels.
15. Predicate must be a relationship phrase, not a sentence.
16. The section field must always be exactly: "{section_label}".
17. Return JSON array only. No markdown. No prose.

GOOD TRIPLE STYLE EXAMPLES:
[
  {{
    "subject": "DravetSyndrome",
    "predicate": "hasTreatment",
    "object": "Fenfluramine",
    "section": "{section_label}"
  }},
  {{
    "subject": "SCN1AVariant",
    "predicate": "isAssociatedWith",
    "object": "DravetSyndrome",
    "section": "{section_label}"
  }},
  {{
    "subject": "KetogenicDietTherapy",
    "predicate": "reduces",
    "object": "SeizureFrequency",
    "section": "{section_label}"
  }}
]

JSON schema:
[
  {{
    "subject": "short label",
    "predicate": "short relation",
    "object": "short label",
    "section": "{section_label}"
  }}
]

SECTION TEXT:
{section_text}
""".strip()


# Backward-compatible wrapper retained in case any other code imports it.
def build_triple_extraction_prompt(article: Article, ontology_csv_text: str, sections: Dict[str, str]) -> str:
    section_blocks = []
    for label in ["Introduction", "Methods", "Results", "Discussion"]:
        text = sections.get(label, "")
        section_blocks.append(f"## {label}\n{text if text else '[EMPTY]'}")
    combined_text = "\n\n".join(section_blocks)
    return build_section_triple_extraction_prompt(
        article=article,
        ontology_csv_text=ontology_csv_text,
        section_label="Discussion",
        section_text=combined_text,
    )


def _parse_json_array(text: str) -> List[dict]:
    text = (text or "").strip()
    if not text:
        return []

    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    m = re.search(r"\[\s*{.*}\s*\]", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def call_gemini_for_triples(prompt: str) -> List[dict]:
    if genai is None:
        raise RuntimeError("google.genai is not available")

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    # Explicitly request JSON and a large output budget. Without this, Gemini
    # can return a very short summary-like response with only 1-2 triples.
    if GenerateContentConfig is not None:
        response = client.models.generate_content(
            model=GEMINI_TRIPLE_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=MAX_GEMINI_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
    else:
        response = client.models.generate_content(
            model=GEMINI_TRIPLE_MODEL,
            contents=prompt,
        )

    return _parse_json_array(getattr(response, "text", "") or "")


def normalize_extracted_triples(items: Iterable[dict]) -> List[dict]:
    normalized: List[dict] = []
    seen = set()

    for item in items:
        subject = _normalize_triple_value(str(item.get("subject", "")))
        predicate = _normalize_triple_value(str(item.get("predicate", "")))
        obj = _normalize_triple_value(str(item.get("object", "")))
        section = _normalize_section(str(item.get("section", "")))

        if not (subject and predicate and obj and section):
            continue

        key = (subject.lower(), predicate.lower(), obj.lower(), section.lower())
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "Subject": subject,
            "Predicate": predicate,
            "Object": obj,
            "Section": section,
        })

    return normalized


def _word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", value or ""))


def _looks_like_sentence(value: str) -> bool:
    value = _normalize_text(value)
    if not value:
        return False
    if any(mark in value for mark in [". ", ";", ":"]):
        return True
    if _word_count(value) > MAX_ENTITY_WORDS:
        return True
    lowered = value.lower()
    sentence_starters = (
        "this study", "the study", "we ", "our ", "these results",
        "the results", "the authors", "it ", "they ", "there ",
    )
    return lowered.startswith(sentence_starters)


def _strip_wrapping_punctuation(value: str) -> str:
    value = _normalize_text(value)
    value = value.strip(" |\t\n\r'\"`*[]{}()")
    return _normalize_text(value)


def _to_camel_like_label(value: str) -> str:
    """Convert simple labels to a compact KG-style label when useful."""
    value = _strip_wrapping_punctuation(value)
    if not value:
        return ""

    if " " not in value and "_" not in value:
        return value

    parts = re.split(r"[\s_]+", value)
    cleaned = []
    for part in parts:
        part = part.strip("-/,;:.()[]{}")
        if not part:
            continue
        if part.isupper() or re.search(r"\d", part):
            cleaned.append(part)
        else:
            cleaned.append(part[:1].upper() + part[1:])
    return "".join(cleaned)


def _normalize_entity_label(value: str) -> str:
    value = _strip_wrapping_punctuation(value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"https?://\S+", "", value).strip()
    return _to_camel_like_label(value)


def _normalize_predicate_label(predicate: str) -> str:
    raw = _strip_wrapping_punctuation(predicate)
    raw = re.sub(r"^(is|are|was|were)\s+", "", raw, flags=re.IGNORECASE)
    raw = _normalize_text(raw)
    lowered = raw.lower()

    if lowered in PREDICATE_NORMALIZATION_MAP:
        return PREDICATE_NORMALIZATION_MAP[lowered]

    if re.search(r"\b(treat|therapy|therapeutic|medication|drug)\b", lowered) and re.search(r"\b(with|for|using|used)\b", lowered):
        return "hasTreatment"
    if re.search(r"\b(associat|link|correlat|relat)\b", lowered):
        return "isAssociatedWith"
    if re.search(r"\b(reduc|decreas|lower)\b", lowered):
        return "reduces"
    if re.search(r"\b(increas|elevat|raise)\b", lowered):
        return "increases"
    if re.search(r"\b(improv)\b", lowered):
        return "improves"
    if re.search(r"\b(worsen|exacerbat)\b", lowered):
        return "worsens"
    if re.search(r"\b(caus|lead|result)\b", lowered):
        return "causes"
    if re.search(r"\b(variant|mutation)\b", lowered):
        return "hasVariant"
    if re.search(r"\b(phenotype|manifest|characteri[sz]ed|present)\b", lowered):
        return "hasPhenotype"
    if re.search(r"\b(comorbid|co-occurr|cooccur)\b", lowered):
        return "hasComorbidity"
    if re.search(r"\b(measur|assess|evaluat)\b", lowered):
        return "isAssessedBy"

    return _to_camel_like_label(raw)


def _qa_rejection_reasons(triple: dict) -> List[str]:
    reasons: List[str] = []
    subject = triple.get("Subject", "")
    predicate = triple.get("Predicate", "")
    obj = triple.get("Object", "")
    section = triple.get("Section", "")

    if not subject:
        reasons.append("missing_subject")
    if not predicate:
        reasons.append("missing_predicate")
    if not obj:
        reasons.append("missing_object")
    if section not in SECTION_PRIORITY:
        reasons.append("invalid_section")
    if section == "Introduction" and not ACCEPT_INTRO_TRIPLES:
        reasons.append("introduction_triples_disabled")

    if subject.lower() in GENERIC_ENTITY_LABELS:
        reasons.append("generic_subject")
    if obj.lower() in GENERIC_ENTITY_LABELS:
        reasons.append("generic_object")
    if predicate.lower() in GENERIC_PREDICATE_LABELS:
        reasons.append("generic_predicate")

    if len(subject) > MAX_ENTITY_LABEL_CHARS:
        reasons.append("subject_too_long")
    if len(obj) > MAX_ENTITY_LABEL_CHARS:
        reasons.append("object_too_long")
    if len(predicate) > MAX_PREDICATE_LABEL_CHARS:
        reasons.append("predicate_too_long")

    if _looks_like_sentence(subject):
        reasons.append("subject_looks_like_sentence")
    if _looks_like_sentence(obj):
        reasons.append("object_looks_like_sentence")
    if _word_count(predicate) > MAX_PREDICATE_WORDS:
        reasons.append("predicate_too_many_words")

    joined = f"{subject} {predicate} {obj}"
    if re.search(r"https?://|www\.|doi:|PMID|PMCID", joined, flags=re.IGNORECASE):
        reasons.append("contains_citation_or_url")
    if any(token in joined for token in ["```", "{", "}", "[EMPTY]"]):
        reasons.append("contains_formatting_artifact")

    if subject and obj and subject.lower() == obj.lower():
        reasons.append("subject_equals_object")

    return reasons


def qa_review_triples(triples: List[dict], pmcid: int) -> Dict[str, object]:
    """
    Normalize and validate triples before permanent KG insertion.

    This automated QA layer rejects malformed/generic/sentence-like triples,
    normalizes predicates to preferred relation labels, optionally rejects
    Introduction triples, and keeps the best section when the same SPO appears
    multiple times in the same article.
    """
    accepted_by_spo: Dict[tuple, dict] = {}
    rejected: List[dict] = []

    for triple in triples:
        normalized = {
            "Subject": _normalize_entity_label(triple.get("Subject", "")),
            "Predicate": _normalize_predicate_label(triple.get("Predicate", "")),
            "Object": _normalize_entity_label(triple.get("Object", "")),
            "Section": _normalize_section(triple.get("Section", "")),
        }

        reasons = _qa_rejection_reasons(normalized)
        if reasons:
            rejected.append({**normalized, "PMCID": pmcid, "Reason": ";".join(reasons)})
            continue

        spo_key = (
            normalized["Subject"].lower(),
            normalized["Predicate"].lower(),
            normalized["Object"].lower(),
        )

        existing = accepted_by_spo.get(spo_key)
        if existing is None:
            accepted_by_spo[spo_key] = normalized
            continue

        existing_priority = SECTION_PRIORITY.get(existing["Section"], 0)
        new_priority = SECTION_PRIORITY.get(normalized["Section"], 0)
        if new_priority > existing_priority:
            rejected.append({**existing, "PMCID": pmcid, "Reason": "lower_priority_duplicate_replaced"})
            accepted_by_spo[spo_key] = normalized
        else:
            rejected.append({**normalized, "PMCID": pmcid, "Reason": "lower_priority_duplicate"})

    accepted = list(accepted_by_spo.values())
    reason_counts: Dict[str, int] = {}
    for row in rejected:
        for reason in str(row.get("Reason", "")).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected_reason_counts": reason_counts,
    }


def append_rejected_triples_to_csv(rejected: List[dict], pmcid: int) -> int:
    if not rejected:
        return 0

    rejected_path = _media_path(REJECTED_TRIPLES_CSV_FILENAME)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rejected = set()
    if rejected_path.exists():
        f = None
        try:
            f, reader = open_csv_dict_reader_with_fallback(rejected_path)
            for row in reader:
                key = (
                    _normalize_triple_value(row.get("Subject", "")).lower(),
                    _normalize_triple_value(row.get("Predicate", "")).lower(),
                    _normalize_triple_value(row.get("Object", "")).lower(),
                    _normalize_section(row.get("Section", "")).lower(),
                    str(row.get("PMCID", "")).strip(),
                    _normalize_text(row.get("Reason", "")).lower(),
                )
                if key[0] and key[1] and key[2] and key[3] and key[4]:
                    existing_rejected.add(key)
        finally:
            if f is not None:
                f.close()

    write_header = not rejected_path.exists()
    added = 0
    with rejected_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "Predicate", "Object", "Section", "PMCID", "Reason"])
        if write_header:
            writer.writeheader()

        for row in rejected:
            key = (
                row.get("Subject", "").lower(),
                row.get("Predicate", "").lower(),
                row.get("Object", "").lower(),
                row.get("Section", "").lower(),
                str(row.get("PMCID", pmcid)),
                _normalize_text(row.get("Reason", "")).lower(),
            )
            if key in existing_rejected:
                continue
            writer.writerow({
                "Subject": row.get("Subject", ""),
                "Predicate": row.get("Predicate", ""),
                "Object": row.get("Object", ""),
                "Section": row.get("Section", ""),
                "PMCID": row.get("PMCID", pmcid),
                "Reason": row.get("Reason", ""),
            })
            existing_rejected.add(key)
            added += 1

    return added


def append_triples_to_csvs(triples: List[dict], pmcid: int) -> Dict[str, int]:
    unique_path = _media_path(KG_CSV_FILENAME)
    ref_path = _media_path(REFERENCE_CSV_FILENAME)

    unique_path.parent.mkdir(parents=True, exist_ok=True)

    existing_unique = set()
    if unique_path.exists():
        f = None
        try:
            f, reader = open_csv_dict_reader_with_fallback(unique_path)
            for row in reader:
                key = (
                    _normalize_triple_value(row.get("Subject", "")).lower(),
                    _normalize_triple_value(row.get("Predicate", "")).lower(),
                    _normalize_triple_value(row.get("Object", "")).lower(),
                )
                if all(key):
                    existing_unique.add(key)
        finally:
            if f is not None:
                f.close()

    existing_refs = set()
    if ref_path.exists():
        f = None
        try:
            f, reader = open_csv_dict_reader_with_fallback(ref_path)
            for row in reader:
                key = (
                    _normalize_triple_value(row.get("Subject", "")).lower(),
                    _normalize_triple_value(row.get("Predicate", "")).lower(),
                    _normalize_triple_value(row.get("Object", "")).lower(),
                    _normalize_section(row.get("Section", "")).lower(),
                    str(row.get("PMCID", "")).strip(),
                )
                if key[0] and key[1] and key[2] and key[3] and key[4]:
                    existing_refs.add(key)
        finally:
            if f is not None:
                f.close()

    write_unique_header = not unique_path.exists()
    write_ref_header = not ref_path.exists()

    unique_added = 0
    refs_added = 0

    # Always write output as UTF-8 so future generated CSVs are consistent.
    with unique_path.open("a", encoding="utf-8", newline="") as uf, ref_path.open("a", encoding="utf-8", newline="") as rf:
        unique_writer = csv.DictWriter(uf, fieldnames=["Subject", "Predicate", "Object"])
        ref_writer = csv.DictWriter(rf, fieldnames=["Subject", "Predicate", "Object", "Section", "PMCID"])

        if write_unique_header:
            unique_writer.writeheader()
        if write_ref_header:
            ref_writer.writeheader()

        for triple in triples:
            unique_key = (
                triple["Subject"].lower(),
                triple["Predicate"].lower(),
                triple["Object"].lower(),
            )
            ref_key = unique_key + (triple["Section"].lower(), str(pmcid))

            if unique_key not in existing_unique:
                unique_writer.writerow({
                    "Subject": triple["Subject"],
                    "Predicate": triple["Predicate"],
                    "Object": triple["Object"],
                })
                existing_unique.add(unique_key)
                unique_added += 1

            if ref_key not in existing_refs:
                ref_writer.writerow({
                    "Subject": triple["Subject"],
                    "Predicate": triple["Predicate"],
                    "Object": triple["Object"],
                    "Section": triple["Section"],
                    "PMCID": pmcid,
                })
                existing_refs.add(ref_key)
                refs_added += 1

    if unique_added or refs_added:
        build_compact_kg_files(force_rebuild=True)

    return {"unique_added": unique_added, "references_added": refs_added}


def extract_triples_for_article(article: Article, ontology_csv_text: str) -> Dict[str, object]:
    """
    Extract triples section-by-section, run automated QA, then append only
    accepted triples to permanent KG CSVs. Rejected triples are written to
    kg_triples_rejected.csv for review/debugging.
    """
    sections = {
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": "",
    }

    xml_path = None
    pdf_path = None

    try:
        xml_path = article_xml_path(article)
        sections = extract_jats_sections(xml_path)
    except Exception:
        xml_path = None

    # If XML is missing, empty, or only abstract-like, fall back to PDF.
    if not xml_sections_have_substantial_text(sections):
        pdf_path = article_pdf_path(article)
        if pdf_path is not None:
            sections = extract_pdf_fallback_sections(pdf_path)

    all_raw_items: List[dict] = []
    section_raw_counts: Dict[str, int] = {}
    section_normalized_counts: Dict[str, int] = {}

    for section_label in ["Introduction", "Methods", "Results", "Discussion"]:
        section_text = (sections.get(section_label, "") or "").strip()

        if not section_text:
            section_raw_counts[section_label] = 0
            section_normalized_counts[section_label] = 0
            continue

        prompt = build_section_triple_extraction_prompt(
            article=article,
            ontology_csv_text=ontology_csv_text,
            section_label=section_label,
            section_text=section_text,
        )

        raw_items = call_gemini_for_triples(prompt)
        section_raw_counts[section_label] = len(raw_items)

        normalized_section_triples = normalize_extracted_triples(raw_items)
        section_normalized_counts[section_label] = len(normalized_section_triples)

        all_raw_items.extend(raw_items)

    raw_normalized_triples = normalize_extracted_triples(all_raw_items)
    qa_result = qa_review_triples(raw_normalized_triples, article.pmcid)
    accepted_triples = qa_result["accepted"]
    rejected_triples = qa_result["rejected"]

    append_counts = append_triples_to_csvs(accepted_triples, article.pmcid)
    rejected_added = append_rejected_triples_to_csv(rejected_triples, article.pmcid)

    if len(accepted_triples) < MIN_TOTAL_TRIPLES_WARNING:
        print(
            f"[TRIPLE WARNING] PMCID {article.pmcid} accepted only {len(accepted_triples)} triples "
            f"after QA. Raw normalized={len(raw_normalized_triples)}. "
            f"Section normalized counts: {section_normalized_counts}. "
            f"Rejected reasons: {qa_result['rejected_reason_counts']}"
        )

    return {
        "pmcid": article.pmcid,
        "raw_triple_count": len(raw_normalized_triples),
        "triple_count": len(accepted_triples),
        "qa_accepted_count": qa_result["accepted_count"],
        "qa_rejected_count": qa_result["rejected_count"],
        "qa_rejected_reasons": qa_result["rejected_reason_counts"],
        "rejected_logged": rejected_added,
        "unique_added": append_counts["unique_added"],
        "references_added": append_counts["references_added"],
        "section_raw_counts": section_raw_counts,
        "section_normalized_counts": section_normalized_counts,
        "triples_preview": accepted_triples[:20],
        "rejected_preview": rejected_triples[:10],
    }


def extract_triples_for_eligible_articles(
    db_alias: str = "dsai",
    limit: Optional[int] = None,
    overwrite_existing_refs: bool = False,
    pmcids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    ontology_csv_text = load_ontology_csv_text()
    articles = get_eligible_articles(db_alias=db_alias, pmcids=pmcids)

    if not overwrite_existing_refs:
        processed = get_processed_pmcids()
        articles = [a for a in articles if a.pmcid not in processed]

    if limit is not None:
        articles = articles[:limit]

    results = []
    errors = []

    for article in articles:
        try:
            results.append(extract_triples_for_article(article, ontology_csv_text))
        except Exception as exc:
            errors.append({
                "pmcid": article.pmcid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

    return {
        "eligible_count": len(articles),
        "processed_count": len(results),
        "error_count": len(errors),
        "items": results,
        "errors": errors,
    }


def parse_pmcid_input(pmcid_text: str) -> List[int]:
    """
    Parse comma/space/newline/semicolon-separated PMCIDs.

    Accepts either numeric IDs (13041439) or prefixed IDs (PMC13041439).
    Invalid tokens are ignored.
    """
    if not pmcid_text:
        return []

    tokens = re.split(r"[\s,;]+", pmcid_text.strip())
    pmcids: List[int] = []
    seen = set()

    for token in tokens:
        cleaned = token.strip().upper()
        if not cleaned:
            continue
        if cleaned.startswith("PMC"):
            cleaned = cleaned[3:]
        cleaned = re.sub(r"\D", "", cleaned)
        if cleaned.isdigit():
            value = int(cleaned)
            if value not in seen:
                seen.add(value)
                pmcids.append(value)

    return pmcids


def extract_triples_manual(
    db_alias: str = "dsai",
    pmcid_text: str = "",
    limit: Optional[int] = None,
    overwrite_existing_refs: bool = False,
) -> Dict[str, object]:
    """
    Manual extraction entry point for Django shell or the web form.

    If pmcid_text is provided, only those PMCIDs are processed. In that case,
    overwrite_existing_refs is commonly set to True when re-running after a
    prompt or QA revision. If pmcid_text is blank, the next eligible articles
    are processed up to the given limit.
    """
    pmcids = parse_pmcid_input(pmcid_text)

    return extract_triples_for_eligible_articles(
        db_alias=db_alias,
        limit=limit if not pmcids else None,
        overwrite_existing_refs=overwrite_existing_refs,
        pmcids=pmcids or None,
    )


def extract_triples_automatic(
    db_alias: str = "dsai",
    limit: int = 20,
) -> Dict[str, object]:
    """
    Automatic incremental extraction entry point for Celery Beat.

    This skips articles that already have rows in kg_triples_references.csv.
    """
    return extract_triples_for_eligible_articles(
        db_alias=db_alias,
        limit=limit,
        overwrite_existing_refs=False,
        pmcids=None,
    )
