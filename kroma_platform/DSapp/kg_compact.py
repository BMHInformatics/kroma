from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from django.conf import settings


KG_CSV_FILENAME = "kg_triples.csv"
REFERENCE_CSV_FILENAME = "kg_triples_references.csv"

NODES_FILENAME = "kg_nodes.tsv"
PREDS_FILENAME = "kg_predicates.tsv"
EDGES_FILENAME = "kg_edges.tsv"
REFS_FILENAME = "kg_refs.tsv"


def _normalize_label(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _paths() -> Dict[str, Path]:
    media_root = Path(settings.MEDIA_ROOT)
    return {
        "kg_csv": media_root / KG_CSV_FILENAME,
        "ref_csv": media_root / REFERENCE_CSV_FILENAME,
        "nodes": media_root / NODES_FILENAME,
        "preds": media_root / PREDS_FILENAME,
        "edges": media_root / EDGES_FILENAME,
        "refs": media_root / REFS_FILENAME,
    }


def build_compact_kg_files(force_rebuild: bool = False) -> Dict[str, Path]:
    """
    Build compact whole-KG TSV files from the lightweight unique KG CSV:

      kg_nodes.tsv       node_id | label
      kg_predicates.tsv  pred_id | label
      kg_edges.tsv       s | p | o
      kg_refs.tsv        placeholder file retained for compatibility

    Source CSV:
      kg_triples.csv
      Columns: Subject, Predicate, Object
    """
    paths = _paths()
    csv_path = paths["kg_csv"]

    if not csv_path.exists():
        raise FileNotFoundError(f"KG CSV not found at {csv_path}")

    target_paths = [paths["nodes"], paths["preds"], paths["edges"], paths["refs"]]

    should_rebuild = force_rebuild or any(not p.exists() for p in target_paths)
    if not should_rebuild:
        csv_mtime = csv_path.stat().st_mtime
        target_oldest = min(p.stat().st_mtime for p in target_paths)
        should_rebuild = csv_mtime > target_oldest

    if not should_rebuild:
        return {
            "nodes": paths["nodes"],
            "predicates": paths["preds"],
            "edges": paths["edges"],
            "refs": paths["refs"],
        }

    df = pd.read_csv(csv_path, dtype=str, encoding="latin-1").fillna("")

    required_cols = {"Subject", "Predicate", "Object"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"KG CSV is missing required columns: {sorted(missing)}")

    for col in ["Subject", "Predicate", "Object"]:
        df[col] = df[col].astype(str).str.strip()

    # Extra safety: deduplicate just in case
    df = df.drop_duplicates(subset=["Subject", "Predicate", "Object"]).reset_index(drop=True)

    unique_nodes = sorted(set(df["Subject"]).union(set(df["Object"])))
    node_to_id = {label: f"n{i+1}" for i, label in enumerate(unique_nodes)}

    unique_preds = sorted(set(df["Predicate"]))
    pred_to_id = {label: f"p{i+1}" for i, label in enumerate(unique_preds)}

    nodes_df = pd.DataFrame({
        "node_id": [node_to_id[label] for label in unique_nodes],
        "label": unique_nodes,
    })
    nodes_df.to_csv(paths["nodes"], sep="\t", index=False)

    preds_df = pd.DataFrame({
        "pred_id": [pred_to_id[label] for label in unique_preds],
        "label": unique_preds,
    })
    preds_df.to_csv(paths["preds"], sep="\t", index=False)

    edges_df = pd.DataFrame({
        "s": df["Subject"].map(node_to_id),
        "p": df["Predicate"].map(pred_to_id),
        "o": df["Object"].map(node_to_id),
    })
    edges_df.to_csv(paths["edges"], sep="\t", index=False)

    # Keep a tiny placeholder file so the existing upload logic still has 4 files if needed
    refs_df = pd.DataFrame({"note": ["Reference lookup now uses kg_triples_references.csv"]})
    refs_df.to_csv(paths["refs"], sep="\t", index=False)

    _load_compact_tables.cache_clear()
    _load_reference_index.cache_clear()

    return {
        "nodes": paths["nodes"],
        "predicates": paths["preds"],
        "edges": paths["edges"],
        "refs": paths["refs"],
    }


def get_compact_kg_signature() -> Tuple[Tuple[str, int, int], ...]:
    files = build_compact_kg_files()
    sig_parts = []
    for path in files.values():
        st = path.stat()
        sig_parts.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(sig_parts)


@lru_cache(maxsize=1)
def _load_compact_tables():
    files = build_compact_kg_files()

    nodes_df = pd.read_csv(files["nodes"], sep="\t", dtype=str).fillna("")
    preds_df = pd.read_csv(files["predicates"], sep="\t", dtype=str).fillna("")
    edges_df = pd.read_csv(files["edges"], sep="\t", dtype=str).fillna("")

    node_label_to_id = {}
    node_id_to_label = {}
    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        label = row["label"]
        norm = _normalize_label(label)
        if norm:
            node_label_to_id[norm] = node_id
        node_id_to_label[node_id] = label

    pred_label_to_id = {}
    pred_id_to_label = {}
    for _, row in preds_df.iterrows():
        pred_id = row["pred_id"]
        label = row["label"]
        norm = _normalize_label(label)
        if norm:
            pred_label_to_id[norm] = pred_id
        pred_id_to_label[pred_id] = label

    edge_set = set()
    edge_rows = []
    for _, row in edges_df.iterrows():
        edge = (row["s"], row["p"], row["o"])
        edge_set.add(edge)
        edge_rows.append(edge)

    return {
        "node_label_to_id": node_label_to_id,
        "node_id_to_label": node_id_to_label,
        "pred_label_to_id": pred_label_to_id,
        "pred_id_to_label": pred_id_to_label,
        "edge_set": edge_set,
        "edge_rows": edge_rows,
    }


@lru_cache(maxsize=1)
def _load_reference_index():
    """
    Load the provenance/reference CSV and build a triple -> supporting rows index.

    Source CSV:
      kg_triples_references.csv
      Columns: Subject, Predicate, Object, Section, Organism, PMCID
    """
    paths = _paths()
    csv_path = paths["ref_csv"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, encoding="latin-1").fillna("")

    required_cols = {"Subject", "Predicate", "Object", "Section", "PMCID"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Reference CSV is missing required columns: {sorted(missing)}")

    for col in ["Subject", "Predicate", "Object", "Section", "PMCID"]:
        df[col] = df[col].astype(str).str.strip()

    triple_to_rows: Dict[Tuple[str, str, str], List[dict]] = {}

    for _, row in df.iterrows():
        key = (
            _normalize_label(row["Subject"]),
            _normalize_label(row["Predicate"]),
            _normalize_label(row["Object"]),
        )
        triple_to_rows.setdefault(key, []).append({
            "Subject": row["Subject"],
            "Predicate": row["Predicate"],
            "Object": row["Object"],
            "Section": row["Section"],
            "PMCID": row["PMCID"],
        })

    return triple_to_rows


def resolve_compact_triples_to_labels(raw_triples_text: str) -> str:
    """
    Converts:
      n123 | p17 | n456
    into:
      SubjectLabel | PredicateLabel | ObjectLabel

    Leaves already-resolved triples unchanged when possible.
    """
    if not raw_triples_text:
        return ""

    tables = _load_compact_tables()
    node_id_to_label = tables["node_id_to_label"]
    pred_id_to_label = tables["pred_id_to_label"]

    resolved_lines = []

    for raw_line in raw_triples_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        s_raw, p_raw, o_raw = parts[0], parts[1], parts[2]

        s_label = node_id_to_label.get(s_raw, s_raw)
        p_label = pred_id_to_label.get(p_raw, p_raw)
        o_label = node_id_to_label.get(o_raw, o_raw)

        resolved_lines.append(f"{s_label} | {p_label} | {o_label}")

    return "\n".join(resolved_lines)



def resolve_reference_pmcids_from_triples(raw_triples_text: str, max_pmcids: int | None = None) -> List[int]:
    """
    Accepts either:
      Subject | Predicate | Object
    or:
      n123 | p17 | n456

    Returns unique PMCIDs from matching provenance rows.

    Preference/order:
      1) PMCIDs from Results-section rows first
      2) Then PMCIDs from all other sections

    Important:
      Earlier versions returned only Results PMCIDs whenever any Results
      row was found. That caused many valid supporting references to be
      discarded. This version keeps Results first but still includes other
      supporting sections.
    """
    if not raw_triples_text:
        return []

    resolved_text = resolve_compact_triples_to_labels(raw_triples_text)
    triple_to_rows = _load_reference_index()

    results_pmcids: List[int] = []
    other_pmcids: List[int] = []
    seen_results = set()
    seen_other = set()

    for raw_line in resolved_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[[") or line.startswith("```"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        key = (
            _normalize_label(parts[0]),
            _normalize_label(parts[1]),
            _normalize_label(parts[2]),
        )

        matching_rows = triple_to_rows.get(key, [])
        if not matching_rows:
            continue

        for row in matching_rows:
            pmcid = row["PMCID"].strip()
            if not pmcid.isdigit():
                continue

            pmcid_int = int(pmcid)
            section = row["Section"].strip().lower()

            if section == "results":
                if pmcid_int not in seen_results:
                    seen_results.add(pmcid_int)
                    results_pmcids.append(pmcid_int)
            else:
                if pmcid_int not in seen_other:
                    seen_other.add(pmcid_int)
                    other_pmcids.append(pmcid_int)

    combined = results_pmcids + [p for p in other_pmcids if p not in seen_results]

    if max_pmcids is not None:
        return combined[:max_pmcids]

    return combined


def compact_ids_from_any_triples(raw_triples_text: str) -> str:
    """
    Convert RAW_TRIPLES into validated compact IDs.

    Accepts either:
      n123 | p17 | n456
    or:
      SubjectLabel | PredicateLabel | ObjectLabel

    Returns only triples that exist in kg_edges.tsv, formatted as:
      n123 | p17 | n456
    """
    if not raw_triples_text:
        return ""

    tables = _load_compact_tables()
    node_label_to_id = tables["node_label_to_id"]
    pred_label_to_id = tables["pred_label_to_id"]
    edge_set = tables["edge_set"]

    validated_lines = []
    seen = set()

    for raw_line in raw_triples_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[[") or line.startswith("```"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        s_raw, p_raw, o_raw = parts[0], parts[1], parts[2]

        # Case 1: already compact IDs
        if re.fullmatch(r"n\d+", s_raw) and re.fullmatch(r"p\d+", p_raw) and re.fullmatch(r"n\d+", o_raw):
            edge = (s_raw, p_raw, o_raw)
        else:
            # Case 2: resolved labels; map them back to compact IDs
            s_id = node_label_to_id.get(_normalize_label(s_raw))
            p_id = pred_label_to_id.get(_normalize_label(p_raw))
            o_id = node_label_to_id.get(_normalize_label(o_raw))

            if not (s_id and p_id and o_id):
                continue

            edge = (s_id, p_id, o_id)

        # Keep only real KG edges
        if edge not in edge_set:
            continue

        if edge not in seen:
            seen.add(edge)
            validated_lines.append(f"{edge[0]} | {edge[1]} | {edge[2]}")

    return "\n".join(validated_lines)


# -------------------------
# Generic disease-agnostic KG retrieval for backend references
# -------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "between",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "in", "into", "is", "it", "its", "may", "more", "most",
    "of", "on", "or", "our", "should", "than", "that", "the", "their", "them",
    "then", "there", "these", "this", "those", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "without",
    "about", "patient", "patients", "child", "children", "caregiver", "caregivers",
    "disease", "disorder", "condition",
}


def _split_camel_and_symbols(text: str) -> str:
    """
    Turn labels such as GeneralizedTonicClonicSeizure or hasComorbidity
    into more searchable text without requiring disease-specific synonyms.
    """
    text = str(text or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return text.lower()


def _canonical_retrieval_token(token: str) -> str:
    """
    Lightweight normalization for retrieval only.

    Important fix: do NOT turn "oligonucleotides" into "oligonucleotid".
    Most biomedical words ending in -ides should simply drop the final s:
      oligonucleotides -> oligonucleotide
      nucleotides -> nucleotide
      peptides -> peptide
    Use -es removal only for common English plural endings such as -ches/-shes/-xes.
    """
    token = (token or "").lower().strip()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize_for_retrieval(text: str) -> List[str]:
    text = _split_camel_and_symbols(text)
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [_canonical_retrieval_token(t) for t in raw_tokens]
    return [
        t for t in tokens
        if len(t) >= 3 and t not in _STOPWORDS
    ]


@lru_cache(maxsize=1)
def _build_retrieval_index():
    """
    Build a lightweight, generic, disease-agnostic searchable index over the
    compact KG. This intentionally uses only local files and Python logic;
    no extra Gemini call is needed.

    The index supports:
      - lexical matching over resolved triple labels
      - simple IDF weighting so rare biomedical terms score higher
      - one-hop graph expansion around the best matched nodes
    """
    tables = _load_compact_tables()
    node_id_to_label = tables["node_id_to_label"]
    pred_id_to_label = tables["pred_id_to_label"]
    edge_rows = tables["edge_rows"]

    docs = []
    df_counter = Counter()

    for edge in edge_rows:
        s, p, o = edge
        s_label = node_id_to_label.get(s, s)
        p_label = pred_id_to_label.get(p, p)
        o_label = node_id_to_label.get(o, o)

        resolved = f"{s_label} | {p_label} | {o_label}"
        searchable = f"{s_label} {p_label} {o_label}"
        tokens = _tokenize_for_retrieval(searchable)
        token_set = set(tokens)

        for tok in token_set:
            df_counter[tok] += 1

        docs.append({
            "edge": edge,
            "compact": f"{s} | {p} | {o}",
            "resolved": resolved,
            "searchable": _split_camel_and_symbols(searchable),
            "tokens": tokens,
            "token_set": token_set,
            "subject_id": s,
            "object_id": o,
        })

    n_docs = max(len(docs), 1)
    idf = {
        tok: math.log((n_docs + 1) / (df + 1)) + 1.0
        for tok, df in df_counter.items()
    }

    node_to_doc_indexes: Dict[str, List[int]] = {}
    edge_to_doc_index: Dict[Tuple[str, str, str], int] = {}
    for idx, doc in enumerate(docs):
        node_to_doc_indexes.setdefault(doc["subject_id"], []).append(idx)
        node_to_doc_indexes.setdefault(doc["object_id"], []).append(idx)
        edge_to_doc_index[doc["edge"]] = idx

    return {
        "docs": docs,
        "idf": idf,
        "node_to_doc_indexes": node_to_doc_indexes,
        "edge_to_doc_index": edge_to_doc_index,
    }


def _score_doc_for_weighted_query(
    doc: dict,
    token_weights: Dict[str, float],
    query_phrases: List[Tuple[str, float]],
    idf: Dict[str, float],
) -> float:
    """
    Score a KG triple against a weighted query.

    User-question terms are intentionally weighted more heavily than answer terms.
    This prevents generic answer wording such as "seizure", "treatment", or
    "clinical" from drowning out the specific concept the user asked about, such
    as "antisense oligonucleotide" or "STK-001".
    """
    if not token_weights:
        return 0.0

    doc_tokens = doc["token_set"]
    overlap = set(token_weights) & doc_tokens
    if not overlap:
        return 0.0

    score = sum(idf.get(tok, 1.0) * token_weights.get(tok, 1.0) for tok in overlap)

    # Reward multi-concept matches in the same triple.
    if len(overlap) >= 2:
        score *= 1.0 + min(len(overlap), 8) * 0.18

    searchable = doc["searchable"]

    # Reward exact phrase occurrence, especially phrases from the user question.
    for phrase, weight in query_phrases:
        if phrase and len(phrase) >= 5 and phrase in searchable:
            score += 3.0 * weight

    # Mildly prefer triples that retain explicit disease context, without making
    # disease-label matches dominate specific biomedical concepts.
    if "dravet" in searchable:
        score += 0.25

    return score


def _weighted_tokens_from_text(text: str, weight: float) -> Counter:
    return Counter({tok: weight for tok in _tokenize_for_retrieval(text)})


def _phrases_from_text(text: str, weight: float) -> List[Tuple[str, float]]:
    cleaned = _split_camel_and_symbols(text)
    phrases: List[Tuple[str, float]] = []

    # N-gram phrases from normalized text. This is disease-agnostic and helps
    # with concepts such as "antisense oligonucleotide", "sudden unexpected death",
    # and future disease-specific terms without a hand-written synonym dictionary.
    tokens = [t for t in re.findall(r"[a-z0-9]+", cleaned) if len(t) >= 3 and t not in _STOPWORDS]
    for n in (2, 3, 4):
        for i in range(0, max(len(tokens) - n + 1, 0)):
            phrase = " ".join(tokens[i:i+n])
            if 5 <= len(phrase) <= 80:
                phrases.append((phrase, weight))

    return phrases


# Generic question-focus extraction for precise reference retrieval.
# These are intentionally disease-agnostic. They protect specific concepts
# such as "antisense oligonucleotide" from being drowned out by broad terms
# such as "Dravet", "seizure", "patient", or "model".
_BROAD_REFERENCE_TOKENS = {
    "dravet", "syndrome", "disease", "disorder", "condition", "patient",
    "patients", "child", "children", "study", "evidence", "article", "paper",
    "research", "clinical", "preclinical", "model", "mouse", "mice", "rat",
    "zebrafish", "human", "seizure", "epilepsy", "epileptic", "severe",
    "early", "year", "life", "first", "prevent", "prevents", "preventing",
    "reduce", "reduces", "reduced", "treat", "treats", "treatment", "therapy",
}

_INTENT_TERMS = {
    "differential": {
        "trigger_words": {
            "distinguish", "distinguishing", "differentiate", "differentiating",
            "difference", "differences", "differential", "versus", "vs", "compare",
            "comparison", "other"
        },
        "search_terms": {
            "diagnosis", "diagnostic", "differential", "phenotype", "phenotypic",
            "feature", "features", "clinical", "onset", "development", "developmental",
            "febrile", "fever", "genetic", "variant", "mutation", "electroclinical",
            "encephalopathy", "classification"
        },
    },
    "therapy_evidence": {
        "trigger_words": {
            "evidence", "prevent", "prevents", "preventing", "reduce", "reduces",
            "reduction", "protect", "protective", "efficacy", "effective", "treat",
            "treatment", "therapy"
        },
        "search_terms": {
            "efficacy", "effective", "protective", "protection", "reduce", "reduction",
            "seizure", "frequency", "survival", "therapy", "treatment", "rescue"
        },
    },
    "mechanism": {
        "trigger_words": {"mechanism", "why", "how", "pathway", "cause", "causes"},
        "search_terms": {"mechanism", "pathway", "expression", "function", "loss", "gain", "channel", "neuron", "interneuron"},
    },
}


def _top_specific_tokens(tokens: List[str], idf: Dict[str, float], limit: int = 8) -> List[str]:
    candidates = []
    for tok in dict.fromkeys(tokens):
        if tok in _BROAD_REFERENCE_TOKENS:
            continue
        candidates.append((idf.get(tok, 1.0), tok))
    candidates.sort(reverse=True)
    return [tok for _, tok in candidates[:limit]]


def extract_query_reference_terms(question_text: str) -> dict:
    """
    Extract disease-agnostic, question-specific retrieval signals.

    Returned fields are used by both KG-triple retrieval and Article metadata
    retrieval. This is not a DS synonym dictionary; it relies on the user's
    exact wording, rare KG-label terms, and generic question intent.
    """
    index = _build_retrieval_index()
    idf = index["idf"]

    q_tokens = _tokenize_for_retrieval(question_text or "")
    q_text = _split_camel_and_symbols(question_text or "")
    q_token_set = set(q_tokens)

    strong_tokens = _top_specific_tokens(q_tokens, idf, limit=8)

    # Preserve exact multi-word concepts from the question. For example,
    # "antisense oligonucleotides" normalizes to "antisense oligonucleotide".
    phrases = []
    phrase_seen = set()
    filtered = [t for t in q_tokens if t not in _STOPWORDS]
    for n in (4, 3, 2):
        for i in range(0, max(len(filtered) - n + 1, 0)):
            parts = filtered[i:i+n]
            if all(t in _BROAD_REFERENCE_TOKENS for t in parts):
                continue
            phrase = " ".join(parts)
            if phrase not in phrase_seen and 5 <= len(phrase) <= 90:
                phrase_seen.add(phrase)
                phrases.append(phrase)

    intent_labels = []
    intent_terms = set()
    for label, cfg in _INTENT_TERMS.items():
        if q_token_set & {_canonical_retrieval_token(t) for t in cfg["trigger_words"]}:
            intent_labels.append(label)
            intent_terms.update(_canonical_retrieval_token(t) for t in cfg["search_terms"])

    return {
        "question_tokens": q_tokens,
        "strong_tokens": strong_tokens,
        "phrases": phrases[:12],
        "intent_labels": intent_labels,
        "intent_terms": sorted(intent_terms),
        "normalized_question": q_text,
    }


def _doc_matches_specific_focus(doc: dict, focus: dict) -> bool:
    """Require fallback references to match the user's specific focus."""
    searchable = doc.get("searchable", "")
    doc_tokens = doc.get("token_set", set())
    strong_tokens = set(focus.get("strong_tokens") or [])
    phrases = focus.get("phrases") or []
    intent_terms = set(focus.get("intent_terms") or [])

    if strong_tokens:
        if doc_tokens & strong_tokens:
            return True
        if any(phrase in searchable for phrase in phrases):
            return True
        return False

    # If the question has no rare biomedical anchor, use intent-level focus
    # rather than retrieving the whole disease neighborhood.
    if intent_terms:
        return bool(doc_tokens & intent_terms)

    # Last resort: allow normal scoring if no specific focus could be extracted.
    return True


def _pmcids_from_scored_compact_triples(
    scored_compact_triples: List[Tuple[str, float]],
    max_pmcids: int | None = 15,
) -> List[int]:
    """
    Convert ranked compact triples into ranked PMCIDs.

    Unlike resolve_reference_pmcids_from_triples(), this preserves evidence
    ranking across triples and avoids returning every PMCID touched by broad
    graph expansion. Results-section rows get a modest boost, but non-Results
    rows are not discarded.
    """
    if not scored_compact_triples:
        return []

    triple_to_rows = _load_reference_index()
    resolved_to_score: Dict[Tuple[str, str, str], float] = {}

    resolved_text = resolve_compact_triples_to_labels(
        "\n".join(compact for compact, _ in scored_compact_triples)
    )
    resolved_lines = [line.strip() for line in resolved_text.splitlines() if line.strip()]

    for resolved_line, (_, score) in zip(resolved_lines, scored_compact_triples):
        parts = [p.strip() for p in resolved_line.split("|")]
        if len(parts) < 3:
            continue
        key = (
            _normalize_label(parts[0]),
            _normalize_label(parts[1]),
            _normalize_label(parts[2]),
        )
        resolved_to_score[key] = max(resolved_to_score.get(key, 0.0), float(score))

    pmcid_scores: Dict[int, float] = {}
    pmcid_best_section: Dict[int, str] = {}

    for key, score in resolved_to_score.items():
        for row in triple_to_rows.get(key, []):
            pmcid = str(row.get("PMCID", "")).strip()
            if not pmcid.isdigit():
                continue
            pmcid_int = int(pmcid)
            section = str(row.get("Section", "")).strip().lower()
            section_boost = 1.25 if section == "results" else 1.0
            pmcid_scores[pmcid_int] = pmcid_scores.get(pmcid_int, 0.0) + score * section_boost
            if section == "results":
                pmcid_best_section[pmcid_int] = "results"
            elif pmcid_int not in pmcid_best_section:
                pmcid_best_section[pmcid_int] = section or "other"

    ranked = sorted(
        pmcid_scores.items(),
        key=lambda item: (
            item[1],
            1 if pmcid_best_section.get(item[0]) == "results" else 0,
        ),
        reverse=True,
    )

    pmcids = [pmcid for pmcid, _ in ranked]
    if max_pmcids is not None:
        pmcids = pmcids[:max_pmcids]
    return pmcids


def retrieve_relevant_kg_support(
    question_text: str,
    answer_text: str = "",
    model_triples_text: str = "",
    max_triples: int = 40,
    max_pmcids: int | None = 15,
    initial_pool: int = 40,
    expansion_per_node: int = 0,
) -> dict:
    """
    Focused provenance recovery for KroMA.

    Design decision:
      - Do NOT use broad backend graph expansion by default. It creates noisy
        references for general DS/seizure/model terms.
      - First trust validated compact triples reported by Gemini in the hidden
        [[RAW_TRIPLES]] block.
      - If those produce too few references, use a focused disease-agnostic
        lexical fallback over KG triples, heavily weighted toward the user's
        original question and rare/specific terms.
      - The fallback does not add one-hop neighbors unless explicitly requested.

    This returns a small, ranked set of supporting triples and PMCIDs. It is not
    intended to exhaustively list every paper connected to the KG neighborhood.
    """
    index = _build_retrieval_index()
    docs = index["docs"]
    idf = index["idf"]
    edge_to_doc_index = index["edge_to_doc_index"]

    selected: List[Tuple[int, float, str]] = []  # doc_idx, score, source
    seen_indexes: Set[int] = set()

    # 1) Model-reported compact triples are the primary provenance signal.
    model_compact = compact_ids_from_any_triples(model_triples_text or "")
    model_edges: List[Tuple[str, str, str]] = []
    for line in model_compact.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            model_edges.append((parts[0], parts[1], parts[2]))

    # Give model triples high but rank-decaying scores so their linked PMCID(s)
    # come first without allowing a single very broad edge to explode references.
    for rank, edge in enumerate(model_edges):
        idx = edge_to_doc_index.get(edge)
        if idx is None or idx in seen_indexes:
            continue
        seen_indexes.add(idx)
        selected.append((idx, 1000.0 - rank, "model"))

    model_compact_text = "\n".join(docs[idx]["compact"] for idx, _, src in selected if src == "model")
    model_pmcids = resolve_reference_pmcids_from_triples(model_compact_text, max_pmcids=max_pmcids)

    # If Gemini gave enough validated triples/references, avoid backend search.
    # This prevents broad lexical matching from drowning out specific evidence
    # such as antisense oligonucleotide papers.
    if len(model_pmcids) >= 3:
        selected = selected[:max_triples]
        scored_compact = [(docs[idx]["compact"], score) for idx, score, _ in selected]
        pmcids = _pmcids_from_scored_compact_triples(scored_compact, max_pmcids=max_pmcids)
        return {
            "compact_triples_text": "\n".join(docs[idx]["compact"] for idx, _, _ in selected),
            "resolved_triples_text": "\n".join(docs[idx]["resolved"] for idx, _, _ in selected),
            "pmcids": pmcids,
            "debug": {
                "strategy": "model_triples_primary",
                "model_edge_count": len(model_edges),
                "selected_triple_count": len(selected),
                "pmcid_count": len(pmcids),
                "max_pmcids": max_pmcids,
            },
        }

    # 2) Focused fallback: lexical retrieval from the user's question, with a
    # smaller contribution from the answer text. No graph expansion by default.
    focus = extract_query_reference_terms(question_text or "")

    token_weights = Counter()
    token_weights.update(_weighted_tokens_from_text(question_text or "", 5.0))
    token_weights.update({tok: 7.0 for tok in focus.get("strong_tokens", [])})
    token_weights.update({tok: 2.0 for tok in focus.get("intent_terms", [])})
    token_weights.update(_weighted_tokens_from_text(answer_text or "", 0.35))

    query_phrases = []
    query_phrases.extend((phrase, 7.0) for phrase in focus.get("phrases", []))
    query_phrases.extend(_phrases_from_text(question_text or "", 5.0))
    query_phrases.extend(_phrases_from_text(answer_text or "", 0.35))

    scored: List[Tuple[float, int]] = []
    for idx, doc in enumerate(docs):
        if idx in seen_indexes:
            continue
        if not _doc_matches_specific_focus(doc, focus):
            continue
        score = _score_doc_for_weighted_query(doc, dict(token_weights), query_phrases, idf)
        if score > 0:
            scored.append((score, idx))

    scored.sort(reverse=True, key=lambda x: x[0])

    if scored:
        best_score = scored[0][0]
        # Keep only focused matches. The relative threshold avoids returning many
        # generic DS/seizure/model references when the question contains a more
        # specific concept such as antisense oligonucleotide.
        min_score = max(4.0, best_score * 0.45)
        for score, idx in scored[:initial_pool]:
            if score < min_score:
                continue
            if idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            selected.append((idx, score, "focused_backend"))
            if len(selected) >= max_triples:
                break

    selected = selected[:max_triples]
    scored_compact = [(docs[idx]["compact"], score) for idx, score, _ in selected]
    pmcids = _pmcids_from_scored_compact_triples(scored_compact, max_pmcids=max_pmcids)

    return {
        "compact_triples_text": "\n".join(docs[idx]["compact"] for idx, _, _ in selected),
        "resolved_triples_text": "\n".join(docs[idx]["resolved"] for idx, _, _ in selected),
        "pmcids": pmcids,
        "debug": {
            "strategy": "model_triples_plus_focused_fallback",
            "question_token_count": len(set(_tokenize_for_retrieval(question_text))),
            "answer_token_count": len(set(_tokenize_for_retrieval(answer_text))),
            "specific_strong_tokens": focus.get("strong_tokens", []),
            "specific_phrases": focus.get("phrases", [])[:6],
            "intent_labels": focus.get("intent_labels", []),
            "direct_match_count": len(scored),
            "model_edge_count": len(model_edges),
            "selected_triple_count": len(selected),
            "pmcid_count": len(pmcids),
            "max_pmcids": max_pmcids,
            "best_backend_score": scored[0][0] if scored else 0,
        },
    }

