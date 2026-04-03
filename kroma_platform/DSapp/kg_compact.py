from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from django.conf import settings


KG_CSV_FILENAME = "kg_triples_unique.csv"
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
      kg_triples_unique.csv
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
    for _, row in edges_df.iterrows():
        edge_set.add((row["s"], row["p"], row["o"]))

    return {
        "node_label_to_id": node_label_to_id,
        "node_id_to_label": node_id_to_label,
        "pred_label_to_id": pred_label_to_id,
        "pred_id_to_label": pred_id_to_label,
        "edge_set": edge_set,
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


def resolve_reference_pmcids_from_triples(raw_triples_text: str) -> List[int]:
    """
    Accepts either:
      Subject | Predicate | Object
    or:
      n123 | p17 | n456

    Returns PMCIDs with this preference:
      1) all unique PMCIDs from matching rows in Section == Results
      2) if no Results rows exist for the matched triples, fall back to all other sections
    """
    if not raw_triples_text:
        return []

    resolved_text = resolve_compact_triples_to_labels(raw_triples_text)
    triple_to_rows = _load_reference_index()

    results_pmcids = []
    fallback_pmcids = []
    seen_results = set()
    seen_fallback = set()

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

        # Prefer Results rows for this triple
        result_rows = [
            r for r in matching_rows
            if r["Section"].strip().lower() == "results"
        ]

        if result_rows:
            for row in result_rows:
                pmcid = row["PMCID"].strip()
                if pmcid.isdigit() and pmcid not in seen_results:
                    seen_results.add(pmcid)
                    results_pmcids.append(int(pmcid))
        else:
            for row in matching_rows:
                pmcid = row["PMCID"].strip()
                if pmcid.isdigit() and pmcid not in seen_fallback:
                    seen_fallback.add(pmcid)
                    fallback_pmcids.append(int(pmcid))

    return results_pmcids if results_pmcids else fallback_pmcids