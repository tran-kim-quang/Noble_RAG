from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_FILES = [
    "02_12_2025_CSBH_574_NOBLE_PALACE_TAY_THANG_LONG_HDBM.md",
    "CONCEPT_THIET_KE_08_02_2025_only_hang_muc_noi_dung.md",
    "fact_sheet_sunshine.md",
]

HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
NEAR_SIGNAL_RE = re.compile(
    r"(gần|gan|thuận tiện|tiep can|tiếp cận|di chuyển|di chuyen|walking|drive)",
    re.IGNORECASE,
)

POI_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hospital": ("benh vien", "bệnh viện", "hospital", "medical"),
    "school": ("truong hoc", "trường học", "school", "giao duc", "giáo dục"),
    "park": ("cong vien", "công viên", "park", "green"),
    "mall": ("mall", "tttm", "trung tam thuong mai", "trung tâm thương mại"),
}

POI_TO_TOPIC = {
    "hospital": "hospital_access",
    "school": "school_access",
    "park": "green_space",
    "mall": "daily_convenience",
}


def normalize_project_slug(raw: str) -> str:
    value = str(raw or "").strip().lower()
    value = re.sub(r"\.(md|markdown|txt|pdf|docx?)$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def infer_project_id_from_hints(hints: str) -> str | None:
    normalized = str(hints or "").strip().lower()
    if not normalized:
        return None
    if any(
        token in normalized
        for token in (
            "sunshine legend city",
            "sunshine legend",
            "fact_sheet_sunshine",
            "ai_factsheet_ss",
            "ss legend city",
        )
    ):
        return "sunshine_legend_city"
    if any(
        token in normalized
        for token in (
            "noble palace tay ho",
            "noble palace tây hồ",
            "tay_ho",
            "tay ho",
            "tây hồ",
            "ciputra",
        )
    ):
        return "noble_palace_tay_ho"
    if any(
        token in normalized
        for token in (
            "noble palace tay thang long",
            "noble palace tây thăng long",
            "tay_thang_long",
            "tay thang long",
            "tây thăng long",
        )
    ):
        return "noble_palace_tay_thang_long"
    return None


def infer_project_id_for_file(file_name: str, content: str) -> str:
    source_hint = str(file_name or "").strip().lower()
    text_hint = str(content or "").strip().lower()[:4000]
    inferred = infer_project_id_from_hints(f"{source_hint}\n{text_hint}")
    if inferred:
        return inferred
    stem_slug = normalize_project_slug(Path(file_name).stem)
    return stem_slug or "unknown_project"


def split_sections(markdown_text: str) -> list[str]:
    cleaned = markdown_text.strip()
    if not cleaned:
        return []

    matches = list(HEADING_RE.finditer(cleaned))
    if not matches:
        return [cleaned]

    sections: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        section = cleaned[start:end].strip()
        if section:
            sections.append(section)
    return sections


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    cursor = 0
    while cursor < len(normalized):
        piece = normalized[cursor : cursor + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if cursor + chunk_size >= len(normalized):
            break
        cursor += step
    return chunks


def detect_poi_types(text: str) -> list[str]:
    lowered = text.lower()
    poi_types: list[str] = []
    for poi_type, keywords in POI_KEYWORDS.items():
        if any(token in lowered for token in keywords):
            poi_types.append(poi_type)
    return list(dict.fromkeys(poi_types))


def extract_distance_text(text: str) -> str | None:
    match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:m|km|phut|phút|min))", text.lower())
    if match:
        return match.group(1)
    return None


def infer_travel_mode(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("di bo", "đi bộ", "walk")):
        return "walk"
    if any(token in lowered for token in ("o to", "ô tô", "lai xe", "drive")):
        return "drive"
    return "unspecified"


def build_proximity_documents(
    piece: str,
    *,
    file_name: str,
    section_idx: int,
    piece_idx: int,
    project_id: str,
) -> list[dict]:
    if not NEAR_SIGNAL_RE.search(piece):
        return []
    poi_types = detect_poi_types(piece)
    if not poi_types:
        return []

    docs: list[dict] = []
    for local_idx, poi_type in enumerate(poi_types, start=1):
        topic = POI_TO_TOPIC.get(poi_type, "general")
        semantic_tags = [f"near_{poi_type}"]
        if poi_type in {"hospital", "school", "park"}:
            semantic_tags.append("family_friendly")
        doc_id = f"md::{Path(file_name).stem}::s{section_idx:03d}::c{piece_idx:03d}::p{local_idx:02d}"
        source = f"{file_name}#s{section_idx:03d}-c{piece_idx:03d}-p{local_idx:02d}"
        docs.append(
            {
                "doc_id": doc_id,
                "source": source,
                "text": piece,
                "metadata": {
                    "project_id": project_id,
                    "origin_file": file_name,
                    "section_index": section_idx,
                    "chunk_index": piece_idx,
                    "type": "proximity_fact",
                    "topic": topic,
                    "poi_type": poi_type,
                    "distance_text": extract_distance_text(piece),
                    "travel_mode": infer_travel_mode(piece),
                    "semantic_tags": semantic_tags,
                },
            }
        )
    return docs


def build_documents(
    data_dir: Path,
    markdown_files: list[str],
    chunk_size: int,
    chunk_overlap: int,
    project_id_override: str,
) -> list[dict]:
    documents: list[dict] = []
    for file_name in markdown_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        file_project_id = (
            normalize_project_slug(project_id_override)
            if str(project_id_override or "").strip()
            else infer_project_id_for_file(file_name=file_path.name, content=content)
        )
        print(f"project_map file={file_path.name} -> project_id={file_project_id}")
        sections = split_sections(content)
        for section_idx, section in enumerate(sections, start=1):
            pieces = chunk_text(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for piece_idx, piece in enumerate(pieces, start=1):
                doc_id = f"md::{file_path.stem}::s{section_idx:03d}::c{piece_idx:03d}"
                source = f"{file_path.name}#s{section_idx:03d}-c{piece_idx:03d}"
                documents.append(
                    {
                        "doc_id": doc_id,
                        "source": source,
                        "text": piece,
                        "metadata": {
                            "project_id": file_project_id,
                            "origin_file": file_path.name,
                            "section_index": section_idx,
                            "chunk_index": piece_idx,
                            "type": "evidence_chunk",
                        },
                    }
                )
                documents.extend(
                    build_proximity_documents(
                        piece,
                        file_name=file_path.name,
                        section_idx=section_idx,
                        piece_idx=piece_idx,
                        project_id=file_project_id,
                    )
                )
    return documents


def ingest_batch(base_url: str, docs: list[dict]) -> int:
    payload = json.dumps({"documents": docs}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/ingest",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return int(parsed.get("ingested", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest markdown project data into retrieval-service/Qdrant.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011", help="Retrieval service base URL.")
    parser.add_argument("--data-dir", default="data", help="Directory containing markdown files.")
    parser.add_argument(
        "--project-id",
        default="",
        help="Optional project_id override for all files. Leave empty to auto-map per file.",
    )
    parser.add_argument("--chunk-size", type=int, default=1400, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of docs per ingest request.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Markdown files relative to --data-dir.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    data_dir = Path(args.data_dir)
    docs = build_documents(
        data_dir=data_dir,
        markdown_files=args.files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        project_id_override=args.project_id,
    )
    if not docs:
        raise RuntimeError("No documents were generated from markdown inputs.")

    total_ingested = 0
    for offset in range(0, len(docs), args.batch_size):
        batch = docs[offset : offset + args.batch_size]
        ingested = ingest_batch(args.base_url, batch)
        total_ingested += ingested
        print(f"batch={offset // args.batch_size + 1} docs={len(batch)} ingested={ingested}")

    print(f"generated_docs={len(docs)}")
    print(f"total_ingested={total_ingested}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"HTTP {exc.code}: {detail[:400]}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"Network error: {exc.reason}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
