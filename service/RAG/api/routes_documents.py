import json
import time
from typing import Optional

import asyncpg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.dependencies import rag
from models.api_models import StorageType, UploadDocumentResponse
from sales.nodes.retrieve_context import refresh_retrieval_caches

router = APIRouter(tags=["documents"])
settings = get_settings()


@router.post("/upload-document", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    storage_type: StorageType = Form(StorageType.VECTOR),
    metadata: Optional[str] = Form(None),
):
    start = time.perf_counter()
    try:
        content = await file.read()
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="File is empty")

        doc_metadata: dict = {}
        if metadata:
            try:
                parsed = json.loads(metadata)
                if isinstance(parsed, dict):
                    doc_metadata.update(parsed)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON metadata")

        doc_metadata["filename"] = file.filename
        doc_metadata["upload_timestamp"] = time.time()

        document_id = await rag.ainsert(text_content, file_paths=file.filename)
        await refresh_retrieval_caches()
        chunks_count = max(1, len(text_content) // settings.chunk_size)
        elapsed = time.perf_counter() - start

        return UploadDocumentResponse(
            status="success",
            document_id=document_id,
            chunks_count=chunks_count,
            storage_type=storage_type.value,
            message=f"Document processed ({chunks_count} chunks in {elapsed:.2f}s)",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")


@router.get("/documents")
async def list_documents(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1-500")

    try:
        conn = await asyncpg.connect(settings.postgres_url)
        rows = await conn.fetch(
            """
            SELECT id, status, file_path, content_length, chunks_count,
                   created_at, updated_at, metadata
            FROM sales.haystack_documents
            WHERE workspace = $1
            ORDER BY updated_at DESC NULLS LAST
            LIMIT $2
            """,
            settings.rag_workspace,
            limit,
        )
        await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

    documents = []
    for row in rows:
        meta = row.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = None
        documents.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "file_path": row.get("file_path"),
                "content_length": row.get("content_length"),
                "chunks_count": row.get("chunks_count"),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
                "metadata": meta,
            }
        )

    return JSONResponse({"workspace": settings.rag_workspace, "count": len(documents), "documents": documents})


@router.get("/documents/track/{track_id}")
async def get_track_status(track_id: str):
    try:
        conn = await asyncpg.connect(settings.postgres_url)
        rows = await conn.fetch(
            """
            SELECT status, COUNT(*) AS count
            FROM sales.haystack_documents
            WHERE workspace = $1 AND track_id = $2
            GROUP BY status
            """,
            settings.rag_workspace,
            track_id,
        )
        await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get track status: {e}")

    status_counts: dict = {}
    total = 0
    for row in rows:
        sv = row.get("status") or "unknown"
        cnt = int(row.get("count") or 0)
        status_counts[sv] = cnt
        total += cnt

    ready = total > 0 and status_counts.get("processing", 0) == 0 and status_counts.get("pending", 0) == 0
    return JSONResponse(
        {
            "workspace": settings.rag_workspace,
            "track_id": track_id,
            "total": total,
            "ready": ready,
            "status_counts": status_counts,
        }
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        await rag.adelete_by_doc_id(doc_id)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Deletion triggered for document {doc_id}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")
