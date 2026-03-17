import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from client.rag_client import RAGClient, StorageType


RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
WEB_UI_HOST = os.getenv("WEB_UI_HOST", "0.0.0.0")
WEB_UI_PORT = int(os.getenv("WEB_UI_PORT", "8501"))

app = FastAPI(title="Noble RAG Web UI", version="1.0.0")
rag_client = RAGClient(url=RAG_SERVICE_URL)

TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


INDEX_HTML = """
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Noble RAG Client UI</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 840px; margin: 24px auto; padding: 0 16px; }
      h1 { margin-bottom: 8px; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
      .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      input[type='text'], textarea, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 8px; }
      textarea { min-height: 120px; }
      button { padding: 8px 12px; border: none; border-radius: 8px; cursor: pointer; }
      .primary { background: #2563eb; color: #fff; }
      .muted { color: #666; font-size: 14px; }
      pre { white-space: pre-wrap; background: #f7f7f7; border-radius: 8px; padding: 10px; }
      .ok { color: #166534; }
      .err { color: #991b1b; }
    </style>
  </head>
  <body>
    <h1>Noble RAG Client UI</h1>
    <p class="muted">Service URL: <span id="serviceUrl"></span></p>

    <div class="card">
      <h2>1) Upload tài liệu (xử lý nền)</h2>
      <div class="row">
        <input id="file" type="file" />
      </div>
      <div class="row" style="margin-top: 8px;">
        <label for="storageType">Storage type</label>
        <select id="storageType">
          <option value="graph">graph</option>
          <option value="vector">vector</option>
          <option value="both">both</option>
        </select>
      </div>
      <div class="row" style="margin-top: 8px;">
        <label for="metadata">Metadata JSON (optional)</label>
        <textarea id="metadata" placeholder='{"source":"web-ui"}'></textarea>
      </div>
      <div class="row" style="margin-top: 8px;">
        <button id="uploadBtn" type="button" class="primary" onclick="startUpload()">Upload</button>
      </div>
      <p id="uploadMsg" class="muted"></p>
      <pre id="uploadResult"></pre>
    </div>

    <div class="card">
      <h2>2) Query RAG</h2>
      <div class="row">
        <textarea id="query" placeholder="Nhập câu hỏi..."></textarea>
      </div>
      <div class="row" style="margin-top: 8px;">
        <button id="queryBtn" type="button" class="primary" onclick="runQuery()">Gửi query</button>
      </div>
      <pre id="queryResult"></pre>
    </div>

    <div class="card">
      <h2>3) Tài liệu đã upload (PostgreSQL)</h2>
      <div class="row" style="margin-top: 8px;">
        <button type="button" class="primary" onclick="refreshDocuments()">Làm mới danh sách</button>
      </div>
      <pre id="docsResult"></pre>
    </div>

    <script>
      const serviceUrl = "__RAG_SERVICE_URL__";
      document.getElementById("serviceUrl").textContent = serviceUrl;

      async function apiFetch(url, options = {}, timeoutMs = 20000) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        try {
          const response = await fetch(url, { ...options, signal: controller.signal });
          const raw = await response.text();
          let data = null;
          if (raw && raw.trim().length > 0) {
            try {
              data = JSON.parse(raw);
            } catch {
              data = { raw };
            }
          }
          return { response, data };
        } finally {
          clearTimeout(timeoutId);
        }
      }

      function formatDocs(payload) {
        if (!payload || !Array.isArray(payload.documents)) {
          return JSON.stringify(payload, null, 2);
        }
        const lines = payload.documents.map((doc, idx) => {
          const fileName = (doc.metadata && doc.metadata.filename) ? doc.metadata.filename : "(unknown)";
          return `${idx + 1}. ${fileName} | status=${doc.status} | chunks=${doc.chunks_count ?? 0} | updated=${doc.updated_at ?? "-"}`;
        });
        return `workspace=${payload.workspace} | total=${payload.count}\\n` + lines.join("\\n");
      }

      async function refreshDocuments() {
        const docsEl = document.getElementById("docsResult");
        docsEl.textContent = "Đang tải danh sách tài liệu...";
        try {
          const { response, data } = await apiFetch("/api/documents", {}, 15000);
          if (!response.ok) {
            docsEl.textContent = JSON.stringify(data, null, 2);
            return;
          }
          docsEl.textContent = formatDocs(data);
        } catch (e) {
          docsEl.textContent = `Lỗi tải danh sách tài liệu: ${e}`;
        }
      }

      async function startUpload() {
        const fileInput = document.getElementById("file");
        const storageType = document.getElementById("storageType").value;
        const metadata = document.getElementById("metadata").value.trim();
        const msgEl = document.getElementById("uploadMsg");
        const resultEl = document.getElementById("uploadResult");
        const uploadBtn = document.getElementById("uploadBtn");

        if (!fileInput.files || fileInput.files.length === 0) {
          msgEl.textContent = "Vui lòng chọn file.";
          msgEl.className = "err";
          return;
        }

        try {
          uploadBtn.disabled = true;
          const formData = new FormData();
          formData.append("file", fileInput.files[0]);
          formData.append("storage_type", storageType);
          formData.append("metadata", metadata);

          msgEl.textContent = "Đang tạo tác vụ upload...";
          msgEl.className = "muted";
          resultEl.textContent = "";

          const { response, data } = await apiFetch("/api/upload", { method: "POST", body: formData }, 30000);

          if (!response.ok) {
            msgEl.textContent = data.detail || "Không tạo được tác vụ upload";
            msgEl.className = "err";
            uploadBtn.disabled = false;
            return;
          }

          const taskId = data.task_id;
          msgEl.textContent = `Đang xử lý nền. Task: ${taskId}`;
          msgEl.className = "muted";

          const timer = setInterval(async () => {
            try {
              const { response: statusRes, data: statusData } = await apiFetch(`/api/tasks/${taskId}`, {}, 15000);
              if (!statusRes.ok || !statusData) {
                msgEl.textContent = `Không lấy được trạng thái task: ${JSON.stringify(statusData)}`;
                msgEl.className = "err";
                uploadBtn.disabled = false;
                clearInterval(timer);
                return;
              }

              msgEl.textContent = `Task ${taskId}: ${statusData.status}`;
              if (statusData.status === "completed") {
                msgEl.className = "ok";
                resultEl.textContent = JSON.stringify(statusData.result, null, 2);
                await refreshDocuments();
                uploadBtn.disabled = false;
                clearInterval(timer);
              }
              if (statusData.status === "failed") {
                msgEl.className = "err";
                resultEl.textContent = JSON.stringify(statusData.error || {}, null, 2);
                uploadBtn.disabled = false;
                clearInterval(timer);
              }
            } catch (e) {
              msgEl.textContent = `Không lấy được trạng thái task: ${e}`;
              msgEl.className = "err";
              uploadBtn.disabled = false;
              clearInterval(timer);
            }
          }, 1200);
        } catch (e) {
          msgEl.textContent = `Lỗi upload: ${e}`;
          msgEl.className = "err";
          uploadBtn.disabled = false;
        }
      }

      async function runQuery() {
        const query = document.getElementById("query").value;
        const resultEl = document.getElementById("queryResult");
        const queryBtn = document.getElementById("queryBtn");

        if (!query.trim()) {
          resultEl.textContent = "Query không được để trống.";
          return;
        }

        resultEl.textContent = "Đang kiểm tra trạng thái upload...";
        queryBtn.disabled = true;

        try {
          const { response: activeRes, data: activeData } = await apiFetch("/api/active-upload-tasks", {}, 15000);
          if (activeRes.ok && activeData.active_count > 0) {
            resultEl.textContent = "Upload vẫn đang xử lý nền. Vui lòng đợi task hoàn tất rồi query lại.";
            return;
          }

          resultEl.textContent = "Đang xử lý query...";
          const { response: res, data } = await apiFetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
          }, 120000);

          if (!res.ok) {
            resultEl.textContent = JSON.stringify(data, null, 2);
            return;
          }

          resultEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resultEl.textContent = `Lỗi query: ${e}`;
        } finally {
          queryBtn.disabled = false;
        }
      }

      refreshDocuments();
    </script>
  </body>
</html>
"""


def _set_task(task_id: str, payload: dict[str, Any]) -> None:
    with TASKS_LOCK:
        TASKS[task_id] = {**TASKS.get(task_id, {}), **payload}


def _wait_track_ready(track_id: str, timeout_sec: int = 120) -> tuple[bool, dict[str, Any]]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] = {}

    while time.time() < deadline:
        response = requests.get(
            f"{RAG_SERVICE_URL}/documents/track/{track_id}",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        if payload.get("ready"):
            return True, payload
        time.sleep(1.2)

    return False, last_payload


def _active_tasks() -> dict[str, dict[str, Any]]:
    with TASKS_LOCK:
        return {
            task_id: task
            for task_id, task in TASKS.items()
      if task.get("status") in {"queued", "running", "indexing"}
        }


def _run_upload_task(task_id: str, file_path: str, storage_type: StorageType, metadata: dict[str, Any]) -> None:
    try:
        _set_task(task_id, {"status": "running", "started_at": time.time()})
        result = rag_client.upload_document(
            file_path=file_path,
            storage_type=storage_type,
            metadata=metadata,
            timeout=600,
        )

        if result is None:
            _set_task(
                task_id,
                {
                    "status": "failed",
                    "error": {"message": "Upload failed. Check RAG service logs."},
                    "finished_at": time.time(),
                },
            )
            return

        track_id = result.get("document_id")
        if not track_id:
            _set_task(
                task_id,
                {
                    "status": "failed",
                    "error": {"message": "Missing track id from upload response."},
                    "finished_at": time.time(),
                },
            )
            return

        _set_task(
            task_id,
            {
                "status": "indexing",
                "track_id": track_id,
                "result": result,
            },
        )

        ready, track_payload = _wait_track_ready(track_id)
        if not ready:
            _set_task(
                task_id,
                {
                    "status": "failed",
                    "error": {
                        "message": "Indexing timeout. Document may still be processing in background.",
                        "track_status": track_payload,
                    },
                    "finished_at": time.time(),
                },
            )
            return

        _set_task(
            task_id,
            {
                "status": "completed",
                "track_status": track_payload,
                "result": result,
                "finished_at": time.time(),
            },
        )
    except Exception as exc:
        _set_task(
            task_id,
            {
                "status": "failed",
                "error": {"message": str(exc)},
                "finished_at": time.time(),
            },
        )
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.replace("__RAG_SERVICE_URL__", RAG_SERVICE_URL))


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
async def health() -> JSONResponse:
    ok = rag_client.check_health()
    return JSONResponse({"ui": "ok", "rag_service": "healthy" if ok else "unhealthy"})


@app.get("/api/documents")
async def list_documents(limit: int = 50) -> JSONResponse:
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/documents", params={"limit": limit}, timeout=15)
        response.raise_for_status()
        return JSONResponse(response.json())
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch documents from RAG service: {exc}") from exc


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    storage_type: str = Form("graph"),
    metadata: str = Form(""),
) -> JSONResponse:
    try:
        storage = StorageType(storage_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid storage_type: {storage_type}") from exc

    parsed_metadata: dict[str, Any] = {}
    if metadata.strip():
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    task_id = uuid.uuid4().hex
    _set_task(
        task_id,
        {
            "status": "queued",
            "filename": file.filename,
            "created_at": time.time(),
            "storage_type": storage.value,
        },
    )

    thread = threading.Thread(
        target=_run_upload_task,
        args=(task_id, temp_path, storage, parsed_metadata),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"task_id": task_id, "status": "queued"})


@app.get("/api/active-upload-tasks")
async def get_active_tasks() -> JSONResponse:
    active = _active_tasks()
    return JSONResponse({
        "active_count": len(active),
        "task_ids": list(active.keys()),
    })


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str) -> JSONResponse:
    with TASKS_LOCK:
        task = TASKS.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return JSONResponse(task)


@app.post("/api/query")
async def query_rag(payload: dict[str, Any]) -> JSONResponse:
    if _active_tasks():
        raise HTTPException(
            status_code=409,
            detail="Upload is still processing. Please query again when task status is completed.",
        )

    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = int(payload.get("top_k", 10))
    structured = bool(payload.get("return_structured_output", False))

    result = rag_client.query(
        query_text=query,
        top_k=top_k,
        return_structured_output=structured,
        timeout=90,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="RAG query failed")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.rag_web_ui:app", host=WEB_UI_HOST, port=WEB_UI_PORT, reload=False)
