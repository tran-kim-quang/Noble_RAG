import json
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st


RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 60


def api_get(path: str, **params: Any) -> requests.Response:
    return requests.get(f"{RAG_SERVICE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)


def api_delete(path: str) -> requests.Response:
    return requests.delete(f"{RAG_SERVICE_URL}{path}", timeout=REQUEST_TIMEOUT)


def api_post_upload(file_name: str, file_content: bytes, storage_type: str, metadata_text: str) -> requests.Response:
    files = {
        "file": (file_name, file_content, "text/plain"),
    }
    data = {
        "storage_type": storage_type,
        "metadata": metadata_text,
    }
    return requests.post(
        f"{RAG_SERVICE_URL}/upload-document",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )


def safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def format_doc_label(doc: Dict[str, Any], index: int) -> str:
    metadata = doc.get("metadata") or {}
    file_name = metadata.get("filename") or doc.get("file_path") or "(unknown)"
    status = doc.get("status") or "unknown"
    chunks = doc.get("chunks_count") or 0
    updated_at = doc.get("updated_at") or "-"
    return f"{index}. {file_name} | status={status} | chunks={chunks} | updated={updated_at}"


st.set_page_config(
    page_title="Documents Upload UI",
    page_icon="D",
    layout="wide",
)

st.title("Documents Upload UI")
st.caption(f"RAG Service: `{RAG_SERVICE_URL}`")

if "last_upload_result" not in st.session_state:
    st.session_state.last_upload_result = None

if "last_track_result" not in st.session_state:
    st.session_state.last_track_result = None

upload_col, docs_col = st.columns([1, 1.15], gap="large")

with upload_col:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Chọn file văn bản UTF-8",
        type=["txt", "md", "csv", "json"],
        accept_multiple_files=False,
    )
    storage_type = st.selectbox(
        "Storage type",
        options=["graph", "vector", "both"],
        index=0,
    )
    metadata_text = st.text_area(
        "Metadata JSON (optional)",
        value='{"source":"documents_upload_ui"}',
        height=120,
    )

    if st.button("Upload Document", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Vui lòng chọn file trước khi upload.")
        else:
            try:
                if metadata_text.strip():
                    json.loads(metadata_text)
                response = api_post_upload(
                    file_name=uploaded_file.name,
                    file_content=uploaded_file.getvalue(),
                    storage_type=storage_type,
                    metadata_text=metadata_text,
                )
                payload = safe_json(response)
                st.session_state.last_upload_result = payload
                if response.ok:
                    st.success("Upload thành công.")
                else:
                    st.error(payload.get("detail") or payload.get("raw") or "Upload thất bại.")
            except json.JSONDecodeError:
                st.error("Metadata phải là JSON hợp lệ.")
            except requests.RequestException as exc:
                st.error(f"Không gọi được RAG service: {exc}")

    if st.session_state.last_upload_result is not None:
        st.markdown("**Kết quả upload**")
        st.json(st.session_state.last_upload_result)

    st.divider()
    st.subheader("Track Status")
    track_id = st.text_input("Track ID", placeholder="Nhập track_id nếu cần kiểm tra")
    if st.button("Kiểm Tra Track", use_container_width=True):
        if not track_id.strip():
            st.warning("Vui lòng nhập track_id.")
        else:
            try:
                response = api_get(f"/documents/track/{track_id.strip()}")
                payload = safe_json(response)
                st.session_state.last_track_result = payload
                if not response.ok:
                    st.error(payload.get("detail") or payload.get("raw") or "Không lấy được track status.")
            except requests.RequestException as exc:
                st.error(f"Không gọi được RAG service: {exc}")

    if st.session_state.last_track_result is not None:
        st.markdown("**Track status**")
        st.json(st.session_state.last_track_result)

with docs_col:
    st.subheader("Documents")
    limit = st.slider("Số lượng documents", min_value=10, max_value=200, value=50, step=10)
    refresh = st.button("Làm Mới Danh Sách", use_container_width=True)

    if refresh or "documents_payload" not in st.session_state:
        try:
            response = api_get("/documents", limit=limit)
            st.session_state.documents_payload = safe_json(response)
            st.session_state.documents_ok = response.ok
        except requests.RequestException as exc:
            st.session_state.documents_payload = {"detail": str(exc)}
            st.session_state.documents_ok = False

    documents_payload = st.session_state.get("documents_payload", {})
    if not st.session_state.get("documents_ok", False):
        st.error(documents_payload.get("detail") or documents_payload.get("raw") or "Không tải được danh sách documents.")
    else:
        documents = documents_payload.get("documents") or []
        st.caption(
            f"workspace=`{documents_payload.get('workspace', '-')}` | total={documents_payload.get('count', 0)}"
        )
        if not documents:
            st.info("Chưa có document nào.")
        else:
            for idx, doc in enumerate(documents, start=1):
                label = format_doc_label(doc, idx)
                with st.expander(label, expanded=False):
                    metadata = doc.get("metadata") or {}
                    st.write(f"**Document ID:** `{doc.get('id')}`")
                    st.write(f"**File path:** `{doc.get('file_path')}`")
                    st.write(f"**Content length:** `{doc.get('content_length')}`")
                    st.write(f"**Created at:** `{doc.get('created_at')}`")
                    st.write(f"**Updated at:** `{doc.get('updated_at')}`")
                    st.write("**Metadata**")
                    st.json(metadata)

                    delete_key = f"delete_{doc.get('id')}"
                    if st.button("Xóa Document", key=delete_key):
                        try:
                            response = api_delete(f"/documents/{doc.get('id')}")
                            payload = safe_json(response)
                            if response.ok:
                                st.success(payload.get("message") or "Đã trigger xóa document.")
                                st.session_state.documents_payload = None
                                st.rerun()
                            else:
                                st.error(payload.get("detail") or payload.get("raw") or "Xóa document thất bại.")
                        except requests.RequestException as exc:
                            st.error(f"Không gọi được RAG service: {exc}")
