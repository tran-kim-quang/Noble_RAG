from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_service.app import create_app
from vision_service.schemas import VisionProfile


class FakeVisionService:
    def __init__(self) -> None:
        self.people = {"Linh": object()}

    def health(self):
        return {"status": "ok", "people_count": len(self.people)}

    def identify_base64(self, image_base64, image_filename=None, image_content_type=None):
        _ = image_base64
        _ = image_filename
        _ = image_content_type
        return VisionProfile(
            recognized=True,
            name="Linh",
            age=31,
            gender="female",
            confidence=0.91,
            source="face_db",
            face_count=1,
            bbox=[10, 20, 90, 100],
            face_id="known-linh",
        )


class ErrorVisionService(FakeVisionService):
    def identify_base64(self, image_base64, image_filename=None, image_content_type=None):
        _ = image_base64
        _ = image_filename
        _ = image_content_type
        raise ValueError("Invalid base64 image payload")


PY313 = sys.version_info >= (3, 13)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_vision_health():
    client = TestClient(create_app(service=FakeVisionService()))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["people_count"] == 1


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_vision_identify():
    client = TestClient(create_app(service=FakeVisionService()))
    res = client.post(
        "/vision/identify",
        json={
            "image_base64": "ZmFrZQ==",
            "image_filename": "frame.jpg",
            "image_content_type": "image/jpeg",
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["recognized"] is True
    assert payload["name"] == "Linh"
    assert payload["gender"] == "female"
    assert payload["source"] == "face_db"
    assert payload["face_id"] == "known-linh"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_vision_identify_bad_payload():
    client = TestClient(create_app(service=ErrorVisionService()))
    res = client.post("/vision/identify", json={"image_base64": "ZmFrZQ=="})
    assert res.status_code == 400
    assert "Invalid base64 image payload" in res.json()["detail"]
