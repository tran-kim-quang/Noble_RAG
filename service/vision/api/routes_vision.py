import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.vision_models import (
    VisionCustomerResponse,
    VisionEnrollResponse,
    VisionIdentifyAndContextResponse,
    VisionIdentifyResponse,
    VisionSessionResponse,
)
from services.customer_context_builder import CustomerContextBuilder
from services.face_service import FaceService
from stores.customer_identity_store import CustomerIdentityStore

router = APIRouter(prefix="/vision", tags=["vision"])
log = logging.getLogger("vision-service")

face_service = FaceService()
identity_store = CustomerIdentityStore()
context_builder = CustomerContextBuilder(identity_store=identity_store)


def _assets_dir() -> str:
    path = "/tmp/noble_vision_assets"
    os.makedirs(path, exist_ok=True)
    return path


def _default_vision_summary(gender_estimate: str, age_group_estimate: str) -> Dict[str, Any]:
    age_map = {
        "young": "20-30",
        "middle_aged": "30-45",
        "elderly": "45+",
    }
    return {
        "age_range": age_map.get(age_group_estimate, "unknown"),
        "gender_guess": gender_estimate,
        "emotion": "unknown",
        "dress_style": "unknown",
        "visible_attributes": [],
        "scene_context": "unknown",
    }


async def _identify_internal(
    *,
    session_id: str,
    source: str,
    channel: str,
    image_bytes: bytes,
    allow_create_unknown: bool,
) -> Dict[str, Any]:
    face = await face_service.detect_single_face(image_bytes)
    basic_attrs = await face_service.extract_basic_attributes(face.cropped_bytes)
    embedding = await face_service.extract_embedding(face.cropped_bytes)

    match = await identity_store.find_best_match(embedding)
    threshold = face_service.settings.vision_match_threshold
    is_existing_customer = bool(match and match.score >= threshold)

    if is_existing_customer:
        customer_id = str(match.customer_id)
        await identity_store.touch_customer(customer_id)
        decision = "matched"
    else:
        if not allow_create_unknown:
            raise HTTPException(status_code=422, detail="unknown_face")
        customer_id = await identity_store.create_customer(
            metadata={
                "created_from": "vision_identify",
                "source": source,
                "channel": channel,
                "gender_estimate": basic_attrs.gender_estimate,
                "age_group_estimate": basic_attrs.age_group_estimate,
            }
        )
        await identity_store.add_face_embedding(
            customer_id=customer_id,
            embedding=embedding,
            embedding_model=face_service.settings.vision_model_name,
            quality_score=face.quality_score,
            metadata={
                "source": source,
                "channel": channel,
                "gender_estimate": basic_attrs.gender_estimate,
                "age_group_estimate": basic_attrs.age_group_estimate,
            },
            is_primary=True,
        )
        decision = "new"

    await identity_store.bind_session(
        customer_id=customer_id,
        session_id=session_id,
        source=source,
        metadata={
            "identify_via": "vision_identify",
            "channel": channel,
            "gender_estimate": basic_attrs.gender_estimate,
            "age_group_estimate": basic_attrs.age_group_estimate,
        },
    )
    await identity_store.log_identity_event(
        customer_id=customer_id,
        session_id=session_id,
        event_type="identify",
        similarity_score=match.score if match else None,
        decision=decision,
        metadata={
            "threshold": threshold,
            "channel": channel,
            "source": source,
            "gender_estimate": basic_attrs.gender_estimate,
            "age_group_estimate": basic_attrs.age_group_estimate,
        },
    )

    try:
        vision_summary = await face_service.extract_vision_summary(image_bytes)
    except Exception:
        vision_summary = _default_vision_summary(
            basic_attrs.gender_estimate,
            basic_attrs.age_group_estimate,
        )

    customer_context = await context_builder.build(customer_id=customer_id, fallback_session_id=session_id)
    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "is_existing_customer": is_existing_customer,
        "matched": is_existing_customer,
        "decision": decision,
        "gender_estimate": basic_attrs.gender_estimate,
        "age_group_estimate": basic_attrs.age_group_estimate,
        "match_score": match.score if (match and is_existing_customer) else None,
        "vision_summary": vision_summary,
        "customer_context": customer_context,
    }


@router.post("/enroll", response_model=VisionEnrollResponse)
async def enroll_customer_face(
    image: UploadFile = File(...),
    customer_id: str = Form(...),
    source: str = Form("camera"),
    note: Optional[str] = Form(None),
):
    if not customer_id.strip():
        raise HTTPException(status_code=400, detail="customer_id cannot be empty")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image is accepted")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")

    customer = await identity_store.get_customer(customer_id.strip())
    if not customer:
        raise HTTPException(status_code=404, detail="customer not found")

    try:
        face = await face_service.detect_single_face(image_bytes)
        embedding = await face_service.extract_embedding(face.cropped_bytes)
    except ValueError as e:
        msg = str(e).strip().lower()
        if "no face" in msg:
            raise HTTPException(status_code=422, detail="no_face_detected")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("vision enroll error: %s", e)
        raise HTTPException(status_code=500, detail="vision enroll processing error")

    image_asset_id = f"img_{uuid.uuid4().hex[:12]}"
    face_embedding_id = await identity_store.add_face_embedding(
        customer_id=customer_id.strip(),
        embedding=embedding,
        embedding_model=face_service.settings.vision_model_name,
        quality_score=face.quality_score,
        metadata={
            "source": source,
            "note": note,
        },
        is_primary=False,
    )

    image_path = os.path.join(_assets_dir(), f"{image_asset_id}.jpg")
    with open(image_path, "wb") as fh:
        fh.write(image_bytes)

    await identity_store.log_identity_event(
        customer_id=customer_id.strip(),
        session_id=f"enroll_{uuid.uuid4().hex[:8]}",
        event_type="enroll",
        similarity_score=None,
        decision="enrolled",
        metadata={
            "source": source,
            "note": note,
            "image_path": image_path,
            "face_embedding_id": face_embedding_id,
        },
    )

    return VisionEnrollResponse(
        customer_id=customer_id.strip(),
        face_detected=True,
        face_count=1,
        best_face_saved=True,
        embedding_saved=True,
        image_asset_id=image_asset_id,
        face_embedding_id=face_embedding_id,
    )


@router.post("/identify", response_model=VisionIdentifyResponse)
async def identify_customer(
    session_id: str = Form(...),
    source: str = Form("camera"),
    channel: str = Form("camera-kiosk"),
    allow_create_unknown: bool = Form(True),
    image: UploadFile = File(...),
):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image is accepted")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")

    try:
        data = await _identify_internal(
            session_id=session_id.strip(),
            source=source,
            channel=channel,
            image_bytes=image_bytes,
            allow_create_unknown=allow_create_unknown,
        )
    except ValueError as e:
        msg = str(e).strip().lower()
        if "no face" in msg:
            raise HTTPException(status_code=400, detail="no_face_detected")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("vision processing error: %s", e)
        raise HTTPException(status_code=500, detail="vision processing error")

    return VisionIdentifyResponse(**data)


@router.post("/identify-and-context", response_model=VisionIdentifyAndContextResponse)
async def identify_and_context(
    session_id: str = Form(...),
    source: str = Form("camera"),
    channel: str = Form("camera-kiosk"),
    allow_create_unknown: bool = Form(True),
    client_trace_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    if client_trace_id:
        log.info("vision identify-and-context client_trace_id=%s", client_trace_id)
    response = await identify_customer(
        session_id=session_id,
        source=source,
        channel=channel,
        allow_create_unknown=allow_create_unknown,
        image=image,
    )
    return VisionIdentifyAndContextResponse(
        matched=response.matched,
        customer_id=response.customer_id,
        session_id=response.session_id,
        vision_context_saved=True,
        vision_summary=response.vision_summary
        or _default_vision_summary(response.gender_estimate, response.age_group_estimate),
        match_score=response.match_score,
    )


@router.get("/customer/{customer_id}", response_model=VisionCustomerResponse)
async def get_customer(customer_id: str):
    customer = await identity_store.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="customer not found")

    context = await context_builder.build(customer_id=customer_id)
    return VisionCustomerResponse(
        customer_id=customer["customer_id"],
        customer_code=customer["customer_code"],
        first_seen_at=customer.get("first_seen_at"),
        last_seen_at=customer.get("last_seen_at"),
        metadata=customer.get("metadata") or {},
        customer_context=context,
    )


@router.get("/session/{session_id}", response_model=VisionSessionResponse)
async def get_customer_by_session(session_id: str):
    data: Optional[dict] = await identity_store.get_customer_by_session(session_id.strip())
    if not data:
        raise HTTPException(status_code=404, detail="session is not bound to any customer")
    customer_id = data["customer_id"]
    customer = await identity_store.get_customer(customer_id)
    customer_context = await context_builder.build(
        customer_id=customer_id,
        fallback_session_id=session_id.strip(),
    )
    known_sessions = await identity_store.list_customer_sessions(customer_id, limit=20)
    return VisionSessionResponse(
        **data,
        customer_code=(customer or {}).get("customer_code"),
        customer_metadata=(customer or {}).get("metadata") or {},
        customer_context=customer_context,
        known_session_count=len(known_sessions),
    )
