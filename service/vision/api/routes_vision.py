import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.vision_models import (
    VisionCustomerResponse,
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


@router.post("/identify", response_model=VisionIdentifyResponse)
async def identify_customer(
    session_id: str = Form(...),
    source: str = Form("camera"),
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
        face = await face_service.detect_single_face(image_bytes)
        basic_attrs = await face_service.extract_basic_attributes(face.cropped_bytes)
        embedding = await face_service.extract_embedding(face.cropped_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("vision processing error: %s", e)
        raise HTTPException(status_code=500, detail="vision processing error")

    match = await identity_store.find_best_match(embedding)
    threshold = face_service.settings.vision_match_threshold
    is_existing_customer = bool(match and match.score >= threshold)

    if is_existing_customer:
        customer_id = match.customer_id
        await identity_store.touch_customer(customer_id)
        decision = "matched"
    else:
        customer_id = await identity_store.create_customer(
            metadata={
                "created_from": "vision_identify",
                "source": source,
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
                "gender_estimate": basic_attrs.gender_estimate,
                "age_group_estimate": basic_attrs.age_group_estimate,
            },
        )
        decision = "new"

    session_id = session_id.strip()
    await identity_store.bind_session(
        customer_id=customer_id,
        session_id=session_id,
        source=source,
        metadata={
            "identify_via": "vision_identify",
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
            "gender_estimate": basic_attrs.gender_estimate,
            "age_group_estimate": basic_attrs.age_group_estimate,
        },
    )

    customer_context = await context_builder.build(customer_id=customer_id, fallback_session_id=session_id)
    return VisionIdentifyResponse(
        session_id=session_id,
        customer_id=customer_id,
        is_existing_customer=is_existing_customer,
        gender_estimate=basic_attrs.gender_estimate,
        age_group_estimate=basic_attrs.age_group_estimate,
        match_score=match.score if (match and is_existing_customer) else None,
        customer_context=customer_context,
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
