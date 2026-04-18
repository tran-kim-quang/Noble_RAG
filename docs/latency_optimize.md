Bạn có thể lưu nguyên nội dung dưới đây thành file `.md`, ví dụ `refactor_test_screen_agents_realtime.md`.

---

# Refactor cụ thể cho `test-screen-agents` theo hướng realtime

## 1. Mục tiêu

Mục tiêu của bản refactor này là giảm mạnh latency cho flow chat của nhánh `test-screen-agents`, đặc biệt cho use case avatar/screen agents.

### Mục tiêu latency

* `consult_discovery`: dưới 4 giây
* `project_grounded`: dưới 8 giây
* turn xin contact / xác nhận follow-up: dưới 2 giây

### Mục tiêu chất lượng

* giữ nguyên contract `/sales/query`
* không phá `QueryResponse`
* rollout được theo từng pha
* có thể rollback bằng config

---

## 2. Bottleneck hiện tại

### 2.1. Mỗi turn thường đang tốn 2 lần gọi model ở orchestrator

Ở `orchestrator_service/app.py`, flow hiện tại vẫn thường là:

* `analyze_turn()` gọi model decider
* nếu không hit fastpath thì `synthesize_assistant_reply()` gọi model lần nữa để viết reply cuối 

Điều này là bottleneck lớn nhất của `consult_discovery`.

### 2.2. `project_grounded` đang chạy retrieval theo 3 pass tuần tự

Trong `retrieval_service/pipeline.py`, `retrieve_project_grounded()` đang chạy:

* `candidate_pass`
* `proximity_pass`
* `evidence_pass`
* rồi mới shape response 

Với collection hiện tại gần như chỉ xoay quanh Noble Palace Tây Thăng Long, kiến trúc này đang quá nặng.

### 2.3. Retrieval còn bị nhân cost do mỗi pass lại embed + search

`retrieve()` sẽ:

* embed query
* rồi mới Qdrant search 

Nếu `EMBEDDING_BACKEND=remote`, 3 pass grounded retrieval tương đương nhiều HTTP roundtrip chỉ để ra một response.

### 2.4. Prompt của decider và synthesis khá dài

Trong `orchestrator_service/app.py`, cả decider prompt lẫn synthesis prompt đều nhét nhiều:

* `lead_state`
* `recent_history`
* grounded snapshot
* rất nhiều rule điều phối và phrasing 

Điều này làm tăng prefill latency đáng kể.

### 2.5. Report trong repo xác nhận latency đang cao

File `inference_continuous_report.jsonl` cho thấy:

* nhiều lượt `consult_discovery` khoảng 12–16 giây
* nhiều lượt `project_grounded` khoảng 21–35 giây 

---

## 3. Nguyên tắc refactor

Refactor nên đi theo thứ tự ưu tiên sau:

1. **Cắt `project_grounded` từ 3-pass retrieval xuống 1-pass**
2. **Bỏ LLM ở các turn có thể quyết định deterministic**
3. **Rút token prefill của decider và synthesis**
4. **Tăng tỷ lệ fastpath, giảm số lượt phải vào `reply_synthesis`**
5. **Chỉ sau đó mới tối ưu model size**

---

## 4. Refactor 1: chuyển `project_grounded` sang single-pass retrieval

## 4.1. Mục tiêu

Thay vì:

* candidate pass
* proximity pass
* evidence pass

chỉ thực hiện:

* 1 lần retrieve rộng
* local shaping để tạo `project_cards`, `trait_tags`, `proximity_facts`, `evidence_chunks`

### Lợi ích

Đây là thay đổi có impact lớn nhất với `project_grounded`.

---

## 4.2. File cần sửa

* `retrieval_service/config.py`
* `retrieval_service/pipeline.py`

---

## 4.3. Sửa `retrieval_service/config.py`

Thêm các config sau vào `Settings`:

```python
project_grounded_mode: str = os.getenv("PROJECT_GROUNDED_MODE", "single_pass").strip().lower()
project_grounded_single_project_id: str = (
    os.getenv("PROJECT_GROUNDED_SINGLE_PROJECT_ID", "noble_palace_tay_thang_long").strip()
)
project_grounded_single_pass_top_k: int = int(os.getenv("PROJECT_GROUNDED_SINGLE_PASS_TOP_K", "8"))
```

### Giá trị khuyến nghị

```env
PROJECT_GROUNDED_MODE=single_pass
PROJECT_GROUNDED_SINGLE_PROJECT_ID=noble_palace_tay_thang_long
PROJECT_GROUNDED_SINGLE_PASS_TOP_K=8
```

---

## 4.4. Sửa `retrieve_project_grounded()` trong `retrieval_service/pipeline.py`

Ở đầu hàm `retrieve_project_grounded(...)`, thêm nhánh dispatch:

```python
if self.settings.project_grounded_mode == "single_pass":
    return self._retrieve_project_grounded_single_pass(
        query=query,
        retrieval_intent=intent,
        top_k=k,
    )
```

---

## 4.5. Thêm hàm `_retrieve_project_grounded_single_pass(...)`

Thêm vào class `RetrievalService`:

```python
def _retrieve_project_grounded_single_pass(
    self,
    query: str,
    retrieval_intent: dict[str, Any],
    top_k: int,
) -> dict:
    single_k = max(top_k, self.settings.project_grounded_single_pass_top_k)
    grounded_query = self._build_grounded_query(query=query, retrieval_intent=retrieval_intent)
    raw = self.retrieve(grounded_query, top_k=single_k)

    evidence_chunks = self._build_evidence_chunks(raw.get("results", []))
    if not evidence_chunks:
        return {
            "route": "project_grounded",
            "project_cards": [],
            "trait_tags": [],
            "proximity_facts": [],
            "evidence_chunks": [],
            "confidence": float(raw.get("confidence", 0.0) or 0.0),
            "low_confidence": bool(raw.get("low_confidence", False)),
        }

    candidate_project_ids = self._dedupe_keep_order(
        [
            str(chunk.get("project_id") or self.settings.project_grounded_single_project_id)
            for chunk in evidence_chunks
        ],
        max_items=4,
    )

    proximity_facts = self._build_proximity_facts(
        chunks=evidence_chunks,
        retrieval_intent=retrieval_intent,
        candidate_project_ids=candidate_project_ids,
    )
    project_cards = self._build_project_cards(
        evidence_chunks=evidence_chunks,
        proximity_facts=proximity_facts,
        retrieval_intent=retrieval_intent,
    )
    trait_tags = self._build_trait_tags(
        evidence_chunks=evidence_chunks,
        query=grounded_query,
        retrieval_intent=retrieval_intent,
    )

    return {
        "route": "project_grounded",
        "project_cards": project_cards,
        "trait_tags": trait_tags,
        "proximity_facts": proximity_facts,
        "evidence_chunks": evidence_chunks[: max(top_k * 2, 8)],
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "low_confidence": bool(raw.get("low_confidence", False)) and not project_cards,
    }
```

---

## 4.6. Kỳ vọng

Sau thay đổi này:

* grounded retrieval chỉ còn 1 query embedding + 1 vector search
* không còn chuỗi 3 pass tuần tự 

---

## 5. Refactor 2: deterministic fastpath cho contact capture

## 5.1. Mục tiêu

Các turn như:

* “mình tên Quang”
* “số điện thoại là…”
* “ok em sắp xếp giúp”

không cần qua LLM decider nữa.

---

## 5.2. File cần sửa

* `orchestrator_service/app.py`

---

## 5.3. Thêm helper `_build_contact_capture_fastpath_analysis(...)`

```python
def _build_contact_capture_fastpath_analysis(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
) -> TurnAnalysis | None:
    _ = recent_history
    extracted_name = _extract_name(message)
    extracted_phone = _extract_phone_contact(message)
    if not extracted_name and not extracted_phone:
        return None

    if lead_state.contact_capture_status not in {"requested", "partial"} and lead_state.sales_state not in {
        "interested",
        "appointment_ready",
    }:
        return None

    consult_reply_parts = []
    if extracted_name:
        consult_reply_parts.append(f"Em cảm ơn anh/chị {extracted_name}.")
    else:
        consult_reply_parts.append("Em đã ghi nhận thông tin liên hệ của anh/chị.")
    if extracted_phone:
        consult_reply_parts.append(f"Em đã lưu số {extracted_phone}.")
    consult_reply_parts.append(
        "Em sẽ dùng thông tin này để sắp xếp tư vấn sâu hơn và chốt khung thời gian phù hợp."
    )

    evidence = [message.strip()] if message.strip() else []

    return TurnAnalysis(
        route="consult_discovery",
        decision_reason="deterministic_contact_capture_fastpath",
        need_update=NeedPainpointDelta(
            summary_delta="Khách đã cung cấp thông tin liên hệ để đi tiếp sang bước tư vấn hoặc hẹn lịch.",
            topics=[TopicWeight(label="đã_cung_cấp_thông_tin_liên_hệ", weight=0.95)],
            evidence=evidence,
        ),
        painpoint_update=NeedPainpointDelta(summary_delta="", topics=[], evidence=evidence),
        routing_signal=RoutingSignal(
            should_route_project=False,
            project_query_hint=None,
            reason="deterministic_contact_capture_fastpath",
        ),
        consult_reply=" ".join(consult_reply_parts),
        query_type="clarification",
        retrieval_readiness="not_ready",
        route_source="deterministic_fastpath",
        engagement_state_after_hint="ready",
        sales_state_after_hint="appointment_ready",
        conversation_goal_hint="confirm_followup",
        extracted_name=extracted_name or lead_state.name,
        extracted_phone=extracted_phone or lead_state.phone_contact,
    )
```

---

## 5.4. Cắm vào đầu `analyze_turn(...)`

```python
fastpath = _build_contact_capture_fastpath_analysis(
    message=message,
    lead_state=lead_state,
    recent_history=recent_history,
)
if fastpath is not None:
    return fastpath
```

---

## 5.5. Kỳ vọng

Các turn cuối funnel sẽ không còn phải gọi decider model, trong khi hiện tại turn nào cũng có nguy cơ đi vào `analyze_turn()` rồi mới ra quyết định 

---

## 6. Refactor 3: giảm token prefill của decider và synthesis

## 6.1. Mục tiêu

Giảm số token model phải đọc ở:

* `analyze_turn()`
* `synthesize_assistant_reply()`

Đây là cách giảm latency ít rủi ro nhất.

---

## 6.2. File cần sửa

* `orchestrator_service/config.py`
* `orchestrator_service/app.py`

---

## 6.3. Thêm config trong `orchestrator_service/config.py`

```python
decider_history_turns: int = int(os.getenv("ORCHESTRATOR_DECIDER_HISTORY_TURNS", "4"))
synthesis_history_turns: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_HISTORY_TURNS", "3"))
synthesis_state_topic_limit: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_STATE_TOPIC_LIMIT", "4"))
grounded_card_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_CARD_LIMIT", "1"))
grounded_trait_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_TRAIT_LIMIT", "3"))
grounded_proximity_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_PROXIMITY_LIMIT", "3"))
grounded_evidence_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_EVIDENCE_LIMIT", "2"))
```

---

## 6.4. Rút `recent_history` trong decider prompt

Hiện `_build_decider_prompt()` đang lấy `recent_history[-6:]` 

Đổi thành:

```python
compact_history = [
    {"role": turn.role, "message": turn.message}
    for turn in recent_history[-settings.decider_history_turns:]
]
```

---

## 6.5. Rút `recent_history` trong synthesis prompt

Hiện `_build_reply_synthesis_prompt()` đang lấy `recent_history[-6:]` 

Đổi thành:

```python
history = [
    {"role": turn.role, "message": turn.message}
    for turn in recent_history[-settings.synthesis_history_turns:]
]
```

---

## 6.6. Rút grounded snapshot

Sửa `_build_grounded_snapshot(...)` để nhận thêm `settings: Settings` rồi cap số item:

```python
for card in project_cards_raw[: settings.grounded_card_limit]:
    ...

snapshot_traits = [
    {"tag": item.get("tag"), "reason": item.get("reason"), "weight": item.get("weight")}
    for item in trait_tags_raw[: settings.grounded_trait_limit]
]

snapshot_proximity = [
    {
        "project_id": item.get("project_id"),
        "poi_type": item.get("poi_type"),
        "poi_name": item.get("poi_name"),
        "proximity_text": item.get("proximity_text"),
    }
    for item in proximity_facts_raw[: settings.grounded_proximity_limit]
]

snapshot_evidence = [
    {"text": item.get("text"), "source": item.get("source"), "topic": item.get("topic")}
    for item in evidence_chunks_raw[: settings.grounded_evidence_limit]
]
```

---

## 6.7. Rút topic count của `lead_state` trong synthesis

Ví dụ:

```python
"need_topics": [{"label": t.label, "weight": t.weight} for t in lead_state.need.topics[:settings.synthesis_state_topic_limit]],
"painpoint_topics": [{"label": t.label, "weight": t.weight} for t in lead_state.painpoint.topics[:settings.synthesis_state_topic_limit]],
```

---

## 7. Refactor 4: grounded template fastpath

## 7.1. Mục tiêu

Một số turn `project_grounded` thực ra không cần phải vào `reply_synthesis()`.

Ví dụ:

* “có gần trường không”
* “có mầm non không”
* “dự án này hợp điểm nào”
* “có loại hình nào”

Nếu đã có `project_cards[0]` và `ask_policy=avoid_question`, có thể trả lời bằng template.

---

## 7.2. File cần sửa

* `orchestrator_service/app.py`

---

## 7.3. Thêm helper `_build_grounded_template_reply(...)`

```python
def _build_grounded_template_reply(
    grounded_result: dict[str, Any] | None,
    reply_plan: ReplyPlan,
) -> str:
    if not grounded_result:
        return ""
    cards = grounded_result.get("project_cards", []) or []
    if not cards:
        return ""

    if _normalize_response_mode(reply_plan.response_mode) not in {
        "grounded_recommendation",
        "value_teaser",
        "soft_next_step",
    }:
        return ""
    if _normalize_ask_policy(reply_plan.ask_policy) != "avoid_question":
        return ""

    first = cards[0]
    project_name = _friendly_project_name(str(first.get("project_id", "Noble Palace Tây Thăng Long")))
    strengths = [str(item).strip() for item in (first.get("strengths") or []) if str(item).strip()]
    summary = str(first.get("summary", "")).strip()

    if strengths:
        reply = f"{project_name} đang là phương án phù hợp để mình tư vấn trước, nổi bật ở {', '.join(strengths[:2])}."
    elif summary:
        reply = summary
    else:
        reply = f"{project_name} đang là phương án phù hợp để mình tư vấn trước."

    return _compact_text(reply, max_words=70)
```

---

## 7.4. Cắm vào flow trước `reply_synthesis`

Trước block:

```python
if not assistant_reply:
    ...
```

thêm:

```python
if not assistant_reply and final_route == "project_grounded":
    assistant_reply = _build_grounded_template_reply(
        grounded_result=grounded_result,
        reply_plan=reply_plan,
    )
    if assistant_reply:
        reply_source = "grounded_template_fastpath"
```

---

## 7.5. Kỳ vọng

Một phần turn `project_grounded` sẽ không còn phải gọi model reply.

---

## 8. Tối ưu model sau khi đã sửa flow

## 8.1. Kết luận

Model nhỏ hơn có giúp, nhưng **chỉ sau khi đã sửa flow**.

Hiện config example trong repo đang dùng model khá nặng:

* orchestrator SGLang example: `google/gemma-3-27b-it` 
* retrieval SGLang example: `Qwen/Qwen3-Embedding-8B` 
* retrieval config mặc định còn để `Qwen/Qwen3-Embedding-4B` 

Nếu chưa cắt 3-pass retrieval mà chỉ giảm model, hiệu quả sẽ không đủ.

---

## 8.2. Khuyến nghị

### Vai trò model

* decider: model nhỏ hơn
* synthesis consult: model nhỏ hoặc template fastpath
* project-grounded synthesis: model vừa
* embedding: nhỏ hơn nhiều so với 4B/8B hiện tại

### Ưu tiên

1. sửa flow
2. giảm prompt
3. rồi mới giảm model size

---

## 9. Thứ tự rollout

## Phase 1

* deterministic contact capture fastpath
* prompt slimming

## Phase 2

* single-pass grounded retrieval

## Phase 3

* grounded template fastpath

## Phase 4

* giảm model size nếu cần thêm

---

## 10. Log nên thêm để đo hiệu quả

### Orchestrator

Theo dõi:

* `analysis_source=llm|deterministic_fastpath|fallback`
* `reply_source=consult_reply_fastpath|grounded_template_fastpath|reply_synthesis`
* `fastpath_gate_reason`

### Retrieval

Theo dõi:

* `project_grounded_mode=single_pass|multi_pass`
* `single_pass_results`
* `single_pass_confidence`

---

## 11. Checklist triển khai

### `retrieval_service/config.py`

* [ ] thêm `PROJECT_GROUNDED_MODE`
* [ ] thêm `PROJECT_GROUNDED_SINGLE_PROJECT_ID`
* [ ] thêm `PROJECT_GROUNDED_SINGLE_PASS_TOP_K`

### `retrieval_service/pipeline.py`

* [ ] dispatch theo `project_grounded_mode`
* [ ] thêm `_retrieve_project_grounded_single_pass()`

### `orchestrator_service/config.py`

* [ ] thêm config cho history/topic/snapshot caps

### `orchestrator_service/app.py`

* [ ] thêm `_build_contact_capture_fastpath_analysis()`
* [ ] cắm contact fastpath vào `analyze_turn()`
* [ ] rút `recent_history` trong decider prompt
* [ ] rút grounded snapshot trong synthesis
* [ ] thêm `_build_grounded_template_reply()`
* [ ] cắm grounded template fastpath trước `reply_synthesis`

---

## 12. Chốt

Bản refactor nên làm ngay theo đúng thứ tự:

1. **Single-pass retrieval cho `project_grounded`**
2. **Deterministic fastpath cho contact capture**
3. **Rút token prefill của decider và synthesis**
4. **Grounded template fastpath**
5. **Sau cùng mới tối ưu model size**

Đây là cách thực dụng nhất để kéo `test-screen-agents` về gần realtime mà chưa cần viết lại toàn bộ kiến trúc.

---

Nếu cần, mình sẽ viết tiếp cho bạn bản **unified diff patch** cho từng file để dán vào code luôn.
