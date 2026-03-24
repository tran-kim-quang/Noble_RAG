
Dưới đây là bản guide theo hướng **sử dụng LangGraph** để refactor hệ thống hiện tại thành **AI sales agent bất động sản**, đủ để bạn dùng làm tài liệu thiết kế hoặc copy vào docs sau.

---

# AI Sales Agent Refactor Guide (LangGraph)

## 1. Mục tiêu

Tài liệu này mô tả hướng refactor hệ thống hiện tại để chuyển từ một RAG assistant chuyên domain bất động sản thành một **AI sales agent bất động sản sử dụng LangGraph**.

Mục tiêu sau refactor:

* Hiểu nhu cầu khách hàng theo quy trình sales.
* Thu thập và lưu hồ sơ lead có cấu trúc.
* Đề xuất sản phẩm phù hợp theo budget, khu vực, loại hình và mục đích mua.
* Xử lý objection theo playbook.
* Chốt bước tiếp theo như gửi shortlist, bảng giá, hẹn call hoặc hẹn đi xem dự án.
* Điều phối toàn bộ hội thoại bằng **graph state machine**, không dồn hết logic vào một chain hay một prompt dài.

---

## 2. Vì sao dùng LangGraph

Use case sales bất động sản không phải là một chain tuyến tính đơn giản. Hệ thống cần:

* quản lý nhiều state hội thoại
* nhớ hồ sơ lead qua nhiều turn
* chuyển nhánh theo intent, objection, buy signal
* retrieve context khác nhau theo từng state
* dễ gắn thêm tools như tính vay, check bảng giá, đặt lịch, CRM

LangGraph phù hợp vì:

1. Quản lý flow có trạng thái tốt hơn chain thông thường.
2. Cho phép tách rõ decision nodes và execution nodes.
3. Dễ kết hợp Redis, Postgres, LightRAG và Whisper trong cùng một graph.
4. Dễ mở rộng sang multi-step sales orchestration mà vẫn kiểm soát được logic.

---

## 3. Hiện trạng hệ thống

Hệ thống hiện tại đã có các thành phần nền tảng tốt:

* `whisper-service` cho speech-to-text
* `rag-service` dùng LightRAG
* `redis` cho chat history
* `postgres` cho storage quan hệ
* `qdrant` cho vector retrieval
* `ollama` cho embedding runtime local
* `rag-ui` cho giao diện chat cơ bản

Kiến trúc này phù hợp với bài toán **domain RAG assistant**, nhưng chưa đủ để trở thành **sales agent** vì còn thiếu:

* LangGraph orchestration layer
* graph state chuẩn cho sales
* lead profile memory có cấu trúc
* product schema cho bất động sản
* objection playbooks
* closing playbooks
* prompt builder theo state
* node-based routing và persistence

---

## 4. Nguyên tắc kiến trúc sau refactor

1. Không nhồi toàn bộ logic sales vào một system prompt dài.
2. Giữ LightRAG làm retrieval engine, không biến nó thành nơi lưu lead profile chính.
3. Dùng Redis cho working memory ngắn hạn và graph/session cache.
4. Dùng Postgres làm nguồn sự thật chính cho lead, session và sales events.
5. Dùng Qdrant/LightRAG cho knowledge retrieval: brochure, FAQ, chính sách, pháp lý, playbook.
6. Dùng LangGraph làm lớp orchestration trung tâm.
7. Rule chuyển state quan trọng nên nằm ở code, không giao hoàn toàn cho LLM.

---

## 5. Kiến trúc tổng thể

```text
User Text / Voice
   ↓
Whisper Service (nếu là voice)
   ↓
LangGraph Sales Orchestrator
   ├── Load chat history / lead profile
   ├── Classify intent
   ├── Extract lead slots
   ├── Resolve sales state
   ├── Retrieve context theo state
   ├── Generate response
   └── Persist history + lead + events
   ↓
UI / TTS / Avatar
```

---

## 6. Folder structure đề xuất

```bash
Noble_RAG/
├── service/
│   ├── RAG/
│   │   ├── app.py
│   │   ├── api/
│   │   │   ├── routes_query.py
│   │   │   ├── routes_documents.py
│   │   │   ├── routes_health.py
│   │   │   └── routes_sales.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── dependencies.py
│   │   ├── rag/
│   │   │   ├── lightrag_client.py
│   │   │   ├── retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── reranker.py
│   │   │   └── document_ingest.py
│   │   ├── sales/
│   │   │   ├── graph.py
│   │   │   ├── graph_state.py
│   │   │   ├── policy.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── state_machine.py
│   │   │   ├── edges.py
│   │   │   ├── nodes/
│   │   │   │   ├── ingest_user_turn.py
│   │   │   │   ├── classify_intent.py
│   │   │   │   ├── extract_lead_slots.py
│   │   │   │   ├── update_lead_profile.py
│   │   │   │   ├── resolve_sales_state.py
│   │   │   │   ├── retrieve_context.py
│   │   │   │   ├── build_response.py
│   │   │   │   ├── persist_turn.py
│   │   │   │   └── finalize_output.py
│   │   ├── memory/
│   │   │   ├── chat_history_store.py
│   │   │   ├── lead_profile_store.py
│   │   │   ├── session_store.py
│   │   │   └── redis_store.py
│   │   ├── models/
│   │   │   ├── api_models.py
│   │   │   ├── lead_schema.py
│   │   │   ├── sales_state.py
│   │   │   ├── product_schema.py
│   │   │   └── response_schema.py
│   │   ├── tools/
│   │   │   ├── tavily_tool.py
│   │   │   ├── calculator_tool.py
│   │   │   └── schedule_tool.py
│   │   └── utils/
│   │       ├── text.py
│   │       ├── time.py
│   │       ├── json_extract.py
│   │       └── validators.py
│   └── whisper-service/
├── data/
│   ├── raw/
│   ├── curated/
│   │   ├── projects/
│   │   ├── personas/
│   │   ├── objection_playbooks/
│   │   ├── closing_playbooks/
│   │   └── sales_prompts/
├── scripts/
├── tests/
└── docs/
```

---

## 7. Graph state

LangGraph cần một state object chung cho toàn bộ pipeline.

### Ví dụ `graph_state.py`

```python
from typing import TypedDict, List, Dict, Optional, Any

class SalesAgentState(TypedDict, total=False):
    session_id: str
    user_text: str
    raw_transcript: Optional[str]

    chat_history: List[Dict[str, str]]

    lead_profile: Dict[str, Any]
    session_context: Dict[str, Any]

    detected_intent: Optional[str]
    objection_type: Optional[str]
    buy_signal: Optional[bool]

    current_sales_state: Optional[str]
    next_sales_state: Optional[str]

    missing_slots: List[str]

    retrieved_context: List[Dict[str, Any]]
    recommended_project_ids: List[str]

    draft_response: Optional[str]
    final_response: Optional[str]

    should_persist: bool
    errors: List[str]
```

---

## 8. Sales state machine

### Danh sách state

* `greeting`
* `qualification`
* `need_discovery`
* `budget_alignment`
* `product_matching`
* `comparison`
* `objection_handling`
* `buy_signal`
* `closing_next_step`
* `follow_up`
* `out_of_scope`

### Mục tiêu từng state

#### `greeting`

* Mở đầu tự nhiên
* Không pitch sản phẩm ngay
* Mời khách chia sẻ nhu cầu

#### `qualification`

* Xác định user mua để ở, đầu tư cho thuê, đầu tư tăng giá, giữ tài sản hay chỉ tham khảo

#### `need_discovery`

* Thu các slot còn thiếu

Slot quan trọng:

* `purpose`
* `budget_range`
* `location_preference`
* `property_type`
* `timeline`

#### `budget_alignment`

* Cân chỉnh kỳ vọng sản phẩm với mức ngân sách thực tế

#### `product_matching`

* Đề xuất 1 đến 3 lựa chọn phù hợp
* Luôn nêu vì sao hợp
* Không ném danh sách dài

#### `comparison`

* So sánh 2 hoặc 3 lựa chọn theo tiêu chí cụ thể: giá, vị trí, pháp lý, tiến độ, thanh toán, tiềm năng cho thuê

#### `objection_handling`

* Xử lý phản đối như giá cao, pháp lý, vị trí, chưa đủ tiền, muốn suy nghĩ thêm

#### `buy_signal`

* Nhận diện tín hiệu mua như hỏi bảng giá, hỏi lịch đi xem, hỏi còn căn hay không

#### `closing_next_step`

* Chốt một bước tiếp theo nhỏ như gửi shortlist, gửi bảng giá, hẹn call, hẹn đi xem

#### `follow_up`

* Nhắc lại mối quan tâm chính và kéo lead quay lại funnel

#### `out_of_scope`

* Từ chối lịch sự và đưa hội thoại quay lại bất động sản

---

## 9. Luật chuyển state

Các rule cốt lõi:

* Không được vào `product_matching` nếu chưa có đủ dữ liệu tối thiểu.
* Dữ liệu tối thiểu trước khi pitch gồm:

  * mục đích mua
  * ngân sách
  * khu vực quan tâm hoặc mức linh hoạt khu vực
  * loại hình sản phẩm
* Khi phát hiện objection thì chuyển sang `objection_handling`.
* Khi phát hiện buy signal thì chuyển sang `closing_next_step`.
* Khi user yêu cầu so sánh thì chuyển sang `comparison`.

### Pseudo logic

```python
if out_of_scope:
    return "out_of_scope"
if missing_required_slots:
    return "need_discovery"
if missing_budget:
    return "budget_alignment"
if detected_intent == "comparison":
    return "comparison"
if detected_intent == "objection":
    return "objection_handling"
if detected_intent == "buy_signal":
    return "closing_next_step"
return "product_matching"
```

Rule chuyển state nên nằm ở code trong `state_machine.py`, không để LLM quyết định toàn bộ.

---

## 10. Các node chính trong LangGraph

### 1. `ingest_user_turn`

Nhiệm vụ:

* nhận user text hoặc transcript
* load chat history từ Redis
* load lead profile từ Redis/Postgres
* chuẩn hóa input

### 2. `classify_intent`

Nhiệm vụ:

* detect greeting
* detect ask recommendation
* detect comparison
* detect objection
* detect buy signal
* detect out_of_scope

### 3. `extract_lead_slots`

Nhiệm vụ:

* rút các slot như budget, location, purpose, property type, timeline, financing need

### 4. `update_lead_profile`

Nhiệm vụ:

* merge slot mới vào lead profile
* update lead temperature nếu cần

### 5. `resolve_sales_state`

Nhiệm vụ:

* áp state machine để quyết định `next_sales_state`

### 6. `retrieve_context`

Nhiệm vụ:

* gọi LightRAG hoặc structured store tùy theo state

### 7. `build_response`

Nhiệm vụ:

* build prompt theo state
* generate response text theo style sales

### 8. `persist_turn`

Nhiệm vụ:

* save history vào Redis
* save lead profile vào Postgres
* log sales events

### 9. `finalize_output`

Nhiệm vụ:

* trả response cho UI, TTS hoặc avatar

---

## 11. LangGraph flow đề xuất

```text
START
  ↓
ingest_user_turn
  ↓
classify_intent
  ↓
extract_lead_slots
  ↓
update_lead_profile
  ↓
resolve_sales_state
  ├── need_discovery ─────→ build_response
  ├── product_matching ───→ retrieve_context → build_response
  ├── comparison ─────────→ retrieve_context → build_response
  ├── objection_handling ─→ retrieve_context → build_response
  ├── closing_next_step ──→ retrieve_context → build_response
  └── out_of_scope ───────→ build_response
  ↓
persist_turn
  ↓
finalize_output
  ↓
END
```

---

## 12. `graph.py` skeleton

```python
from langgraph.graph import StateGraph, END
from .graph_state import SalesAgentState
from .nodes.ingest_user_turn import ingest_user_turn
from .nodes.classify_intent import classify_intent
from .nodes.extract_lead_slots import extract_lead_slots
from .nodes.update_lead_profile import update_lead_profile
from .nodes.resolve_sales_state import resolve_sales_state
from .nodes.retrieve_context import retrieve_context
from .nodes.build_response import build_response
from .nodes.persist_turn import persist_turn
from .nodes.finalize_output import finalize_output
from .edges import route_after_state_resolution

builder = StateGraph(SalesAgentState)

builder.add_node("ingest_user_turn", ingest_user_turn)
builder.add_node("classify_intent", classify_intent)
builder.add_node("extract_lead_slots", extract_lead_slots)
builder.add_node("update_lead_profile", update_lead_profile)
builder.add_node("resolve_sales_state", resolve_sales_state)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("build_response", build_response)
builder.add_node("persist_turn", persist_turn)
builder.add_node("finalize_output", finalize_output)

builder.set_entry_point("ingest_user_turn")
builder.add_edge("ingest_user_turn", "classify_intent")
builder.add_edge("classify_intent", "extract_lead_slots")
builder.add_edge("extract_lead_slots", "update_lead_profile")
builder.add_edge("update_lead_profile", "resolve_sales_state")

builder.add_conditional_edges(
    "resolve_sales_state",
    route_after_state_resolution,
    {
        "retrieve_context": "retrieve_context",
        "build_response": "build_response",
    },
)

builder.add_edge("retrieve_context", "build_response")
builder.add_edge("build_response", "persist_turn")
builder.add_edge("persist_turn", "finalize_output")
builder.add_edge("finalize_output", END)

sales_graph = builder.compile()
```

---

## 13. Lead schema

### `LeadProfile`

```python
from pydantic import BaseModel
from typing import Optional, List

class LeadProfile(BaseModel):
    lead_id: str
    name: Optional[str] = None
    purpose: Optional[str] = "khong_ro"
    property_type: Optional[str] = "khong_ro"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_text: Optional[str] = None
    location_preference: List[str] = []
    region_flexibility: Optional[bool] = None
    financing_need: Optional[bool] = None
    financing_ratio: Optional[float] = None
    timeline: Optional[str] = "khong_ro"
    legal_sensitivity: Optional[str] = None
    risk_appetite: Optional[str] = None
    key_needs: List[str] = []
    key_concerns: List[str] = []
    objections: List[str] = []
    recommended_projects: List[str] = []
    shortlisted_projects: List[str] = []
    rejected_projects: List[str] = []
    current_state: str
    lead_temperature: Optional[str] = "cold"
    preferred_contact_channel: Optional[str] = "chat"
    contact_phone: Optional[str] = None
    last_user_intent: Optional[str] = None
    last_next_action: Optional[str] = None
```

### `SessionContext`

```python
from pydantic import BaseModel
from typing import Optional, List

class SessionContext(BaseModel):
    session_id: str
    current_state: str
    previous_state: Optional[str] = None
    last_agent_action: Optional[str] = None
    last_retrieved_context_ids: List[str] = []
    conversation_turn_count: int = 0
```

---

## 14. Lưu dữ liệu ở đâu

### Redis

Dùng cho dữ liệu nóng, phục vụ session đang chạy:

* `chat_history:{session_id}`
* `session_context:{session_id}`
* `lead_profile_cache:{session_id}`
* `graph_checkpoint:{session_id}`
* `last_recommendations:{session_id}`

### Postgres

Dùng làm nguồn sự thật chính cho dữ liệu sales:

* `lead_profiles`
* `conversation_sessions`
* `conversation_turns`
* `sales_events`
* `appointment_requests`

### Qdrant / LightRAG

Dùng cho retrieval knowledge:

* brochure
* FAQ
* pháp lý
* chính sách bán hàng
* objection playbooks
* closing playbooks
* comparison knowledge

Không dùng Qdrant làm nơi chính để lưu lead profile.

---

## 15. Product schema

Nên có dữ liệu dự án dạng có cấu trúc song song với tài liệu text.

```python
from pydantic import BaseModel
from typing import Optional, List, Dict

class ProductProject(BaseModel):
    project_id: str
    project_name: str
    city: str
    district: str
    ward: Optional[str] = None
    property_types: List[str]
    target_personas: List[str]
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    legal_status: Optional[str] = None
    handover_time: Optional[str] = None
    payment_policy: Optional[str] = None
    bank_support: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []
    fit_rules: List[str] = []
    disqualify_rules: List[str] = []
    objection_answers: Dict = {}
    cta_recommendation: Optional[str] = None
```

---

## 16. Retrieval strategy theo state

### `need_discovery`

Retrieve nhẹ:

* discovery question bank
* buyer persona prompts

### `product_matching`

Retrieve:

* project schema
* product catalog
* pricing band
* project docs liên quan

### `comparison`

Retrieve:

* 2 hoặc 3 project schemas
* comparison snippets
* khác biệt về giá, pháp lý, vị trí, thanh toán

### `objection_handling`

Retrieve:

* objection playbook
* FAQ liên quan
* policy hoặc legal docs nếu cần

### `closing_next_step`

Retrieve:

* CTA playbook
* shortlist templates
* bảng giá hoặc brochure phù hợp

Node `retrieve_context` nên chọn đúng nguồn theo `next_sales_state`.

---

## 17. Prompt sales

### System prompt chính

```text
Bạn là AI Sales Agent bất động sản chuyên tư vấn các sản phẩm/dự án trong hệ thống Noble.

MỤC TIÊU CHÍNH
- Hiểu đúng nhu cầu thật của khách hàng.
- Đề xuất sản phẩm phù hợp nhất dựa trên hồ sơ khách và dữ liệu retrieve được.
- Xử lý băn khoăn một cách trung thực, tinh tế, không ép mua.
- Dẫn hội thoại tới bước tiếp theo rõ ràng như gửi shortlist, gửi bảng giá, hẹn call, hẹn xem dự án.

VAI TRÒ
- Bạn không phải chatbot FAQ thuần túy.
- Bạn là chuyên viên sales tư vấn theo quy trình.
- Bạn cần chủ động dẫn dắt nhưng vẫn tự nhiên và tôn trọng khách.

NGUYÊN TẮC BẮT BUỘC
1. Không bịa thông tin về giá, pháp lý, ưu đãi, tiến độ, tồn kho.
2. Chỉ sử dụng thông tin có trong context, knowledge base hoặc tool results.
3. Nếu thiếu dữ liệu, hãy nói rõ chưa đủ thông tin và hỏi thêm hoặc đề xuất bước tiếp theo phù hợp.
4. Không cam kết lợi nhuận, không hứa chắc tăng giá, không dùng lời lẽ thao túng.
5. Không pitch sản phẩm cụ thể khi chưa hiểu tối thiểu:
   - mục đích mua
   - ngân sách
   - khu vực quan tâm hoặc mức linh hoạt khu vực
   - loại hình sản phẩm
6. Sau mỗi lượt trả lời, cố gắng tạo ra một bước tiến nhỏ trong sales funnel.

PHONG CÁCH
- Xưng "em", gọi khách là "Anh/Chị".
- Tự nhiên, gọn, rõ.
- Không chào hỏi máy móc ở mọi lượt.
- Không trả lời như robot.
- Không liệt kê quá nhiều lựa chọn cùng lúc.
- Ưu tiên 1 đến 3 phương án tốt nhất.
```

### Prompt theo state

#### NEED_DISCOVERY

```text
TRẠNG THÁI HIỆN TẠI: NEED_DISCOVERY
NHIỆM VỤ
- Thu thêm thông tin còn thiếu để có thể tư vấn chính xác.
- Chỉ hỏi tối đa 2 câu ngắn.
- Không đề xuất dự án cụ thể trừ khi đã có đủ dữ liệu cơ bản.
```

#### PRODUCT_MATCHING

```text
TRẠNG THÁI HIỆN TẠI: PRODUCT_MATCHING
NHIỆM VỤ
- Đề xuất tối đa 3 lựa chọn phù hợp.
- Mỗi lựa chọn phải có:
  1. tên dự án/sản phẩm
  2. vì sao phù hợp với khách
  3. 1 điểm cần lưu ý
```

#### OBJECTION_HANDLING

```text
TRẠNG THÁI HIỆN TẠI: OBJECTION_HANDLING
NHIỆM VỤ
- Xác định phản đối chính.
- Đồng cảm trước, giải thích sau.
- Không tranh luận tay đôi.
- Không ép mua.
- Sau khi xử lý, mở ra 1 bước tiếp theo hợp lý.
```

#### CLOSING_NEXT_STEP

```text
TRẠNG THÁI HIỆN TẠI: CLOSING_NEXT_STEP
NHIỆM VỤ
- Chốt một bước tiếp theo nhỏ, rõ ràng.
- Ưu tiên CTA phù hợp với mức nóng của lead.
- Không tạo cảm giác ép.
```

---

## 18. API endpoints nên thêm

Tạo `routes_sales.py` với các endpoint:

* `POST /sales/chat`
* `POST /sales/chat/stream`
* `GET /sales/lead/{session_id}`
* `PATCH /sales/lead/{session_id}`
* `GET /sales/state/{session_id}`
* `POST /sales/recommendations/refresh`
* `POST /sales/followup/generate`

`/sales/chat` và `/sales/chat/stream` sẽ gọi `sales_graph.invoke(...)` hoặc `sales_graph.astream(...)`.

---

## 19. Tách trách nhiệm khỏi `app.py`

Sau refactor:

* `app.py`: chỉ init FastAPI, dependency injection, route registration
* `sales/graph.py`: định nghĩa LangGraph
* `sales/graph_state.py`: khai báo state
* `sales/state_machine.py`: resolve state
* `sales/prompt_builder.py`: dựng prompt theo state
* `sales/nodes/*`: node xử lý từng bước
* `memory/chat_history_store.py`: đọc ghi history
* `memory/lead_profile_store.py`: đọc ghi lead profile
* `rag/retriever.py`: retrieval logic

---

## 20. Dữ liệu curated nên có

### Project data

Mỗi dự án nên có file JSON riêng trong `data/curated/projects/`.

### Objection playbooks

Ví dụ:

* `gia_cao.yaml`
* `phap_ly.yaml`
* `vi_tri.yaml`
* `chua_du_tien.yaml`
* `suy_nghi_them.yaml`

### Closing playbooks

Ví dụ:

* `book_site_visit.yaml`
* `book_call.yaml`
* `send_shortlist.yaml`

### Sales prompts

Tách riêng prompt hệ thống, prompt theo state, refusal rules.

---

## 21. Roadmap triển khai

### Phase 1

* Tách `app.py`
* Thêm `graph_state.py`
* Thêm `SalesState`
* Thêm `LeadProfile`
* Thêm `sales/graph.py`
* Thêm `POST /sales/chat`

### Phase 2

* Thêm các node cơ bản:

  * `ingest_user_turn`
  * `classify_intent`
  * `extract_lead_slots`
  * `update_lead_profile`
  * `resolve_sales_state`
  * `build_response`
  * `persist_turn`
* Thêm graph persistence cơ bản

### Phase 3

* Thêm `retrieve_context` theo state
* Thêm `product_schema.py`
* Thêm `product_matcher.py`
* Thêm persist Postgres cho lead profile và sales events

### Phase 4

* Thêm objection playbooks
* Thêm closing playbooks
* Thêm streaming graph output
* Thêm evaluation test cases

### Phase 5

* UI hiển thị state, lead profile và retrieved products
* Kết nối TTS hoặc avatar
* Theo dõi conversion funnel

---

## 22. Definition of Done

Một bản refactor được coi là hoàn tất khi:

* Agent biết hỏi đủ nhu cầu trước khi pitch
* Agent lưu lead profile có cấu trúc
* Agent điều phối hội thoại qua LangGraph thay vì một chain đơn
* Agent đề xuất sản phẩm bằng logic phù hợp, không trả lời kiểu brochure bot
* Agent xử lý objection bằng playbook
* Agent biết chốt bước tiếp theo
* Retrieval được dùng theo state, không dùng một kiểu chung cho mọi case
* Prompt được tách thành system prompt + state prompt + retrieved context
* Node transition quan trọng được kiểm soát bằng code

---

## 23. Kết luận

Refactor đúng theo hướng LangGraph không phải là kéo dài prompt hiện tại, mà là thêm một **graph-based sales brain layer** phía trên LightRAG.

Công thức đúng là:

**Whisper + LightRAG + LangGraph + Redis history + Lead Profile + Sales State Machine + Prompt Builder + Playbooks + Postgres persistence**

Khi hoàn thành, hệ thống sẽ chuyển từ một **RAG assistant bất động sản** sang một **AI sales agent bất động sản có kịch bản, có state, có orchestration và khả năng dẫn dắt hội thoại thực sự**.

---
