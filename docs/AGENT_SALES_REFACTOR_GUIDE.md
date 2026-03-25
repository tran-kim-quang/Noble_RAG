# Tài liệu triển khai AI Sales Agent Bất động sản theo Script Engine + LangGraph

## 1. Mục tiêu

Tài liệu này mô tả cách triển khai hệ thống AI Sales Agent bất động sản theo hướng:

- điều khiển bằng **workflow/script engine**
- orchestration bằng **LangGraph**
- không phụ thuộc vào việc “siết prompt”
- tách rõ **state / step / action / validator / retrieval / persistence**

Mục tiêu là để agent **follow đúng kịch bản sale** theo tài liệu nghiệp vụ, thay vì chỉ “nói giống sales”.

---

## 2. Nguyên tắc thiết kế

### 2.1. Không dùng prompt làm bộ điều khiển chính
Prompt chỉ còn vai trò phụ:
- diễn đạt câu trả lời tự nhiên ở một số bước
- hỗ trợ giải thích, objection handling, closing

Prompt **không được quyết định**:
- đang ở bước nào
- nên hỏi slot nào tiếp theo
- có được tư vấn chưa
- được phép nhắc dự án nào
- được phép chốt hay chưa

Các quyết định này phải nằm ở **code**.

### 2.2. Phân tầng điều khiển
Hệ thống phải tách thành các tầng sau:

1. **slot extraction**
2. **script step resolution**
3. **action selection**
4. **candidate retrieval**
5. **response rendering**
6. **response validation**
7. **persistence**

### 2.3. “State” và “Step” là hai khái niệm khác nhau
- **sales_state**: trạng thái lớn như greeting, discovery, matching, objection, closing
- **script_step**: bước cụ thể trong kịch bản sale

Muốn follow đúng tài liệu sale, phải lấy **script_step** làm trung tâm.

---

## 3. Kịch bản sale được encode như thế nào

Từ tài liệu nghiệp vụ, hệ thống cần encode đúng 10 chặng chính:

1. mở đầu – chào hỏi & tạo thiện cảm
2. khảo sát nhu cầu qualify nhanh
3. khai thác sâu theo nhánh mua ở / đầu tư
4. đề xuất 2 phương án phù hợp
5. tư vấn chi tiết từng phương án
6. kiểm tra mức độ quan tâm
7. xử lý từ chối
8. chốt mềm
9. chốt hành động
10. kết thúc & chăm sóc

Quan trọng nhất:
- **hỏi từng câu một**
- khách trả lời xong mới hỏi tiếp
- nhánh mua ở và đầu tư phải tách riêng
- matching phải ra đúng **2 phương án**
- objection và closing phải theo đúng trình tự script

---

## 4. Kiến trúc tổng thể

```text
User Input (Text / Voice)
  -> Whisper (nếu là voice)
  -> LangGraph Orchestrator
      -> ingest_user_turn
      -> classify_and_extract
      -> update_lead_profile
      -> resolve_sales_state
      -> resolve_script_step
      -> decide_response_action
      -> retrieve_candidates_if_needed
      -> render_response_from_action
      -> validate_response
      -> persist_turn
  -> UI / TTS / Avatar
```

---

## 5. Folder structure đề xuất

```bash
service/RAG/
├── api/
│   ├── routes_sales.py
│   └── routes_query.py
├── sales/
│   ├── graph.py
│   ├── graph_state.py
│   ├── state_machine.py
│   ├── step_resolver.py
│   ├── action_policy.py
│   ├── response_validator.py
│   ├── response_templates.py
│   ├── product_policy.py
│   ├── prompt_builder.py
│   ├── script_loader.py
│   ├── scripts/
│   │   └── sales_script.yaml
│   └── nodes/
│       ├── ingest_user_turn.py
│       ├── classify_and_extract.py
│       ├── update_lead_profile.py
│       ├── resolve_sales_state.py
│       ├── resolve_script_step.py
│       ├── decide_response_action.py
│       ├── retrieve_candidates.py
│       ├── render_response.py
│       ├── validate_response.py
│       ├── persist_turn.py
│       └── finalize_output.py
├── memory/
│   ├── chat_history_store.py
│   ├── lead_profile_store.py
│   └── session_store.py
├── models/
│   ├── api_models.py
│   ├── lead_schema.py
│   ├── sales_state.py
│   ├── script_step.py
│   ├── response_action.py
│   └── product_schema.py
└── rag/
    ├── retriever.py
    ├── candidate_retriever.py
    └── reranker.py
```

---

## 6. Dữ liệu trung tâm của graph

### 6.1. SalesAgentState

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
    buy_signal: bool
    extracted_slots: Dict[str, Any]

    current_sales_state: Optional[str]
    next_sales_state: Optional[str]

    current_script_step: Optional[str]
    next_script_step: Optional[str]

    response_action: Optional[str]

    missing_slots: List[str]
    retrieved_candidates: List[Dict[str, Any]]
    retrieved_context: List[Dict[str, Any]]

    draft_response: Optional[str]
    final_response: Optional[str]

    validation_errors: List[str]
    errors: List[str]
```

---

## 7. Lead schema

```python
from pydantic import BaseModel
from typing import Optional, List

class LeadProfile(BaseModel):
    lead_id: str
    name: Optional[str] = None

    purpose: Optional[str] = "khong_ro"
    property_type: Optional[str] = "khong_ro"

    family_member_count: Optional[int] = None
    children_count: Optional[int] = None

    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_text: Optional[str] = None

    location_preference: List[str] = []
    timeline: Optional[str] = None
    financing_need: Optional[bool] = None

    investment_horizon: Optional[str] = None
    expected_return: Optional[str] = None
    risk_preference: Optional[str] = None

    key_concerns: List[str] = []
    special_needs: List[str] = []

    shortlisted_projects: List[str] = []
    rejected_projects: List[str] = []

    current_state: str = "greeting"
    current_script_step: str = "S1_opening"
    lead_temperature: str = "cold"

    last_user_intent: Optional[str] = None
    last_action: Optional[str] = None
```

---

## 8. Sales state và script step

### 8.1. Sales state

```python
class SalesState(str, Enum):
    GREETING = "greeting"
    QUALIFY = "qualify"
    LIVE_DISCOVERY = "live_discovery"
    INVEST_DISCOVERY = "invest_discovery"
    MATCHING = "matching"
    DETAILING = "detailing"
    INTEREST_CHECK = "interest_check"
    OBJECTION = "objection"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"
```

### 8.2. Script step

```python
class ScriptStep(str, Enum):
    S1_OPENING = "S1_opening"
    S2_ASK_PURPOSE = "S2_ask_purpose"
    S3_ASK_LOCATION = "S3_ask_location"
    S4_ASK_BUDGET = "S4_ask_budget"
    S5_ASK_TIMELINE = "S5_ask_timeline"

    L1_LIVE_ASK_FAMILY_SIZE = "L1_live_ask_family_size"
    L2_LIVE_ASK_PRIORITY = "L2_live_ask_priority"
    L3_LIVE_ASK_SPECIAL_NEEDS = "L3_live_ask_special_needs"

    I1_INVEST_ASK_HORIZON = "I1_invest_ask_horizon"
    I2_INVEST_ASK_EXPECTED_RETURN = "I2_invest_ask_expected_return"
    I3_INVEST_ASK_RISK_PREFERENCE = "I3_invest_ask_risk_preference"

    M1_MATCH_TWO_OPTIONS = "M1_match_two_options"
    M2_EXPLAIN_OPTION_DETAIL = "M2_explain_option_detail"
    M3_INTEREST_CHECK = "M3_interest_check"

    O1_HANDLE_PRICE_OBJECTION = "O1_handle_price_objection"
    O2_HANDLE_INDECISION = "O2_handle_indecision"

    C1_SOFT_CLOSE = "C1_soft_close"
    C2_HARD_CLOSE_LIGHT = "C2_hard_close_light"
    F1_FOLLOWUP_CLOSEOUT = "F1_followup_closeout"
```

---

## 9. Script engine: encode kịch bản thành YAML

### 9.1. Mục tiêu
Thay vì để tài liệu sale nằm ở dạng text, cần chuyển thành config có thể chạy được.

### 9.2. Ví dụ `sales_script.yaml`

```yaml
steps:
  - id: S1_opening
    state: greeting
    action: ASK_PURPOSE
    ask_one_by_one: true
    exit_conditions: []
    next:
      default: S2_ask_purpose

  - id: S2_ask_purpose
    state: qualify
    action: ASK_PURPOSE
    ask_one_by_one: true
    exit_conditions:
      - purpose
    next:
      when_present: S3_ask_location
      when_missing: S2_ask_purpose

  - id: S3_ask_location
    state: qualify
    action: ASK_LOCATION
    ask_one_by_one: true
    exit_conditions:
      - location_preference
    next:
      when_present: S4_ask_budget
      when_missing: S3_ask_location

  - id: S4_ask_budget
    state: qualify
    action: ASK_BUDGET
    ask_one_by_one: true
    exit_conditions:
      - budget_text_or_range
    next:
      when_present: S5_ask_timeline
      when_missing: S4_ask_budget

  - id: S5_ask_timeline
    state: qualify
    action: ASK_TIMELINE
    ask_one_by_one: true
    exit_conditions:
      - timeline
    next:
      when_live: L1_live_ask_family_size
      when_invest: I1_invest_ask_horizon

  - id: L1_live_ask_family_size
    state: live_discovery
    action: ASK_FAMILY_SIZE
    exit_conditions:
      - family_member_count
    next:
      when_present: L2_live_ask_priority

  - id: L2_live_ask_priority
    state: live_discovery
    action: ASK_LIVE_PRIORITY
    exit_conditions:
      - key_concerns
    next:
      when_present: L3_live_ask_special_needs

  - id: L3_live_ask_special_needs
    state: live_discovery
    action: ASK_SPECIAL_NEEDS
    exit_conditions:
      - live_branch_complete
    next:
      when_present: M1_match_two_options

  - id: I1_invest_ask_horizon
    state: invest_discovery
    action: ASK_INVEST_HORIZON
    exit_conditions:
      - investment_horizon
    next:
      when_present: I2_invest_ask_expected_return

  - id: I2_invest_ask_expected_return
    state: invest_discovery
    action: ASK_EXPECTED_RETURN
    exit_conditions:
      - expected_return
    next:
      when_present: I3_invest_ask_risk_preference

  - id: I3_invest_ask_risk_preference
    state: invest_discovery
    action: ASK_RISK_PREFERENCE
    exit_conditions:
      - risk_preference
    next:
      when_present: M1_match_two_options

  - id: M1_match_two_options
    state: matching
    action: MATCH_TWO_OPTIONS
    requires_candidates: true
    max_candidates: 2
    next:
      default: M2_explain_option_detail

  - id: M2_explain_option_detail
    state: detailing
    action: EXPLAIN_OPTION_DETAIL
    requires_candidates: true
    next:
      default: M3_interest_check

  - id: M3_interest_check
    state: interest_check
    action: CHECK_INTEREST
    next:
      if_price_objection: O1_handle_price_objection
      if_indecision: O2_handle_indecision
      if_positive_interest: C1_soft_close

  - id: O1_handle_price_objection
    state: objection
    action: HANDLE_PRICE_OBJECTION
    next:
      default: C1_soft_close

  - id: O2_handle_indecision
    state: objection
    action: HANDLE_INDECISION
    next:
      default: F1_followup_closeout

  - id: C1_soft_close
    state: closing
    action: SOFT_CLOSE
    next:
      if_positive_interest: C2_hard_close_light
      default: F1_followup_closeout

  - id: C2_hard_close_light
    state: closing
    action: HARD_CLOSE_LIGHT
    next:
      default: F1_followup_closeout

  - id: F1_followup_closeout
    state: follow_up
    action: FOLLOWUP_CLOSEOUT
    next:
      default: F1_followup_closeout
```

---

## 10. ResponseAction: hẹp và có kiểm soát

```python
class ResponseAction(str, Enum):
    ASK_PURPOSE = "ask_purpose"
    ASK_LOCATION = "ask_location"
    ASK_BUDGET = "ask_budget"
    ASK_TIMELINE = "ask_timeline"

    ASK_FAMILY_SIZE = "ask_family_size"
    ASK_LIVE_PRIORITY = "ask_live_priority"
    ASK_SPECIAL_NEEDS = "ask_special_needs"

    ASK_INVEST_HORIZON = "ask_invest_horizon"
    ASK_EXPECTED_RETURN = "ask_expected_return"
    ASK_RISK_PREFERENCE = "ask_risk_preference"

    MATCH_TWO_OPTIONS = "match_two_options"
    EXPLAIN_OPTION_DETAIL = "explain_option_detail"
    CHECK_INTEREST = "check_interest"

    HANDLE_PRICE_OBJECTION = "handle_price_objection"
    HANDLE_INDECISION = "handle_indecision"

    SOFT_CLOSE = "soft_close"
    HARD_CLOSE_LIGHT = "hard_close_light"
    FOLLOWUP_CLOSEOUT = "followup_closeout"
```

Luật quan trọng:
- mỗi `script_step` chỉ có **1 action chính**
- action được chọn bằng code, không do LLM chọn

---

## 11. Template cho các bước hỏi

Các bước hỏi không nên dùng LLM tự do.

### `response_templates.py`

```python
RESPONSE_TEMPLATES = {
    "ASK_PURPOSE": "Không biết hiện tại Anh/Chị đang quan tâm đến bất động sản để ở hay đầu tư ạ?",
    "ASK_LOCATION": "Mình đang quan tâm khu vực nào ạ?",
    "ASK_BUDGET": "Tầm tài chính dự kiến của mình khoảng bao nhiêu để em lọc sản phẩm phù hợp nhất ạ?",
    "ASK_TIMELINE": "Thời gian dự kiến mình mua là trong khoảng nào ạ?",

    "ASK_FAMILY_SIZE": "Gia đình mình có khoảng bao nhiêu thành viên ạ?",
    "ASK_LIVE_PRIORITY": "Anh/Chị ưu tiên yếu tố nào hơn: tiện ích sống, môi trường hay khả năng di chuyển ạ?",
    "ASK_SPECIAL_NEEDS": "Mình có yêu cầu đặc biệt nào về không gian sống không ạ?",

    "ASK_INVEST_HORIZON": "Mình ưu tiên đầu tư ngắn hạn hay dài hạn ạ?",
    "ASK_EXPECTED_RETURN": "Kỳ vọng lợi nhuận của mình khoảng bao nhiêu phần trăm một năm ạ?",
    "ASK_RISK_PREFERENCE": "Anh/Chị ưu tiên an toàn vốn hay tăng trưởng mạnh ạ?",

    "CHECK_INTEREST": "Không biết với 2 phương án này, Anh/Chị đang cảm thấy phù hợp hơn với hướng nào ạ? Em có thể đi sâu hơn vào phương án mình quan tâm.",
    "SOFT_CLOSE": "Dạ, nếu để chọn trong thời điểm này, Anh/Chị đang nghiêng về phương án nào hơn để em hỗ trợ sâu hơn ạ?",
    "HARD_CLOSE_LIGHT": "Hiện tại bên em đang có chính sách hỗ trợ giữ chỗ ưu tiên cho khách hàng quan tâm sớm. Anh/Chị có muốn em hỗ trợ giữ trước một vị trí đẹp để mình cân nhắc thêm không ạ?",
    "FOLLOWUP_CLOSEOUT": "Em cảm ơn Anh/Chị đã dành thời gian trao đổi ạ. Em xin phép gửi thông tin chi tiết qua Zalo/Email để mình tiện tham khảo, khi cần thêm hỗ trợ Anh/Chị cứ nhắn em bất cứ lúc nào ạ."
}
```

Template giúp tránh:
- reset lời chào
- hỏi sai slot
- hỏi nhiều ý cùng lúc
- suy diễn khu vực / dự án

---

## 12. Những bước nào được dùng LLM

### 12.1. Không dùng LLM hoặc chỉ template
- opening
- hỏi purpose
- hỏi location
- hỏi budget
- hỏi timeline
- hỏi family size
- hỏi live priority
- hỏi special needs
- hỏi invest horizon
- hỏi expected return
- hỏi risk preference
- check interest
- soft close
- hard close nhẹ
- kết thúc chăm sóc

### 12.2. Dùng LLM nhưng phải constrained
- `MATCH_TWO_OPTIONS`
- `EXPLAIN_OPTION_DETAIL`
- `HANDLE_PRICE_OBJECTION`
- `HANDLE_INDECISION`

Ở các action này, LLM chỉ được diễn đạt trong phạm vi:
- candidate list đã retrieve
- insight hiện có trong lead profile
- facts đã retrieve từ KB

---

## 13. Product policy

Mục tiêu: ngăn tư vấn lệch nhu cầu.

### Ví dụ `product_policy.py`

```python
def allowed_candidates(lead_profile, candidates):
    purpose = lead_profile.get("purpose")
    concerns = set(lead_profile.get("key_concerns") or [])
    property_type = lead_profile.get("property_type")

    filtered = []
    for c in candidates:
        c_type = c.get("property_type")

        if purpose == "mua_o" and c_type == "shophouse":
            continue

        if property_type not in (None, "", "khong_ro") and c_type != property_type:
            continue

        if "gần trường học" in concerns and not c.get("family_friendly", False):
            continue

        filtered.append(c)

    return filtered
```

Policy này phải chạy trước khi render recommendation.

---

## 14. Retrieval strategy theo step

### 14.1. Với bước hỏi slot
Không cần retrieve hoặc chỉ retrieve question bank.

### 14.2. Với bước matching
`M1_match_two_options` phải retrieve **candidate list có cấu trúc**, không chỉ một cục text.

Ví dụ output từ retriever:

```json
[
  {
    "project_id": "noble_palace_tay_thang_long",
    "project_name": "Noble Palace Tây Thăng Long",
    "property_type": "nha_pho",
    "fit_score": 0.82,
    "fit_reasons": ["phù hợp mua ở", "gần tiện ích gia đình"],
    "risk_notes": ["cần kiểm tra thêm mức tài chính"],
    "source_snippets": ["..."]
  },
  {
    "project_id": "project_b",
    "project_name": "...",
    "property_type": "can_ho",
    "fit_score": 0.79,
    "fit_reasons": ["gần trường học", "phù hợp gia đình trẻ"],
    "risk_notes": ["..."],
    "source_snippets": ["..."]
  }
]
```

### 14.3. Với objection handling
Retrieve đúng playbook objection và facts hỗ trợ.

### 14.4. Với closing
Retrieve CTA / policy / asset cần gửi.

---

## 15. Render response theo action

### 15.1. Với action dạng hỏi slot
Trả template trực tiếp.

### 15.2. Với action dạng recommend / explain / objection
Dùng renderer có khung dữ liệu:

```python
render_input = {
    "action": "MATCH_TWO_OPTIONS",
    "lead_profile": lead_profile,
    "candidates": filtered_candidates,
    "allowed_project_ids": [c["project_id"] for c in filtered_candidates],
    "must_not_ask_more_slots": True,
    "must_not_greet": True,
    "must_use_retrieved_facts_only": True,
}
```

LLM chỉ được diễn đạt lại.

---

## 16. Response validator

Đây là lớp bắt buộc để hệ ổn định.

### 16.1. Ví dụ rule validator

#### Nếu action = `ASK_LOCATION`
- phải có đúng 1 câu hỏi
- không được chào lại
- không được hỏi budget/timeline
- không được nêu dự án/khu vực mẫu

#### Nếu action = `MATCH_TWO_OPTIONS`
- chỉ được nhắc tối đa 2 dự án
- các dự án phải nằm trong `allowed_project_ids`
- không được hỏi slot mới
- không được nhắc loại hình không phù hợp policy

#### Nếu action = `HANDLE_PRICE_OBJECTION`
- phải có phần đồng cảm
- phải có phần phương án tài chính khác hoặc lựa chọn thay thế
- không được hard close ngay

#### Nếu action = `SOFT_CLOSE`
- phải hỏi nghiêng về phương án nào
- không được quay lại hỏi slot

### 16.2. Hành vi khi fail validation
- regenerate 1 lần với prompt sửa lỗi
- nếu vẫn fail → fallback về template / response deterministic

---

## 17. LangGraph flow chi tiết

```text
START
  ↓
ingest_user_turn
  ↓
classify_and_extract
  ↓
update_lead_profile
  ↓
resolve_sales_state
  ↓
resolve_script_step
  ↓
decide_response_action
  ├── nếu action hỏi slot  ─────────→ render_response_from_template
  └── nếu action cần dữ liệu ───────→ retrieve_candidates_if_needed
                                         ↓
                                   apply_product_policy
                                         ↓
                                   render_response_constrained
  ↓
validate_response
  ├── valid   → persist_turn
  └── invalid → regenerate_or_fallback
  ↓
finalize_output
  ↓
END
```

---

## 18. Các node cần thêm

### `resolve_script_step.py`
Trách nhiệm:
- đọc `current_script_step`
- đọc slot đã có / slot còn thiếu
- đọc intent / objection / buy signal
- trả về `next_script_step`

### `decide_response_action.py`
Trách nhiệm:
- map `next_script_step` → `response_action`
- xác định lượt này có được hỏi tiếp hay phải recommend/close

### `retrieve_candidates.py`
Trách nhiệm:
- retrieve candidates có cấu trúc cho bước matching/detailing/objection/closing

### `response_validator.py`
Trách nhiệm:
- validate response theo action + step + allowed candidate list

### `response_templates.py`
Trách nhiệm:
- chứa template cứng cho các bước hỏi và close cơ bản

### `product_policy.py`
Trách nhiệm:
- lọc candidate theo purpose / property type / insight / concern

---

## 19. Pseudo-code cho orchestration

```python
async def handle_sales_turn(session_id: str, user_text: str):
    state = await ingest_user_turn(session_id, user_text)

    state = await classify_and_extract(state)
    state = update_lead_profile(state)

    state = resolve_sales_state(state)
    state = resolve_script_step(state)
    state = decide_response_action(state)

    action = state["response_action"]

    if action in TEMPLATE_ACTIONS:
        response = render_template(action, state)
    else:
        candidates = await retrieve_candidates(state)
        candidates = apply_product_policy(state["lead_profile"], candidates)
        response = await render_constrained(action, state, candidates)

    validation = validate_response(response, state)
    if not validation.ok:
        response = fallback_response(action, state, validation)

    state["final_response"] = response
    await persist_turn(state)
    return state
```

---

## 20. Persistence

### Redis
Dùng cho:
- chat history
- session context
- script step hiện tại
- graph checkpoint

### Postgres
Dùng cho:
- lead profile
- conversation sessions
- conversation turns
- sales events
- objection logs
- recommendation history

### Ghi thêm các trường mới
Nên lưu thêm:
- `current_script_step`
- `last_response_action`
- `last_recommended_project_ids`
- `last_validation_errors`

---

## 21. API contract đề xuất

### `POST /sales/chat`
Request:

```json
{
  "session_id": "optional",
  "message": "Gia đình mình có 4 người, 2 con nhỏ"
}
```

Response:

```json
{
  "session_id": "sales_regression_001",
  "response": "Anh/Chị đang ưu tiên mua để ở hay đầu tư ạ?",
  "sales_state": "qualify",
  "script_step": "S2_ask_purpose",
  "response_action": "ASK_PURPOSE",
  "lead_profile": {...},
  "missing_slots": [...]
}
```

### `POST /sales/chat/stream`
- stream theo node hoặc theo chunk cuối
- không hardcode opening chunk giống nhau cho mọi state

---

## 22. Regression test bắt buộc

### Case 1 – opening
Input: `xin chào`
Expected:
- step = `S2_ask_purpose`
- action = `ASK_PURPOSE`
- đúng 1 câu hỏi
- không hỏi property type
- không nhắc dự án

### Case 2 – gia đình
Input: `gia đình mình có 4 người, 2 con nhỏ`
Expected:
- không chào lại
- nếu purpose thiếu → hỏi `ASK_PURPOSE`
- không reset opening

### Case 3 – purpose + location
Input: `mình mua để ở, ưu tiên khu vực Hà Nội gần trường học`
Expected:
- step tiếp theo là hỏi budget hoặc timeline
- không fallback “chưa có dữ liệu từ kho dự án”

### Case 4 – đủ qualify, sang matching
Input: đầy đủ purpose + location + budget + timeline
Expected:
- step = `M1_match_two_options`
- output đúng 2 phương án
- không hỏi thêm slot nếu script chưa yêu cầu

### Case 5 – objection price
Input: `giá hơi cao`
Expected:
- step = `O1_handle_price_objection`
- có đồng cảm
- có mở ra phương án tài chính khác
- không chốt cứng ngay

---

## 23. Roadmap triển khai

### Phase 1
- thêm `script_step.py`
- thêm `response_action.py`
- thêm `sales_script.yaml`
- thêm `step_resolver.py`
- thêm `action_policy.py`

### Phase 2
- thêm `response_templates.py`
- chuyển các bước hỏi sang template deterministic
- thêm `response_validator.py`

### Phase 3
- thêm `candidate_retriever.py`
- thêm `product_policy.py`
- giới hạn matching còn đúng 2 candidates

### Phase 4
- constrained renderer cho recommend / explain / objection / close
- thêm structured response contract

### Phase 5
- regression test theo conversation
- đánh giá tỉ lệ fail validation
- tối ưu streaming theo state/action

---

## 24. Definition of Done

Hoàn tất khi hệ thống đạt được:

- follow đúng step trong kịch bản sale
- hỏi từng câu một ở phase discovery
- không reset greeting giữa cuộc hội thoại
- không hỏi sai slot
- không tư vấn trước khi đủ điều kiện theo script
- matching đúng 2 phương án
- tư vấn chỉ bám candidate/context được retrieve
- objection handling và closing đúng trình tự
- output qua được validator trước khi trả cho user

---

## 25. Kết luận

Cách triển khai đúng không phải là “viết prompt tốt hơn”, mà là:

- biến tài liệu sale thành **`sales_script.yaml`**
- dùng LangGraph để điều phối **state + step + action**
- dùng template cho các bước hỏi
- dùng policy cho product filtering
- dùng validator để chặn output lệch script

Công thức cuối cùng:

**LangGraph + Script Engine + Action Templates + Product Policy + Response Validator + Structured Retrieval + Persistence**

Đây là hướng đủ ổn định để AI agent thật sự đi theo kịch bản sale, thay vì chỉ “nói giống sales”.
