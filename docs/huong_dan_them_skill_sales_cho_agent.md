# Hướng dẫn sửa luồng hệ thống hiện tại để thêm skill sales cho agent

Tài liệu này dùng để phác thảo code trên nhánh `test-screen-agents`, theo hướng **giữ kiến trúc 2 routes hiện tại** nhưng bổ sung thêm lớp **sales skill** để agent:

- bớt trả lời kiểu brochure
- biết khách đang ở đâu trong funnel
- biết lượt này cần làm gì
- biết lúc nào nên tư vấn, lúc nào nên gợi mở, lúc nào nên mời bước tiếp theo, lúc nào handoff cho sales thật

---

## 1. Bài toán hiện tại

Trên code hiện tại, hệ thống đã có:

- `consult_discovery`
- `project_grounded`
- `query_type`
- `retrieval_readiness`
- `response_mode`
- `ask_policy`
- chain `consult -> project`

Tuy nhiên hệ vẫn thiếu lớp **sales behavior** nên model còn:

- trả lời an toàn, hơi “cho có”
- giới thiệu dự án kiểu chung chung
- không có “đích hội thoại” rõ trong từng lượt
- chưa biết dẫn khách đi tiếp trong funnel
- chưa biết khi nào nên nurture, khi nào nên handoff

Tóm lại:

> Route đã khá ổn, nhưng skill sales chưa được mô hình hóa thành state + response strategy.

---

## 2. Mục tiêu của lần sửa này

Cần thêm cho agent 4 năng lực:

1. **Biết khách đang ở state nào trong funnel sales**
2. **Biết mục tiêu hội thoại của lượt hiện tại là gì**
3. **Biết nên phản hồi theo mode nào**
4. **Biết next step phù hợp nhất là gì**

Tức là nâng hệ từ:

- `route -> answer`

thành:

- `route -> sales_state -> conversation_goal -> response_mode -> answer`

---

## 3. Nguyên tắc thiết kế

### 3.1. Không thay kiến trúc 2 routes

Giữ nguyên:

- `consult_discovery`
- `project_grounded`

Skill sales là **lớp nằm trên route**, không thay route.

### 3.2. Không nhét nguyên tài liệu sales vào một prompt dài

Không nên làm kiểu:

- lấy toàn bộ tài liệu sales
- nhét vào system prompt
- hy vọng model “biết bán hàng hơn”

Cách đúng là:

- map tài liệu sales thành **state machine**
- map thành **conversation goal**
- map thành **response mode**

### 3.3. Không biến agent thành bot hỏi form

Skill sales không có nghĩa là mỗi lượt phải hỏi:

- mua để ở hay đầu tư
- ngân sách bao nhiêu
- khu vực nào
- thời gian nào

Cách đúng là:

- chỉ hỏi khi thật sự cần
- hoặc mở tò mò bằng insight/hook
- hoặc recommendation grounded
- hoặc mời bước tiếp theo rõ ràng

---

## 4. Lớp mới cần thêm: `sales_state`

Cần thêm một state machine chuyên cho funnel sales.

### 4.1. Đề xuất các state

```text
unknown
exploring
need_identified
qualified
interested
appointment_ready
nurture
handoff
```

### 4.2. Ý nghĩa từng state

#### `unknown`
Chưa biết khách là ai, chưa rõ nhu cầu.

#### `exploring`
Khách mới tìm hiểu, hỏi chung, chưa lộ nhu cầu đủ rõ.

#### `need_identified`
Đã biết ít nhất 1–2 trục chính như:
- ở thực / đầu tư
- ưu tiên môi trường sống / tiện ích / an toàn / giữ giá
- gia đình có con nhỏ
- thích tài sản an toàn hoặc tăng trưởng

#### `qualified`
Đã đủ dữ kiện để tư vấn có định hướng.
Ví dụ:
- purpose khá rõ
- product fit tương đối rõ
- query đã có dạng project matching hoặc project specific

#### `interested`
Khách phản hồi tích cực, hỏi sâu, thể hiện mức quan tâm cao hơn.

#### `appointment_ready`
Đã đến lúc mời bước tiếp theo như:
- nhận tài liệu cá nhân hóa
- gọi với advisor
- xem dự án / nhà mẫu / site visit

#### `nurture`
Khách chưa xuống bước tiếp theo nhưng còn tiềm năng, cần nuôi.

#### `handoff`
Chuyển cho sales thật.

---

## 5. Cần sửa schema gì

## 5.1. Sửa `LeadState`

Hiện tại `LeadState` mới có:

- `name`
- `phone_contact`
- `need`
- `painpoint`

Cần mở rộng thêm:

```python
class LeadState(BaseModel):
    name: str | None = None
    phone_contact: str | None = None
    need: NeedPainpointState = Field(default_factory=NeedPainpointState)
    painpoint: NeedPainpointState = Field(default_factory=NeedPainpointState)

    sales_state: Literal[
        "unknown",
        "exploring",
        "need_identified",
        "qualified",
        "interested",
        "appointment_ready",
        "nurture",
        "handoff",
    ] = "unknown"

    lead_level: Literal["exploratory", "interested", "qualified", "hot"] | None = None
    last_conversation_goal: str | None = None
    next_best_action: str | None = None
```

### Ý nghĩa

- `sales_state`: state chính trong funnel
- `lead_level`: nhãn mức nóng/lạnh kiểu sales
- `last_conversation_goal`: goal của lượt trước
- `next_best_action`: hệ đang muốn kéo khách sang bước nào tiếp theo

---

## 5.2. Sửa `DecisionTrace`

Thêm vào:

```python
sales_state_before: str | None = None
sales_state_after: str | None = None
conversation_goal: str | None = None
response_mode: str | None = None
```

Mục đích:
- debug được khách đi từ state nào sang state nào
- biết lượt đó model đang nói theo mode gì

---

## 6. Thêm lớp `conversation_goal`

Đây là “điểm đích” của từng lượt chat.

### 6.1. Đề xuất goal set

```text
build_trust
discover_need
surface_priority
show_fit
handle_concern
invite_next_step
nurture_lead
handoff_to_human
```

### 6.2. Ý nghĩa

#### `build_trust`
Tạo thiện cảm, cho khách thấy được hiểu, được tư vấn tử tế.

#### `discover_need`
Làm rõ nhu cầu và động cơ mua.

#### `surface_priority`
Làm nổi bật điều khách quan tâm nhất.

#### `show_fit`
Cho khách thấy dự án này hợp với họ ở điểm nào.

#### `handle_concern`
Xử lý băn khoăn / objection.

#### `invite_next_step`
Mời bước tiếp theo rõ ràng.

#### `nurture_lead`
Giữ cuộc trò chuyện và tăng readiness mà không ép.

#### `handoff_to_human`
Đưa khách sang sales thật.

---

## 7. Nâng cấp `response_mode`

Hiện mode hiện tại còn ít. Cần mở rộng thành:

```python
_RESPONSE_MODES = {
    "warm_welcome",
    "discover_need",
    "value_teaser",
    "recommendation",
    "handle_objection",
    "next_step_invite",
    "nurture_followup",
    "meeting_invite",
}
```

### 7.1. Ý nghĩa từng mode

#### `warm_welcome`
Chào mở đầu tự nhiên, không quá sales, tạo thiện cảm.

#### `discover_need`
Hỏi nhẹ, không checklist, để hiểu khách hơn.

#### `value_teaser`
Mở ra một góc hấp dẫn của dự án, tạo tò mò, chưa đi quá sâu.

#### `recommendation`
Đưa ra nhận định/gợi ý phù hợp dựa trên grounded data hoặc state hiện có.

#### `handle_objection`
Phản hồi ngắn, chắc, logic, không tranh cãi.

#### `next_step_invite`
Mời bước tiếp theo rõ ràng nhưng mềm.

#### `nurture_followup`
Nuôi lead, tăng thiện cảm, không ép chốt.

#### `meeting_invite`
Đề xuất buổi hẹn hoặc handoff cho người thật.

---

## 8. Map state machine từ tài liệu sales vào system hiện tại

## 8.1. Map `query_type` -> `sales_state`

Giữ nguyên 4 `query_type` hiện có:

- `advisory_strategy`
- `project_matching`
- `project_specific`
- `clarification`

Nhưng dùng chúng như tín hiệu để cập nhật state.

### advisory_strategy
Ví dụ:
- “mình muốn được tư vấn”
- “mua để đầu tư nên chọn như thế nào”

Map:
- `unknown -> exploring`
- `exploring -> need_identified`

### project_matching
Ví dụ:
- “gần bệnh viện”
- “gần trường học”
- “hợp gia đình có con nhỏ”

Map:
- `need_identified -> qualified`
- `qualified -> interested`

### project_specific
Ví dụ:
- “giới thiệu qua về dự án”
- “pháp lý sao”
- “dự án này có gì nổi bật”

Map:
- `qualified -> interested`
- hoặc `interested -> appointment_ready`

### clarification
Ví dụ:
- “ok”
- “ừ mình thiên về an toàn hơn”

Map:
- không reset state
- chỉ nối tiếp logic cũ

---

## 8.2. Map `sales_state` -> `conversation_goal`

### `unknown`
- goal = `build_trust`

### `exploring`
- goal = `discover_need`

### `need_identified`
- goal = `surface_priority` hoặc `show_fit`

### `qualified`
- goal = `show_fit`

### `interested`
- goal = `handle_concern` hoặc `invite_next_step`

### `appointment_ready`
- goal = `invite_next_step`

### `nurture`
- goal = `nurture_lead`

### `handoff`
- goal = `handoff_to_human`

---

## 8.3. Map `sales_state + goal` -> `response_mode`

### unknown + build_trust
- `response_mode = warm_welcome`

### exploring + discover_need
- `response_mode = discover_need`

### need_identified + show_fit
- `response_mode = value_teaser`

### qualified + show_fit
- `response_mode = recommendation`

### interested + handle_concern
- `response_mode = handle_objection`

### interested / appointment_ready + invite_next_step
- `response_mode = next_step_invite`

### nurture + nurture_lead
- `response_mode = nurture_followup`

### handoff + handoff_to_human
- `response_mode = meeting_invite`

---

## 9. Cần thêm 3 hàm mới

## 9.1. `update_sales_state(...)`

Mục tiêu:
- đọc state cũ
- đọc query_type
- đọc lead_state
- đọc grounded_result
- suy ra state mới

Pseudo-code:

```python
def update_sales_state(
    previous_state: str,
    query_type: str,
    lead_state: LeadState,
    grounded_result: dict[str, Any] | None,
    recent_history: list[HistoryTurn],
) -> str:
    # 1. nếu chưa rõ gì -> exploring
    # 2. nếu đã có purpose / preference rõ -> need_identified
    # 3. nếu query matching/specific + retrieval hữu ích -> qualified
    # 4. nếu user hỏi sâu, phản hồi tích cực -> interested
    # 5. nếu đủ dấu hiệu để mời next step -> appointment_ready
    # 6. nếu chưa chốt được bước nào nhưng còn tiềm năng -> nurture
    # 7. nếu hỏi quá sâu hoặc cần người thật -> handoff
```

---

## 9.2. `select_conversation_goal(...)`

Mục tiêu:
- từ `sales_state`
- chọn goal phù hợp cho lượt hiện tại

Pseudo-code:

```python
def select_conversation_goal(
    sales_state: str,
    query_type: str,
    grounded_result: dict[str, Any] | None,
) -> str:
    ...
```

---

## 9.3. `select_next_best_action(...)`

Mục tiêu:
- xác định bước tiếp theo tốt nhất trong funnel

Đề xuất action set:

```text
continue_discovery
show_project_fit
handle_concern
invite_brochure
invite_call
invite_site_visit
handoff_human
```

Pseudo-code:

```python
def select_next_best_action(
    sales_state: str,
    conversation_goal: str,
    grounded_result: dict[str, Any] | None,
) -> str:
    ...
```

---

## 10. Cần sửa `build_reply_plan()`

Hiện tại `build_reply_plan()` đang quyết định mode chủ yếu theo:
- có grounded result hay không
- low_confidence
- advisory hay consult

Cần đổi sang thứ tự mới:

### Bước 1
Đọc `sales_state`

### Bước 2
Chọn `conversation_goal`

### Bước 3
Map sang `response_mode`

### Bước 4
Chọn `ask_policy`

---

## 10.1. Quy tắc chọn `ask_policy`

### `avoid_question`
Dùng khi:
- đang ở `recommendation`
- đang ở `value_teaser`
- đã có grounded data đủ tốt
- 1–2 lượt gần đây đã hỏi rồi
- đang `meeting_invite`

### `allow_question`
Dùng khi:
- đang `discover_need`
- đang `nurture_followup`
- đang `handle_objection`

### `must_clarify`
Chỉ dùng khi:
- thiếu đúng 1 điểm critical để đi tiếp
- ví dụ chưa đủ để biết nên tư vấn theo ở thực hay đầu tư

---

## 11. Cần sửa `ReplyPlan`

Hiện `ReplyPlan` có:

- `response_mode`
- `ask_policy`
- `focus`
- `question_focus`

Cần thêm:

```python
@dataclass(frozen=True)
class ReplyPlan:
    response_mode: str
    ask_policy: str
    focus: str
    question_focus: str
    conversation_goal: str
    next_best_action: str
```

---

## 12. Cần sửa prompt decider

Hiện decider prompt đã có:
- `query_type`
- `retrieval_readiness`
- `start_route`
- `should_route_project`

Cần bổ sung output schema:

```json
{
  "sales_state_after": "unknown|exploring|need_identified|qualified|interested|appointment_ready|nurture|handoff",
  "conversation_goal": "build_trust|discover_need|surface_priority|show_fit|handle_concern|invite_next_step|nurture_lead|handoff_to_human"
}
```

### 12.1. Rule cần thêm vào prompt decider

- Không chỉ quyết định route, mà còn phải quyết định khách đang ở giai đoạn sales nào.
- Mỗi lượt phải có một mục tiêu hội thoại rõ ràng.
- Nếu khách còn sớm trong funnel, ưu tiên xây thiện cảm và khám phá nhu cầu.
- Nếu khách đã đủ fit, ưu tiên recommendation hoặc mời bước tiếp theo.
- Nếu khách hỏi quá sâu hoặc đã sẵn sàng, có thể chuyển sang handoff.
- Không lặp brochure intro nhiều lượt liên tiếp.

---

## 13. Cần sửa prompt synthesis cuối

Prompt synthesis hiện tại đã có `response_mode` và `ask_policy`, nhưng cần thêm:

- `sales_state`
- `conversation_goal`
- `next_best_action`

### 13.1. Rule mới cho prompt synthesis

- Mỗi lượt phải mở ra **1 góc khám phá mới**, không lặp lại phần giới thiệu tổng quan cũ.
- Không lặp mô thức “dự án này có vị trí tốt, tiện ích tốt” qua nhiều lượt.
- Nếu state là `exploring`, ưu tiên tạo thiện cảm và tò mò.
- Nếu state là `need_identified`, ưu tiên cho khách thấy dự án hợp ở điểm nào.
- Nếu state là `qualified`, ưu tiên recommendation grounded.
- Nếu state là `interested`, ưu tiên xử lý băn khoăn hoặc mời step tiếp theo.
- Nếu state là `appointment_ready`, phải có CTA mềm nhưng rõ.
- Nếu state là `handoff`, không ôm tiếp quá lâu, mà mời gặp người thật.

---

## 14. Handoff rule

Theo tài liệu sales, có những đoạn không nên để bot ôm quá lâu.

### 14.1. Set `sales_state = handoff` khi:

- user hỏi tài chính chi tiết
- user hỏi pháp lý sâu
- user hỏi căn cụ thể / booking / cọc / deal
- user muốn gặp trực tiếp / xem thực tế
- user tỏ ra nóng / VIP / mất kiên nhẫn
- bot không đủ dữ liệu để trả lời có trách nhiệm

### 14.2. Khi `handoff`

- `response_mode = meeting_invite`
- `ask_policy = avoid_question`
- `next_best_action = handoff_human`

---

## 15. Thay đổi luồng tổng thể trong `query()`

Flow mới đề xuất:

### Phase 1 — Analyze
- lấy `query_type`
- lấy `retrieval_readiness`
- lấy `route`
- lấy `sales_state_after` sơ bộ
- lấy `conversation_goal`

### Phase 2 — Execute
- chạy `consult_discovery` hoặc `project_grounded`
- chain nếu cần

### Phase 3 — State update
- merge `need/painpoint`
- gọi `update_sales_state(...)`
- gọi `select_conversation_goal(...)`
- gọi `select_next_best_action(...)`

### Phase 4 — Reply planning
- `build_reply_plan(...)`
- chọn `response_mode`
- chọn `ask_policy`

### Phase 5 — Synthesis
- sinh câu trả lời cuối theo strategy sales

---

## 16. Mapping thực chiến cho các case phổ biến

## Case 1
User:

```text
xin chào, mình muốn được tư vấn
```

Expected:
- `route = consult_discovery`
- `sales_state = exploring`
- `conversation_goal = build_trust`
- `response_mode = warm_welcome`

Reply nên:
- chào tự nhiên
- mở một góc đáng chú ý của dự án
- không vội hỏi checklist

---

## Case 2
User:

```text
giới thiệu qua cho mình về dự án
```

Expected:
- `route = consult_discovery` hoặc `project_grounded` tùy data
- `sales_state = need_identified`
- `conversation_goal = show_fit`
- `response_mode = value_teaser`

Reply nên:
- không đọc brochure
- chọn 1–2 điểm đáng xem nhất
- tạo cảm giác “muốn nghe tiếp”

---

## Case 3
User:

```text
có căn hộ nào gần bệnh viện không
```

Expected:
- `query_type = project_matching`
- `final_route = project_grounded`
- `sales_state = qualified`
- `conversation_goal = show_fit`
- `response_mode = recommendation`

Reply nên:
- grounded
- cho thấy dự án phù hợp ở đâu
- không hỏi form nếu chưa cần

---

## Case 4
User:

```text
giá thế nào, có thể xem trực tiếp không?
```

Expected:
- `sales_state = appointment_ready` hoặc `handoff`
- `conversation_goal = invite_next_step`
- `response_mode = next_step_invite` hoặc `meeting_invite`

Reply nên:
- đưa bước tiếp theo rõ ràng
- không vòng vo

---

## 17. Trình tự sửa code đề xuất

## Bước 1
Sửa `schemas.py`
- thêm `sales_state`, `lead_level`, `last_conversation_goal`, `next_best_action` vào `LeadState`
- thêm `sales_state_before`, `sales_state_after`, `conversation_goal`, `response_mode` vào `DecisionTrace`

## Bước 2
Sửa `app.py`
- mở rộng `_RESPONSE_MODES`
- mở rộng `ReplyPlan`
- thêm `update_sales_state()`
- thêm `select_conversation_goal()`
- thêm `select_next_best_action()`

## Bước 3
Sửa `build_reply_plan()`
- không chọn mode chỉ dựa vào route/grounded nữa
- chọn theo `sales_state + conversation_goal`

## Bước 4
Sửa decider prompt
- output thêm `sales_state_after`
- output thêm `conversation_goal`

## Bước 5
Sửa synthesis prompt
- thêm `sales_state`
- thêm `conversation_goal`
- thêm `next_best_action`
- thêm rule chống brochure-style lặp lại

## Bước 6
Bổ sung log trace
- state trước / sau
- goal
- response_mode
- next_best_action

---

## 18. Mục tiêu sau khi sửa xong

Sau khi hoàn tất, hệ thống nên chuyển từ kiểu:

- route đúng nhưng reply nhạt
- hỏi chưa đúng nhịp
- không kéo khách đi trong funnel

thành:

- route đúng
- biết khách đang ở state nào
- biết mục tiêu của lượt hiện tại
- biết nên nói theo mode nào
- biết khi nào nên mời bước tiếp theo
- biết khi nào nên handoff

Công thức cuối cùng:

```text
route = có cần grounding không
sales_state = khách đang ở đâu trong funnel
conversation_goal = lượt này phải đạt gì
response_mode = nên nói theo kiểu nào
next_best_action = muốn kéo khách sang bước nào tiếp
```

---

## 19. Kết luận

Không cần làm lại toàn bộ hệ thống.

Chỉ cần nâng kiến trúc hiện tại từ:

```text
query_type + route + reply
```

thành:

```text
query_type + route + sales_state + conversation_goal + response_mode + next_best_action + reply
```

thì agent sẽ bắt đầu có **skill sales thực sự**, thay vì chỉ là chatbot biết consult và grounded.
