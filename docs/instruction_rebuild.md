Dưới đây là **chỗ cần sửa cụ thể** trên branch hiện tại và **cách sửa đúng** để assistant có thể:

* chỉ xin **tên / số điện thoại** khi khách đã đủ nóng
* lưu lại vào `LeadState` JSON
* không hỏi contact quá sớm như bot sales

Hiện code đã có:

* `LeadState.name`, `LeadState.phone_contact` 
* `_extract_name()` và `_extract_phone_contact()` để parse trực tiếp từ message 

Thiếu ở đây là: **policy hội thoại** để chủ động hỏi contact đúng lúc.

---

# 1) `schemas.py` — thêm mục tiêu và action cho contact capture

## Cần sửa

Trong `LeadState.last_conversation_goal`, thêm:

* `capture_contact`
* `confirm_followup`

Trong `LeadState.next_best_action`, thêm:

* `ask_name`
* `ask_phone`
* `ask_name_and_phone`
* `schedule_followup`

Trong `DecisionTrace.response_mode`, thêm:

* `contact_capture`
* `followup_confirm`

## Sửa đúng

Tìm các enum hiện tại trong `schemas.py` và thêm như sau:

```python
last_conversation_goal: Literal[
    "build_trust",
    "discover_need",
    "surface_priority",
    "show_fit",
    "handle_concern",
    "invite_next_step",
    "nurture_lead",
    "handoff_to_human",
    "capture_contact",
    "confirm_followup",
] | None = None
```

```python
next_best_action: Literal[
    "continue_discovery",
    "show_project_fit",
    "handle_concern",
    "invite_brochure",
    "invite_call",
    "invite_site_visit",
    "handoff_human",
    "ask_name",
    "ask_phone",
    "ask_name_and_phone",
    "schedule_followup",
] | None = None
```

```python
response_mode: Literal[
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
    "grounded_recommendation",
    "handle_concern",
    "soft_next_step",
    "nurture_followup",
    "meeting_invite",
    "contact_capture",
    "followup_confirm",
] | None = None
```

## Nên thêm nữa

Trong `LeadState`, thêm 1 field nhẹ để debug/contact workflow:

```python
contact_capture_status: Literal["unknown", "requested", "partial", "complete"] = "unknown"
```

---

# 2) `merge_lead_state()` — đang giữ thiếu engagement fields

## Vấn đề

Hàm `merge_lead_state()` hiện đang giữ:

* `sales_state`
* `lead_level`
* `last_conversation_goal`
* `next_best_action`

nhưng **không giữ lại**:

* `engagement_state`
* `engagement_confidence` 

Điều này dễ làm state bị reset tạm thời trước khi state engine set lại.

## Sửa đúng

Trong `merge_lead_state()`, thêm:

```python
engagement_state=lead_state.engagement_state,
engagement_confidence=lead_state.engagement_confidence,
contact_capture_status=getattr(lead_state, "contact_capture_status", "unknown"),
```

Nếu bạn chưa thêm `contact_capture_status`, bỏ dòng đó.

---

# 3) Thêm helper để biết khi nào phải xin contact

## Cần tạo mới trong `app.py`

### A. Hàm kiểm tra còn thiếu contact gì

```python
def _missing_contact_fields(lead_state: LeadState) -> tuple[bool, bool]:
    missing_name = not bool((lead_state.name or "").strip())
    missing_phone = not bool((lead_state.phone_contact or "").strip())
    return missing_name, missing_phone
```

### B. Hàm nhận diện follow-up intent

Đây là **soft rule hợp lý**, không phải route rule chính.

```python
def _message_requests_followup(message: str) -> bool:
    lowered = (message or "").strip().lower()
    signals = [
        "đi xem", "xem thực tế", "gặp trực tiếp", "hẹn",
        "gọi lại", "liên hệ", "tư vấn kỹ hơn", "tư vấn sâu hơn",
        "gửi thông tin", "gửi bảng giá", "đặt lịch", "trao đổi thêm",
    ]
    return any(token in lowered for token in signals)
```

## Sửa đúng

Dùng helper này **không để route**, mà để:

* promote `sales_state`
* chọn `conversation_goal`
* chọn `next_best_action`

---

# 4) `update_sales_state()` — chỉ lên `appointment_ready` khi có follow-up signal rõ

## Vấn đề

Hiện đang có rule:

```python
if state == "interested" and grounded_fit and _has_structured_need(lead_state):
    state = "appointment_ready"
```

Rule này làm hệ lên `appointment_ready` quá sớm 

## Sửa đúng

Thay bằng logic chặt hơn:

```python
if state == "interested":
    if _message_requests_followup(message):
        state = "appointment_ready"
    elif lead_state.next_best_action in {"invite_call", "invite_site_visit", "handoff_human"}:
        state = "appointment_ready"
```

### Lưu ý

Vì hiện `update_sales_state()` chưa nhận `message`, bạn cần đổi signature:

```python
def update_sales_state(
    previous_state: str,
    message: str,
    query_type: str,
    lead_state: LeadState,
    grounded_result: dict[str, Any] | None,
    recent_history: list[HistoryTurn],
    routed_to_project: bool,
    suggested_state: str | None = None,
) -> str:
```

Và update chỗ gọi trong `query()`.

---

# 5) `select_conversation_goal()` — thêm goal lấy contact

## Vấn đề

Hiện hàm này chưa nhìn vào việc:

* khách đã muốn follow-up chưa
* còn thiếu tên/số chưa 

## Sửa đúng

Đổi signature để truyền cả `lead_state` và `message`:

```python
def select_conversation_goal(
    sales_state: str,
    engagement_state: str,
    query_type: str,
    grounded_result: dict[str, Any] | None,
    lead_state: LeadState,
    message: str,
    suggested_goal: str | None = None,
) -> str:
```

Thêm ưu tiên ở đầu hàm:

```python
missing_name, missing_phone = _missing_contact_fields(lead_state)

if sales_state in {"appointment_ready", "handoff"} or _message_requests_followup(message):
    if missing_name or missing_phone:
        return "capture_contact"
    return "confirm_followup"
```

## Vì sao đây là cách đúng

* Không xin contact ở giai đoạn `cold/warm`
* Chỉ xin khi user đã **muốn đi tiếp**
* Nếu đã có đủ contact thì chuyển sang goal xác nhận bước follow-up

---

# 6) `select_next_best_action()` — quyết định hỏi tên hay hỏi số

## Vấn đề

Hiện `next_best_action` chưa có action cho contact capture 

## Sửa đúng

Đổi signature để nhận `lead_state`:

```python
def select_next_best_action(
    sales_state: str,
    engagement_state: str,
    conversation_goal: str,
    grounded_result: dict[str, Any] | None,
    lead_state: LeadState,
) -> str:
```

Thêm logic ở đầu hàm:

```python
missing_name, missing_phone = _missing_contact_fields(lead_state)

if conversation_goal == "capture_contact":
    if missing_name and missing_phone:
        return "ask_name_and_phone"
    if missing_name:
        return "ask_name"
    if missing_phone:
        return "ask_phone"

if conversation_goal == "confirm_followup":
    return "schedule_followup"
```

---

# 7) `_select_question_focus()` — hỏi đúng thứ đang thiếu

## Vấn đề

Hiện `question_focus` chỉ biết:

* mục tiêu sử dụng
* ưu tiên
* mức sẵn sàng
* băn khoăn... 

## Sửa đúng

Thêm vào đầu hàm:

```python
if action == "ask_name":
    return "tên xưng hô thuận tiện"
if action == "ask_phone":
    return "số điện thoại liên hệ"
if action == "ask_name_and_phone":
    return "tên và số điện thoại liên hệ"
if action == "schedule_followup":
    return "thời điểm tiện để mình liên hệ lại"
```

---

# 8) `_RESPONSE_MODES` + `_select_response_mode()` — thêm mode xin contact

## Cần sửa

Trong `_RESPONSE_MODES`, thêm:

* `contact_capture`
* `followup_confirm`

## Sửa đúng

```python
_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
    "grounded_recommendation",
    "handle_concern",
    "soft_next_step",
    "nurture_followup",
    "meeting_invite",
    "contact_capture",
    "followup_confirm",
}
```

Trong `_select_response_mode()` thêm ở đầu:

```python
if goal == "capture_contact":
    return "contact_capture"
if goal == "confirm_followup":
    return "followup_confirm"
```

---

# 9) `build_reply_plan()` — bắt buộc hỏi đúng 1 câu khi đang lấy contact

## Vấn đề

Hiện `ask_policy` đang map theo engagement và response mode chung, chưa có branch riêng cho contact capture 

## Sửa đúng

Sau khi ra `response_mode`, thêm override:

```python
if response_mode == "contact_capture":
    ask_policy = "must_clarify"
elif response_mode == "followup_confirm":
    ask_policy = "allow_question"
```

Và nên cho `contact_capture` bypass một số rule anti-question hiện có, vì đây là trường hợp **được phép hỏi trực tiếp**.

Ví dụ:

```python
if response_mode == "contact_capture":
    return ReplyPlan(
        response_mode=response_mode,
        ask_policy="must_clarify",
        focus=_build_reply_focus(...),
        question_focus=_select_question_focus(...),
        conversation_goal=goal,
        next_best_action=action,
    )
```

Cách này gọn nhất.

---

# 10) `_build_reply_synthesis_prompt()` — thêm rule hỏi tên/số trực diện nhưng lịch sự

## Vấn đề

Prompt hiện chưa biết mode `contact_capture` và `followup_confirm` 

## Sửa đúng

Thêm vào phần “Ràng buộc do orchestrator cung cấp”:

```text
- response_mode=contact_capture: khi khách đã muốn đi xem, muốn được tư vấn sâu hơn hoặc cần follow-up, hãy xin thông tin liên hệ trực tiếp nhưng lịch sự.
- Nếu thiếu cả name và phone_contact, có thể hỏi gọn cả hai trong cùng một câu.
- Luôn nêu rõ lý do xin thông tin: để sắp xếp tư vấn sâu hơn, gửi tài liệu phù hợp hoặc đặt lịch hẹn.
- response_mode=followup_confirm: xác nhận bước tiếp theo ngắn gọn, không hỏi lan man thêm.
```

### Mẫu câu nên hướng model tới

* thiếu cả 2:

  * “Nếu mình muốn em sắp xếp tư vấn kỹ hơn, anh/chị cho em xin tên và số điện thoại liên hệ thuận tiện nhất nhé?”
* thiếu số:

  * “Em xin số điện thoại thuận tiện để gửi thông tin và sắp xếp trao đổi kỹ hơn với anh/chị nhé?”
* thiếu tên:

  * “Em xin phép lưu tên mình để tiện xưng hô và hỗ trợ sát hơn nhé?”

---

# 11) `query()` — update chỗ gọi các hàm đã đổi signature

Sau khi sửa các hàm trên, trong `query()` bạn phải update các call:

## A. `update_sales_state(...)`

Thêm `message=message`

## B. `select_conversation_goal(...)`

Thêm:

* `lead_state=final_state`
* `message=message`

## C. `select_next_best_action(...)`

Thêm:

* `lead_state=final_state`

Ví dụ:

```python
sales_state_after = update_sales_state(
    previous_state=sales_state_before,
    message=message,
    query_type=_normalize_query_type(analysis.query_type),
    lead_state=final_state,
    grounded_result=grounded_result,
    recent_history=payload.recent_history,
    routed_to_project=(final_route == "project_grounded"),
    suggested_state=analysis.sales_state_after_hint,
)
```

```python
conversation_goal = select_conversation_goal(
    sales_state=sales_state_after,
    engagement_state=engagement_state_after,
    query_type=_normalize_query_type(analysis.query_type),
    grounded_result=grounded_result,
    lead_state=final_state,
    message=message,
    suggested_goal=analysis.conversation_goal_hint,
)
```

```python
next_best_action = _normalize_next_best_action(
    select_next_best_action(
        sales_state=sales_state_after,
        engagement_state=engagement_state_after,
        conversation_goal=conversation_goal,
        grounded_result=grounded_result,
        lead_state=final_state,
    )
)
```

---

# 12) `LeadState` update cuối request — set `contact_capture_status`

Trong đoạn `final_state = final_state.model_copy(update={...})`, thêm:

```python
"contact_capture_status": (
    "complete" if final_state.name and final_state.phone_contact
    else "partial" if final_state.name or final_state.phone_contact
    else "requested" if conversation_goal == "capture_contact"
    else getattr(final_state, "contact_capture_status", "unknown")
),
```

---

# 13) Không nên sửa ở đâu

## Không nên

* Không dùng keyword heuristic để quyết định route/contact ngay từ đầu
* Không xin số điện thoại ở `cold` / `warm` chỉ vì user hỏi dự án
* Không biến `discover_need` thành mode thu thập form

## Nên

* Chỉ xin contact khi:

  * `appointment_ready`
  * `handoff`
  * follow-up intent rõ
* Nếu user đã chủ động gửi số/tên thì regex hiện tại tự extract và lưu luôn 

---

# 14) Thứ tự sửa đúng nhất

1. `schemas.py`
2. `merge_lead_state()`
3. thêm helper `_missing_contact_fields()` và `_message_requests_followup()`
4. sửa `update_sales_state()`
5. sửa `select_conversation_goal()`
6. sửa `select_next_best_action()`
7. sửa `_select_question_focus()`
8. sửa `_RESPONSE_MODES` + `_select_response_mode()`
9. sửa `build_reply_plan()`
10. sửa `_build_reply_synthesis_prompt()`
11. sửa chỗ gọi trong `query()`

---

# 15) Tiêu chí đúng sau khi sửa

## Case 1

User:

> mình muốn được tư vấn kỹ hơn, có thể hẹn trao đổi không

Expected:

* `sales_state -> appointment_ready`
* `conversation_goal -> capture_contact`
* `next_best_action -> ask_name_and_phone` hoặc `ask_phone`
* reply hỏi contact trực tiếp, lịch sự

## Case 2

User:

> ok anh là Quang

Expected:

* `LeadState.name = "Quang"`
* nếu chưa có số:

  * `conversation_goal -> capture_contact`
  * `next_best_action -> ask_phone`

## Case 3

User:

> số của mình là 0988123456

Expected:

* `LeadState.phone_contact = "0988123456"`
* nếu đã đủ tên + số:

  * `conversation_goal -> confirm_followup`
  * `next_best_action -> schedule_followup`

---

Nếu bạn muốn, mình sẽ viết tiếp cho bạn **một patch pseudo-code hoàn chỉnh** cho đúng 6 hàm:

* `update_sales_state`
* `select_conversation_goal`
* `select_next_best_action`
* `_select_question_focus`
* `_select_response_mode`
* `build_reply_plan`
