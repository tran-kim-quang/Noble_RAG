# Tối ưu latency cho `consult_discovery` theo hướng deterministic, không heuristic

## 1. Mục tiêu

Mục tiêu của tài liệu này là mô tả chi tiết cách tối ưu latency cho nhánh `consult_discovery` trên branch `test-screen-agents` mà vẫn giữ được chất lượng của toàn bộ flow orchestrator.

Nguyên tắc chính:

- Không dùng heuristic keyword để quyết định fast-path.
- Không bỏ qua state machine.
- Không bỏ qua `reply_planning`.
- Chỉ cắt bỏ lần gọi model cuối khi lần gọi model đầu đã tạo ra `consult_reply` đủ dùng.

---

## 2. Tình trạng hiện tại của flow

Hiện tại một request `consult_discovery` đang đi theo luồng:

1. `analyze_turn()` gọi model để tạo:
   - `query_type`
   - `retrieval_readiness`
   - `start_route`
   - `need_update`
   - `painpoint_update`
   - `engagement_state_after_hint`
   - `sales_state_after_hint`
   - `conversation_goal_hint`
   - `consult_reply`
2. `run_consult_discovery()` chỉ pass-through `analysis.consult_reply`.
3. `merge_lead_state()` cập nhật state.
4. `sales_state_engine` cập nhật:
   - `engagement_state`
   - `sales_state`
   - `conversation_goal`
   - `next_best_action`
5. `build_reply_plan()` chọn:
   - `response_mode`
   - `ask_policy`
   - `question_focus`
6. `synthesize_assistant_reply()` lại gọi model thêm một lần để viết câu trả lời cuối.
7. Nếu cần, hệ còn gọi model thêm cho:
   - `_normalize_reply_to_accented_vietnamese()`
   - `_rewrite_reply_by_policy()`

Vấn đề là với các turn consult đơn giản, bước 6 và các bước rewrite phía sau thường không tạo thêm đủ giá trị để bù lại chi phí latency.

---

## 3. Ý tưởng tối ưu đúng

### Công thức tối ưu

Với các turn chỉ kết thúc ở `consult_discovery` và không cần grounded retrieval, thay vì:

- gọi model ở `analyze_turn`
- rồi lại gọi model ở `reply_synthesis`

thì dùng luôn `analysis.consult_reply` làm câu trả lời cuối, sau khi đã đi qua:

- `sales_state_engine`
- `reply_planning`
- `local sanitize`

Nói cách khác:

- vẫn giữ orchestration
- vẫn giữ state transition
- vẫn giữ `response_mode` và `ask_policy`
- chỉ bỏ lần generate cuối nếu không cần thiết

---

## 4. Điều kiện deterministic để vào fast-path

Fast-path **không được quyết định bằng keyword**. Chỉ được quyết định từ các biến đã được orchestrator tính ra.

### Được vào fast-path khi thỏa tất cả điều kiện

- `final_route == "consult_discovery"`
- `chained_from_consult == False`
- `grounded_result is None`
- `reply_plan.ask_policy != "must_clarify"`
- `reply_plan.response_mode` thuộc nhóm an toàn

### Nhóm `response_mode` an toàn trong phase đầu

- `warm_welcome`
- `value_teaser`
- `discover_need`

### Không được vào fast-path nếu thuộc một trong các trường hợp sau

- `final_route == "project_grounded"`
- `chained_from_consult == True`
- `grounded_result is not None`
- `reply_plan.ask_policy == "must_clarify"`
- `reply_plan.response_mode` thuộc nhóm:
  - `contact_capture`
  - `followup_confirm`
  - `meeting_invite`
  - `grounded_recommendation`
  - `handle_concern`

Lưu ý: đây là gate **deterministic**, không phải heuristic.

---

## 5. Luồng mới sau khi tối ưu

## Luồng cũ

```text
analyze_turn (LLM)
-> consult_discovery
-> sales_state_engine
-> reply_planning
-> reply_synthesis (LLM)
-> maybe normalize/rewrite (LLM)
-> return
```

## Luồng mới

```text
analyze_turn (LLM)
-> consult_discovery
-> sales_state_engine
-> reply_planning
-> if can_fastpath:
     use consult_reply
     local sanitize only
   else:
     reply_synthesis
-> return
```

### Lợi ích

Với consult-only turns:

- giảm từ 2–4 model calls xuống còn 1 model call
- giữ nguyên state logic
- giảm latency rõ rệt ở các câu mở đầu, tư vấn chiến lược, clarification nhẹ

---

## 6. Quy trình sửa code chi tiết

## Bước 1: thêm hàm gate cho fast-path

Thêm vào `orchestrator_service/app.py`:

```python
def _can_fastpath_consult_reply(
    final_route: str,
    chained_from_consult: bool,
    grounded_result: dict[str, Any] | None,
    reply_plan: ReplyPlan,
) -> bool:
    if final_route != "consult_discovery":
        return False
    if chained_from_consult:
        return False
    if grounded_result is not None:
        return False
    if reply_plan.ask_policy == "must_clarify":
        return False

    return reply_plan.response_mode in {
        "warm_welcome",
        "value_teaser",
        "discover_need",
    }
```

### Vai trò của hàm này

- Chỉ dùng state + planning output.
- Không phụ thuộc vào message keyword.
- Không đoán bằng pattern text.
- Giữ logic orchestration sạch.

---

## Bước 2: thêm hàm chuẩn hóa local cho `consult_reply`

```python
def _prepare_fastpath_consult_reply(
    analysis: TurnAnalysis,
    reply_plan: ReplyPlan,
) -> str:
    reply = (analysis.consult_reply or "").strip()
    if not reply:
        return ""

    reply = _sanitize_reply_for_policy(
        reply=reply,
        ask_policy=reply_plan.ask_policy,
        single_project_mode=True,
        question_focus=reply_plan.question_focus,
    )
    return _compact_text(reply, max_words=120)
```

### Mục đích

- Dùng lại `consult_reply` từ decider.
- Không gọi thêm model.
- Vẫn áp đúng `ask_policy`.
- Vẫn compact output để an toàn.

---

## Bước 3: sửa block `reply_synthesis` trong `query()`

Trong `query()`, hiện tại sau `reply_plan = build_reply_plan(...)` hệ luôn đi vào `reply_synthesis`.

Cần thay bằng logic:

```python
used_fastpath_consult_reply = False
assistant_reply = ""

if _can_fastpath_consult_reply(
    final_route=final_route,
    chained_from_consult=chained_from_consult,
    grounded_result=grounded_result,
    reply_plan=reply_plan,
):
    used_fastpath_consult_reply = True
    assistant_reply = _prepare_fastpath_consult_reply(
        analysis=analysis,
        reply_plan=reply_plan,
    )
else:
    with start_observation(
        langfuse_enabled,
        name="orchestrator.reply_synthesis",
        as_type="generation",
        model=settings.decider_model,
        input={
            "final_route": final_route,
            "response_mode": reply_plan.response_mode,
            "ask_policy": reply_plan.ask_policy,
        },
    ) as reply_obs:
        assistant_reply = synthesize_assistant_reply(
            message=message,
            recent_history=payload.recent_history,
            lead_state=final_state,
            analysis=analysis,
            final_route=final_route,
            decision_reason=reason,
            reply_plan=reply_plan,
            grounded_result=grounded_result,
            settings=settings,
        )
        reply_obs.update(
            output={
                "assistant_reply_chars": len(assistant_reply),
                "assistant_reply_preview": assistant_reply[:240],
            }
        )
```

### Kết quả

- Turn consult-only đủ điều kiện sẽ không gọi `reply_synthesis`.
- Turn phức tạp vẫn giữ nguyên flow cũ.

---

## Bước 4: log rõ source của câu trả lời

Thêm vào `request_obs.update(...)`:

```python
"used_fastpath_consult_reply": used_fastpath_consult_reply,
"reply_source": "consult_reply_fastpath" if used_fastpath_consult_reply else "reply_synthesis",
```

### Vì sao cần

Để đo:

- bao nhiêu turn dùng fast-path
- latency giảm ở nhóm đó bao nhiêu
- quality của nhóm fast-path có bị tụt không

---

## Bước 5: chỉnh prompt decider để `consult_reply` usable hơn

Hiện `consult_reply` được sinh trong `_build_decider_prompt()` và trả ra trong `TurnAnalysis`.

Khi bật fast-path, `consult_reply` phải đủ chất lượng để trả thẳng.

Nên thêm vào phần rule của `consult_reply` trong prompt:

```text
- consult_reply phải đủ dùng như câu trả lời cuối cho các turn consult đơn giản.
- Với advisory_strategy hoặc clarification nhẹ, consult_reply nên usable ngay, không phụ thuộc vào bước viết lại phía sau.
- Tránh các câu mơ hồ phụ thuộc vào grounded_context hoặc vào bước rewrite sau đó.
```

### Mục đích

- Tăng chất lượng ngay từ lần generate đầu.
- Giảm phụ thuộc vào `reply_synthesis`.

---

## 7. Những gì fast-path vẫn phải giữ nguyên

Fast-path **không** được bỏ qua các bước sau:

- `merge_lead_state()`
- `update_engagement_state()`
- `update_sales_state()`
- `select_conversation_goal()`
- `select_next_best_action()`
- `build_reply_plan()`

### Lý do

Nếu bỏ qua các bước này thì hệ sẽ:

- mất state consistency
- sai `response_mode`
- sai `ask_policy`
- làm hỏng logic assistant/sales funnel

Fast-path đúng chỉ là:

> bỏ lần generate cuối, không bỏ orchestration

---

## 8. Những gì không được dùng để quyết định fast-path

Không dùng:

- keyword kiểu “xin chào”, “ok”, “mình muốn tư vấn”
- regex intent shortcut
- heuristic score theo message text
- hard rules từ token list

### Vì sao

- dễ sai context
- làm vỡ cấu trúc state-driven hiện tại
- trái với định hướng architecture đã thống nhất

Fast-path chỉ nên dùng output của orchestrator:

- `final_route`
- `chained_from_consult`
- `grounded_result`
- `response_mode`
- `ask_policy`

---

## 9. Mở rộng phase 2

Sau khi phase 1 ổn định, có thể mở rộng fast-path cho:

- `consultive_recommendation`

### Điều kiện để mở phase 2

- log cho thấy `consult_reply` đủ ổn định
- không có nhiều retry quality issue
- response không bị khô hoặc brochure-style quá mức

### Không nên mở sớm cho

- `handle_concern`
- `meeting_invite`
- `contact_capture`
- `followup_confirm`
- `grounded_recommendation`

---

## 10. Kết hợp với tối ưu latency tiếp theo

Sau khi xong fast-path cho consult, nên tối ưu tiếp theo thứ tự này:

### 10.1. Bỏ model-call hậu xử lý

Cân nhắc loại bỏ hoặc giới hạn:

- `_normalize_reply_to_accented_vietnamese()`
- `_rewrite_reply_by_policy()`

Thay bằng local sanitize.

### 10.2. Tách model router và model reply

- model nhỏ cho `analyze_turn`
- model tốt hơn cho `reply_synthesis`

### 10.3. Dynamic retrieval budget

- consult-only: không retrieval
- `project_specific`: `top_k` nhỏ
- `project_matching`: `top_k` vừa

---

## 11. Checklist rollout

## Phase 1

- [ ] thêm `_can_fastpath_consult_reply()`
- [ ] thêm `_prepare_fastpath_consult_reply()`
- [ ] sửa block `reply_synthesis` trong `query()`
- [ ] log `used_fastpath_consult_reply`
- [ ] update prompt decider cho `consult_reply`

## Phase 2

- [ ] benchmark consult-only trước/sau
- [ ] review quality 30–50 sample turns
- [ ] mở thêm `consultive_recommendation` nếu ổn

## Phase 3

- [ ] xem xét bỏ `_rewrite_reply_by_policy()`
- [ ] xem xét bỏ `_normalize_reply_to_accented_vietnamese()`
- [ ] đo lại latency end-to-end và per-phase

---

## 12. Tiêu chí đánh giá thành công

### Về latency

- consult-only turn giảm đáng kể so với hiện tại
- số lần gọi model trung bình trên consult-only turn giảm từ 2–4 xuống 1

### Về chất lượng

- `query_type` không đổi
- `engagement_state` không đổi
- `sales_state` không đổi
- `conversation_goal` không đổi
- `ask_policy` vẫn được tôn trọng
- không tăng lỗi brochure-style quá mạnh
- không tăng câu hỏi thừa

### Về vận hành

- log phân biệt được `consult_reply_fastpath` và `reply_synthesis`
- dễ rollback nếu cần

---

## 13. Kết luận

Fast-path đúng cho `consult_discovery` là:

- không dùng heuristic
- không phá state machine
- không bỏ reply planning
- chỉ bỏ lần generate cuối nếu turn kết thúc ở consult-only và `reply_plan` cho thấy câu trả lời đầu đã đủ dùng

Công thức ngắn gọn:

```text
State-driven orchestration giữ nguyên
+ Deterministic fast-path gate
+ Reuse consult_reply
+ Local sanitize only
= giảm latency mạnh mà ít rủi ro chất lượng nhất
```
