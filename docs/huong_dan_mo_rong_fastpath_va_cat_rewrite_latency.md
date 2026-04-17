# Hướng dẫn sửa source: mở rộng fast-path an toàn + cắt normalize/rewrite latency

Tài liệu này bám theo source hiện tại của nhánh `test-screen-agents`, đặc biệt là `orchestrator_service/app.py`.

## 1) Kết luận ngắn về bottleneck hiện tại

Fast-path cho `consult_discovery` đã được thêm vào code, nhưng hiệu quả chưa rõ vì:

1. Fast-path hiện chỉ cho 3 `response_mode`:
   - `warm_welcome`
   - `value_teaser`
   - `discover_need`
2. Nhiều consult turn thực tế rơi vào `consultive_recommendation`, nên vẫn bị đẩy sang `reply_synthesis`.
3. Khi đã vào `reply_synthesis`, code vẫn có thể gọi model thêm ở:
   - `_normalize_reply_to_accented_vietnamese(...)`
   - `_rewrite_reply_by_policy(...)`

Vì vậy, để latency giảm rõ rệt, cần sửa đồng thời 2 phần:

- **Mở rộng fast-path một cách an toàn**
- **Cắt các model-call hậu xử lý trong synthesize path**

---

## 2) Mục tiêu sau khi sửa

### Với consult-only turn
Mục tiêu là giảm flow từ:

```text
analyze_turn (LLM)
-> consult_discovery
-> sales_state_engine
-> reply_planning
-> reply_synthesis (LLM)
-> normalize/rewrite (LLM thêm)
-> return
```

thành:

```text
analyze_turn (LLM)
-> consult_discovery
-> sales_state_engine
-> reply_planning
-> fast-path local sanitize
-> return
```

### Với synth path còn lại
Nếu vẫn phải đi `reply_synthesis`, mục tiêu là chỉ còn:

```text
reply_synthesis (1 model call)
-> local sanitize
-> fallback local nếu cần
```

không còn:
- rewrite bằng model
- thêm dấu bằng model

---

## 3) Phần A — Mở rộng fast-path an toàn

## 3.1 File cần sửa
- `orchestrator_service/app.py`

## 3.2 Chỗ hiện tại cần chỉnh
Hiện trong file có:

- `_FASTPATH_CONSULT_RESPONSE_MODES`
- `_evaluate_consult_reply_fastpath(...)`
- `_prepare_fastpath_consult_reply(...)`
- block chọn `reply_source` trong `query()`

### Trạng thái hiện tại
```python
_FASTPATH_CONSULT_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
}
```

Đây là danh sách quá hẹp.

---

## 3.3 Sửa danh sách safe mode

### Mục tiêu
Mở fast-path thêm cho các consult turn kiểu tư vấn nhẹ nhưng chưa cần synthesis riêng.

### Nên sửa thành
```python
_FASTPATH_CONSULT_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
}
```

### Vì sao `consultive_recommendation` nên được mở
Đây là mode rất thường gặp ở consult route khi user hỏi như:
- mua để đầu tư nên chọn thế nào
- gia đình có con nhỏ nên ưu tiên gì
- mình thiên về dòng tiền đều đặn

Nếu không mở mode này, đa số consult turn vẫn rơi vào `reply_synthesis`.

---

## 3.4 Không mở fast-path cho các mode sau
Giữ nguyên là **không fast-path** cho:

- `grounded_recommendation`
- `handle_concern`
- `soft_next_step` (pha 1)
- `meeting_invite`
- `contact_capture`
- `followup_confirm`

### Vì sao
- `grounded_recommendation`: cần phrasing bám dữ liệu retrieval rõ hơn
- `handle_concern`: thường nhạy hơn, cần câu chữ chắc hơn
- `soft_next_step`: dễ biến thành CTA vụng nếu consult_reply chưa đủ khéo
- `contact_capture`, `followup_confirm`, `meeting_invite`: nên giữ control chặt

---

## 3.5 Giữ gate deterministic, không dùng heuristic

### Hàm hiện tại
`_evaluate_consult_reply_fastpath(...)`

### Giữ nguyên triết lý
Gate chỉ nên dựa trên:
- `final_route`
- `chained_from_consult`
- `grounded_result`
- `reply_plan.ask_policy`
- `reply_plan.response_mode`

### Không thêm keyword matching
Không dùng:
- token list
- regex intent heuristic
- keyword warm/interest/ready

### Bản nên giữ
```python
def _evaluate_consult_reply_fastpath(
    final_route: str,
    chained_from_consult: bool,
    grounded_result: dict[str, Any] | None,
    reply_plan: ReplyPlan,
) -> tuple[bool, str]:
    if final_route != "consult_discovery":
        return False, "final_route_not_consult_discovery"
    if chained_from_consult:
        return False, "chained_from_consult"
    if grounded_result is not None:
        return False, "grounded_result_present"
    if _normalize_ask_policy(reply_plan.ask_policy) == "must_clarify":
        return False, "ask_policy_must_clarify"

    response_mode = _normalize_response_mode(reply_plan.response_mode)
    if response_mode not in _FASTPATH_CONSULT_RESPONSE_MODES:
        return False, f"response_mode_not_safe:{response_mode}"

    return True, "eligible"
```

---

## 3.6 Tăng chất lượng `consult_reply` để đủ dùng cho fast-path

### File cần sửa
- `orchestrator_service/app.py`
- hàm `_build_decider_prompt(...)`

### Chỗ cần tăng yêu cầu
Trong phần `Quy tac consult_reply`, thêm yêu cầu rõ hơn cho `consultive_recommendation`:

```text
- Với consultive_recommendation, consult_reply phải usable ngay như câu trả lời cuối cho consult-only turn.
- Nếu không cần grounding dự án hoặc không cần xin contact/hẹn gặp, consult_reply phải đủ để trả thẳng cho user.
- Tránh tạo câu trả lời phụ thuộc vào bước reply_synthesis phía sau.
```

### Mục tiêu
Khi fast-path bật cho `consultive_recommendation`, `consult_reply` không bị quá “thô” hoặc phụ thuộc rewrite.

---

## 3.7 Không thay đổi vị trí gate trong flow

Fast-path phải được quyết định **sau**:
- `sales_state_engine`
- `select_conversation_goal`
- `select_next_best_action`
- `build_reply_plan`

Không dời gate lên ngay sau `analyze_turn()`.

### Vì sao
Nếu gate quá sớm thì sẽ bỏ qua:
- `engagement_state`
- `sales_state`
- `conversation_goal`
- `ask_policy`

Điều đó làm giảm quality.

---

## 4) Phần B — Cắt normalize/rewrite latency

## 4.1 File cần sửa
- `orchestrator_service/app.py`

## 4.2 Vấn đề hiện tại trong synth path
Trong `synthesize_assistant_reply(...)`, sau lần generate chính, code hiện có thể gọi thêm model ở 2 chỗ:

1. `_normalize_reply_to_accented_vietnamese(...)`
2. `_rewrite_reply_by_policy(...)`

Đây là nguyên nhân làm synth path vẫn nặng.

---

## 4.3 Hướng sửa đúng

### Nguyên tắc
- **Không rewrite bằng model nữa**
- **Không normalize dấu bằng model nữa**
- Chỉ dùng:
  - local sanitize
  - fallback local

---

## 4.4 Sửa `_normalize_reply_to_accented_vietnamese(...)`

### Hiện tại
Hàm này gọi model lại nếu phát hiện output không dấu.

### Nên sửa
Đổi hàm thành no-op:

```python
def _normalize_reply_to_accented_vietnamese(reply: str, settings: Settings) -> str:
    _ = settings
    return (reply or "").strip()
```

### Vì sao
- Prompt synthesis đã yêu cầu tiếng Việt có dấu
- Model tốt như Kimi/Gemma chat thường đã đủ ổn
- Cost latency của một model-call chỉ để thêm dấu là không đáng

### Nếu muốn rollback an toàn
Có thể thêm cờ config:
- `ORCHESTRATOR_ENABLE_ACCENT_NORMALIZE_MODEL=false`

Nhưng phiên bản tối ưu nhất là bỏ hẳn.

---

## 4.5 Sửa `_rewrite_reply_by_policy(...)`

### Mục tiêu
Không dùng model để viết lại reply nữa.

### Cách làm
Giữ hàm này tạm thời cho tương thích, nhưng không gọi nữa.

Có thể thay body thành:

```python
def _rewrite_reply_by_policy(reply: str, single_project_mode: bool, ask_policy: str, settings: Settings) -> str:
    _ = single_project_mode
    _ = ask_policy
    _ = settings
    return (reply or "").strip()
```

Hoặc giữ nguyên hàm nhưng loại bỏ hoàn toàn chỗ gọi trong `synthesize_assistant_reply(...)`.

---

## 4.6 Sửa `synthesize_assistant_reply(...)`

### Mục tiêu flow mới
Từ:

```text
generate
-> normalize by model
-> sanitize local
-> maybe rewrite by model
-> sanitize local
-> maybe fallback
```

thành:

```text
generate
-> sanitize local
-> if still bad: fallback local
```

### Bản sửa nên làm
Trong `synthesize_assistant_reply(...)`, thay block xử lý reply bằng logic sau:

```python
response_payload = _call_model_generate(
    settings=settings,
    prompt=prompt,
    temperature=max(0.0, min(1.0, settings.decider_temperature + 0.18)),
    response_format="json",
)

if isinstance(response_payload, dict):
    reply = str(response_payload.get("assistant_reply", "")).strip()
else:
    parsed_obj = _extract_json_object(str(response_payload))
    reply = str(parsed_obj.get("assistant_reply", "")).strip()

reply = _sanitize_reply_for_policy(
    reply=reply,
    ask_policy=reply_plan.ask_policy,
    single_project_mode=single_project_mode,
    question_focus=reply_plan.question_focus,
)

if _reply_needs_retry(
    reply=reply,
    single_project_mode=single_project_mode,
    ask_policy=reply_plan.ask_policy,
):
    fallback_reply = _build_reply_fallback(
        message=message,
        analysis=analysis,
        grounded_result=grounded_result,
    )
    reply = _sanitize_reply_for_policy(
        reply=fallback_reply,
        ask_policy=reply_plan.ask_policy,
        single_project_mode=single_project_mode,
        question_focus=reply_plan.question_focus,
    )

if reply:
    return _compact_text(reply, max_words=120)
```

### Quan trọng
Bỏ hoàn toàn các đoạn:
- `_normalize_reply_to_accented_vietnamese(...)`
- `_rewrite_reply_by_policy(...)`

khỏi flow chạy chính.

---

## 4.7 Đơn giản hóa `_reply_needs_retry(...)`

### Hiện tại
`_reply_needs_retry(...)` đang phụ thuộc `_needs_policy_rewrite(...)` và còn mang ý nghĩa “có nên rewrite bằng model nữa không”.

### Nên đổi nghĩa
Nó chỉ nên trả lời:
> reply hiện tại có quá tệ để phải rơi về fallback local không?

### Gợi ý bản mới
```python
def _reply_needs_retry(reply: str, single_project_mode: bool, ask_policy: str) -> bool:
    cleaned = (reply or "").strip()
    if not cleaned:
        return True

    normalized_ask_policy = _normalize_ask_policy(ask_policy)
    question_marks = cleaned.count("?")

    if normalized_ask_policy == "avoid_question" and question_marks > 0:
        return True
    if normalized_ask_policy == "must_clarify" and question_marks == 0:
        return True
    if question_marks > 1:
        return True

    lowered = cleaned.lower()
    technical_terms = ["route", "retrieval", "metadata", "payload", "confidence", "schema", "vector"]
    if any(term in lowered for term in technical_terms):
        return True

    if single_project_mode and re.search(r"\bnhiều\s+(lựa chọn|dự án|căn hộ)\b", lowered):
        return True
    if single_project_mode and re.search(r"\bkhu\s*vực\b|\bquận\b", lowered):
        return True

    return False
```

### Ý nghĩa mới
- `True` = dùng fallback local
- `False` = accept reply hiện tại

Không còn bước “rewrite thêm một lần nữa”.

---

## 5) Phần C — Điều chỉnh prompt để giảm fallback

## 5.1 File cần sửa
- `orchestrator_service/app.py`
- hàm `_build_reply_synthesis_prompt(...)`

## 5.2 Vấn đề hiện tại
Prompt hiện đang rất chặt:
- 2–4 câu
- tối đa 75 từ
- nhịp `reflect -> insight -> invite`
- nhiều rule nhỏ cùng lúc

Điều này làm model dễ sinh reply vi phạm nhẹ, rồi bị đẩy sang rewrite/fallback.

## 5.3 Nên nới một chút

### Sửa rule:
Từ:
```text
- Phản hồi 2-4 câu ngắn, tối đa 75 từ.
- Mỗi lượt phải theo nhịp reflect -> insight -> invite.
```

Thành:
```text
- Phản hồi 2-4 câu ngắn, ưu tiên dưới 90 từ.
- Mỗi lượt ưu tiên reflect và insight; invite chỉ dùng khi phù hợp với response_mode và conversation_goal.
```

### Vì sao
- giảm xác suất reply bị coi là “cần sửa lại”
- tăng khả năng một lần generate là usable luôn

---

## 6) Phần D — Logging để xác minh tối ưu có thật sự chạy

## 6.1 Giữ các log mới đã thêm
Hiện branch đã có:
- `used_fastpath_consult_reply`
- `reply_source`
- `fastpath_gate_reason`

Tiếp tục giữ các field này.

## 6.2 Nên log thêm trong synth path
Trong `reply_obs.update(...)`, thêm:

```python
"used_model_rewrite": False,
"used_model_accent_normalize": False,
```

Sau khi bỏ 2 step kia, log này giúp xác nhận synth path đã gọn hơn thật.

---

## 7) Checklist sửa source cụ thể

## Bước 1
Trong `app.py`, sửa `_FASTPATH_CONSULT_RESPONSE_MODES`:

```python
_FASTPATH_CONSULT_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
}
```

## Bước 2
Giữ nguyên `_evaluate_consult_reply_fastpath(...)` theo kiểu deterministic.

## Bước 3
Trong `_build_decider_prompt(...)`, thêm yêu cầu mạnh hơn cho `consult_reply` của consult-only turn.

## Bước 4
Biến `_normalize_reply_to_accented_vietnamese(...)` thành no-op.

## Bước 5
Loại bỏ việc gọi `_rewrite_reply_by_policy(...)` trong `synthesize_assistant_reply(...)`.

## Bước 6
Đơn giản hóa `synthesize_assistant_reply(...)` thành:
- 1 lần generate
- 1 lần local sanitize
- fallback local nếu cần

## Bước 7
Nới prompt synthesis từ 75 từ lên khoảng 90 từ và không bắt buộc `invite` mọi lượt.

---

## 8) Kỳ vọng sau khi sửa

### Consult-only turn
Nếu `response_mode` là:
- `warm_welcome`
- `value_teaser`
- `discover_need`
- `consultive_recommendation`

thì `reply_source` nên chuyển nhiều hơn sang:
- `consult_reply_fastpath`

### Synth path còn lại
Dù vẫn phải synthesize, tổng latency cũng giảm vì:
- không còn accent normalize bằng model
- không còn rewrite bằng model

---

## 9) Cách rollout an toàn

### Phase 1
- mở fast-path cho `consultive_recommendation`
- bỏ accent normalize bằng model
- giữ fallback local

### Phase 2
- bỏ hẳn rewrite bằng model
- nới prompt synthesis một chút

### Phase 3
- đo lại log với các field:
  - `reply_source`
  - `fastpath_gate_reason`
  - `response_mode`
  - `ask_policy`

---

## 10) Tiêu chí thành công

### Thành công về latency
- consult-only turn phải có tỷ lệ `reply_source=consult_reply_fastpath` tăng rõ
- synth path không còn xuất hiện model hậu xử lý

### Thành công về quality
- reply không bị quá khô
- không lặp brochure intro
- không vi phạm ask_policy
- không xuất hiện technical terms

---

## 11) Chốt ngắn

Muốn tối ưu branch hiện tại mà vẫn an toàn, thứ tự sửa đúng là:

1. **Mở fast-path thêm cho `consultive_recommendation`**
2. **Không dùng model để normalize dấu nữa**
3. **Không dùng model để rewrite theo policy nữa**
4. **Chỉ giữ local sanitize + fallback local**

Đây là cách giảm latency thực tế rõ nhất mà chưa phá cấu trúc orchestrator hiện tại.
