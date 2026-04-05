# Sale Agent Latency Optimization Guide

## Mục tiêu

Tài liệu này tổng hợp các điểm nghẽn hiện tại trong branch `feature/sale-agent` và đề xuất thứ tự tối ưu để giảm:

- **TTFT** (time to first token / time to first useful response)
- **Tổng thời gian phản hồi** của `/sales/chat` và `/sales/chat/stream`
- **Số lượng LLM round-trip** trên mỗi turn

Ưu tiên chính là giảm số lần gọi LLM trước khi hệ thống có thể trả về nội dung hữu ích cho người dùng.

---

## Kết luận nhanh

Điểm nghẽn lớn nhất hiện tại không nằm ở template rendering hay validator, mà nằm ở **tầng hiểu user turn trước khi trả lời**.

Cụ thể:

1. `fast_parse_user_turn` đang có thể tạo ra **1 đến 2 LLM call** cho mỗi turn.
2. Nếu `fast_parse` chưa đủ chắc, graph còn rơi tiếp sang `classify_and_extract`, tạo thành **3 LLM call liên tiếp chỉ để hiểu intent/slot**.
3. Với các nhánh grounded (`project_qa`, `comparison`, `objection_handling`, `closing_next_step`), hệ thống có thể lại thêm:
   - 1 LLM call trong retrieval helper
   - 1 LLM call để generate final response
4. `/sales/chat/stream` hiện **chưa stream thật** cho sales flow; chỉ phát một câu ack cố định rồi chờ toàn bộ graph chạy xong.

Hệ quả là nhiều lượt chat đang bị chậm ngay từ giai đoạn **hiểu ý khách**, trước cả khi bắt đầu phát nội dung trả lời thực sự.

---

## Flow hiện tại gây trễ ở đâu

### 1. Graph tổng quát

Flow chính:

`ingest_user_turn -> fast_parse_user_turn -> classify_and_extract? -> update_lead_profile -> resolve_sales_state -> resolve_script_step -> decide_response_action -> retrieve_context?/build_response -> validate_response -> persist_turn -> finalize_output`

Ý nghĩa:

- `fast_parse_user_turn` là chốt chặn đầu tiên
- nếu chưa tự tin thì mới rơi sang `classify_and_extract`
- sau đó mới đến retrieval hoặc response generation

### 2. Bottleneck số 1: `fast_parse_user_turn`

Hiện tại node này làm nhiều việc hơn mong đợi:

- rule-based extraction mỏng
- luôn gọi `_micro_understand_turn_with_llm(...)`
- nếu `should_escalate=true` thì gọi thêm `_deep_understand_turn_with_llm(...)`
- sau đó mới quyết định lane A / B / C

Điều này khiến các turn rất ngắn như:

- `quận 7`
- `2 người`
- `chưa có con`
- `để ở`

vẫn có thể tốn 1 LLM call, thậm chí 2 LLM call, dù bản chất có thể xử lý bằng rule + session state.

### 3. Bottleneck số 2: fallback `classify_and_extract`

Nếu `fast_parse` ra `lane_c` hoặc confidence thấp, hệ thống chạy tiếp `classify_and_extract`.

Như vậy ở case mơ hồ, chuỗi call có thể là:

1. micro understand
2. deep understand
3. classify + extract

Đây là phần đắt nhất trong sales flow vì chưa tạo ra câu trả lời nào cho user nhưng đã tiêu tốn 2-3 round-trip LLM.

### 4. Bottleneck số 3: grounded path bị gọi LLM kép

Ở các nhánh như:

- `project_qa`
- `comparison`
- `objection_handling`
- `closing_next_step`

hệ thống có thể đi theo dạng:

1. resolve project / retrieval helper
2. RAG summary hoặc retrieval generation
3. final response generation

Nếu không khóa chặt deterministic path, một lượt hỏi grounded có thể tốn thêm 1-2 LLM call sau giai đoạn parsing.

### 5. Bottleneck cảm nhận: stream chưa thật

`/sales/chat/stream` hiện phát:

- 1 câu `thinking_ack`
- rồi chờ `_run_sales_flow()` hoàn tất
- sau đó mới bắt đầu stream response thật theo chunk

Kết quả:

- user có cảm giác hệ thống “đã phản hồi” nhưng thực tế chưa thấy nội dung hữu ích sớm hơn
- `fast_ack_response` trong graph không giúp giảm TTFT thực tế nếu nó không được flush ra client ngay khi có

---

## Phân loại impact

### P0 — Tác động lớn nhất, nên làm ngay

#### P0.1. Cho phép `llm_model_func()` forward `max_tokens`

Hiện các call như micro/deep parse có truyền `max_tokens`, nhưng tầng `llm_model_func()` chưa forward tham số này xuống payload của provider OpenAI-compatible.

Hệ quả:

- micro parse không thực sự “micro”
- deep parse không bị chặn độ dài output
- tăng token out không cần thiết
- kéo dài latency

**Việc cần làm:**

- thêm `max_tokens` vào payload của `llm_model_func()`
- giữ default nhỏ cho các task parser JSON

**Đề xuất:**

- micro parse: `max_tokens=120-160`
- deep parse: `max_tokens=180-220`
- classify_and_extract: `max_tokens=220-280`

---

#### P0.2. Không gọi LLM cho lane A thật ngắn

Các case sau nên đi **rule-only**, không gọi `_micro_understand_turn_with_llm(...)`:

- trả lời numeric slot trực tiếp
- trả lời location ngắn sau khi agent đang hỏi vị trí
- trả lời `chưa có con`, `0`, `không`
- trả lời mục đích rõ ràng như `để ở`, `đầu tư`

**Nguyên tắc:**

Nếu:

- current state là `need_discovery`
- current step đang hỏi đúng slot
- user text ngắn
- parser đã trích được slot rõ

thì trả lane A ngay.

**Lợi ích:**

- cắt 1 LLM call ở các turn discovery đơn giản
- giảm mạnh latency trung bình trong hội thoại thu thập nhu cầu

---

#### P0.3. Nếu đã chạy deep parse thì không chạy lại `classify_and_extract`

Hiện deep parse và `classify_and_extract` có phần nhiệm vụ chồng lấn:

- hiểu turn role
- hiểu intent
- suy ra slot updates

Khi `deep_understand_turn_with_llm()` đã chạy thành công và confidence đủ dùng, không nên lại đẩy sang `classify_and_extract` chỉ vì lane/cutoff cứng.

**Việc cần làm:**

- thêm cờ như `semantic_parse_done=true`
- trong `route_after_fast_parse`, nếu đã có deep parse usable thì bỏ qua `classify_and_extract`

**Mục tiêu:**

- biến case mơ hồ từ `2-3 LLM calls` còn `1-2`

---

#### P0.4. Stream thật cho sales flow

Cần thay đổi `/sales/chat/stream` để:

- phát `fast_ack_response` thật sự nếu graph đã đi qua nhánh lane B
- hoặc stream token từ `build_response` nếu response cuối là LLM-generated
- tránh pattern hiện tại: ack giả -> chờ xong toàn bộ -> mới phát

**Lợi ích:**

- giảm TTFT cảm nhận
- giữ UX tốt ngay cả khi backend vẫn còn vài bước nội bộ

---

### P1 — Tác động tốt, nên làm sau P0

#### P1.1. Tách parser nhiệm vụ rõ ràng

Hiện có ba khối cùng làm semantic understanding:

- rule parse
- micro parse
- deep parse
- classify_and_extract

Nên refactor theo nguyên tắc:

- **rule parser**: slot rõ, không LLM
- **micro parser**: chỉ phân biệt answer-slot / ask-project / objection / buy-signal
- **deep parser**: chỉ dùng khi mơ hồ thật
- **classify_and_extract**: fallback cuối cùng, không chạy mặc định

---

#### P1.2. Giảm khối lượng prompt ở parser

Prompt của micro/deep parse hiện khá dài vì chứa:

- nhiều schema field
- nhiều quy tắc
- ví dụ
- working memory
- history gần đây

Có thể giảm bằng cách:

- rút bớt field ít dùng trong parser phase
- tách `semantic_move` và `slot_updates` khỏi các field không cần ở lượt đầu
- giới hạn history vào đúng 2-3 turn gần nhất
- rút gọn instruction wording

**Mục tiêu:** giảm input tokens cho parser call.

---

#### P1.3. Hạn chế LLM helper trong `retrieve_context`

Riêng `project_qa` đang có thêm helper để resolve project name từ lịch sử.

Nên ưu tiên thứ tự:

1. match từ `resolved_project_name`
2. match từ known project list bằng heuristic
3. chỉ khi vẫn fail mới gọi LLM helper

**Tránh:** cứ vào `project_qa` là gọi LLM để resolve project name.

---

#### P1.4. Không generate lại khi deterministic path đã đủ tốt

Trong `build_response`, các action như:

- `catalog_overview`
- `match_options`
- `project_qa` khi đủ `project_facts`
- `comparison` khi đủ facts

đã có deterministic scaffolding khá tốt.

Cần ưu tiên deterministic path trước, chỉ dùng LLM khi thật sự cần diễn đạt mềm hơn.

---

### P2 — Tối ưu bổ sung

#### P2.1. Telemetry latency theo node

Cần log hoặc trace riêng cho từng node:

- `fast_parse_user_turn`
- `classify_and_extract`
- `retrieve_context`
- `build_response`
- `validate_response`
- `persist_turn`

Và chi tiết hơn theo từng sub-call:

- micro parse duration
- deep parse duration
- classify duration
- rag query duration
- llm generation duration

**Nếu không có số đo theo node, rất khó biết tối ưu nào thật sự hiệu quả.**

---

#### P2.2. Budget theo route

Thiết lập budget mềm cho từng route:

- discovery simple turn: tối đa 0-1 LLM call
- discovery ambiguous turn: tối đa 1-2 LLM call
- project_qa simple fact: ưu tiên deterministic / 0-1 LLM call
- grounded generation phức tạp: tối đa 2 LLM call sau retrieval

---

## Lộ trình sửa đề xuất

## Phase 1 — Quick wins

1. Forward `max_tokens` trong `llm_model_func()`
2. Bypass micro LLM cho short slot reply rõ ràng
3. Nếu deep parse usable thì skip `classify_and_extract`
4. Log latency từng node

**Kỳ vọng:** giảm latency trung bình rõ rệt mà ít phải đổi kiến trúc.

---

## Phase 2 — Flow simplification

1. Rút gọn prompt micro/deep parse
2. Giảm field JSON ở parser layer
3. Đẩy `project_qa` sang heuristic-first thay vì LLM-first ở bước resolve name
4. Tăng deterministic coverage cho `project_qa` / `comparison`

---

## Phase 3 — UX latency

1. Stream thật cho sales flow
2. Flush fast ack thật từ graph
3. Nếu build_response là LLM-generated thì stream token trực tiếp thay vì chờ xong toàn bộ

---

## Mức ưu tiên sửa file

### Ưu tiên 1

- `service/RAG/core/dependencies.py`
- `service/RAG/sales/nodes/fast_parse_user_turn.py`
- `service/RAG/sales/edges.py`

### Ưu tiên 2

- `service/RAG/api/routes_sales.py`
- `service/RAG/sales/nodes/retrieve_context.py`
- `service/RAG/sales/nodes/build_response.py`

### Ưu tiên 3

- `service/RAG/sales/prompt_builder.py`
- `service/RAG/sales/nodes/validate_response.py`

---

## KPI nên đo sau khi tối ưu

### KPI kỹ thuật

- median latency `/sales/chat`
- p95 latency `/sales/chat`
- TTFT `/sales/chat/stream`
- số LLM call trung bình / turn
- số token input/output trung bình / turn
- tỷ lệ turn rơi vào `classify_and_extract`
- tỷ lệ turn dùng deep parse

### KPI hành vi

- số turn để đi hết discovery
- tỷ lệ user drop sau 1-2 lượt đầu
- tỷ lệ user tiếp tục sau câu hỏi discovery

---

## Kỳ vọng hiệu quả

Nếu làm đúng theo thứ tự P0 trước, có thể kỳ vọng:

- giảm đáng kể latency ở discovery turns đơn giản
- giảm số case bị 2-3 LLM calls chỉ để hiểu intent
- cải thiện rõ cảm giác phản hồi ở endpoint stream
- giảm chi phí token không cần thiết ở parser layer

---

## Kết luận

Muốn tối ưu latency của sale-agent trong branch `feature/sale-agent`, ưu tiên số 1 là:

> **giảm số LLM call trước khi có thể quyết định response action**

Không nên tập trung trước vào validator hay template rendering, vì phần đắt nhất hiện tại đang là:

- `fast_parse_user_turn`
- `classify_and_extract`
- các helper LLM trong retrieval path
- và cách stream hiện tại chưa flush được nội dung hữu ích sớm

Thứ tự làm tốt nhất:

1. sửa `llm_model_func` để nhận `max_tokens`
2. cho short discovery reply đi rule-only
3. tránh deep parse xong lại classify thêm lần nữa
4. stream thật cho sales flow

---

## Gợi ý bước tiếp theo

Sau tài liệu này, nên tạo tiếp 1 checklist implementation ngắn với các patch cụ thể theo từng file để đội dev có thể sửa lần lượt và benchmark trước/sau.