# Sale Agent Latency Implementation Plan

> Tài liệu này là bản implementation-oriented để đội dev bám vào khi sửa code.
> 
> Nó **thay thế về mặt định hướng triển khai** cho phần diễn đạt cũ kiểu `heuristic-first for project` trong tài liệu latency trước đó.

---

## 1. Mục tiêu thực tế

Giảm latency của sale-agent mà **không hy sinh chất lượng semantic understanding**.

Mục tiêu không phải là:

- thay LLM bằng hardcode keyword ở các case mơ hồ
- hardcode tên dự án hoặc keyword domain
- ép mọi route đi deterministic nếu câu hỏi còn thiếu ngữ cảnh

Mục tiêu đúng là:

- cắt các **LLM call dư thừa**
- giảm **token input/output không cần thiết**
- chuyển các case rõ ràng sang deterministic path
- giữ LLM cho những nơi LLM thật sự tạo ra giá trị: **disambiguation**, **semantic understanding**, **final phrasing**

---

## 2. Điều KHÔNG nên làm

### 2.1. Không làm `heuristic-first` theo nghĩa hardcode keyword

Không nên thiết kế theo kiểu:

- có từ `pháp lý` thì nghĩ user đang hỏi project facts
- có từ `shophouse` thì nghĩ đúng một loại dự án nào đó
- có từ `quận 7` thì map vào một project cụ thể
- có pattern text giống tên dự án thì tự chốt project luôn

Lý do:

1. Không bền khi kho tài liệu mở rộng.
2. Tài liệu mới có thể thêm nhiều dự án cùng họ tên.
3. Người dùng có thể nói tắt, gọi sai chính tả, hoặc nhắc lại bằng đại từ như `dự án đó`, `cái hôm qua em nói`.
4. Nếu heuristic sai rồi mới retrieval/generation tiếp, model rất dễ trả lời trơn tru trên **entity sai**.

### 2.2. Không fallback thẳng sang full retrieval khi entity chưa rõ

Nếu chưa resolve được project mà đã chạy retrieval tổng quát, rủi ro lớn là:

- retrieve nhầm project
- LLM tóm tắt dựa trên context sai
- câu trả lời nghe vẫn hợp lý nhưng factual grounding sai

Nguyên tắc:

> **Entity chưa rõ -> resolve entity trước -> rồi mới retrieve sâu**

---

## 3. Nguyên tắc đúng cho `project_qa`

### 3.1. Không phải heuristic-first, mà là `memory / registry / disambiguation`

Flow đúng cho `project_qa` hoặc `comparison`:

1. **Session memory first**
   - nếu turn trước đã chốt rõ project và turn hiện tại vẫn cùng chủ đề, reuse project đó
2. **Registry first**
   - exact match / normalized match / alias match trên danh mục project được sinh từ corpus
3. **High-confidence candidate selection**
   - nếu chỉ ra được 1 candidate mạnh, dùng candidate đó
4. **LLM disambiguation**
   - nếu có 2-3 candidate gần nhau hoặc context nhiều-turn mơ hồ
5. **Ask user or safe fallback**
   - nếu vẫn chưa đủ chắc thì hỏi lại hoặc trả về overview an toàn

### 3.2. Heuristic chỉ được dùng để giảm search space

Heuristic được phép dùng ở đây chỉ là:

- normalized string match
- alias lookup
- fuzzy overlap nhẹ
- reuse resolved entity từ session
- candidate ranking từ project registry

Heuristic **không được** tự quyết semantic meaning của câu.

Nói ngắn gọn:

- heuristic để **lọc candidate**
- LLM để **chọn candidate đúng khi còn mơ hồ**

---

## 4. Fallback logic chuẩn

### 4.1. Với `project_qa`

Nếu confidence của candidate thấp:

- **không** đi thẳng sang retrieval tổng quát
- **không** generate câu trả lời như thể đã biết đúng dự án
- **nên** fallback sang LLM disambiguation nhỏ

Sau LLM disambiguation:

- nếu chọn được đúng 1 project -> load đúng doc chunks rồi trả lời
- nếu vẫn ambiguous -> hỏi lại user
- nếu user chỉ hỏi tổng quan chung chung -> có thể chuyển sang catalog overview an toàn

### 4.2. Với `comparison`

Nếu mới thấy 1 project rõ còn project còn lại mơ hồ:

- không so sánh ngay
- dùng LLM disambiguation cho project thứ 2
- nếu vẫn không chắc, hỏi lại tên dự án còn thiếu

### 4.3. Với `product_matching`

`product_matching` không cần resolve 1 project duy nhất như `project_qa`.

Ở đây nên ưu tiên:

- project facts đã cache
- filter theo lead profile
- deterministic ranking trước
- chỉ dùng LLM để phrasing hoặc reasoning mềm hơn nếu cần

---

## 5. Project registry mới nên có gì

Không được phụ thuộc vào keyword do dev nghĩ ra.

Phải build một **project registry** từ kho tài liệu / project facts đã ingest.

### 5.1. Cấu trúc đề xuất

```python
ProjectRegistryItem = {
    "project_id": str,
    "canonical_name": str,
    "aliases": list[str],
    "doc_ids": list[str],
    "source_files": list[str],
    "normalized_name": str,
}
```

### 5.2. Alias lấy từ đâu

Alias không nên hardcode thủ công toàn bộ.

Alias có thể lấy từ:

- canonical name
- normalized name bỏ dấu / bỏ ký tự đặc biệt
- các heading/tên xuất hiện lặp lại trong doc
- các biến thể rút gọn an toàn được extract tự động
- một alias map thủ công nhỏ cho các case business-critical

### 5.3. Registry refresh khi nào

- build cùng lúc với `_load_project_facts()`
- refresh khi `refresh_retrieval_caches()` chạy
- không rebuild registry ở mỗi turn chat

---

## 6. Thiết kế confidence và ngưỡng

### 6.1. Không dùng ngưỡng thấp để auto-retrieve

Nếu heuristic score thấp, hành vi đúng là:

- **fallback sang disambiguation LLM**, không phải retrieval rộng

### 6.2. Đề xuất mức hành vi

#### Case A: confidence rất cao

Ví dụ:

- exact match canonical name
- alias match rõ ràng
- session resolved project khớp tốt với current turn

Hành động:

- dùng luôn candidate
- skip LLM disambiguation

#### Case B: confidence trung bình

Ví dụ:

- 2-3 candidate cùng họ tên Noble
- user nói tắt
- current turn phụ thuộc nhiều vào history

Hành động:

- gọi `LLM disambiguation` với candidate list đã rút gọn

#### Case C: confidence thấp

Hành động:

- không retrieve sâu
- nếu câu hỏi broad/general -> overview safe path
- nếu câu hỏi fact-specific -> hỏi lại user để xác nhận project

---

## 7. LLM disambiguation phải nhỏ và rẻ

Đây không phải full reasoning pass.

Đây chỉ là một micro-task để chọn entity đúng từ danh sách candidate ngắn.

### 7.1. Input

- current user text
- 3-4 turns gần nhất
- 2-3 candidate names
- resolved project trong session nếu có

### 7.2. Output schema

```json
{
  "selected_project": null | "<canonical project name>",
  "status": "selected" | "ambiguous" | "none",
  "confidence": 0.0
}
```

### 7.3. Nguyên tắc

- không cho model bịa project mới
- chỉ được chọn trong candidate list
- nếu không chắc thì trả `ambiguous` hoặc `none`
- `max_tokens` nhỏ, chỉ đủ cho JSON

---

## 8. Concrete code changes theo file

## 8.1. `service/RAG/core/dependencies.py`

### Việc phải làm

- forward `max_tokens`
- log provider/model/max_tokens ở parser tasks nếu cần debug

### Kết quả mong muốn

- micro parser thật sự nhỏ
- disambiguation LLM thật sự rẻ

---

## 8.2. `service/RAG/sales/nodes/fast_parse_user_turn.py`

### Việc phải làm

- cho short slot reply đi rule-only nếu state/step rõ
- không gọi micro parse cho các case như `2 người`, `quận 7`, `0`, `chưa có con`, `để ở`
- giữ deep parse cho case semantic mơ hồ thật

### Không làm

- không hardcode keyword dự án ở node này
- không cố resolve project entity ở parser nếu chưa cần

---

## 8.3. `service/RAG/sales/edges.py`

### Việc phải làm

- nếu deep parse đã xong và đủ confidence thì skip `classify_and_extract`
- tránh parser chồng parser

---

## 8.4. `service/RAG/sales/nodes/retrieve_context.py`

Đây là file quan trọng nhất cho phần `project_qa`.

### A. Bỏ tư duy `LLM-first` cho project resolution

Không nên vào `project_qa` là gọi ngay `_resolve_project_name_from_user_context(...)` như hiện tại.

Thay bằng flow:

1. thử reuse `state.resolved_project_name`
2. thử exact/normalized/alias match từ registry
3. thử candidate ranking từ registry
4. nếu ambiguous -> gọi `LLM disambiguation`
5. nếu vẫn ambiguous -> block an toàn hoặc hỏi lại user

### B. Thêm project registry helpers

Các helper đề xuất:

- `_build_project_registry(project_facts)`
- `_normalize_project_alias(text)`
- `_resolve_project_candidates_from_registry(state, registry, max_items=3)`
- `_select_high_confidence_candidate(candidates)`
- `_disambiguate_project_with_llm(state, candidates)`

### C. Tách rõ `project_qa` và `comparison`

#### `project_qa`

Cần đúng 1 project rõ.

Nếu chưa rõ:

- broad question -> safe overview path
- fact question -> hỏi lại user

#### `comparison`

Cần đúng 2 project rõ.

Nếu chưa đủ 2:

- disambiguate
- hoặc hỏi lại user

### D. Không full-retrieve khi entity chưa rõ

Đây là quy tắc cứng.

Nếu project chưa rõ mà câu hỏi dạng:

- pháp lý
- bảng giá
- số căn
- tiến độ
- thanh toán

thì **không** chạy retrieval rộng rồi generate như thể đã biết đúng dự án.

---

## 8.5. `service/RAG/sales/nodes/build_response.py`

### Việc phải làm

- tăng deterministic path cho `project_qa` và `comparison` khi đã có `project_facts`
- chỉ dùng LLM phrasing khi cần mềm hóa output
- nếu state đánh dấu `project_qa_blocked` thì phải trả response an toàn, không suy diễn

---

## 8.6. `service/RAG/api/routes_sales.py`

### Việc phải làm

- stream thật cho sales flow
- nếu graph ra fast ack sớm thì flush luôn
- nếu final response là LLM-generated thì stream token, không chờ full graph xong rồi mới chunk

---

## 9. Safe fallback messages

### 9.1. Khi project chưa rõ nhưng user hỏi fact-specific

Ví dụ fallback tốt:

- `Để em trả lời chính xác về pháp lý hoặc bảng giá, Anh/Chị cho em xin đúng tên dự án Noble mình đang quan tâm nhé?`

### 9.2. Khi user hỏi broad overview

Ví dụ fallback tốt:

- `Nếu Anh/Chị đang muốn xem tổng quan các dự án Noble hiện có, em có thể gửi nhanh một shortlist ngắn để mình chọn đúng dự án rồi em đi sâu tiếp ạ.`

---

## 10. Telemetry bắt buộc phải thêm

Để team benchmark trước/sau, cần log các field sau:

- `project_resolution_source`
  - `session_memory`
  - `registry_exact`
  - `registry_alias`
  - `registry_ranked`
  - `llm_disambiguation`
  - `user_clarification`
  - `unresolved`

- `project_resolution_confidence`
- `candidate_count_before_llm`
- `llm_disambiguation_called`
- `retrieval_skipped_due_to_unresolved_entity`
- `full_retrieval_used`

---

## 11. Thứ tự implement khuyến nghị

### Phase A — ít rủi ro, làm ngay

1. forward `max_tokens`
2. rule-only cho short discovery replies
3. skip `classify_and_extract` khi deep parse đã đủ
4. thêm telemetry

### Phase B — sửa `project_qa` đúng hướng

1. build project registry từ `project_facts`
2. thêm memory/registry resolution path
3. thêm LLM disambiguation nhỏ
4. chặn full retrieval nếu entity chưa rõ
5. thêm safe fallback messages

### Phase C — UX stream

1. stream fast ack thật
2. stream final LLM output thật

---

## 12. Định nghĩa thành công

Implementation này được coi là thành công khi:

1. latency giảm rõ ở discovery turns đơn giản
2. số call parser chồng nhau giảm
3. `project_qa` không còn trả lời trên project sai vì retrieval khi entity chưa rõ
4. team có thể thêm tài liệu mới mà không cần cập nhật bộ keyword hardcode

---

## 13. Kết luận triển khai

Tư duy đúng để implement là:

> **Không heuristic-first theo keyword. Không LLM-first cho mọi case.**
> 
> **Phải là memory-first + registry-first + LLM disambiguation khi cần.**

Đây là cách cân bằng được cả 3 thứ:

- latency
- chất lượng semantic understanding
- khả năng mở rộng khi corpus thay đổi

---

## 14. Bước tiếp theo nên làm ngay

Sau doc này, đội dev nên tạo thêm 1 checklist patch-level gồm:

- diff mong muốn cho `retrieve_context.py`
- helper signatures mới
- telemetry fields cần log
- bộ test cases cho `project_qa` / `comparison` / `product_matching`

Nếu không có checklist patch-level, team rất dễ quay lại kiểu tối ưu bằng keyword heuristic và làm giảm chất lượng toàn hệ thống.