# Kế hoạch 3 sprint đầu để sửa runtime, router và retrieval cho agent tư vấn BĐS

## 1. Mục tiêu của tài liệu

Tài liệu này chốt lại kế hoạch triển khai cho **3 sprint đầu** của nhánh `test-screen-agents` để xử lý các vấn đề đang gặp phải:

- runtime hiện chưa bám đúng flow `consult_discovery -> maybe project_grounded`
- router đang phân biệt route kém, thiên về "thiếu clarity thì hỏi thêm" hơn là hiểu **loại query**
- retrieval hiện chưa đủ giàu cho các query kiểu **POI / proximity** như:
  - gần bệnh viện
  - gần trường học
  - gần công viên
  - thuận tiện cho gia đình có con nhỏ
- hành vi test sẽ chuyển sang **manual test trực tiếp**, không tiếp tục phát triển auto test trong giai đoạn này

Tài liệu này chỉ tập trung vào **3 sprint đầu tiên** để team có thể bắt đầu triển khai ngay.

---

## 2. Nguyên tắc bắt buộc

### 2.1. Không dùng hard rules trong production routing

Không được dùng các rule cứng kiểu:

```python
if "bệnh viện" in message:
    route = "project_grounded"
```

Thay vào đó, route phải được quyết định bởi:

- structured output từ LLM decider
- state hiện tại của cuộc trò chuyện
- mức sẵn sàng để retrieval (`retrieval_readiness`)
- kết quả retrieval thực tế

### 2.2. Route phải dựa vào `query_type`, không dựa vào thiếu slot

Hệ thống phải phân biệt được ít nhất 4 nhóm:

- `advisory_strategy`
- `project_matching`
- `project_specific`
- `clarification`

Thiếu budget, khu vực hoặc timeline **không được tự động đẩy query về consult**.

### 2.3. Missing slots chỉ ảnh hưởng cách trả lời, không quyết định route cuối

Ví dụ:

- user hỏi: `có căn hộ nào gần bệnh viện không?`
- nếu knowledge base đã có thể trả shortlist theo filter này
- thì phải ưu tiên grounding trước
- chỉ hỏi lại khi retrieval yếu hoặc chưa đủ để trả lời hữu ích

### 2.4. Giai đoạn này ưu tiên manual test trực tiếp

Trong 3 sprint đầu:

- **không mở rộng thêm auto test**
- **không lấy auto test làm tiêu chí chính để chốt behavior**
- validation sẽ dựa trên:
  - manual test script
  - interactive chat test
  - log trace
  - checklist case thực tế

Auto test hiện có có thể giữ lại để tham khảo, nhưng **không phải trọng tâm phát triển của giai đoạn này**.

---

## 3. Phạm vi của 3 sprint đầu

### Sprint 1
Sửa **runtime orchestration** để flow chạy đúng theo kiến trúc 2 routes đã chốt.

### Sprint 2
Sửa **router prompt + decider output schema** để route theo `query_type` thay vì tư duy "thiếu clarity thì hỏi thêm".

### Sprint 3
Làm **retrieval schema giàu hơn cho POI / proximity** và hỗ trợ query matching thực tế.

---

# Sprint 1 — Sửa runtime orchestration

## 1.1. Mục tiêu

Làm cho runtime thực sự bám theo flow:

```text
Input
-> analyze turn
-> consult_discovery hoặc project_grounded
-> nếu consult_discovery và should_route_project=true
   thì chain tiếp sang project_grounded trong cùng request
-> merge state
-> return final response
```

Mục tiêu của sprint này là sửa **luồng chạy**, chưa tập trung tối ưu retrieval.

---

## 1.2. Vấn đề hiện tại

Hiện runtime đang có xu hướng:

- nếu `analysis.route == consult_discovery`
- thì return luôn
- `routing_signal.should_route_project` chỉ tồn tại ở payload
- chưa trở thành cơ chế điều phối thực sự

Hệ quả:

- consult route không thể tự chuyển sang project route trong cùng request
- runtime bị lệch so với flow kiến trúc đã thống nhất
- user hỏi matching query nhưng vẫn có thể bị giữ lại ở consult

---

## 1.3. Kết quả mong muốn sau sprint 1

Sau sprint 1, hệ thống phải làm được:

1. `consult_discovery` có thể chạy như một **phase 1**
2. sau consult, state được merge ngay
3. nếu `should_route_project == true` thì chạy tiếp `project_grounded`
4. final response có thể là `project_grounded` dù start route là consult
5. log phải thể hiện rõ:
   - `start_route`
   - `final_route`
   - `chained_from_consult`
   - `decision_reason`

---

## 1.4. Việc cần làm

### A. Tách rõ các phase runtime

Refactor `query()` trong orchestrator thành 3 phase rõ ràng:

#### Phase 1 — Analyze
- đọc `message`
- đọc `lead_state`
- đọc `recent_history`
- gọi decider để sinh plan ban đầu

#### Phase 2 — Execute
- nếu start route là `consult_discovery`
  - chạy consult
  - merge state
  - kiểm tra `should_route_project`
  - nếu true thì chạy tiếp `project_grounded`
- nếu start route là `project_grounded`
  - chạy project route luôn

#### Phase 3 — Finalize
- merge state cuối cùng
- build response
- attach trace info vào log

---

### B. Thêm `decision_trace`

Mỗi request cần có trace nội bộ tối thiểu:

```json
{
  "query_type": "...",
  "retrieval_readiness": "...",
  "start_route": "consult_discovery",
  "final_route": "project_grounded",
  "chained_from_consult": true,
  "route_source": "llm_decider",
  "decision_reason": "..."
}
```

Trace này không nhất thiết trả toàn bộ ra ngoài API, nhưng phải log được đầy đủ để team debug manual.

---

### C. Merge state theo 2 bước nếu chain

Nếu consult route chain sang project route:

1. merge state sau consult
2. dùng state mới đó để build retrieval intent
3. merge state lần nữa sau project grounded

Mục tiêu:
- retrieval intent luôn dùng state mới nhất
- project route tận dụng được `need/painpoint` vừa được làm rõ trong consult

---

### D. Giữ `force_route` chỉ như công cụ debug

`force_route` vẫn có thể giữ lại cho manual test, nhưng:

- không được dùng để che bug route
- không được xem là behavior chuẩn của runtime

---

## 1.5. File dự kiến sửa

- `orchestrator_service/app.py`
- `orchestrator_service/schemas.py`

---

## 1.6. Manual test cho sprint 1

### Cách test
Dùng `continuous_system_inference_test.py` ở chế độ interactive.

### Checklist test trực tiếp

#### Case 1
```text
Mình muốn được tư vấn, mình muốn mua với mục đích để đầu tư thì nên lựa chọn như thế nào là hợp lý
```
Kỳ vọng:
- start route: consult_discovery
- final route: consult_discovery hoặc chain sang project nếu history đủ mạnh
- reply mang tính định hướng, không hỏi kiểu form

#### Case 2
```text
Có căn hộ nào gần bệnh viện không
```
Kỳ vọng:
- không bị consult return luôn theo kiểu hỏi ngân sách/khu vực máy móc
- nếu decider thấy đủ thì đi project_grounded ngay
- hoặc consult ngắn rồi chain project trong cùng request

#### Case 3
```text
Ừ mình thiên về an toàn hơn
```
Kỳ vọng:
- nếu trước đó context đang nói về đầu tư thì query này phải dùng được history
- không được xử lý như câu độc lập vô nghĩa

### Tiêu chí pass sprint 1
- runtime có chain consult -> project được
- log thể hiện rõ start/final route
- manual test không còn cảm giác flow bị gãy giữa consult và project

---

# Sprint 2 — Sửa router prompt theo `query_type`

## 2.1. Mục tiêu

Đổi tư duy router từ:

> "chưa đủ clarity thì consult"

sang:

> "đây là loại query gì, và với loại query đó thì retrieval đã sẵn sàng chưa?"

Sprint này tập trung vào **prompt decider + schema output của decider**.

---

## 2.2. Vấn đề hiện tại

Prompt decider hiện thiên về:

- làm rõ thêm
- giữ consult nếu thấy thiếu context
- tránh retrieval quá sớm

Hệ quả:

- query matching như `gần bệnh viện`, `gần trường học` dễ bị xem là chưa rõ
- model ưu tiên hỏi thêm thay vì thử grounding

---

## 2.3. Kết quả mong muốn sau sprint 2

Decider phải trả được ít nhất 4 trường cốt lõi:

```json
{
  "query_type": "advisory_strategy | project_matching | project_specific | clarification",
  "retrieval_readiness": "not_ready | soft_ready | ready",
  "start_route": "consult_discovery | project_grounded",
  "decision_reason": "..."
}
```

Mục tiêu là để runtime hiểu được:
- đây là query chiến lược
- hay query matching dự án
- hay query hỏi thẳng vào một dự án cụ thể

---

## 2.4. Việc cần làm

### A. Thiết kế lại output schema của decider

Output mới nên gồm:

```json
{
  "query_type": "advisory_strategy | project_matching | project_specific | clarification",
  "retrieval_readiness": "not_ready | soft_ready | ready",
  "start_route": "consult_discovery | project_grounded",
  "should_route_project": true,
  "project_query_hint": "string | null",
  "decision_reason": "string",
  "need_update": {
    "summary_delta": "string",
    "topics": [{"label": "string", "weight": 0.0}],
    "evidence": ["string"]
  },
  "painpoint_update": {
    "summary_delta": "string",
    "topics": [{"label": "string", "weight": 0.0}],
    "evidence": ["string"]
  },
  "consult_reply": "string"
}
```

---

### B. Viết lại prompt giải thích rõ từng query type

#### `advisory_strategy`
Ví dụ:
- nên đầu tư như thế nào
- nên chọn thế nào là hợp lý
- nên bắt đầu từ đâu

#### `project_matching`
Ví dụ:
- căn nào gần bệnh viện
- dự án nào gần trường học
- dự án nào hợp gia đình có con nhỏ
- căn nào phù hợp để đầu tư an toàn

#### `project_specific`
Ví dụ:
- dự án A pháp lý sao
- dự án B có gì nổi bật
- Noble Palace Tây Thăng Long phù hợp ai

#### `clarification`
Ví dụ:
- câu rất ngắn
- phụ thuộc nặng vào turn trước
- không đủ ngữ cảnh nếu đứng một mình

---

### C. Định nghĩa rõ `retrieval_readiness`

#### `ready`
Đã đủ để gọi retrieval và trả grounded answer ngay.

#### `soft_ready`
Có thể retrieval trước; nếu evidence yếu thì chỉ hỏi lại 1 câu focused.

#### `not_ready`
Chưa nên retrieval; cần consult trước.

---

### D. Thêm few-shot examples cho decider

Bắt buộc có ít nhất các ví dụ sau trong prompt hoặc tài liệu kèm prompt:

1. `Mua để đầu tư thì nên chọn như thế nào?`
   - query_type: `advisory_strategy`
   - start_route: `consult_discovery`

2. `Có căn hộ nào gần bệnh viện không?`
   - query_type: `project_matching`
   - start_route: `project_grounded` hoặc `consult -> project`

3. `Dự án A pháp lý sao?`
   - query_type: `project_specific`
   - start_route: `project_grounded`

4. `Ừ mình thiên về an toàn hơn`
   - query_type: `clarification`
   - phụ thuộc history

---

### E. Giảm tính "slot-filling" trong consult reply

`consult_reply` phải tuân thủ:

- không hỏi checklist
- không hỏi lặp lại budget / khu vực / mục đích nếu user vừa nói rồi
- không tự động dùng form-style response khi query thực chất là project matching
- nếu user hỏi filter dự án rõ, consult chỉ nên đóng vai trò bridge ngắn, không phải route cuối

---

### F. Sửa fallback strategy

Trong sprint này, fallback không nên tự đoán route quá mạnh.

Hướng đề xuất:
- fallback chỉ dùng khi decider fail thật
- fallback phải log `route_source=fallback`
- manual test cần đo riêng tần suất rơi vào fallback

---

## 2.5. File dự kiến sửa

- `orchestrator_service/app.py`
- prompt builder trong orchestrator
- cấu trúc parse output của decider

---

## 2.6. Manual test cho sprint 2

### Checklist test trực tiếp

#### Nhóm A — strategy
```text
Mình muốn mua để đầu tư thì nên lựa chọn như thế nào là hợp lý
```
Kỳ vọng:
- query_type = advisory_strategy
- start_route = consult_discovery
- reply có framework ngắn, không ép project ngay

#### Nhóm B — matching
```text
Có căn hộ nào gần bệnh viện không
```
Kỳ vọng:
- query_type = project_matching
- không còn bị xem như thiếu clarity đơn thuần

```text
Có căn nào gần trường học và bệnh viện không
```
Kỳ vọng:
- query_type = project_matching
- start_route ưu tiên project_grounded

#### Nhóm C — specific
```text
Dự án Noble Palace Tây Thăng Long có gì nổi bật
```
Kỳ vọng:
- query_type = project_specific
- project_grounded

### Tiêu chí pass sprint 2
- decider output ổn định hơn
- query matching không còn thường xuyên bị consult nuốt mất
- reply consult không hỏi kiểu form nếu query thuộc nhóm project matching

---

# Sprint 3 — Làm retrieval schema giàu hơn cho POI / proximity

## 3.1. Mục tiêu

Làm cho retrieval không chỉ semantic chung chung, mà hiểu và phục vụ được các query như:

- gần bệnh viện
- gần trường học
- gần công viên
- thuận tiện cho gia đình có con nhỏ
- tiện sinh hoạt hằng ngày

Sprint này tập trung vào **schema ingest + retrieval payload**.

---

## 3.2. Vấn đề hiện tại

Hiện retrieval mới mạnh ở các nhóm như:
- price
- legal
- amenities
- bank
- timeline
- location

Nhưng chưa có lớp dữ liệu rõ cho:
- POI
- proximity
- school/hospital access
- family-oriented environment
- daily convenience

Điều đó khiến query matching theo proximity chưa grounded tốt.

---

## 3.3. Kết quả mong muốn sau sprint 3

Sau sprint 3, retrieval layer phải có thể trả về payload giàu hơn, ví dụ:

```json
{
  "project_cards": [...],
  "trait_tags": [...],
  "proximity_facts": [...],
  "evidence_chunks": [...],
  "confidence": 0.82,
  "low_confidence": false
}
```

Để `project_grounded` có thể:
- shortlist dự án
- giải thích lý do phù hợp
- chỉ ra bằng chứng liên quan tới bệnh viện/trường học/POI

---

## 3.4. Việc cần làm

### A. Thêm loại object mới: `proximity_fact`

Schema gợi ý:

```json
{
  "project_id": "...",
  "poi_type": "hospital | school | park | mall",
  "poi_name": "...",
  "proximity_text": "...",
  "distance_text": "...",
  "travel_mode": "walk | drive | unspecified",
  "evidence_source": "...",
  "semantic_tags": ["near_hospital", "family_friendly"]
}
```

Không bắt buộc phải có distance số. Nếu tài liệu chỉ ghi:
- gần bệnh viện X
- gần trường Y
thì vẫn đủ để tạo semantic proximity fact.

---

### B. Làm giàu `project_card`

`project_card` sau sprint 3 nên có thêm:

- `area`
- `product_types`
- `fit_personas`
- `family_fit_score`
- `investor_fit_score`
- `proximity_tags`
- `key_pois`

Mục tiêu là để retrieval layer trả được thứ gần với tư duy tư vấn hơn, không chỉ raw chunks.

---

### C. Mở rộng topic taxonomy

Cần thêm các nhóm mới:

- `hospital_access`
- `school_access`
- `medical_access`
- `education_access`
- `family_living`
- `green_space`
- `daily_convenience`
- `commute`

Các topic này dùng cho:
- enrich metadata
- trait tags
- retrieval planner
- advisory synthesis

---

### D. Bổ sung extractor trong offline ingest

Trong pipeline ingest:

1. parse docs dự án
2. detect project_id
3. detect POI mentions
4. classify POI type
5. tạo `proximity_fact`
6. ghi thêm metadata vào `evidence_chunk`
7. enrich `project_card`

Nếu chưa thể làm NLP extractor tốt ngay, sprint 3 có thể bắt đầu từ mức đơn giản:
- regex / heuristic mềm trong offline pipeline
- rồi refine ở sprint sau

---

### E. Chuyển retrieval sang multi-pass

Thay vì chỉ:
- embed query
- lấy chunks gần nghĩa

hãy làm 3 pass logic:

#### Pass 1 — candidate projects
Lấy `project_card` candidates phù hợp query.

#### Pass 2 — proximity facts
Lấy `proximity_fact` của các candidate đó.

#### Pass 3 — evidence chunks
Lấy bằng chứng text để support answer.

Mục tiêu:
- answer grounded hơn
- không phải nhồi tất cả hy vọng vào top semantic chunks

---

### F. Nâng retrieval intent

`build_retrieval_intent` cần tiến tới dạng structured hơn, ví dụ:

```json
{
  "goal": "project_matching",
  "semantic_focus": ["hospital_access", "apartment"],
  "persona_hint": ["family_friendly"],
  "poi_types": ["hospital"],
  "filters": {}
}
```

Sprint 3 chưa cần full planner hoàn hảo, nhưng phải bắt đầu tách intent ra khỏi chỉ một string text dài.

---

## 3.5. File dự kiến sửa

- `retrieval_service/pipeline.py`
- `retrieval_service/schemas.py`
- script ingest hiện có
- hoặc thêm script ingest entity/proximity mới trong `scripts/`

---

## 3.6. Manual test cho sprint 3

### Checklist test trực tiếp

#### Case 1 — hospital proximity
```text
Có căn hộ nào gần bệnh viện không
```
Kỳ vọng:
- route cuối là project_grounded
- payload grounded có ít nhất 1 candidate project hoặc proximity evidence
- reply không chỉ nói chung chung về tiện ích

#### Case 2 — school + hospital
```text
Có những căn hộ nào gần khu vực trường học và bệnh viện?
```
Kỳ vọng:
- shortlist grounded tốt hơn case 1
- nếu evidence còn yếu, câu clarify phải focused, không hỏi form

#### Case 3 — family lens
```text
Nhà mình có con nhỏ, muốn ưu tiên môi trường sống ổn và gần trường học
```
Kỳ vọng:
- retrieval tận dụng được family-oriented signals
- grounded answer ưu tiên family fit + school access

### Tiêu chí pass sprint 3
- retrieval trả được dữ liệu hữu ích cho query POI/proximity
- project grounded reply có lý do + bằng chứng tốt hơn
- query hospital/school không còn bị trả lời kiểu semantic quá chung

---

# 4. Cách tổ chức manual test trong 3 sprint đầu

## 4.1. Công cụ test chính

Dùng:

- `continuous_system_inference_test.py` ở chế độ interactive
- log runtime của orchestrator
- log retrieval service
- JSONL report để review thủ công

## 4.2. Không phát triển thêm auto test ở giai đoạn này

Trong 3 sprint đầu:

- không thêm unit test mới như tiêu chí chính
- không thêm e2e auto test mới như tiêu chí bắt buộc
- team sẽ review behavior bằng manual test checklist
- nếu cần giữ test cũ thì chỉ giữ để tham khảo, không mở rộng tiếp

## 4.3. Bộ manual test tối thiểu phải chạy mỗi cuối sprint

### Nhóm 1 — advisory
- mua để đầu tư nên chọn thế nào
- mua để ở nên bắt đầu từ đâu

### Nhóm 2 — project matching
- có căn nào gần bệnh viện không
- có căn nào gần trường học không
- có căn nào phù hợp gia đình có con nhỏ không

### Nhóm 3 — project specific
- dự án X pháp lý sao
- dự án Y có gì nổi bật

### Nhóm 4 — context continuation
- ừ mình thiên về an toàn hơn
- khu vực đó có hợp với gia đình không

---

# 5. Phân công đề xuất

## Backend orchestration owner
Phụ trách:
- sprint 1
- phần runtime chaining
- decision trace
- decider output schema

## Prompt / LLM routing owner
Phụ trách:
- sprint 2
- viết lại prompt decider
- thiết kế examples cho query_type
- review consult reply behavior

## Retrieval / Data owner
Phụ trách:
- sprint 3
- proximity schema
- offline ingest enrich
- retrieval payload mới

## QA manual owner
Phụ trách:
- chạy manual checklist
- tổng hợp log sai route
- tổng hợp response fail cases
- cập nhật danh sách test prompt thực tế

---

# 6. Definition of Done theo từng sprint

## Sprint 1 Done
- runtime có chain consult -> project trong cùng request
- log hiển thị được start_route/final_route
- manual test thấy flow không còn gãy giữa consult và project

## Sprint 2 Done
- decider output có `query_type` và `retrieval_readiness`
- query matching không còn thường xuyên bị consult nuốt
- consult reply ít form-like hơn rõ rệt

## Sprint 3 Done
- retrieval có schema proximity cơ bản
- query hospital/school trả grounded answer khá hơn rõ rệt
- project grounded payload có thể giải thích theo POI / proximity

---

# 7. Gợi ý thứ tự triển khai thực tế

## Tuần 1
- bắt đầu sprint 1
- sửa runtime orchestration
- thêm decision trace
- chạy manual test 4 case cơ bản

## Tuần 2
- bắt đầu sprint 2
- sửa decider prompt
- sửa output schema
- chạy lại manual test cùng 4 case + thêm continuation cases

## Tuần 3
- bắt đầu sprint 3
- enrich retrieval schema
- ingest proximity facts
- test trực tiếp với hospital/school/family queries

---

# 8. Kết luận

Trong 3 sprint đầu, team không nên cố tối ưu mọi thứ cùng lúc.

Thứ tự đúng là:

1. **sửa runtime cho đúng flow**
2. **sửa router cho đúng query type**
3. **sửa retrieval để answer được POI / proximity**

Và trong giai đoạn này:

> **ưu tiên manual test trực tiếp hơn auto test**

vì vấn đề hiện tại là vấn đề behavior, route quality và grounded quality trong hội thoại thực tế, không phải chỉ là contract correctness.
