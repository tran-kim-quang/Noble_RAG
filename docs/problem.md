Đúng, mình thấy 7 vấn đề bạn nêu đều phản ánh rất sát code hiện tại. Tin tốt là branch mới đã có khung LangGraph khá ổn: `app.py` đã mount cả `query_router` và `sales_router`, còn `sales/graph.py` đã tách flow thành các node `ingest_user_turn → classify_and_extract → update_lead_profile → resolve_sales_state → retrieve_context/build_response → persist_turn`.  

Nhưng hiện tại hệ vẫn còn “sales-flavored RAG assistant” nhiều hơn là “sales agent bị ràng buộc chặt”. Gốc vấn đề nằm ở 4 điểm:

* Prompt vẫn cho phép agent “tư vấn sơ bộ” ngay cả trong `need_discovery`. 
* State machine còn quá mềm: chỉ cần hết `missing_slots` là có thể rơi vào `product_matching`, và nếu intent không rõ thì default vẫn là `product_matching`. 
* Retrieval chưa dùng budget/property_type dù classifier đã trích xuất được, và `retrieve_context` chỉ đổ một khối text từ `query_rag` vào prompt, chưa có cơ chế “must cite retrieved facts only”.  
* Hai endpoint `/query/stream` và `/sales/chat` đang gọi cùng một graph nhưng khác contract, trong khi `/query/stream` còn hardcode opening chunk.  

Mình sẽ gom thành 3 nhóm hành động như bạn đề xuất, nhưng mình thêm một nhóm thứ 4 là “context grounding”, vì đây là phần quyết định agent có thực sự là sales agent an toàn hay không.

## 1. Siết prompt

### Vấn đề

Trong `need_discovery`, prompt hiện ghi: “Nếu đã có thể tư vấn sơ bộ từ dữ liệu dự án, hãy tư vấn ngắn trước rồi chỉ hỏi thêm đúng phần còn thiếu.” Điều này chính là lỗ khiến model có thể tự suy diễn hoặc nhảy sang tư vấn khi chưa đủ điều kiện. 

System prompt cũng đang nói “Nếu khách hỏi về dự án, phải ưu tiên tư vấn dự án trước rồi mới xin thêm thông tin còn thiếu”, nên nó vô tình đẩy model sang mode trả lời sớm. 

### Cách sửa

Trong `SYSTEM_PROMPT`, thay logic “ưu tiên tư vấn dự án trước” bằng:

```text
Nếu chưa đủ điều kiện pitch theo state machine, không được tư vấn dự án cụ thể.
Chỉ được:
- xác nhận nhu cầu đã hiểu
- hỏi thêm đúng slot còn thiếu
- tóm tắt bước tiếp theo
```

Trong `_STATE_PROMPTS["need_discovery"]`, đổi thành:

```text
TRẠNG THÁI: NEED_DISCOVERY
NHIỆM VỤ
- Chỉ hỏi để lấy các slot còn thiếu.
- Không tư vấn dự án, không nêu ví dụ dự án, không nêu ví dụ khu vực.
- Không suy diễn địa điểm, loại hình, ngân sách hoặc chân dung gia đình nếu khách chưa nói.
- Mỗi lượt tối đa 1 câu hỏi chính; tối đa 1 câu phụ nếu thật sự cần.
- Nếu câu khách rất ngắn hoặc chỉ là chào hỏi, chỉ hỏi 1 câu duy nhất.
```

Với `greeting`, prompt hiện khá chung và chưa ép agent mở đầu ngắn. 
Nên sửa thành:

```text
TRẠNG THÁI: GREETING
NHIỆM VỤ
- Chào khách tự nhiên trong 1 câu ngắn.
- Sau đó hỏi đúng 1 câu mở để khách nói nhu cầu.
- Không hỏi nhiều ý cùng lúc.
- Không liệt kê tiêu chí, không pitch, không xin quá nhiều thông tin ở lượt đầu.
```

### Nên thêm output contract mềm trong prompt

Ở cuối `build_prompt`, thêm yêu cầu:

```text
RÀNG BUỘC HÌNH THỨC:
- Nếu state là greeting: trả lời tối đa 2 câu, chỉ có 1 câu hỏi.
- Nếu state là need_discovery: trả lời tối đa 2 câu, chỉ hỏi về các slot đang thiếu.
- Nếu state là product_matching/comparison/objection_handling/closing_next_step: chỉ dùng thông tin có trong NGỮ CẢNH TỪ KHO TÀI LIỆU.
- Nếu không có retrieved context, không được nêu tên dự án hay dữ kiện dự án.
```

Điểm quan trọng nhất: bỏ hoàn toàn mọi wording kiểu “tư vấn sơ bộ trước” ở `need_discovery`. Đó là chỗ đang làm state mềm đi.

---

## 2. Siết state machine

### Vấn đề

`resolve_next_state` hiện quá permissive. Nếu không rơi vào các nhánh intent đặc biệt, cuối cùng nó trả về `product_matching`. Điều này làm agent có xu hướng pitch sớm. 

Ngoài ra, “required slots for pitch” hiện được tính ở `update_lead_profile`, nhưng từ prompt mình thấy hệ đang coi 4 slot lõi là:

* family_member_count
* children_count
* purpose
* location_preference
  thay vì các slot sale thực dụng hơn như budget, purpose, property_type, location.  

### Cách sửa

Đổi rule cứng trong `resolve_next_state` như sau:

```python
def resolve_next_state(state):
    intent = state.get("detected_intent") or ""
    missing = state.get("missing_slots") or []
    buy_signal = bool(state.get("buy_signal", False))
    current_state = state.get("current_sales_state") or "greeting"
    retrieved_context = state.get("retrieved_context") or []
    lead = state.get("lead_profile") or {}

    if intent == "out_of_scope":
        return "out_of_scope"

    # greeting chỉ dành cho lượt rất đầu hoặc greeting thực sự
    if current_state == "greeting" and intent in ("greeting", "", "other"):
        return "greeting"

    # objection/comparison/buy_signal có ưu tiên cao hơn
    if intent == "comparison":
        return "comparison"
    if intent == "objection":
        return "objection_handling"
    if buy_signal or intent == "buy_signal":
        return "closing_next_step"

    # chưa đủ slot => không được pitch
    if missing:
        return "need_discovery"

    # đủ slot nhưng chưa có context => vẫn không pitch
    if intent == "ask_recommendation" and not retrieved_context:
        return "product_matching"

    # follow-up thì tiếp state cũ nếu hợp lệ
    if intent == "follow_up":
        return current_state if current_state != "greeting" else "need_discovery"

    # fallback an toàn
    return "need_discovery"
```

### Đổi tập slot bắt buộc

Mình khuyên slot tối thiểu để pitch là:

* `purpose`
* `location_preference`
* `property_type`
* một trong `budget_text / budget_min / budget_max`

`family_member_count` và `children_count` là slot tăng chất lượng, không nên là slot cốt lõi để unlock pitch. Hiện logic lead temperature đang dựa quá nhiều vào family/children/location/purpose. 
Điều này không sát sales bằng budget + purpose + property_type + location.

### Thêm “response policy by state” trong code

Ngoài prompt, thêm một lớp code để ép số câu hỏi:

* `greeting`: tối đa 1 câu hỏi
* `need_discovery`: tối đa 2 câu, chỉ 1 câu hỏi chính
* `product_matching`: không được hỏi slot mới nếu đã đủ slot
* `objection_handling`: không hỏi lan man ngoài objection hiện tại

Cách nhanh nhất là post-process ở `build_response`:

* đếm dấu `?`
* nếu state = `greeting` mà có >1 dấu hỏi, trim lại hoặc regenerate bằng prompt ngắn hơn
* nếu state = `need_discovery` mà phát hiện tên địa danh không nằm trong profile/context, reject và regenerate

---

## 3. Siết context grounding

### Vấn đề

`retrieve_context` hiện build query theo `family_member_count`, `children_count`, `purpose`, `location`, `objection_type`, `user_text`, nhưng bỏ qua `budget` và `property_type` dù classifier đã extract được các field này.  

Nặng hơn, `retrieve_context` chỉ trả về một item dạng `{"content": raw_answer, "source": "lightrag"}`. Tức là bạn đang retrieve thành một khối answer đã được tổng hợp sẵn, rồi LLM lại tổng hợp lần hai trong `build_response`. Đây là double-generation, dễ drift khỏi context.  

### Cách sửa

Có 3 mức:

#### Mức 1: sửa query template ngay

Thêm `budget` và `property_type` vào template `product_matching`:

```python
"product_matching": (
    "Tư vấn dự án Noble phù hợp với mục đích {purpose}, "
    "loại hình {property_type}, ngân sách {budget}, "
    "ưu tiên khu vực {location}, gia đình {family_size} người, "
    "có {children_count} con nhỏ. "
    "Chỉ nêu phương án phù hợp với thông tin này."
)
```

#### Mức 2: thêm cờ `has_retrieved_context`

Trong `build_prompt`, nếu state là `product_matching`, `comparison`, `objection_handling`, `closing_next_step` mà `retrieved_context` rỗng, ép trả lời như:

* “Em chưa có đủ dữ liệu từ kho dự án để đề xuất chính xác.”
* rồi chuyển sang CTA hoặc xin thông tin bổ sung

Tức là không cho model được nói về dự án nếu không có context.

#### Mức 3: đổi contract response thành structured

Hiện `build_response` gọi model trả text free-form. 
Nên chuyển sang JSON schema tối thiểu:

```json
{
  "response_text": "...",
  "used_context": true,
  "mentioned_projects": [],
  "asked_slots": [],
  "next_action": "need_discovery|recommend|compare|close"
}
```

Sau đó validate:

* nếu `state in {"product_matching","comparison","objection_handling"}` và `used_context == false` thì reject
* nếu `mentioned_projects` có giá trị nhưng `retrieved_context` rỗng thì reject
* nếu `asked_slots` chứa slot không nằm trong `missing_slots` thì reject

Đây là lớp “guardrail thật”, thay vì tin prompt.

---

## 4. Siết `/query/stream`, `/sales/chat`, và client payload

### Vấn đề 1: opening chunk cố định

`/query/stream` luôn stream trước chunk `*(Em đang phân tích nhu cầu của Anh/Chị...)*`, bất kể state nào. 
Điều này làm stream không thật “state-aware”.

### Cách sửa

Hoặc bỏ opening chunk hoàn toàn, hoặc sinh opening chunk theo state sau khi chạy một pre-pass rất nhẹ.

Cách đơn giản:

* không yield gì trước khi có result
* hoặc yield typing-neutral như `"..."`

Cách tốt hơn:

* `sales_graph.astream()` để stream theo node
* sau node `resolve_sales_state`, mới yield opening chunk theo state:

Ví dụ:

* `greeting`: `"*(Em đang chuẩn bị câu chào phù hợp...)*"`
* `need_discovery`: `"*(Em đang rà lại thông tin còn thiếu để hỏi gọn nhất...)*"`
* `product_matching`: `"*(Em đang đối chiếu hồ sơ với dữ liệu dự án...)*"`

Hiện `/query/stream` đang dùng `ainvoke`, nên chưa tận dụng được graph stream. 

### Vấn đề 2: `/sales/chat` và `/query/stream` chồng vai

Hai endpoint này cùng gọi một graph nhưng khác payload:

* `/sales/chat` yêu cầu `session_id` và `message`
* `/query/stream` nhận `query` hoặc `message`, `session_id` optional.   

Điều này dễ gây lệch client.

### Cách sửa

Chuẩn hóa một contract chung:

* `message: str`
* `session_id: Optional[str]`
* `stream: bool = false`

Sau đó:

* `/sales/chat` là endpoint chính
* `/query/stream` chỉ là alias backward-compatible, hoặc bỏ hẳn sau khi client migrate

Nếu giữ cả hai:

* dùng chung một internal function như `run_sales_graph(request)`
* dùng chung cùng schema request

### Vấn đề 3: 422 và 400

`QueryRequest` cho `query: str = ""`, `message: Optional[str] = None`, nên 422 chủ yếu đến từ client gửi sai `Content-Type`, sai shape JSON, hoặc field types không match; còn 400 đến từ trường hợp cả `query` lẫn `message` rỗng sau strip.  

`SalesChatRequest` lại bắt buộc `session_id` và `message`, nên client nào còn gọi theo contract cũ sẽ dễ 422.  

### Cách sửa

Cho `SalesChatRequest` thành:

```python
class SalesChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    query: Optional[str] = None
    raw_transcript: Optional[str] = None
```

và trong route:

* `user_text = (request.message or request.query or "").strip()`
* nếu không có `session_id` thì tự sinh
* log warning cho payload legacy

Điều này sẽ giảm cả 422 lẫn 400 từ client cũ.

---

## Ưu tiên sửa theo thứ tự

### Nhóm A — sửa ngay, ít rủi ro

1. Bỏ câu “tư vấn sơ bộ” trong `need_discovery`. 
2. Siết `greeting` chỉ còn 1 câu hỏi. 
3. Thêm `budget` và `property_type` vào retrieval query.  
4. Đổi fallback cuối của state machine từ `product_matching` sang `need_discovery`. 
5. Chuẩn hóa payload `message/query/session_id`.   

### Nhóm B — tăng độ cứng

1. Structured output cho `build_response`. 
2. Validator theo state sau khi model trả lời.
3. Chặn mention dự án/địa danh nếu không có retrieved context.
4. State-aware stream thay vì opening chunk cố định. 

### Nhóm C — kiến trúc sạch hơn

1. Gộp `/query/stream` thành wrapper của `/sales/chat?stream=true`.
2. Chuyển sang `sales_graph.astream()` để stream theo node.
3. Sau này tách `query_router` thành backward compatibility בלבד, còn sales dùng một router chuẩn.

---

## Chẩn đoán ngắn theo từng issue của bạn

* “model còn suy diễn ở need_discovery”: đúng, vì prompt vẫn cho phép “tư vấn sơ bộ”. 
* “greeting hỏi hơi nhiều”: đúng, vì state prompt chưa ép 1 câu hỏi và không có code validator. 
* “stream chunk mở đầu chưa thông minh”: đúng, vì đang hardcode 1 opening chunk trước khi graph chạy. 
* “422/400 trong log”: đúng, vì `/sales/chat` và `/query/stream` đang dùng 2 contract khác nhau, trong đó `/sales/chat` còn bắt buộc `session_id`.  
* “phụ thuộc nhiều vào prompt”: đúng, vì state machine hiện chỉ xác định state, chưa có response guardrails hay validator.  
* “tư vấn dự án chưa bám tuyệt đối context retrieve”: đúng, vì retrieval đang trả một khối answer tổng hợp rồi model tự diễn giải thêm. 
* “/sales/chat và /query/stream chồng vai”: đúng, cả hai cùng gọi `sales_graph.ainvoke()` nhưng khác payload và khác mode trả về.  

## Kết luận

Code mới của bạn đã đi đúng hướng hơn rất nhiều: đây đã là LangGraph sales flow thật sự, không còn chỉ là RAG chatbot. Nhưng để thành “sales agent stateful chặt”, mình sẽ ưu tiên như sau:

* trước hết siết prompt để cấm hoàn toàn tư vấn trong `need_discovery`
* sau đó siết state machine để default an toàn là `need_discovery`, không phải `product_matching`
* tiếp theo thêm validator/output schema để mọi tư vấn dự án phải chứng minh là dựa trên retrieved context
* cuối cùng chuẩn hóa payload và hợp nhất contract giữa `/sales/chat` với `/query/stream`

Mình có thể làm tiếp bước cụ thể nhất là viết cho bạn một bản patch đề xuất cho 4 file:
`prompt_builder.py`, `state_machine.py`, `retrieve_context.py`, `api_models.py`.
