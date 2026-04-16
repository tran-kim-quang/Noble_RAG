# Hướng dẫn update hệ thống theo `query_type` + `engagement_state` + `response_mode`

## 1. Đánh giá nhanh nhánh `test-screen-agents`

### 1.1. Điểm đã đúng hướng

Nhánh hiện tại đã đi đúng hướng ở phần kiến trúc nền:

- Đã có `query_type` để phân biệt khách đang hỏi kiểu gì: `advisory_strategy`, `project_matching`, `project_specific`, `clarification`.
- Đã có `retrieval_readiness` để tách câu nào nên grounding ngay, câu nào chưa cần.
- Đã có 2 routes `consult_discovery` và `project_grounded`.
- Đã có cơ chế chain `consult -> project` trong cùng request.
- Đã có `response_mode` và `ask_policy` để bắt đầu kiểm soát cách trả lời.

Đây là nền tảng đúng để phát triển skill sales tiếp.

### 1.2. Điểm chưa đủ cho skill sales

Hiện tại hệ vẫn chưa đủ "phong thái sales tư vấn" vì:

- `LeadState` mới chỉ có `name`, `phone_contact`, `need`, `painpoint`, chưa có lớp thể hiện mức độ mở lòng / hứng thú / readiness của khách.
- `DecisionTrace` chưa ghi được trạng thái sales hoặc strategy của lượt trả lời.
- `build_reply_plan()` hiện đang chọn `response_mode` chủ yếu theo grounded result, low confidence và query type, chưa dựa trên mức độ hứng thú của khách.
- `_RESPONSE_MODES` hiện còn quá ít: `inform_only`, `recommendation`, `clarify_light`, `meeting_invite`.
- Prompt synthesis vẫn thiên về "an toàn, ngắn, grounded" hơn là "niềm nở, khơi mở, tạo tò mò".

## Kết luận ngắn

Nhánh `test-screen-agents` **đã đúng hướng về route và grounding**, nhưng **chưa hoàn thiện skill sales/conversation psychology**.

---

## 2. Tư duy kiến trúc mới

Sau lần update này, hệ cần được hiểu theo 3 lớp:

### `query_type = khách đang hỏi gì`
Dùng để quyết định route và retrieval strategy.

### `engagement_state = khách đang mở lòng tới đâu`
Dùng để quyết định khách đang lạnh, đang ấm lên hay đã hứng thú đủ để đi bước tiếp.

### `response_mode = lượt này nên nói kiểu gì`
Dùng để quyết định giọng phản hồi và mục tiêu hội thoại ở đúng thời điểm.

## Nguyên tắc quan trọng

- `query_type` **không thay thế** `engagement_state`.
- `engagement_state` **không thay thế** `route`.
- `response_mode` **không phải route**, mà là chiến lược nói chuyện của lượt hiện tại.

---

# 3. Lớp 1 — `query_type = khách đang hỏi gì`

Giữ nguyên 4 nhóm hiện tại:

- `advisory_strategy`
- `project_matching`
- `project_specific`
- `clarification`

## 3.1. `advisory_strategy`
Các câu kiểu:
- muốn được tư vấn
- nên đầu tư thế nào
- gia đình có con nhỏ thì nên chọn gì
- nên bắt đầu từ đâu

### Ý nghĩa
Khách đang hỏi theo hướng chiến lược, định hướng, chọn phương án.

### Route gợi ý
- ưu tiên `consult_discovery`

---

## 3.2. `project_matching`
Các câu kiểu:
- có gần bệnh viện không
- gần trường học không
- loại nào phù hợp gia đình có con nhỏ
- sản phẩm nào hợp dòng tiền đều

### Ý nghĩa
Khách đang match nhu cầu với sản phẩm/dự án.

### Route gợi ý
- ưu tiên `project_grounded`
- hoặc `consult -> project` nếu cần bridge ngắn

---

## 3.3. `project_specific`
Các câu kiểu:
- dự án có những loại hình nào
- có shophouse không
- pháp lý sao
- phân khu nào nổi bật

### Ý nghĩa
Khách đang hỏi trực tiếp vào facts của dự án.

### Route gợi ý
- `project_grounded`

---

## 3.4. `clarification`
Các câu kiểu:
- ok
- ừ
- mình ưu tiên dòng tiền đều đặn
- mình thích an toàn hơn

### Ý nghĩa
Khách đang nối tiếp ý cũ, không nên reset flow.

### Route gợi ý
- dùng history + state để xác định đang nối vào consult hay project

---

# 4. Lớp 2 — `engagement_state = khách đang mở lòng tới đâu`

Đề xuất không dùng 2 nhãn quá thô kiểu "hứng thú / chưa hứng thú", mà dùng 4 mức:

- `cold`
- `warm`
- `interested`
- `ready`

Có thể thêm `engagement_confidence` nếu muốn.

---

## 4.1. `cold`
### Dấu hiệu
- mới mở lời
- câu ngắn, chung chung
- chưa phản hồi cảm xúc rõ
- chưa cho thấy muốn đi sâu

### Ví dụ
- "xin chào, mình muốn được tư vấn"
- "ok"
- "giới thiệu qua cho mình"

### Mục tiêu hội thoại
- tạo thiện cảm
- không ép hỏi form
- không CTA mạnh
- mở 1 hook nhẹ để khách muốn nói tiếp

---

## 4.2. `warm`
### Dấu hiệu
- bắt đầu chia sẻ bối cảnh cá nhân
- nói về con nhỏ, đầu tư, an toàn, dòng tiền, ở thực
- trả lời theo hướng có hợp tác

### Ví dụ
- "gia đình mình đang có 2 con nhỏ"
- "mình muốn mua để đầu tư"
- "mình ưu tiên dòng tiền đều đặn"

### Mục tiêu hội thoại
- đào sâu đúng 1 trục quan tâm
- cho khách thấy dự án phù hợp ở điểm nào
- tiếp tục nuôi hứng thú

---

## 4.3. `interested`
### Dấu hiệu
- hỏi sâu vào sản phẩm/dự án
- hỏi loại hình, phân khu, shophouse, trường mầm non, pháp lý, giá trị khai thác
- có dấu hiệu so sánh / cân nhắc thật

### Ví dụ
- "có những loại hình sản phẩm nào"
- "dự án có shophouse không"
- "căn shophouse phù hợp mục đích nào hơn"

### Mục tiêu hội thoại
- recommendation rõ hơn
- trả grounded hơn
- gợi bước tiếp theo mềm

---

## 4.4. `ready`
### Dấu hiệu
- hỏi sâu kiểu gần ra quyết định
- muốn xem cụ thể
- hỏi so sánh phương án
- muốn xem thực tế / cần bảng tính / cần người thật hỗ trợ

### Ví dụ
- "mình muốn xem layout cụ thể"
- "giá từng loại thế nào"
- "có thể đi xem dự án khi nào"

### Mục tiêu hội thoại
- đẩy sang next step
- CTA rõ hơn nhưng vẫn mềm
- có thể handoff cho sales thật

---

# 5. Lớp 3 — `response_mode = lượt này nên nói kiểu gì`

Hiện tại code mới có 4 mode. Nên mở rộng thành:

- `warm_welcome`
- `value_teaser`
- `discover_need`
- `consultive_recommendation`
- `grounded_recommendation`
- `handle_concern`
- `soft_next_step`
- `meeting_invite`
- `nurture_followup`

---

## 5.1. `warm_welcome`
Dùng khi khách còn `cold`.

### Cách nói
- chào niềm nở
- ấm áp
- không nói brochure dài
- không hỏi form

### Mục tiêu
Làm khách muốn nói tiếp.

---

## 5.2. `value_teaser`
Dùng khi khách còn `cold` hoặc mới `warm`.

### Cách nói
- nhấn 1 góc hấp dẫn cụ thể
- không nói hết
- tạo cảm giác "dự án này có cái đáng để nghe tiếp"

### Mục tiêu
Tạo tò mò.

---

## 5.3. `discover_need`
Dùng khi khách đã `warm` nhưng need chưa rõ.

### Cách nói
- phản chiếu điều khách vừa nói
- hỏi mềm đúng 1 ý nếu cần
- không checklist

### Mục tiêu
Làm rõ thêm 1 trục quan trọng.

---

## 5.4. `consultive_recommendation`
Dùng khi need đã rõ, nhưng chưa cần facts quá cụ thể.

### Cách nói
- tư vấn như consultant
- nêu logic chọn sản phẩm
- giải thích vì sao

### Mục tiêu
Làm khách thấy được hiểu.

---

## 5.5. `grounded_recommendation`
Dùng khi `project_grounded` đã có result tốt.

### Cách nói
- grounded
- có nhận định cụ thể
- nói ngắn nhưng chắc

### Mục tiêu
Biến data thành gợi ý thật sự hữu ích.

---

## 5.6. `handle_concern`
Dùng khi khách có nghi ngại, băn khoăn hoặc đang cân nhắc.

### Cách nói
- giải tỏa nhẹ
- không tranh luận
- không push sale mạnh

### Mục tiêu
Giữ nhịp tin tưởng.

---

## 5.7. `soft_next_step`
Dùng khi khách đã `interested` hoặc gần `ready`.

### Cách nói
- mời bước tiếp theo mềm
- không ép chốt
- không lặp “đi xem cuối tuần” ở mọi lượt

### Mục tiêu
Đẩy hội thoại sang bước sâu hơn.

---

## 5.8. `meeting_invite`
Dùng khi khách `ready` hoặc cần phân tích quá sâu.

### Cách nói
- mời hẹn rõ ràng
- lý do hợp lý
- không lạm dụng ở quá sớm

### Mục tiêu
Chuyển bước hiệu quả.

---

## 5.9. `nurture_followup`
Dùng khi khách chưa đủ nóng nhưng vẫn có tín hiệu tốt.

### Cách nói
- gợi mở nhẹ
- nhấn đúng điểm phù hợp
- không pressure

### Mục tiêu
Nuôi lead.

---

# 6. Mapping chính: `engagement_state -> response_mode -> ask_policy`

## 6.1. `cold`
### response_mode ưu tiên
- `warm_welcome`
- `value_teaser`

### ask_policy
- `allow_question` hoặc `avoid_question`

### Quy tắc
- không hỏi form
- không CTA mạnh
- nếu hỏi thì chỉ 1 câu rất mềm
- nhiều trường hợp tốt hơn là **không hỏi**, chỉ mở một hook

---

## 6.2. `warm`
### response_mode ưu tiên
- `discover_need`
- `consultive_recommendation`
- `value_teaser`

### ask_policy
- `allow_question`

### Quy tắc
- có thể hỏi thêm 1 ý nếu thật sự giúp tư vấn tốt hơn
- ưu tiên phản chiếu nhu cầu trước rồi mới hỏi

---

## 6.3. `interested`
### response_mode ưu tiên
- `grounded_recommendation`
- `handle_concern`
- `soft_next_step`

### ask_policy
- `avoid_question` hoặc `allow_question`

### Quy tắc
- nếu đã có enough grounding thì không cần hỏi nữa
- có thể CTA mềm
- không vội hẹn trực tiếp ở mọi lượt

---

## 6.4. `ready`
### response_mode ưu tiên
- `soft_next_step`
- `meeting_invite`

### ask_policy
- `avoid_question`

### Quy tắc
- tập trung chốt bước tiếp theo
- không mở thêm loop hội thoại không cần thiết

---

# 7. Cách xác định `engagement_state`

Không nên chỉ dựa vào 1 message cuối. Nên dựa vào:

- message hiện tại
- recent_history
- lead_state.need
- lead_state.painpoint
- số lần user chủ động hỏi sâu
- tín hiệu hợp tác của user

## Gợi ý logic mềm

### `cold`
Nếu:
- mới vào hội thoại
- câu chung chung
- chưa thấy reveal nhu cầu rõ

### `warm`
Nếu:
- user bắt đầu reveal nhu cầu, gia đình, đầu tư, painpoint
- trả lời mang tính hợp tác

### `interested`
Nếu:
- user hỏi trực tiếp facts dự án / loại hình / sản phẩm / phân khu
- có follow-up nhiều lượt liên tiếp

### `ready`
Nếu:
- user hỏi cụ thể để ra quyết định
- muốn xem thực tế
- muốn nhận phương án / phân tích sâu / gặp người thật

---

# 8. Hướng dẫn hoàn thiện skill sales

Phần này là hướng dẫn bổ sung để agent không chỉ route đúng, mà còn có phong thái sales tư vấn đúng hơn.

## 8.1. Mục tiêu của skill sales

Agent cần đạt 4 mục tiêu:

1. **tạo thiện cảm**
2. **khám phá nhu cầu mà không hỏi cứng**
3. **cho khách thấy dự án phù hợp ở đâu**
4. **mời bước tiếp theo đúng lúc**

Nếu thiếu một trong bốn mục tiêu này, agent sẽ dễ rơi vào một trong hai thái cực:
- chỉ trả lời cho có
- hoặc hỏi dồn như form

---

## 8.2. Các năng lực sales cần bổ sung

### A. Đồng cảm / phản chiếu
Khi user nói:
- có 2 con nhỏ
- ưu tiên dòng tiền đều đặn
- thiên về an toàn

agent không nên nhảy thẳng sang facts. Trước tiên phải phản chiếu nhẹ:
- "với gia đình có 2 bé nhỏ thì tiêu chí an toàn và tiện đưa đón thường sẽ được ưu tiên hơn"
- "nếu ưu tiên dòng tiền đều đặn thì cách chọn sản phẩm sẽ khác với người thiên về tăng giá dài hạn"

### B. Hook / tạo tò mò
Mỗi lượt nên có 1 hook nhỏ, ví dụ:
- một điểm đáng chú ý
- một contrast
- một lời hứa mở ra góc nhìn tiếp theo

Ví dụ:
- "điểm hay của dự án này không chỉ là vị trí, mà là cách quy hoạch phù hợp cho nhóm khách mua ở thực"
- "nếu nhìn theo dòng tiền thì có vài loại hình đáng xem hơn so với cách nhìn thông thường"

### C. Gợi mở mềm
Không phải lúc nào cũng hỏi. Có 3 kiểu gợi mở:

- **không hỏi, chỉ mở hook**
- **hỏi mở 1 câu**
- **CTA nhẹ**

### D. Chọn đúng thời điểm CTA
CTA chỉ nên đẩy mạnh khi:
- khách đã `interested`
- hoặc `ready`

Nếu khách còn `cold/warm`, CTA quá sớm sẽ làm agent mất duyên.

---

## 8.3. Mẫu tư duy phản hồi theo skill sales

Mỗi lượt agent nên đi theo thứ tự:

1. **đón nhận tín hiệu của user**
2. **phản chiếu hoặc đồng cảm ngắn**
3. **đưa 1 nhận định hoặc hook cụ thể**
4. **chọn kết thúc phù hợp**
   - không hỏi
   - hỏi mềm
   - CTA nhẹ

### Công thức ngắn

`reflect -> insight -> invite`

Trong đó:
- `reflect`: phản chiếu đúng tín hiệu user vừa nói
- `insight`: cho khách thấy giá trị
- `invite`: mời bước tiếp theo phù hợp với engagement_state

---

## 8.4. Các lỗi sales cần tránh

### Lỗi 1: brochure-style
Ví dụ:
- "dự án nổi bật với vị trí đắc địa, tiện ích hoàn chỉnh..."

### Lỗi 2: hỏi form
Ví dụ:
- "anh/chị quan tâm khu vực nào, ngân sách bao nhiêu, thời điểm mua khi nào?"

### Lỗi 3: CTA quá sớm
Ví dụ:
- cứ 1-2 lượt là mời đi xem dự án / gặp trực tiếp

### Lỗi 4: không bám tín hiệu user
Ví dụ:
- user nói về con nhỏ nhưng agent quay lại nói chung về dự án

### Lỗi 5: thiếu consistency sản phẩm
Ví dụ:
- lúc thì nói như dự án căn hộ
- lúc thì nói như dự án shophouse

---

## 8.5. Update cần làm trong code để hoàn thiện skill sales

### A. Thêm `engagement_state` vào `LeadState`
Gợi ý:

```python
engagement_state: Literal["cold", "warm", "interested", "ready"] = "cold"
engagement_confidence: float | None = None
```

### B. Thêm vào `DecisionTrace`

```python
engagement_state_before: str | None = None
engagement_state_after: str | None = None
response_mode: str | None = None
ask_policy: str | None = None
```

### C. Mở rộng `_RESPONSE_MODES`

Thay:
- `inform_only`
- `recommendation`
- `clarify_light`
- `meeting_invite`

Bằng:
- `warm_welcome`
- `value_teaser`
- `discover_need`
- `consultive_recommendation`
- `grounded_recommendation`
- `handle_concern`
- `soft_next_step`
- `meeting_invite`
- `nurture_followup`

### D. Sửa `build_reply_plan()`
Thứ tự chọn plan nên là:

1. đọc `query_type`
2. đọc `engagement_state`
3. đọc `grounded_result`
4. suy ra `response_mode`
5. suy ra `ask_policy`

Không nên chọn mode chủ yếu theo grounded result như hiện tại.

### E. Sửa prompt synthesis
Prompt cần nhận thêm:
- `engagement_state`
- `response_mode`
- `ask_policy`

Và thêm rule:
- mỗi lượt phải có ít nhất 1 hook hoặc 1 insight mới
- không lặp brochure intro
- không CTA quá sớm nếu engagement_state chưa đủ
- nếu user vừa bộc lộ một nhu cầu/painpoint thì phải phản chiếu lại trước

---

# 9. Gợi ý pseudo-flow mới

## Phase 1 — Analyze
Xác định:
- `query_type`
- `retrieval_readiness`
- `start_route`
- `engagement_state_before`
- `engagement_state_after`

## Phase 2 — Execute
- chạy `consult_discovery` hoặc `project_grounded`
- merge state

## Phase 3 — Reply planning
Chọn:
- `response_mode`
- `ask_policy`
- `focus`
- `question_focus`

## Phase 4 — Reply synthesis
Sinh câu trả lời theo:
- khách đang hỏi gì
- khách đang mở lòng tới đâu
- lượt này nên nói kiểu gì

---

# 10. Mapping nhanh để code luôn

## `query_type = advisory_strategy`
- `cold` -> `warm_welcome` hoặc `value_teaser`
- `warm` -> `discover_need` hoặc `consultive_recommendation`
- `interested` -> `consultive_recommendation`
- `ready` -> `soft_next_step`

## `query_type = project_matching`
- `cold` -> `value_teaser`
- `warm` -> `consultive_recommendation`
- `interested` -> `grounded_recommendation`
- `ready` -> `soft_next_step`

## `query_type = project_specific`
- `cold` -> `value_teaser`
- `warm` -> `grounded_recommendation`
- `interested` -> `grounded_recommendation` hoặc `handle_concern`
- `ready` -> `meeting_invite`

## `query_type = clarification`
- giữ engagement_state cũ
- nối tiếp mode của turn trước, không reset

---

# 11. Kết luận

Bản update đúng cho hệ hiện tại là:

- `query_type` để biết khách đang hỏi gì
- `engagement_state` để biết khách đang mở lòng tới đâu
- `response_mode` để biết lượt này nên nói kiểu gì
- `ask_policy` để biết có nên hỏi hay không

Và để hoàn thiện skill sales, cần bổ sung thêm:

- phản chiếu cảm xúc / nhu cầu
- hook tạo tò mò
- CTA đúng nhịp
- tránh brochure-style và hỏi form

## Câu chốt

Nếu làm đúng cấu trúc này, agent sẽ bớt:
- hỏi cứng
- trả lời cho có
- brochure-style
- CTA quá sớm

và sẽ gần hơn với một **trợ lý tư vấn viên bất động sản có duyên sales**.
