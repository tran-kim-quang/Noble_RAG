Ngắn gọn, sửa theo 5 ý này:

1. **Bỏ mặc định “cuối câu phải hỏi”**
   Đổi rule từ:

   * “chỉ đặt tối đa 1 câu hỏi mở ở cuối”
     thành:
   * “chỉ hỏi khi thật sự thiếu 1 thông tin quan trọng”
   * “không mặc định kết thúc bằng câu hỏi”

2. **Để orchestrator quyết định kiểu phản hồi trước**
   Thêm biến như:

   * `response_mode = inform_only | recommendation | clarify_light | meeting_invite`
   * `ask_policy = avoid_question | allow_question | must_clarify`

3. **Nếu đã đủ dữ liệu thì ưu tiên kết thúc bằng nhận định/gợi ý, không hỏi nữa**
   Ví dụ với query cụ thể thì model chỉ cần:

   * nêu điểm phù hợp
   * nêu 1 nhận định grounded
   * dừng

4. **Không truyền `consult_seed_reply` nguyên xi vào prompt cuối**
   Vì seed này thường mang nhịp “khai thác thêm”, làm model lặp pattern hỏi cuối.
   Thay bằng guidance ngắn kiểu:

   * `focus`
   * `response_mode`
   * `ask_policy`

5. **Thêm rule chống lặp pattern hỏi**
   Ví dụ:

   * nếu 1–2 lượt gần đây đã kết thúc bằng câu hỏi thì lượt này ưu tiên không hỏi
   * nếu user vừa trả lời ngắn cho câu hỏi trước thì tiếp tục tư vấn trước, chưa hỏi tiếp

Mấu chốt nhất là:

> **đừng để prompt tự quyết có hỏi hay không**
> **hãy để orchestrator truyền `response_mode` + `ask_policy`, prompt chỉ diễn đạt cho tự nhiên**
