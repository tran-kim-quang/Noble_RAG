# Noble Vision Service (MVP)

Service nhận diện khách hàng từ ảnh và map `customer_id` với `session_id`.

## Endpoints

- `POST /vision/identify`
- `GET /vision/customer/{customer_id}`
- `GET /vision/session/{session_id}`

## Chạy local

```bash
cd service/vision
pip install -r requirements.txt
python app.py
```

Mặc định service chạy ở `0.0.0.0:8020`.

## Ghi chú MVP

- Embedding hiện tại dùng deterministic hashing để hoàn thiện flow backend.
- Có thể thay bằng model face embedding thật ở phase tiếp theo mà không đổi API.
