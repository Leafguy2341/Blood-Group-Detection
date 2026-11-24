# Hướng dẫn sử dụng và triển khai

## 1. Chạy local

Từ thư mục dự án:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Sau đó mở:
- http://127.0.0.1:8080/ để dùng giao diện web
- http://127.0.0.1:8080/docs để xem Swagger UI

## 2. API

### Upload ảnh để dự đoán
```bash
curl -X POST http://127.0.0.1:8080/predict -F "file=@your_image.png"
```

## 3. Triển khai lên Render

1. Tạo repository trên GitHub.
2. Đăng nhập Render và chọn New > Web Service.
3. Connect repository.
4. Cấu hình:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 10000`
5. Deploy.

## 4. Triển khai lên Railway

1. Đăng nhập Railway.
2. New Project > Deploy from GitHub repo.
3. Cấu hình Start Command:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. Deploy.

## 5. Dùng ngrok (test nhanh)

```bash
ngrok http 8000
```

Sau đó dùng URL được ngrok cung cấp để truy cập từ bên ngoài mạng.

## 6. Điểm nổi bật của giao diện mới

- Xem trước ảnh ngay khi chọn file
- Hiển thị kết quả theo thẻ trực quan
- Tối ưu cho mobile
- Hỗ trợ API và giao diện web cùng lúc
