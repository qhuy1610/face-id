# Face Recognition Project

## Mô tả
- Bài cuối kì môn học  
- Mục đích: tạo nền tảng cho các ứng dụng nhận diện khuôn mặt cao hơn

---

## Hướng dẫn sử dụng

### 1️ Thu thập dữ liệu khuôn mặt
```bash
python capture_faces.py
```
- Nhập **ID người** (ví dụ: 1, 2, ...)  
- Camera sẽ mở và tự động lưu ảnh khuôn mặt vào thư mục `dataset/`  

### 2️ Train mô hình nhận diện
```bash
python train_faces.py
```
- Chạy sau khi đã thu thập đủ dữ liệu  
- File `trainer/trainer.yml` sẽ được tạo, lưu model nhận diện khuôn mặt  
- Có thể chỉnh sửa `names` trong file để ánh xạ **ID → Tên người**

### 3️ Nhận diện khuôn mặt trực tiếp
```bash
python face_recognition.py
```
- Camera mở ra, nhận diện khuôn mặt  
- Hiển thị **tên người** và **độ chính xác (%)** trên màn hình  
- Nhấn `ESC` để thoát

---

## Lưu ý
- Tạo sẵn 2 folder: `dataset` và `trainer` trước khi chạy chương trình  
- Mọi vấn đề khác, tham khảo GPT nếu cần  

---

## Cấu trúc project gợi ý
```
face-recognition/
│
├─ dataset/             # chứa ảnh khuôn mặt
├─ trainer/             # chứa model trainer.yml sau khi train
├─ capture_faces.py     # file thu thập ảnh
├─ train_faces.py       # file train mô hình
├─ face_recognition.py  # file nhận diện trực tiếp
└─ README.md            # hướng dẫn sử dụng
```

## Author
G12 - ML_12-ELC3006_50K29.2
