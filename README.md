

# -------------------------------------------------
# 1. Chuẩn bị cấu trúc thư mục
# -------------------------------------------------
# project/
# ├── 1.py                     # thu thập ảnh khuôn mặt
# ├── 2.py                     # train mô hình CNN
# ├── 3.py                     # điểm danh
# ├── deploy.prototxt.txt      # model face detection (OpenCV)
# ├── res10_300x300_ssd_iter_140000.caffemodel
# ├── dataset/                 # tự tạo hoặc để chương trình tạo
# ├── trainer/                 # lưu model CNN sau khi train
# └── thong tin cac lop.txt    # lưu thông tin lớp & sinh viên

# -------------------------------------------------
# 2. Sử dụng file 1.py – Thu thập dữ liệu
# -------------------------------------------------
# Mục đích:
# - Mở webcam
# - Phát hiện khuôn mặt
# - Chụp và lưu ảnh sinh viên
# - Lưu thông tin lớp và sinh viên

python 1.py

# Khi chạy sẽ nhập:
# - tên lớp
# - id sinh viên
# - tên sinh viên

# Kết quả:
# dataset/<ten_lop>/User.<id>.<count>.jpg
# thong tin cac lop.txt được cập nhật

# Lưu ý:
# - Chạy 1.py cho từng sinh viên
# - Có thể chạy nhiều lần để thêm sinh viên

# -------------------------------------------------
# 3. Sử dụng file 2.py – Train mô hình CNN
# -------------------------------------------------
# Mục đích:
# - Đọc toàn bộ ảnh trong dataset/
# - Gán nhãn theo lớp + sinh viên
# - Huấn luyện mô hình CNN
# - Lưu model để nhận diện

python 2.py

# Kết quả:
# trainer/cnn_face_model.h5
# trainer/label_map.json

# Lưu ý:
# - Chỉ chạy sau khi đã chụp xong tất cả sinh viên
# - Khi thêm sinh viên mới → cần chạy lại 2.py

# -------------------------------------------------
# 4. Sử dụng file 3.py – Điểm danh
# -------------------------------------------------
# Mục đích:
# - Mở webcam
# - Nhận diện khuôn mặt bằng CNN
# - Điểm danh theo lớp được chọn
# - Xuất file kết quả

python 3.py

# Khi chạy:
# - chọn lớp cần điểm danh

# Kết quả:
# diem_danh_<ten_lop>_<thoi_gian>.txt

# -------------------------------------------------
# 5. Thứ tự sử dụng bắt buộc
# -------------------------------------------------
# 1.py -> 2.py -> 3.py
# =================================================

# -------------------------------------------------
# Author / Contact
# -------------------------------------------------
# Email : qhuy161026@gmail.com
# -------------------------------------------------
