import cv2
import os

# Nạp mô hình deep learning có sẵn của OpenCV
configFile = "deploy.prototxt.txt"
modelFile  = "res10_300x300_ssd_iter_140000.caffemodel"
DATASET_DIR = 'dataset'



# Nạp model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Nhập ID người dùng
face_id = input("Nhập ID người để lưu khuôn mặt: ").strip()
count = 0
MAX_COUNT = 150
# Mở webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # đổi về 0 nếu chỉ có 1 webcam

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Tạo blob cho mạng DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    # Duyệt qua tất cả các phát hiện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Chỉ nhận khi độ tin cậy > 0.6
        if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")

                # Cắt vùng khuôn mặt
                face = frame[y1:y2, x1:x2]

                # Kiểm tra hợp lệ để lưu
                if face.size > 0 and count < MAX_COUNT:
                    count += 1
                    save_path = os.path.join(DATASET_DIR, f"User.{face_id}.{count}.jpg")
                    cv2.imwrite(save_path, face)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Saved: {count}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Face Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Tự động dừng khi đủ ảnh
    if count >= MAX_COUNT:
        print("Đã lưu đủ ảnh.")
        break

cam.release()
cv2.destroyAllWindows()
