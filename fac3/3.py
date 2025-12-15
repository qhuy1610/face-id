import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- Tải model nhận diện khuôn mặt của OpenCV (deep learning) ---
configFile = "deploy.prototxt.txt"
modelFile  = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# --- Tải model CNN của bạn (đã train trước đó) ---
MODEL_PATH = 'trainer/cnn_face_model.h5'
IMG_SIZE = (64, 64)
model = load_model(MODEL_PATH)

# --- Tên ID ---
names = {
    1: "cr7",
    2: "minh chim be",
    3: "quan",
    4: "nguyen",
    5: "tuan anh"
}

# --- Mở webcam ---
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

font = cv2.FONT_HERSHEY_SIMPLEX
print("\nBắt đầu nhận diện bằng deep learning. Nhấn ESC để thoát.\n")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    (h, w) = frame.shape[:2]

    # --- Tạo blob để đưa vào mạng deep learning ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # num_faces = 0

    # --- Duyệt qua tất cả khuôn mặt ---
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.85:  # ngưỡng tin cậy
            # num_faces += 1
            box = detections[0, 0, i, 3:7] * ([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)

            # Cắt khuôn mặt
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Tiền xử lý khuôn mặt cho CNN
            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)).resize(IMG_SIZE)
            face_array = np.array(face_img, 'float32') / 255.0
            face_array = np.expand_dims(face_array, axis=(0, -1))  # (1,64,64,1)

            # Dự đoán bằng model CNN của bạn
            predictions = model.predict(face_array)
            id_ = np.argmax(predictions)
            confidence_cnn = predictions[0][id_] * 100

            if confidence_cnn > 90:
                name = names.get(id_ + 1, "Unknown")
            else:
                name = "Unknown"

            # Vẽ khung và tên
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x1 + 5, y1 - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Acc: {round(confidence_cnn)}%", (x1 + 5, y2 + 25), font, 0.7, (255, 255, 0), 1)

    # --- Hiển thị số khuôn mặt và cảnh báo nếu có nhiều hơn 1 ---
    # cv2.putText(frame, f"Detected: {num_faces}", (20, 40), font, 1, (255, 0, 0), 2)
    # if num_faces > 1:
    #     cv2.putText(frame, "Canh bao: Nhieu hon 1 khuon mat!", (20, 80), font, 0.8, (0, 0, 255), 2)

    cv2.imshow("Deep Learning Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

print("\nThoát chương trình.")
cam.release()
cv2.destroyAllWindows()
