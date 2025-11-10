import cv2
import numpy as np
from PIL import Image
import os

DATASET_DIR = 'dataset'        # thư mục chứa ảnh đã chụp từ file 1
TRAINER_DIR = 'trainer'        # nơi lưu model sau khi train
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"



# Tạo model 
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("LBPH Face Recognizer created successfully.")


#Tạo bộ phát hiện khuôn mặt
detector = cv2.CascadeClassifier(CASCADE_PATH)



#Hàm đọc ảnh và nhãn 
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # try:
            # Chuyển ảnh sang grayscale
            PIL_img = Image.open(imagePath).convert('L') # cái này qua ảnh xám,đỡ tốn dữ liệu
            img_numpy = np.array(PIL_img, 'uint8')

            # Lấy ID từ tên file: User.<id>.<count>.jpg
            filename = os.path.split(imagePath)[-1]
            id = int(filename.split('.')[1])

            # Nhận diện khuôn mặt trong ảnh
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        # except Exception as e:
        #     print(f"[WARN] Bỏ qua ảnh lỗi: {imagePath} ({e})")

    return faceSamples, ids

print("\nĐang train dữ liệu khuôn mặt. Vui lòng đợi...")

faces, ids = getImagesAndLabels(DATASET_DIR)

# if len(faces) == 0:
#     print("[ERROR] Không tìm thấy ảnh hợp lệ trong folder dataset.")
#     exit(1)

# Train mô hình
recognizer.train(faces, np.array(ids))

# Lưu mô hình
model_path = os.path.join(TRAINER_DIR, 'trainer.yml')
recognizer.write(model_path)

print(f"\n Hoàn tất. Đã train {len(np.unique(ids))} khuôn mặt.")
# print(f"Mô hình đã lưu tại: {model_path}")
