import cv2
import os


DATASET_DIR='dataset' #path folder chứa dataset


# tạo camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

# Load cascade nhận diện khuôn mặt, đây là thư viện, model train sẵn ở đây, lên gg nghiên cứu thêm
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Nhập ID
face_id=input("Nhập ID người để lưu khuôn mặt: ").strip()

count=0
MAX_COUNT=150 # số ảnh tối đa lưu, càng nhiều càng tốt, càng nặng máy lag thì vl

print("\nBắt đầu cắt ảnh, Hãy nhìn vào camera và xoay...")

while True:
    ret, img = cam.read()
    if not ret:
        break
    img = cv2.flip(img, 1) # chống lệch mặt, về 0 thì bth
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    num_face = len(faces)

    # Vẽ rectangle và lưu ảnh cắt
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if face_id != "":
            count += 1
            face_img = gray[y:y+h, x:x+w]
            save_path = os.path.join(DATASET_DIR, f"User.{face_id}.{count}.jpg")
            cv2.imwrite(save_path, face_img)

    # Hiển thị số người trên màn hình
    cv2.putText(img, f"Detected: {num_face}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Detection", img)

    # ESC để thoát hoặc đủ 200 ảnh
    if cv2.waitKey(1) & 0xFF == 27 or count >= MAX_COUNT:
        break

print(f"Hoàn tất. Tổng số ảnh đã lưu: {count}")
print(save_path)
cam.release()
cv2.destroyAllWindows()
