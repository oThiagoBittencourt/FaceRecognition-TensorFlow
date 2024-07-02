import cv2
import os

base_dir = 'dataset'

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eyes_cascade.detectMultiScale(roi_gray)
    left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
    right_eyes = right_eye_cascade.detectMultiScale(roi_gray)

    if len(eyes) < 2 or len(left_eyes) < 1 or len(right_eyes) < 1:
        return None

    features = []
    for (ex, ey, ew, eh) in eyes:
        features.append(ex + ew // 2)
        features.append(ey + eh // 2)
    for (ex, ey, ew, eh) in left_eyes:
        features.append(ex + ew // 2)
        features.append(ey + eh // 2)
    for (ex, ey, ew, eh) in right_eyes:
        features.append(ex + ew // 2)
        features.append(ey + eh // 2)

    return features

def capture_images(username, num_images=300):
    user_dir = os.path.join(base_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        features = extract_features(frame)

        if features is not None:
            count += 1
            img_name = f"{username}_{count}.jpg"
            img_path = os.path.join(user_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Captured image {count}/{num_images}")

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter username: ")
    capture_images(username)
