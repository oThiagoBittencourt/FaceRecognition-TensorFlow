import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import joblib

base_dir = 'dataset'

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    features = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eyes_cascade.detectMultiScale(roi_gray)
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) < 2 or len(left_eyes) < 1 or len(right_eyes) < 1:
            features.extend([0, 0, 0, 0])
        else:
            for (ex, ey, ew, eh) in eyes:
                features.append(ex + ew // 2)
                features.append(ey + eh // 2)
            for (ex, ey, ew, eh) in left_eyes:
                features.append(ex + ew // 2)
                features.append(ey + eh // 2)
            for (ex, ey, ew, eh) in right_eyes:
                features.append(ex + ew // 2)
                features.append(ey + eh // 2)

    while len(features) < 12:
        features.extend([0, 0])

    return features[:12]

data = []
labels = []
for user in os.listdir(base_dir):
    user_path = os.path.join(base_dir, user)
    if os.path.isdir(user_path):
        for img_file in os.listdir(user_path):
            img_path = os.path.join(user_path, img_file)
            features = extract_features(img_path)
            data.append(features)
            labels.append(user)

data = np.array(data)
labels = np.array(labels)

scaler = StandardScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")

joblib.dump(grid, 'face_recognition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
