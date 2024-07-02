import cv2
import numpy as np
import joblib

model = joblib.load('face_recognition_model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    features = extract_features(frame)

    if features is not None:
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        confidence = model.decision_function(features)
        prediction = model.predict(features)

        confidence_value = np.max(confidence)

        if confidence_value > 0.75:
            name = prediction[0]
        else:
            name = 'Unknown'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()