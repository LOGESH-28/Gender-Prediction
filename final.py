import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Load Models ---

# Load pre-trained MobileNetV2 model
mobilenet_model = load_model('saved_models/mobilenetv2_gender.h5')

# Load pre-trained InceptionV3 model
inception_model = load_model('saved_models/inceptionv3_gender.h5')

# Load SSR-Net model
from SSR_net_model import SSR_net_general
ssr_model = SSR_net_general(64, 64, 1, 1.0, 1.0)
ssr_model.load_weights('saved_models/ssrnet_gender.h5')

# --- Preprocess face image ---
def preprocess_face(face, size):
    face_resized = cv2.resize(face, size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    return np.expand_dims(face_normalized, axis=0)

# --- Predict gender from a single face image ---
def predict_gender(face):
    try:
        face_mobilenet = preprocess_face(face, (224, 224))
        face_inception = preprocess_face(face, (224, 224))
        face_ssr = preprocess_face(face, (64, 64))

        pred_m = mobilenet_model.predict(face_mobilenet, verbose=0)[0][0]
        pred_i = inception_model.predict(face_inception, verbose=0)[0][0]
        pred_s = ssr_model.predict(face_ssr, verbose=0)[0][0]

        predictions = [
            ('MobileNetV2', pred_m),
            ('InceptionV3', pred_i),
            ('SSR-Net', pred_s)
        ]

        # Format: (model_name, confidence_score_towards_predicted_gender, predicted_gender)
        formatted_preds = [(name, prob if prob > 0.5 else 1 - prob, 'Female' if prob > 0.5 else 'Male') for name, prob in predictions]
        formatted_preds.sort(key=lambda x: x[1], reverse=True)

        # Take top 2
        top2 = formatted_preds[:2]
        genders = [g for _, _, g in top2]
        final_gender = max(set(genders), key=genders.count)

        print("\nPredictions:")
        for name, conf, gender in formatted_preds:
            print(f"{name}: {gender} ({conf * 100:.2f}% confidence)")
        print(f"Final Gender (Top 2 vote): {final_gender}")

        return final_gender
    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Unknown"

# --- Real-time webcam gender prediction ---
def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            gender = predict_gender(face)
            cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Gender Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
