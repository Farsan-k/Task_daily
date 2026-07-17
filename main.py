import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -----------------------------
# Load Models
# -----------------------------

# Detects PERSONS (not faces)
face_model = YOLO("yolov8n.pt")

# Load emotion model without compiling
emotion_model = load_model(
    "fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = face_model(frame)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            # Only keep PERSON class
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Convert to grayscale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Resize to the model's expected size
            face = cv2.resize(face, (64, 64))

            # Normalize
            face = face.astype("float32") / 255.0

            # Add channel dimension
            face = np.expand_dims(face, axis=-1)

            # Add batch dimension
            face = np.expand_dims(face, axis=0)

            prediction = emotion_model.predict(face, verbose=0)

            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            confidence = prediction[0][emotion_index]

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        f"{emotion} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()