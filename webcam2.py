import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load ONNX emotion model
session = ort.InferenceSession("emotion_model.onnx")
input_name = session.get_inputs()[0].name

# SAME ORDER AS TRAINING
class_labels = ['angry', 'happy', 'sad', 'relaxed']

# Distance estimation constants
KNOWN_DISTANCE_CM = 100
KNOWN_HEIGHT_PX = 150
FOCAL_LENGTH = (KNOWN_HEIGHT_PX * KNOWN_DISTANCE_CM) / KNOWN_HEIGHT_PX  # = 100

def estimate_distance(bbox_height):
    if bbox_height == 0:
        return None
    return (FOCAL_LENGTH * KNOWN_DISTANCE_CM) / bbox_height

def preprocess_for_onnx(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, classes=[16])  # Detect dogs

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
            confidence = float(box.conf[0])

            if confidence > 0.5:
                bbox_height = y2 - y1
                distance_cm = estimate_distance(bbox_height)

                dog_roi = frame[y1:y2, x1:x2]
                if dog_roi.size == 0:
                    continue

                # Preprocess dog face for ONNX model
                img = preprocess_for_onnx(dog_roi)

                # Predict emotion
                pred = session.run(None, {input_name: img})[0]
                predicted_class = class_labels[np.argmax(pred)]

                # Draw labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_class, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if distance_cm:
                    cv2.putText(frame, f"{int(distance_cm)} cm", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Dog Emotion + Distance (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

