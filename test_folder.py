import tensorflow as tf
import numpy as np
import cv2
import os
import csv

MODEL_PATH = "models/mobilenetv2_final.h5"
CLASS_LABELS = ["angry", "happy", "relaxed", "sad"]

def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def test_folder(folder):
    model = tf.keras.models.load_model(MODEL_PATH)

    report = []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print("\n📌 Testing", len(files), "images in:", folder)
    print("---------------------------------------------")

    for f in files:
        path = os.path.join(folder, f)
        img = load_and_preprocess(path)
        if img is None:
            print("❌ Could not read:", path)
            continue

        pred = model.predict(img, verbose=0)
        cls = CLASS_LABELS[np.argmax(pred)]
        conf = float(np.max(pred)) * 100

        print(f"{f} → {cls} ({conf:.2f}%)")
        report.append([f, cls, conf])

    # Save CSV report
    with open("folder_test_report.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "predicted_class", "confidence"])
        writer.writerows(report)

    print("\n📄 Saved report: folder_test_report.csv")

if __name__ == "__main__":
    FOLDER = input("Enter folder (example: dataset/angry): ")
    test_folder(FOLDER)
