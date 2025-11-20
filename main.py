import cv2
import numpy as np
import asyncio
import uvicorn
import sqlite3
import base64
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# ========================================================
# FASTAPI + CORS
# ========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve detection images
app.mount("/detections", StaticFiles(directory="detections"), name="detections")

# ========================================================
# LOAD MODELS
# ========================================================
yolo_model = YOLO("yolov8n.pt")
emotion_model = load_model("models/mobilenetv2_final.h5")

with open("models/labels.json", "r") as f:
    class_labels = json.load(f)

print("Loaded classes:", class_labels)

# ========================================================
# DISTANCE ESTIMATION
# ========================================================
KNOWN_DISTANCE_CM = 100
FOCAL_LENGTH = 100

def estimate_distance(h):
    return (FOCAL_LENGTH * KNOWN_DISTANCE_CM) / h if h > 0 else None

# ========================================================
# DATABASE SETUP
# ========================================================
DB_PATH = "surveillance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lat REAL,
        lon REAL,
        emotion TEXT,
        distance REAL,
        confidence REAL,
        thumbnail_path TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ========================================================
# LOG DETECTION ENDPOINT
# ========================================================
@app.post("/log_detection")
async def log_detection(
    lat: float = Body(...),
    lon: float = Body(...),
    emotion: str = Body(...),
    distance: float = Body(...),
    confidence: float = Body(...),
    image_base64: str = Body(...)
):
    try:
        img_bytes = base64.b64decode(image_base64)
        filename = f"detections/{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"

        with open(filename, "wb") as f:
            f.write(img_bytes)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detections (lat, lon, emotion, distance, confidence, thumbnail_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (lat, lon, emotion, distance, confidence, filename, datetime.now().isoformat()))

        conn.commit()
        conn.close()

        return {"status": "success"}

    except Exception as e:
        return {"error": str(e)}

# ========================================================
# SURVEILLANCE DATA
# ========================================================
@app.get("/surveillance_data")
def surveillance_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT lat, lon, emotion, distance, confidence, thumbnail_path, timestamp FROM detections")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "lat": r[0],
            "lon": r[1],
            "emotion": r[2],
            "distance": r[3],
            "confidence": r[4],
            "thumbnail": r[5],
            "timestamp": r[6]
        })

    return data

# ========================================================
# AUTO DISCONNECT HANDLER
# ========================================================
async def await_disconnected(request: Request):
    if await request.is_disconnected():
        return True
    await asyncio.sleep(0.01)
    return False

# ========================================================
# VIDEO STREAM GENERATOR
# ========================================================
async def generate_frames(request: Request, lat: float, lon: float):
    cap = cv2.VideoCapture(0)

    try:
        while True:
            if await await_disconnected(request):
                break

            success, frame = cap.read()
            if not success:
                break

            results = yolo_model(frame, classes=[16])

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < 0.50:
                        continue

                    h = y2 - y1
                    distance = estimate_distance(h)
                    roi = frame[y1:y2, x1:x2]

                    if roi.size > 0:
                        img = cv2.resize(roi, (224, 224))
                        arr = image.img_to_array(img)
                        arr = np.expand_dims(arr, 0) / 255.0
                        pred = emotion_model.predict(arr, verbose=0)
                        emotion = class_labels[np.argmax(pred)]

                        _, buf = cv2.imencode(".jpg", roi)
                        thumb_b64 = base64.b64encode(buf).decode()

                        asyncio.create_task(
                            log_detection(lat, lon, emotion, distance, conf * 100, thumb_b64)
                        )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    finally:
        cap.release()

# ========================================================
# VIDEO FEED ROUTE
# ========================================================
@app.get("/video_feed")
async def video_feed(request: Request, lat: float, lon: float):
    return StreamingResponse(
        generate_frames(request, lat, lon),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ========================================================
# CAMERA PAGE
# ========================================================
@app.get("/camera", response_class=HTMLResponse)
def camera_page():
    return """
    <html>
    <body>
        <script>
            navigator.geolocation.getCurrentPosition(pos => {
                window.location.href =
                    "/camera_stream?lat=" + pos.coords.latitude + "&lon=" + pos.coords.longitude;
            });
        </script>
        <p style="color:white">Getting location...</p>
    </body>
    </html>
    """

# ========================================================
# CAMERA STREAM PAGE (FIXED BACK BUTTON)
# ========================================================
@app.get("/camera_stream", response_class=HTMLResponse)
def camera_stream(lat: float, lon: float):
    return f"""
    <html>
    <body style="margin:0;background:black;">

        <img id="cam" 
             src="/video_feed?lat={lat}&lon={lon}" 
             style="width:100vw;height:100vh;object-fit:contain">

        <button onclick="stopAndGoHome()" 
                style="position:fixed;top:20px;left:20px;padding:10px 20px;
                       font-size:16px;border-radius:8px;z-index:999;">
            Back Home
        </button>

        <script>
            function stopAndGoHome() {{
                const cam = document.getElementById('cam');
                cam.src = "";
                setTimeout(() => {{
                    window.location.href = "/?skipIntro=1";

                }}, 200);
            }}
        </script>

    </body>
    </html>
    """

# ========================================================
# SURVEILLANCE PAGE
# ========================================================
@app.get("/surveillance", response_class=HTMLResponse)
def surveillance():
    with open("surveillance.html", encoding="utf-8") as f:
        return f.read()

# ========================================================
# HOMEPAGE
# ========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    with open("homepage.html", encoding="utf-8") as f:
        return f.read()

# ========================================================
# ANALYZE UPLOAD (UPLOAD PAGE)
# ========================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Unable to read the uploaded image."}

        h, w = frame.shape[:2]
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = yolo_model(frame, classes=[16])
        detections = []

        lat, lon = 0.0, 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.25:
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                img = cv2.resize(roi, (224, 224))
                arr = image.img_to_array(img)
                arr = np.expand_dims(arr, 0) / 255.0
                pred = emotion_model.predict(arr, verbose=0)
                emotion = class_labels[np.argmax(pred)]

                bbox_h = y2 - y1
                distance = estimate_distance(bbox_h)

                _, buf = cv2.imencode(".jpg", roi)
                b64_thumb = base64.b64encode(buf).decode()

                asyncio.create_task(
                    log_detection(
                        lat, lon, emotion, distance, conf * 100, b64_thumb
                    )
                )

                detections.append({
                    "emotion": emotion,
                    "confidence": round(conf * 100, 2),
                    "distance": round(distance, 2) if distance else None
                })

        if not detections:
            return {"message": "No dog detected."}

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

# ========================================================
# RUN APP
# ========================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
