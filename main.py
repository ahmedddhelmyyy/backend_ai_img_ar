from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import numpy as np
import cv2
import io

# -------------------- APP --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- LOAD MODEL ONCE --------------------
model = YOLO("best.pt")   # path to your trained model

# -------------------- ENDPOINT --------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image bytes
    data = await file.read()

    # Decode image
    np_img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model.predict(img, conf=0.4)

    # Draw predictions
    annotated = results[0].plot()

    # Encode back to JPEG
    success, buf = cv2.imencode(".jpg", annotated)

    return Response(buf.tobytes(), media_type="image/jpeg")
