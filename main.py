from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("best.pt")

@app.get("/")
def root():
    return {"status": "YOLO backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        data = await file.read()

        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        np_img = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        results = model.predict(img, conf=0.4)

        annotated = results[0].plot()

        success, buf = cv2.imencode(".jpg", annotated)
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed")

        return Response(buf.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
