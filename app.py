import os
import time
import uuid
import logging
import uvicorn
import requests

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from tracker import ObjectTracker
from pydantic import BaseModel

app = FastAPI()

# Initialize the ObjectTracker with the YOLO model path and threshold
model_path = "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/models/yolov8x.pt"
tracker = ObjectTracker(model_path=model_path, threshold=0.8)

logging.basicConfig(level=logging.INFO) 

class TrackInfo(BaseModel):
    videoPath: str
    intime: float
    outtime: float
    xPer: float
    yPer: float
    wPer: float
    hPer: float 
    expFrame: int
    objId: int



@app.get("/api/status")
async def segment_app(request: Request): 
    return {'status': "OK"}

@app.post("/api/track_with_initial_yolo_check")
async def track_with_initial_yolo_check_endpoint(request: TrackInfo):
    try:
        
        #local_video_path = await upload_video(request.videoPath)
        local_video_path = "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/video/1917.mp4"
        result = tracker.track_with_initial_yolo_check(
            local_video_path,
            request.xPer,
            request.yPer,
            request.wPer,
            request.hPer,
            request.intime,
            request.outtime,
            request.expFrame,
            request.objId
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=6060, reload=True)
