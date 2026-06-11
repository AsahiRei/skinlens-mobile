from fastapi import HTTPException, UploadFile
from services.detection_services import DetectionService
import os
from pathlib import Path
import requests

UPLOAD_DIR = "backend/upload"

def detect_from_upload(file: UploadFile, threshold: float = 0.5):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {allowed_extensions}")
        
        # Read file bytes
        contents = file.file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Perform detection
        result = DetectionService.detect_from_bytes(contents, threshold=threshold)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result["data"]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

def detect_from_path(file_path: str, threshold: float = 0.5):
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        result = DetectionService.detect_from_file(file_path, threshold=threshold)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result["data"]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

def detect_from_url(image_url: str, threshold: float = 0.5):
    try:
        if not image_url:
            raise HTTPException(status_code=400, detail="No URL provided")
        
        # Download image from URL
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="URL request timed out")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
        
        content_type = response.headers.get("content-type", "").lower()
        if "image" not in content_type:
            raise HTTPException(status_code=400, detail="URL does not return an image")

        image_bytes = response.content
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image from URL")
        
        # Perform detection
        result = DetectionService.detect_from_bytes(image_bytes, threshold=threshold)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result["data"]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
