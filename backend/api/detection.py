from fastapi import APIRouter, UploadFile, File, Query
from controllers.detection_controller import detect_from_upload, detect_from_path, detect_from_url

router = APIRouter(prefix="/api", tags=["detection"])

@router.post("/detect/upload")
async def detect_upload(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    return detect_from_upload(file, threshold=threshold)

@router.post("/detect/path")
async def detect_path(
    file_path: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    return detect_from_path(file_path, threshold=threshold)

@router.post("/detect/url")
async def detect_url(
    image_url: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    return detect_from_url(image_url, threshold=threshold)
