"""
backend/routes/vision.py  (v5)
────────────────────────────────
Endpoints:
  POST /vision/pose        → MoveNet pose detection + rep counting
  POST /vision/predict     → PyTorch ExerciseNet classification (image)
  POST /vision/classify    → PyTorch ExerciseNet from raw keypoints JSON
  POST /vision/reset       → Reset rep counter session
"""

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from backend.services.cv_service import run_pose_inference, reset_rep_counter
from backend.services.cv_pytorch import predict_exercise, predict_from_keypoints

router = APIRouter(prefix="/vision", tags=["Computer Vision"])


# ─── Existing: MoveNet pose detection + rep counting ──────────────────────────

@router.post("/pose")
async def pose_inference(
    file:       UploadFile = File(...),
    exercise:   str        = Form(default="squat"),
    session_id: str        = Form(default="default"),
):
    """
    Pose detection + rep counting using TensorFlow MoveNet.
    Returns 17 keypoints, rep count, posture stage, and form feedback.
    """
    image_bytes = await file.read()
    result      = run_pose_inference(image_bytes, exercise, session_id)
    return result


# ─── NEW: PyTorch exercise classification (image) ─────────────────────────────

@router.post("/predict")
async def pytorch_predict(file: UploadFile = File(...)):
    """
    Classify exercise from an image using PyTorch ExerciseNet (93.5% accuracy).

    Supports 15 exercises:
    squat, deadlift, bench_press, push_up, pull_up, shoulder_press, plank,
    barbell_biceps_curl, lat_pulldown, lateral_raise, leg_extension,
    leg_raises, romanian_deadlift, t_bar_row, tricep_dips

    Returns: exercise_id, exercise_name, confidence, top_3, posture_feedback,
             form_score (0–1), source
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp",
                                  "image/jpg", None):
        raise HTTPException(status_code=422,
                            detail=f"Unsupported file type: {file.content_type}")
    image_bytes = await file.read()
    if len(image_bytes) < 100:
        raise HTTPException(status_code=422, detail="Image file too small or empty")
    return predict_exercise(image_bytes)


# ─── NEW: PyTorch exercise classification (keypoints) ─────────────────────────

class KeypointRequest(BaseModel):
    keypoints:  List[float]   # 51-dim vector (17 joints × x, y, confidence)
    session_id: Optional[str] = "default"

@router.post("/classify")
async def classify_from_keypoints(req: KeypointRequest):
    """
    Classify exercise from 51-dim MoveNet keypoint vector.
    Use this endpoint in conjunction with /vision/pose — pipe the keypoints
    directly into ExerciseNet for real-time classification with no extra image upload.

    Input: {"keypoints": [x0, y0, c0, x1, y1, c1, ...]}  (51 floats)
    Returns: same as /vision/predict
    """
    if len(req.keypoints) < 17:
        raise HTTPException(status_code=422,
                            detail="Need at least 17 keypoint values (51 recommended)")
    return predict_from_keypoints(req.keypoints)


# ─── Reset rep counter ────────────────────────────────────────────────────────

@router.post("/reset")
async def reset_counter(session_id: str = Query(default="default")):
    """Reset the rep counter for a given session."""
    reset_rep_counter(session_id)
    return {"status": "ok", "message": f"Counter reset for '{session_id}'"}
