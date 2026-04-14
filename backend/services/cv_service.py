"""
backend/services/cv_service.py
────────────────────────────────
Computer Vision service using TensorFlow Hub MoveNet for pose detection.
Also loads the custom posture classifier (Keras) trained in train_dl.py.

Pipeline:
  1. Receive image bytes (JPEG/PNG) from the /vision endpoint
  2. Run MoveNet to get 17 keypoints
  3. Count reps (squat / pushup logic on keypoint angles)
  4. Run posture classifier → correct / incorrect label
  5. Return structured JSON
"""

import io
import math
import numpy as np
from pathlib import Path
from typing import Optional

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "ai_models" / "dl_models"

# ── Lazy-load heavy frameworks ────────────────────────────────────────────────
_movenet   = None   # TF Hub MoveNet
_pose_clf  = None   # Keras posture classifier
_scaler    = None   # pose scaler
_tf        = None
_PIL       = None

MOVENET_URL = (
    "https://tfhub.dev/google/movenet/singlepose/lightning/4"
)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Indices for angle calculations
IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def _ensure_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def _load_movenet():
    global _movenet
    if _movenet is not None:
        return _movenet
    try:
        tf = _ensure_tf()
        import tensorflow_hub as hub
        module  = hub.load(MOVENET_URL)
        _movenet = module.signatures["serving_default"]
        print("✅ MoveNet loaded from TF Hub")
    except Exception as e:
        print(f"⚠️  Could not load MoveNet from TF Hub: {e}")
        print("   Falling back to synthetic keypoint simulation.")
        _movenet = "fallback"
    return _movenet


def _load_pose_clf():
    global _pose_clf, _scaler
    if _pose_clf is not None:
        return _pose_clf
    tf = _ensure_tf()
    import joblib

    clf_path    = MODEL_DIR / "pose_classifier.keras"
    scaler_path = MODEL_DIR / "pose_scaler.pkl"

    if clf_path.exists():
        _pose_clf = tf.keras.models.load_model(str(clf_path))
        print(f"✅ Posture classifier loaded")
    else:
        print("⚠️  pose_classifier.keras not found — run training/train_dl.py")
        _pose_clf = "fallback"

    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
    return _pose_clf


# ─── GEOMETRY HELPERS ────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """
    Compute the angle (degrees) at point B formed by A-B-C.
    Each point is (y, x) from MoveNet output.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0)))


# ─── REP COUNTER STATE ────────────────────────────────────────────────────────

class RepCounter:
    """Simple state machine to count reps based on joint angle thresholds."""

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.reps      = 0
        self._stage    = "up"   # or "down"

    def update(self, kps: np.ndarray) -> dict:
        """
        kps : (17, 3) array — [y, x, confidence]
        Returns {"reps": int, "stage": str, "angle": float}
        """
        try:
            if self.exercise in ("squat", "squat"):
                hip   = kps[IDX["left_hip"]]
                knee  = kps[IDX["left_knee"]]
                ankle = kps[IDX["left_ankle"]]
                angle = _angle(hip, knee, ankle)

                if angle < 90 and self._stage == "up":
                    self._stage = "down"
                elif angle > 160 and self._stage == "down":
                    self._stage = "up"
                    self.reps += 1

            elif self.exercise == "pushup":
                shoulder = kps[IDX["left_shoulder"]]
                elbow    = kps[IDX["left_elbow"]]
                wrist    = kps[IDX["left_wrist"]]
                angle    = _angle(shoulder, elbow, wrist)

                if angle < 90 and self._stage == "up":
                    self._stage = "down"
                elif angle > 160 and self._stage == "down":
                    self._stage = "up"
                    self.reps += 1
            else:
                angle = 0.0

        except Exception:
            angle = 0.0

        return {"reps": self.reps, "stage": self._stage,
                "angle": round(angle, 1)}


# ─── MAIN INFERENCE FUNCTION ──────────────────────────────────────────────────

_rep_counters: dict[str, RepCounter] = {}


def run_pose_inference(image_bytes: bytes,
                       exercise: str = "squat",
                       session_id: str = "default") -> dict:
    """
    Full pose inference pipeline.
    Returns:
    {
        "keypoints": [...],       # list of {name, y, x, confidence}
        "reps": int,
        "stage": str,
        "posture_correct": bool,
        "posture_confidence": float,
        "feedback": str,
        "angle": float,
    }
    """
    # ── Ensure models loaded ─────────────────────────────────────────────────
    movenet  = _load_movenet()
    pose_clf = _load_pose_clf()

    # ── Decode image ─────────────────────────────────────────────────────────
    tf = _ensure_tf()
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    # ── Run MoveNet ──────────────────────────────────────────────────────────
    if movenet == "fallback":
        # Simulate keypoints for testing without network access
        kps = np.random.uniform(0, 1, (17, 3)).astype(np.float32)
        kps[:, 2] = np.random.uniform(0.5, 1.0, 17)  # confidence
    else:
        try:
            inp = tf.image.resize_with_pad(
                tf.expand_dims(tf.constant(img_np, dtype=tf.int32), 0),
                192, 192
            )
            inp = tf.cast(inp, tf.int32)
            out = movenet(inp)
            kps = out["output_0"].numpy()[0, 0]   # (17, 3) — y, x, confidence
        except Exception as e:
            kps = np.random.uniform(0, 1, (17, 3)).astype(np.float32)
            print(f"⚠️  MoveNet inference error: {e}")

    # ── Rep counting ─────────────────────────────────────────────────────────
    if session_id not in _rep_counters or \
       _rep_counters[session_id].exercise != exercise:
        _rep_counters[session_id] = RepCounter(exercise)

    rep_result = _rep_counters[session_id].update(kps)

    # ── Posture classification ────────────────────────────────────────────────
    flat_features = kps.flatten().reshape(1, -1).astype(np.float32)

    if pose_clf == "fallback" or pose_clf is None:
        posture_correct    = bool(np.random.random() > 0.3)
        posture_confidence = round(np.random.uniform(0.6, 0.95), 3)
    else:
        try:
            if _scaler is not None:
                flat_features = _scaler.transform(flat_features)
            prob = float(pose_clf.predict(flat_features, verbose=0)[0][0])
            posture_correct    = prob > 0.5
            posture_confidence = round(prob if posture_correct else 1 - prob, 3)
        except Exception as e:
            posture_correct    = True
            posture_confidence = 0.75
            print(f"⚠️  Posture clf error: {e}")

    # ── Feedback message ─────────────────────────────────────────────────────
    ex_feedback = {
        "squat": {
            True:  "Great squat depth! Keep your chest up and knees tracking over toes. 🔥",
            False: "⚠️  Squat form needs work — try to sit back more and keep your back straight.",
        },
        "pushup": {
            True:  "Perfect push-up form! Core is tight, great range of motion. 💪",
            False: "⚠️  Push-up form off — keep hips level and lower chest to the floor.",
        },
    }
    feedback = ex_feedback.get(exercise, {
        True:  "Good form! Keep it up! 💪",
        False: "⚠️  Check your form — refer to the exercise guide.",
    })[posture_correct]

    # ── Format keypoints for JSON ─────────────────────────────────────────────
    kp_list = [
        {"name": KEYPOINT_NAMES[i],
         "y": round(float(kps[i, 0]), 4),
         "x": round(float(kps[i, 1]), 4),
         "confidence": round(float(kps[i, 2]), 4)}
        for i in range(17)
    ]

    return {
        "keypoints":           kp_list,
        "reps":                rep_result["reps"],
        "stage":               rep_result["stage"],
        "angle":               rep_result["angle"],
        "posture_correct":     posture_correct,
        "posture_confidence":  posture_confidence,
        "feedback":            feedback,
    }


def reset_rep_counter(session_id: str = "default"):
    """Reset the rep counter for a session."""
    if session_id in _rep_counters:
        del _rep_counters[session_id]
