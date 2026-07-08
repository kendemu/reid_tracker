#!/usr/bin/env python3
"""
ReID system with pluggable embedder strategy:
  - RT-DETR (HuggingFace transformers) for person detection
  - ByteTrack (supervision 0.29) for temporal tracking
  - Embedder strategy: DINOv3 (local hub)
  - Cosine similarity gallery with flat-matrix batched matching

Usage:
  python reid.py 0                            # webcam
  python reid.py 0 --embedder dinov3 --dinov3-weights ./dinov3_vitb16.pth
"""

import argparse
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import supervision as sv
from PIL import Image

import onnxruntime as ort
import os
import json

# ── defaults ──────────────────────────────────────────────────────────────────

def _models_dir() -> Path:
    """
    Return the directory that holds all ONNX model files.

    Priority:
      1. Installed ROS2 package share directory (ament_index_python).
      2. Source-tree sibling 'models/' folder (standalone / colcon build --symlink-install).
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory('reid_tracker')) / 'models'
        if p.exists():
            return p
    except Exception:
        pass
    # Source tree: reid_tracker/reid_tracker/fullreid.py → reid_tracker/models/
    return Path(__file__).parent.parent / 'models'

_MODELS_DIR = _models_dir()

RTDETR_MODEL    = str(_MODELS_DIR / "rtdetr_r50vd.onnx")
DINOV3_REPO     = str(_MODELS_DIR)   # vestigial; kept for CLI compat
DINOV3_WEIGHTS  = str(_MODELS_DIR / "dinov3.onnx")
DINOV3_BACKBONE = "dinov3_vitb16"

DETECT_CONFIDENCE    = 0.5
REID_SIM_THRESHOLD   = 0.82   # cosine similarity to accept a re-ID match (raised: 0.75 was too permissive)
GID_MERGE_THRESHOLD  = 0.90   # centroid cosine threshold to merge a newly created GID into an existing one
EMB_NOVELTY_THRESHOLD = 0.85  # body embedding cosine threshold for gallery novelty
KEYFRAME_MIN_INTERVAL = 60    # minimum frames between keyframe candidates
MATCH_RATE_THRESHOLD = 0.40   # LightGlue: max match rate below this → new gallery entry
GALLERY_DUMP_DIR     = "gallery_dump"
MAX_GALLERY_PER_GID  = 10     # max entries per global_id; evicts most-central on overflow
MATCH_STRATEGY       = "top1" # "top1" | "centroid"
LIGHTGLUE_ONNX       = str(_MODELS_DIR / "superpoint_lightglue_pipeline.ort.onnx")
IOU_GATE_THRESHOLD   = 0.0   # min IoU vs gid last-bbox to attempt re-ID (0 = disabled)
STABLE_IOU_THRESHOLD = 0.70  # per-track: bbox IoU above this counts as a stable frame
STABLE_MIN_FRAMES    = 5     # consecutive stable frames → skip gallery update
PENDING_CONFIRM_FRAMES = 5  # frames to buffer before committing to a new GID
RTMO_MODEL           = str(_MODELS_DIR / "rtmo.onnx")
POSE_CONF_THRESHOLD  = 0.50  # keypoint confidence below this → joint not visible
POSE_MIN_QUALITY     = "upper_body"  # "full_body" | "upper_body" | "none"

FACE_CONF_THRESHOLD      = 0.65   # YuNet score threshold; sqrt formula pushes background to ~0.58
FACE_KEYFRAME_THRESHOLD  = 0.95   # SFace cosine sim; add face keyframe when best < this
                                   # SFace is very rotation-robust: same face scores 0.97-1.00,
                                   # clearly different viewpoints score ~0.88; 0.95 sits between them
FACE_SIM_THRESHOLD       = 0.35   # SFace cosine sim threshold for face-only re-ID match
MAX_FACE_GALLERY_PER_GID = 10
FACE_MATCH_STRATEGY      = "face_top1_dino_top1_avg"
# Strategies: face_top1_dino_top1_avg | face_body_all_gallery |
#             face_body_face_gallery | face_only_top1
YUNET_MODEL  = str(_MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL  = str(_MODELS_DIR / "face_recognition_sface_2021dec.onnx")

# ImageNet normalisation used by DINOv3
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]




def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ── pose quality ──────────────────────────────────────────────────────────────

class PoseQuality(IntEnum):
    NONE       = 0   # skeleton absent or too partial to trust
    UPPER_BODY = 1   # face + both shoulders visible
    FULL_BODY  = 2   # both shoulders + both hips visible

# COCO-17 keypoint indices
_KP_FACE      = [0, 1, 2]   # nose, left_eye, right_eye
_KP_SHOULDERS = [5, 6]      # left_shoulder, right_shoulder
_KP_HIPS      = [11, 12]    # left_hip, right_hip

def _coco17_quality(kps: np.ndarray, conf_thr: float) -> PoseQuality:
    """kps: (17, 3) — (x, y, confidence). Returns the visible body coverage."""
    vis = kps[:, 2] >= conf_thr
    if vis[_KP_SHOULDERS].all() and vis[_KP_HIPS].all():
        return PoseQuality.FULL_BODY
    if vis[_KP_FACE].any() and vis[_KP_SHOULDERS].all():
        return PoseQuality.UPPER_BODY
    return PoseQuality.NONE


class PoseCheckerBase(ABC):
    @abstractmethod
    def check(self, crop: np.ndarray) -> PoseQuality: ...

    def check_frame(self, frame: np.ndarray, det_boxes: np.ndarray) -> list:
        """Default: run check() on each crop. Override for full-frame efficiency."""
        h, w = frame.shape[:2]
        out = []
        for box in det_boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            out.append(self.check(crop))
        return out


class RTMOPoseChecker(PoseCheckerBase):
    """
    Pose checker backed by RTMO (ONNX).
    check_frame() runs inference ONCE on the full frame, then maps each
    RT-DETR bbox to the nearest RTMO detection by IoU — no per-crop inference.
    """
    _IOU_MATCH_MIN = 0.30   # min IoU to associate an RTMO bbox with a RT-DETR bbox

    def __init__(
        self,
        model_path:     str   = RTMO_MODEL,
        device:         str   = "cuda",
        conf_threshold: float = POSE_CONF_THRESHOLD,
        input_size:     tuple = (640, 640),
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for RTMOPoseChecker. "
                "Install with: pip install onnxruntime-gpu"
            ) from exc

        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        self._session      = ort.InferenceSession(model_path, providers=providers)
        self._input_name   = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        inp_type           = self._session.get_inputs()[0].type
        self._input_dtype  = np.float16 if "float16" in inp_type else np.float32
        self._input_size   = input_size
        self.conf_threshold = conf_threshold
        print(
            f"[RTMOPoseChecker] {model_path}  "
            f"providers={self._session.get_providers()}"
        )

    def _preprocess(self, img: np.ndarray):
        h, w   = img.shape[:2]
        th, tw = self._input_size
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        rgb    = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = rgb.transpose(2, 0, 1)[None].astype(self._input_dtype)
        return tensor, scale

    def _run(self, img: np.ndarray) -> list:
        """Run RTMO on img; return [{bbox, score, keypoints}] above conf threshold."""
        tensor, scale = self._preprocess(img)
        outs  = self._session.run(self._output_names, {self._input_name: tensor})
        dets  = outs[0][0]   # (N, 5)  — x1,y1,x2,y2,score in scaled coords
        kpts  = outs[1][0]   # (N, 17, 3)
        results = []
        for i, det in enumerate(dets):
            score = float(det[4])
            if score < self.conf_threshold:
                continue
            bbox = (det[:4] / scale).astype(np.float32)
            kp   = kpts[i].copy()
            kp[:, :2] /= scale
            results.append({"bbox": bbox, "score": score, "keypoints": kp})
        return results

    def check(self, crop: np.ndarray) -> PoseQuality:
        """Standalone single-crop check (used for debugging / fallback)."""
        if crop.size == 0:
            return PoseQuality.NONE
        results = self._run(crop)
        if not results:
            return PoseQuality.NONE
        best = max(results, key=lambda r: r["score"])
        return _coco17_quality(best["keypoints"], self.conf_threshold)

    def check_frame(self, frame: np.ndarray, det_boxes: np.ndarray) -> list:
        """
        Single RTMO inference on the full frame.
        Each RT-DETR box is paired with the RTMO detection of highest IoU.
        Falls back to NONE when no RTMO detection overlaps sufficiently.
        """
        results = self._run(frame)
        qualities = []
        for det_box in det_boxes:
            best_iou, best_kpts = 0.0, None
            for r in results:
                iou = bbox_iou(det_box, r["bbox"])
                if iou > best_iou:
                    best_iou, best_kpts = iou, r["keypoints"]
            if best_kpts is not None and best_iou >= self._IOU_MATCH_MIN:
                qualities.append(_coco17_quality(best_kpts, self.conf_threshold))
            else:
                qualities.append(PoseQuality.NONE)
        return qualities


# ── face detection + embedding (YuNet + SFace via DeepFace) ──────────────────

class FaceEmbedder:
    """
    Per-frame face detection (YuNet) + face embedding (SFace) via pure ONNX Runtime.
    No TensorFlow dependency. CUDA-accelerated via the same ONNX Runtime used for RTMO.

    YuNet input:  (1, 3, 640, 640) float32 BGR 0-255, letterboxed
    YuNet outputs: cls_S / obj_S / bbox_S / kps_S for S in {8, 16, 32}
    SFace input:  (1, 3, 112, 112) float32 BGR, normalised (x-127.5)/128
    SFace output: (1, 128) float32 raw embedding
    """

    # InsightFace 112×112 reference landmarks (right_eye, left_eye, nose, right_mouth, left_mouth)
    _REF_KPS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    _INPUT_HW = (640, 640)   # YuNet model fixed input size

    def __init__(
        self,
        conf_threshold: float = FACE_CONF_THRESHOLD,
        nms_threshold:  float = 0.3,
        yunet_path:     str   = YUNET_MODEL,
        sface_path:     str   = SFACE_MODEL,
        device:         str   = "cuda",
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required; install onnxruntime-gpu") from exc

        for label, path in [("YuNet", yunet_path), ("SFace", sface_path)]:
            if not Path(path).exists():
                raise RuntimeError(
                    f"[FaceEmbedder] {label} model not found: {path}\n"
                    f"  Place the ONNX file in: {_MODELS_DIR}"
                )

        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        self._det_session = ort.InferenceSession(yunet_path, providers=providers)
        self._rec_session = ort.InferenceSession(sface_path, providers=providers)

        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold

        # Pre-compute anchor centres for each stride (based on 640×640 input)
        H, W = self._INPUT_HW
        self._anchors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for stride in (8, 16, 32):
            rows, cols = np.mgrid[0 : H // stride, 0 : W // stride]
            self._anchors[stride] = (
                (cols * stride).ravel().astype(np.float32),  # cx
                (rows * stride).ravel().astype(np.float32),  # cy
            )

        print(f"[FaceEmbedder] YuNet + SFace ready (pure ONNX)  "
              f"providers={self._det_session.get_providers()}")

    # ── preprocessing / postprocessing helpers ────────────────────────────────

    @staticmethod
    def _letterbox(frame: np.ndarray, target_hw: tuple[int, int]):
        """Resize with padding to target (H, W). Returns (blob_NCHW, scale, pad_x, pad_y)."""
        h, w   = frame.shape[:2]
        th, tw = target_hw
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_x  = (tw - nw) // 2
        pad_y  = (th - nh) // 2
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(frame, (nw, nh))
        blob   = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # NCHW BGR 0-255
        return blob, scale, pad_x, pad_y

    def _decode_stride(
        self,
        cls: np.ndarray, obj: np.ndarray,
        bbox: np.ndarray, kps: np.ndarray,
        stride: int, scale: float, pad_x: int, pad_y: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Decode one stride's raw outputs → (boxes_xyxy, scores, kps) in original coords."""
        # score = sqrt( sigmoid(cls) * sigmoid(obj) )  — matches OpenCV's formula
        score = np.sqrt(
            (1.0 / (1.0 + np.exp(-cls[:, 0]))) *
            (1.0 / (1.0 + np.exp(-obj[:, 0])))
        )
        mask = score >= self.conf_threshold
        if not mask.any():
            return None

        s     = score[mask]
        bx    = bbox[mask]
        kp    = kps[mask]
        cx_a, cy_a = self._anchors[stride][0][mask], self._anchors[stride][1][mask]

        # Decode boxes from letterboxed-640 space → original image coords
        cx = (cx_a + bx[:, 0] * stride - pad_x) / scale
        cy = (cy_a + bx[:, 1] * stride - pad_y) / scale
        bw = np.exp(bx[:, 2]) * stride / scale
        bh = np.exp(bx[:, 3]) * stride / scale
        boxes = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)

        # Decode 5 landmarks
        kp_dec = np.empty((len(s), 10), dtype=np.float32)
        for i in range(5):
            kp_dec[:, i*2]   = (cx_a + kp[:, i*2]   * stride - pad_x) / scale
            kp_dec[:, i*2+1] = (cy_a + kp[:, i*2+1] * stride - pad_y) / scale

        return boxes, s, kp_dec

    def _align_crop(self, frame: np.ndarray, kp5: np.ndarray) -> Optional[np.ndarray]:
        """Affine-warp face to standard 112×112 using 5 detected landmarks."""
        M, _ = cv2.estimateAffinePartial2D(kp5, self._REF_KPS, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

    # ── public API ────────────────────────────────────────────────────────────

    def detect_frame(
        self, frame: np.ndarray
    ) -> list[tuple[np.ndarray, torch.Tensor, tuple[int, int, int, int]]]:
        """
        Detect ALL faces in the full frame.
        Returns list of (face_crop_bgr, L2-norm SFace embedding, (x, y, w, h))
        where (x, y, w, h) are FRAME-space coordinates.

        Call once per frame before the per-person loop; then call
        pick_for_person() per person — only one YuNet inference total.
        """
        if frame is None or frame.size == 0:
            return []

        blob, scale, pad_x, pad_y = self._letterbox(frame, self._INPUT_HW)

        try:
            outs = self._det_session.run(None, {"input": blob})
        except Exception as exc:
            print(f"[FaceEmbedder] detect error: {exc}", file=sys.stderr)
            return []

        out_names = [o.name for o in self._det_session.get_outputs()]
        out = {n: v[0] for n, v in zip(out_names, outs)}   # strip batch dim

        all_boxes, all_scores, all_kps = [], [], []
        for stride in (8, 16, 32):
            res = self._decode_stride(
                out[f"cls_{stride}"], out[f"obj_{stride}"],
                out[f"bbox_{stride}"], out[f"kps_{stride}"],
                stride, scale, pad_x, pad_y,
            )
            if res is not None:
                all_boxes.append(res[0]);  all_scores.append(res[1]);  all_kps.append(res[2])

        if not all_boxes:
            if not hasattr(self, "_diag_printed"):
                self._diag_printed = True
                print(f"[FaceEmbedder] first frame: 0 face(s)  threshold={self.conf_threshold}",
                      flush=True)
            return []

        boxes  = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kps    = np.concatenate(all_kps)

        # NMS (cv2.dnn.NMSBoxes takes xywh format)
        x1y1 = boxes[:, :2];  wh = boxes[:, 2:] - x1y1
        indices = cv2.dnn.NMSBoxes(
            bboxes          = np.concatenate([x1y1, wh], axis=1).tolist(),
            scores          = scores.tolist(),
            score_threshold = self.conf_threshold,
            nms_threshold   = self.nms_threshold,
        )
        indices = np.array(indices).ravel() if len(indices) else np.array([], dtype=int)

        if not hasattr(self, "_diag_printed"):
            self._diag_printed = True
            print(f"[FaceEmbedder] first frame: {len(indices)} face(s) after NMS  "
                  f"scores={[f'{scores[i]:.2f}' for i in indices]}  "
                  f"threshold={self.conf_threshold}", flush=True)

        h_orig, w_orig = frame.shape[:2]
        results = []
        for idx in indices:
            x1i = max(0,      int(boxes[idx, 0]))
            y1i = max(0,      int(boxes[idx, 1]))
            x2i = min(w_orig, int(boxes[idx, 2]))
            y2i = min(h_orig, int(boxes[idx, 3]))
            fw, fh = x2i - x1i, y2i - y1i
            if fw < 10 or fh < 10:
                continue

            # Align face to 112×112 using the 5 landmarks
            kp5 = kps[idx].reshape(5, 2)
            aligned = self._align_crop(frame, kp5)
            if aligned is None:
                continue

            # SFace embedding via ONNX Runtime
            try:
                inp  = ((aligned.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
                feat = self._rec_session.run(None, {"data": inp})[0]   # (1, 128)
                emb  = F.normalize(torch.from_numpy(feat[0]).float(), dim=0)
            except Exception as exc:
                print(f"[FaceEmbedder] embed error: {exc}", file=sys.stderr)
                continue

            face_bgr = frame[y1i:y2i, x1i:x2i].copy()
            results.append((face_bgr, emb, (x1i, y1i, fw, fh)))

        return results

    def pick_for_person(
        self,
        frame_faces: list[tuple],
        person_xyxy: np.ndarray,
    ) -> Optional[tuple[np.ndarray, torch.Tensor, tuple[int, int, int, int]]]:
        """
        Return the largest face whose center falls within person_xyxy, or None.
        Pass the list from detect_frame() — no extra inference needed.
        """
        px1, py1, px2, py2 = map(int, person_xyxy)
        best: Optional[tuple] = None
        for entry in frame_faces:
            _, _, (fx, fy, fw, fh) = entry
            cx, cy = fx + fw // 2, fy + fh // 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if best is None or fw * fh > best[2][2] * best[2][3]:
                    best = entry
        return best


# ── embedder strategy interface ───────────────────────────────────────────────

class EmbedderBase(ABC):
    """
    Strategy interface for appearance embedders.
    Each concrete strategy encapsulates a model + preprocessing.
    """

    @abstractmethod
    def embed(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        Args:
            crop: BGR numpy array, shape (H, W, 3).
        Returns:
            L2-normalised float32 tensor of shape (D,) on CPU,
            or None when the crop is too small / an error occurs.
        """

    def embed_batch(self, crops: list) -> list:
        """Embed multiple crops. Override for a true batched forward pass."""
        return [self.embed(c) for c in crops]


# ── concrete strategy: DINOv3 ─────────────────────────────────────────────────
class DinoV3Embedder(EmbedderBase):
    """
    Appearance embedder backed by DINOv3 ONNX model.
    Runs inference via ONNX Runtime and returns PyTorch CPU Tensors 
    for downstream compatibility.
    """

    def __init__(self, model_path: str = "dinov3.onnx", device: str = None, **kwargs):
        """
        Args:
            model_path (str): Path to the local 'dinov3.onnx' file.
            device (str): Execution device ('cuda' or 'cpu').
            **kwargs: Absorbs legacy parameters (repo, weights, backbone_name) to avoid breaking changes.
        """
        # Backwards compatibility: fallback to 'weights' parameter if passed instead of model_path
        if "weights" in kwargs and model_path == "dinov3.onnx":
            model_path = kwargs["weights"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map device selection to ONNX Execution Providers
        if "cuda" in self.device.lower():
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"[DinoV3Embedder] Loading ONNX model from {model_path} on {providers[0]} ...")
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

        # ImageNet normalization constants matching PyTorch defaults
        self._IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print("[DinoV3Embedder] Ready.")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Transforms BGR image crop to ImageNet-normalized CHW float32 array."""
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res  = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        blob = (res.astype(np.float32) / 255.0 - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return blob.transpose(2, 0, 1)

    def _l2_normalize(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Computes element-wise L2 normalization across specified axis."""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(norm, 1e-12)

    def embed(self, crop: np.ndarray) -> torch.Tensor | None:
        """Extracts embedding for a single crop and returns it as a CPU torch.Tensor."""
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        try:
            blob = self._preprocess(crop)
            batch = np.expand_dims(blob, axis=0)  # Shape: (1, 3, 224, 224)
            
            # Run ONNX inference
            outputs = self.session.run(None, {"images": batch})
            cls_token = outputs[0][0]  # Strip batch dimension -> Shape: (D,)
            
            # Normalize and wrap vector to a CPU PyTorch Tensor
            normalized_np = self._l2_normalize(cls_token)
            return torch.from_numpy(normalized_np)
            
        except Exception as exc:
            print(f"[DinoV3Embedder] embed error: {exc}", file=sys.stderr)
            return None

    def embed_batch(self, crops: list[np.ndarray]) -> list[torch.Tensor | None]:
        """Processes a list of crops simultaneously and returns a list of CPU torch.Tensors."""
        results = [None] * len(crops)
        valid_idx, blobs = [], []
        
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            try:
                blobs.append(self._preprocess(crop))
                valid_idx.append(i)
            except Exception:
                pass
                
        if not blobs:
            return results
            
        try:
            batch = np.stack(blobs)  # Shape: (N, 3, 224, 224)
            
            # Run ONNX batch inference
            outputs = self.session.run(None, {"images": batch})
            embs = outputs[0]  # Shape: (N, D)
            
            normalized_embs = self._l2_normalize(embs, axis=-1)
            
            # Disperse normalized vectors into their proper index slots as PyTorch Tensors
            for out_i, emb in zip(valid_idx, normalized_embs):
                results[out_i] = torch.from_numpy(emb)
                
        except Exception as exc:
            print(f"[DinoV3Embedder] embed_batch error: {exc}", file=sys.stderr)
            
        return results
"""
class DinoV3Embedder(EmbedderBase):

    _transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    def __init__(
        self,
        repo:    str = DINOV3_REPO,
        weights: str = DINOV3_WEIGHTS,
        backbone_name: str = DINOV3_BACKBONE,
        device:  Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo_path    = Path(repo)
        weights_path = Path(weights)

        if not repo_path.exists():
            print(f"[DinoV3Embedder] ERROR: repo not found at {repo_path}", file=sys.stderr)
            print("  Clone: git clone https://github.com/facebookresearch/dinov3.git",
                  file=sys.stderr)
            sys.exit(1)
        if not weights_path.exists():
            print(f"[DinoV3Embedder] ERROR: weights not found at {weights_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[DinoV3Embedder] Loading {backbone_name} from {repo_path} on {self.device} ...")
        self.backbone = torch.hub.load(
            str(repo_path),
            backbone_name,
            source="local",
            weights=str(weights_path),
        ).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print("[DinoV3Embedder] Ready.")

    @torch.no_grad()
    def embed(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        try:
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = self._transform(image).unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
            feats = self.backbone.forward_features(x)
            cls = feats["x_norm_clstoken"].float().squeeze(0)          # (D,)
            return F.normalize(cls, dim=-1).cpu()
        except Exception as exc:
            print(f"[DinoV3Embedder] embed error: {exc}", file=sys.stderr)
            return None

    @torch.no_grad()
    def embed_batch(self, crops: list) -> list:
        results = [None] * len(crops)
        valid_idx, tensors = [], []
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            try:
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensors.append(self._transform(image))
                valid_idx.append(i)
            except Exception:
                pass
        if not tensors:
            return results
        try:
            batch = torch.stack(tensors).to(self.device)          # (N, 3, 224, 224)
            feats = self.backbone.forward_features(batch)
            cls   = F.normalize(feats["x_norm_clstoken"].float(), dim=-1).cpu()  # (N, D)
            for out_i, emb in zip(valid_idx, cls):
                results[out_i] = emb
        except Exception as exc:
            print(f"[DinoV3Embedder] embed_batch error: {exc}", file=sys.stderr)
        return results
"""

# ── factory ───────────────────────────────────────────────────────────────────

def make_embedder(
    name:            str,
    device:          Optional[str] = None,
    dinov3_repo:     str = DINOV3_REPO,
    dinov3_weights:  str = DINOV3_WEIGHTS,
    dinov3_backbone: str = DINOV3_BACKBONE,
) -> EmbedderBase:
    if name == "dinov3":
        return DinoV3Embedder(dinov3_weights, device)
    raise ValueError(f"Unknown embedder {name!r}. Choose 'dinov3'.")


# ── LightGlue descriptor matcher ─────────────────────────────────────────────

class LightGlueDescriptor:
    """
    Gallery keyframe novelty detector using SuperPoint + LightGlue (single ONNX pipeline).

    Returns match_rate ∈ [0, 1]: fraction of the 1024 SuperPoint keypoints in
    crop_a that found a confident correspondence in crop_b.
        High rate → same view  → don't add new gallery entry
        Low rate  → new view   → add new gallery entry

    Model I/O (superpoint_lightglue_pipeline.ort.onnx):
      input:  images    (2, 1, H, W) float32 grayscale [0, 1]
      output: keypoints (2, 1024, 2) int64
              matches   (M, 3)       int64  — [pair_idx, kp_idx0, kp_idx1]
              mscores   (M,)         float32
    """

    MAX_KP = 1024  # fixed by SuperPoint in the pipeline

    def __init__(
        self,
        onnx_path: str = LIGHTGLUE_ONNX,
        device:    str = "cpu",
        img_size:  int = 320,
    ):
        import onnxruntime as ort

        self.img_size = img_size
        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if "cuda" in device else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        print(f"[LightGlueDescriptor] {Path(onnx_path).name}  "
              f"providers={self._sess.get_providers()}")

    def _to_blob(self, crop: np.ndarray) -> np.ndarray:
        """BGR uint8 → (1, 1, H, W) float32 [0, 1]."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.img_size, self.img_size))
        return gray.astype(np.float32)[None, None] / 255.0

    def match_rate(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Returns fraction of keypoints in crop a matched to crop b.
        Returns 1.0 (treat as same view) when either crop is empty.
        """
        if a.size == 0 or b.size == 0:
            return 1.0
        batch = np.concatenate([self._to_blob(a), self._to_blob(b)], axis=0)
        _kpts, matches, _mscores = self._sess.run(None, {"images": batch})
        # matches: (M, 3) — all rows have pair_idx=0 for a single pair
        return float(len(matches)) / self.MAX_KP


# ── gallery ───────────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    global_id:    int
    embedding:    torch.Tensor    # (D,) L2-normalised float32 — for cosine matching
    crop:         np.ndarray      # BGR crop — for LightGlue/SSIM comparison
    frame_idx:    int             # frame when this entry was captured
    crop_path:    str          = field(default="")
    pose_quality: "PoseQuality" = field(default_factory=lambda: PoseQuality.NONE)
    has_face:     bool          = field(default=False)


@dataclass
class FaceEntry:
    global_id: int
    embedding: torch.Tensor   # L2-normalized SFace 128-dim
    face_crop: np.ndarray     # aligned face BGR
    frame_idx: int
    crop_path: str = field(default="")


class ReIDGallery:
    """
    Multi-entry gallery per global_id.

    Entry lifecycle:
    - Added when embedding novelty (cosine sim < emb_novelty_threshold) or pose upgrade detected,
      or when a gid has no entries yet.
    - When entries exceed max_per_gid, the most-central entry is evicted (highest mean pairwise sim).
    - Lost-track entries are kept forever — re-entries always find a match.

    Matching strategies (dispatch via match()):
    - top1    : query vs every entry; best single-entry cosine similarity wins.
    - centroid: query vs per-gid mean embedding; closest centroid wins.
    """

    def __init__(
        self,
        sim_threshold:        float = REID_SIM_THRESHOLD,
        emb_novelty_threshold: float = EMB_NOVELTY_THRESHOLD,
        keyframe_min_interval: int   = KEYFRAME_MIN_INTERVAL,
        match_rate_threshold: float = MATCH_RATE_THRESHOLD,
        max_per_gid:          int   = MAX_GALLERY_PER_GID,
        match_strategy:       str   = MATCH_STRATEGY,
        gallery_dir:          Optional[str] = GALLERY_DUMP_DIR,
        descriptor:           Optional[LightGlueDescriptor] = None,
        iou_gate_threshold:   float = IOU_GATE_THRESHOLD,
    ):
        self.sim_threshold         = sim_threshold
        self.emb_novelty_threshold = emb_novelty_threshold
        self.keyframe_min_interval = keyframe_min_interval
        self.match_rate_threshold  = match_rate_threshold
        self.max_per_gid           = max_per_gid
        self.match_strategy        = match_strategy
        self.iou_gate_threshold    = iou_gate_threshold
        self._descriptor           = descriptor
        self._by_gid: dict[int, list[GalleryEntry]] = defaultdict(list)
        self._last_seen: dict[int, int] = {}
        self._gid_last_bbox: dict[int, np.ndarray] = {}
        # Flat embedding matrix for batched cosine similarity (rebuilt on gallery change)
        self._cache_dirty = True
        self._flat_embs: Optional[torch.Tensor] = None
        self._flat_gids: list[int] = []
        self._centroid_matrix: Optional[torch.Tensor] = None
        self._centroid_gid_list: list[int] = []

        self._dump_dir: Optional[Path] = None
        if gallery_dir:
            self._dump_dir = Path(gallery_dir)
            self._dump_dir.mkdir(parents=True, exist_ok=True)

    # ── private helpers ────────────────────────────────────────────────────

    def _save_crop(self, gid: int, crop: np.ndarray, frame_idx: int) -> str:
        if self._dump_dir is None or crop.size == 0:
            return ""
        path = self._dump_dir / f"gid{gid:04d}_f{frame_idx:07d}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    def _evict_most_central(self, entries: list[GalleryEntry]) -> None:
        """
        Evict the most redundant entry: the one with the highest mean cosine
        similarity to all other entries (most similar to the rest → least unique).
        Uses already-computed L2-normalised embeddings — no extra model calls.
        """
        n = len(entries)
        stack   = torch.stack([e.embedding for e in entries])   # (n, D)
        sim_mat = (stack @ stack.T).numpy()                     # (n, n) pairwise cos-sim
        mean_sims = [(float(sim_mat[i].sum()) - 1.0) / (n - 1) for i in range(n)]
        remove_idx = int(np.argmax(mean_sims))
        removed = entries.pop(remove_idx)
        self._cache_dirty = True
        if removed.crop_path:
            try:
                Path(removed.crop_path).unlink(missing_ok=True)
            except OSError:
                pass
        print(
            f"[Gallery] G{removed.global_id:04d}  evicted f{removed.frame_idx:07d}"
            f"  mean-cos={mean_sims[remove_idx]:.3f}  remaining={len(entries)}"
        )

    def _add_entry(
        self,
        gid:          int,
        embedding:    torch.Tensor,
        crop:         np.ndarray,
        frame_idx:    int,
        pose_quality: "PoseQuality" = None,
        has_face:     bool = False,
    ) -> GalleryEntry:
        if pose_quality is None:
            pose_quality = PoseQuality.NONE
        path  = self._save_crop(gid, crop, frame_idx)
        entry = GalleryEntry(gid, embedding, crop.copy(), frame_idx, path, pose_quality, has_face)
        entries = self._by_gid[gid]
        entries.append(entry)
        self._cache_dirty = True
        if len(entries) > self.max_per_gid:
            self._evict_most_central(entries)
        return entry

    def _rebuild_cache(self) -> None:
        """Rebuild flat embedding matrix and centroid matrix from current gallery state."""
        gids: list[int] = []
        embs: list[torch.Tensor] = []
        c_embs: list[torch.Tensor] = []
        c_gids: list[int] = []
        for gid, entries in self._by_gid.items():
            if not entries:
                continue
            for e in entries:
                gids.append(gid)
                embs.append(e.embedding)
            stack = torch.stack([e.embedding for e in entries])
            c_embs.append(F.normalize(stack.mean(dim=0), dim=-1))
            c_gids.append(gid)
        if embs:
            self._flat_embs = torch.stack(embs)
            self._flat_gids = gids
        else:
            self._flat_embs = None
            self._flat_gids = []
        if c_embs:
            self._centroid_matrix = torch.stack(c_embs)
            self._centroid_gid_list = c_gids
        else:
            self._centroid_matrix = None
            self._centroid_gid_list = []
        self._cache_dirty = False

    # ── public API ─────────────────────────────────────────────────────────

    def touch(self, gid: int, frame_idx: int, bbox: Optional[np.ndarray] = None) -> None:
        """Update last-seen and optionally the gid's last-known bbox."""
        self._last_seen[gid] = frame_idx
        if bbox is not None:
            self._gid_last_bbox[gid] = bbox.copy()

    def check_needs_keyframe(
        self,
        gid:          int,
        crop:         np.ndarray,
        pose_quality: "PoseQuality" = None,
    ) -> bool:
        """
        Cheap pre-filter: should we bother computing a body embedding this frame?
        Uses only the frame-interval gate (O(1)) regardless of whether LightGlue
        is loaded — LightGlue runs later inside update_or_add, not here.
        """
        if pose_quality is None:
            pose_quality = PoseQuality.NONE
        if pose_quality < PoseQuality.UPPER_BODY:
            return False
        if crop.size == 0:
            return False
        entries = self._by_gid.get(gid)
        if not entries:
            return True

        # Trigger 1 — frame-interval gate (cheap dict lookup, no model calls).
        last_kf_frame = max(e.frame_idx for e in entries)
        novel_view = (self._last_seen.get(gid, last_kf_frame) - last_kf_frame) >= self.keyframe_min_interval

        # Trigger 2 — pose coverage upgrade.
        pose_upgrade = pose_quality > max(e.pose_quality for e in entries)

        return novel_view or pose_upgrade

    def update_or_add(
        self,
        gid:          int,
        embedding:    Optional[torch.Tensor],
        crop:         np.ndarray,
        frame_idx:    int,
        bbox:         Optional[np.ndarray] = None,
        pose_quality: "PoseQuality" = None,
        has_face:     bool = False,
    ) -> bool:
        """
        Attempt to add a new keyframe for this gid.  Two independent triggers:

          1. Visual novelty (LightGlue / SSIM): crop is different from all
             existing keyframes.
          2. Pose-coverage upgrade: current frame shows more body than the best
             existing keyframe (e.g. full-body seen for the first time).

        Head-only (NONE quality) is never added — the caller should already
        guard with quality_ok, but this enforces it at the gallery level too.
        Returns True if a new entry was added.
        """
        if pose_quality is None:
            pose_quality = PoseQuality.NONE
        self._last_seen[gid] = frame_idx
        if bbox is not None:
            self._gid_last_bbox[gid] = bbox.copy()
        if embedding is None or crop.size == 0:
            return False
        # Never store head-only crops.
        if pose_quality < PoseQuality.UPPER_BODY:
            return False

        entries = self._by_gid[gid]
        if not entries:
            self._add_entry(gid, embedding, crop, frame_idx, pose_quality, has_face)
            return True

        # Trigger 1 — visual novelty.
        if self._descriptor is not None:
            max_rate   = max(self._descriptor.match_rate(crop, e.crop) for e in entries)
            novel_view = max_rate < self.match_rate_threshold
            sg_label   = f"sg-rate {max_rate:.3f}"
        else:
            stack    = torch.stack([e.embedding for e in entries])
            max_sim  = float(F.cosine_similarity(embedding.unsqueeze(0), stack).max())
            novel_view = max_sim < self.emb_novelty_threshold
            sg_label   = f"emb {max_sim:.3f}"

        # Trigger 2 — pose coverage upgrade.
        best_gallery_quality = max(e.pose_quality for e in entries)
        pose_upgrade = pose_quality > best_gallery_quality

        if novel_view or pose_upgrade:
            reason = []
            if novel_view:    reason.append(sg_label)
            if pose_upgrade:  reason.append(
                f"pose {best_gallery_quality.name}→{pose_quality.name}"
            )
            print(
                f"[Gallery] G{gid:04d}  new keyframe ({', '.join(reason)})"
                f"  total={len(entries) + 1}"
            )
            self._add_entry(gid, embedding, crop, frame_idx, pose_quality, has_face)
            return True

        return False

    # ── GID merge helpers ──────────────────────────────────────────────────

    def merge_gid(self, src_gid: int, dst_gid: int) -> None:
        """Move all body entries from src_gid into dst_gid, evict to cap, remove src."""
        src_entries = self._by_gid.pop(src_gid, [])
        if src_entries:
            dst = self._by_gid[dst_gid]
            dst.extend(src_entries)
            while len(dst) > self.max_per_gid:
                self._evict_most_central(dst)
        keep_frame = max(self._last_seen.get(src_gid, 0), self._last_seen.get(dst_gid, 0))
        self._last_seen.pop(src_gid, None)
        if keep_frame:
            self._last_seen[dst_gid] = keep_frame
        src_bbox = self._gid_last_bbox.pop(src_gid, None)
        if src_bbox is not None and dst_gid not in self._gid_last_bbox:
            self._gid_last_bbox[dst_gid] = src_bbox
        self._cache_dirty = True

    def find_merge_candidate(
        self, new_gid: int, threshold: float
    ) -> Optional[tuple[int, int, float]]:
        """After a new GID is registered, check if it's near-duplicate of an existing GID.
        Returns (new_gid, existing_gid, sim) if centroid similarity >= threshold, else None.
        The caller should merge new_gid → existing_gid (keep the older one)."""
        if self._cache_dirty:
            self._rebuild_cache()
        if self._centroid_matrix is None or len(self._centroid_gid_list) < 2:
            return None
        try:
            idx = self._centroid_gid_list.index(new_gid)
        except ValueError:
            return None
        new_vec = self._centroid_matrix[idx]
        sims = F.cosine_similarity(new_vec.unsqueeze(0), self._centroid_matrix)
        best_sim, best_gid = -1.0, None
        for gid, sim in zip(self._centroid_gid_list, sims.tolist()):
            if gid != new_gid and sim > best_sim:
                best_sim, best_gid = sim, gid
        if best_gid is not None and best_sim >= threshold:
            return (new_gid, best_gid, best_sim)
        return None

    # ── combined face+body matching ────────────────────────────────────────

    def match_with_face(
        self,
        body_emb:     Optional[torch.Tensor],
        face_emb:     Optional[torch.Tensor],
        face_gallery: "FaceGallery",
        exclude_ids:  set,
        strategy:     str,
        query_bbox:   Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, float]]:
        """
        Match using body embedding, face embedding, or both.
        Returns (gid, score) or None. Falls back to body-only match() when face_emb is None.

        Strategies:
          face_top1_dino_top1_avg  — avg(face_top1, body_top1) per gid (default)
          face_body_all_gallery    — avg(face_top1, body_top1) against all entries
          face_body_face_gallery   — avg(face_top1, body_top1) using body entries
                                     captured with face visible (has_face=True)
          face_only_top1           — face similarity only; uses FACE_SIM_THRESHOLD
        """
        if face_emb is None:
            return self.match(body_emb, exclude_ids, query_bbox) if body_emb is not None else None

        effective_exclude = set(exclude_ids)
        if query_bbox is not None and self.iou_gate_threshold > 0.0:
            for gid, last_bbox in self._gid_last_bbox.items():
                if gid not in effective_exclude and bbox_iou(query_bbox, last_bbox) < self.iou_gate_threshold:
                    effective_exclude.add(gid)

        all_gids = set(self._by_gid.keys()) | set(face_gallery._by_gid.keys())

        # Batch-compute face sims and body sims once — avoids N_gids × torch.stack loops.
        face_sim_by_gid = face_gallery.all_top1_sims(face_emb)

        body_sim_by_gid: dict[int, float] = {}
        if body_emb is not None and strategy != "face_only_top1":
            if self._cache_dirty:
                self._rebuild_cache()
            if self._flat_embs is not None:
                flat_sims = F.cosine_similarity(body_emb.unsqueeze(0), self._flat_embs)
                for gid_r, sim in zip(self._flat_gids, flat_sims.tolist()):
                    if sim > body_sim_by_gid.get(gid_r, -1.0):
                        body_sim_by_gid[gid_r] = sim

        best_gid, best_score = None, -1.0

        for gid in all_gids:
            if gid in effective_exclude:
                continue
            body_entries = self._by_gid.get(gid, [])
            face_sim = face_sim_by_gid.get(gid, 0.0)

            if strategy == "face_only_top1":
                score = face_sim

            elif strategy == "face_body_face_gallery":
                # Body: only entries captured when face was visible (small subset — stack directly)
                fb_entries = [e for e in body_entries if e.has_face]
                body_sim = 0.0
                if fb_entries and body_emb is not None:
                    bstack = torch.stack([e.embedding for e in fb_entries])
                    body_sim = float(F.cosine_similarity(body_emb.unsqueeze(0), bstack).max())
                parts = []
                if face_gallery.has_entries(gid):
                    parts.append(face_sim)
                if fb_entries and body_emb is not None:
                    parts.append(body_sim)
                if not parts:
                    continue
                score = sum(parts) / len(parts)

            else:  # face_body_all_gallery and face_top1_dino_top1_avg
                body_sim = body_sim_by_gid.get(gid, 0.0)
                parts = []
                if face_gallery.has_entries(gid):
                    parts.append(face_sim)
                if body_entries and body_emb is not None:
                    parts.append(body_sim)
                if not parts:
                    continue
                score = sum(parts) / len(parts)

            if score > best_score:
                best_score, best_gid = score, gid

        if best_gid is None:
            return None
        threshold = face_gallery.sim_threshold if strategy == "face_only_top1" else self.sim_threshold
        return (best_gid, best_score) if best_score >= threshold else None

    # ── matching strategies ────────────────────────────────────────────────

    def _match_top1(
        self, embedding: torch.Tensor, exclude_ids: set[int]
    ) -> Optional[tuple[int, float]]:
        """top1 via pre-built flat matrix — one batched matmul instead of N_gids loops."""
        if self._cache_dirty:
            self._rebuild_cache()
        if self._flat_embs is None:
            return None
        sims = F.cosine_similarity(embedding.unsqueeze(0), self._flat_embs)  # (N_total,)
        per_gid: dict[int, float] = {}
        for gid, sim in zip(self._flat_gids, sims.tolist()):
            if gid not in exclude_ids and sim > per_gid.get(gid, -1.0):
                per_gid[gid] = sim
        if not per_gid:
            return None
        best_gid = max(per_gid, key=per_gid.__getitem__)
        best_sim = per_gid[best_gid]
        return (best_gid, best_sim) if best_sim >= self.sim_threshold else None

    def _match_centroid(
        self, embedding: torch.Tensor, exclude_ids: set[int]
    ) -> Optional[tuple[int, float]]:
        """centroid via pre-built centroid matrix — one batched matmul."""
        if self._cache_dirty:
            self._rebuild_cache()
        if self._centroid_matrix is None:
            return None
        sims = F.cosine_similarity(embedding.unsqueeze(0), self._centroid_matrix)  # (N_gids,)
        best_gid: Optional[int] = None
        best_sim = -1.0
        for gid, sim in zip(self._centroid_gid_list, sims.tolist()):
            if gid not in exclude_ids and sim > best_sim:
                best_sim = sim
                best_gid = gid
        return (best_gid, best_sim) if best_gid is not None and best_sim >= self.sim_threshold else None

    def match(
        self,
        embedding:   torch.Tensor,
        exclude_ids: set[int],
        query_bbox:  Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, float]]:
        """Dispatch to matching strategy with optional IoU gate. Returns (gid, score) or None."""
        effective_exclude = set(exclude_ids)
        if query_bbox is not None and self.iou_gate_threshold > 0.0:
            for gid, last_bbox in self._gid_last_bbox.items():
                if gid not in effective_exclude and bbox_iou(query_bbox, last_bbox) < self.iou_gate_threshold:
                    effective_exclude.add(gid)
        if self.match_strategy == "centroid":
            return self._match_centroid(embedding, effective_exclude)
        return self._match_top1(embedding, effective_exclude)


# ── face gallery ──────────────────────────────────────────────────────────────

class FaceGallery:
    """
    Per-GID face gallery backed by SFace embeddings.
    A new face keyframe is added when the best cosine similarity to existing
    entries falls below keyframe_threshold (i.e. a visually different face view).
    """

    def __init__(
        self,
        keyframe_threshold: float = FACE_KEYFRAME_THRESHOLD,
        sim_threshold:      float = FACE_SIM_THRESHOLD,
        max_per_gid:        int   = MAX_FACE_GALLERY_PER_GID,
        gallery_dir:        Optional[str] = None,
    ):
        self.keyframe_threshold = keyframe_threshold
        self.sim_threshold      = sim_threshold
        self.max_per_gid        = max_per_gid
        self._by_gid: dict[int, list[FaceEntry]] = defaultdict(list)
        self._dump_dir: Optional[Path] = None
        if gallery_dir:
            self._dump_dir = Path(gallery_dir) / "face"
            self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dirty = True
        self._flat_embs: Optional[torch.Tensor] = None
        self._flat_gids: list[int] = []

    def _save_crop(self, gid: int, face_crop: np.ndarray, frame_idx: int) -> str:
        if self._dump_dir is None or face_crop.size == 0:
            return ""
        path = self._dump_dir / f"gid{gid:04d}_f{frame_idx:07d}.jpg"
        cv2.imwrite(str(path), face_crop)
        return str(path)

    def update_or_add(
        self,
        gid:       int,
        emb:       torch.Tensor,
        face_crop: np.ndarray,
        frame_idx: int,
    ) -> tuple[bool, float]:
        """Returns (added, sim) where sim is the best cosine similarity
        against existing entries (-1.0 if this is the first entry)."""
        entries = self._by_gid[gid]
        if not entries:
            path = self._save_crop(gid, face_crop, frame_idx)
            entries.append(FaceEntry(gid, emb, face_crop.copy(), frame_idx, path))
            self._cache_dirty = True
            print(f"[FaceGallery] G{gid:04d} f{frame_idx}: first face keyframe", flush=True)
            return True, -1.0
        stack   = torch.stack([e.embedding for e in entries])
        max_sim = float(F.cosine_similarity(emb.unsqueeze(0), stack).max())
        if max_sim >= self.keyframe_threshold:
            return False, max_sim   # near-identical — skip but still return the sim
        # Evict most-central (most redundant) entry if at capacity
        if len(entries) >= self.max_per_gid:
            n = len(entries)
            sim_mat   = (stack @ stack.T).numpy()
            mean_sims = [(float(sim_mat[i].sum()) - 1.0) / max(n - 1, 1) for i in range(n)]
            removed = entries.pop(int(np.argmax(mean_sims)))
            if removed.crop_path:
                try:
                    Path(removed.crop_path).unlink(missing_ok=True)
                except OSError:
                    pass
        path = self._save_crop(gid, face_crop, frame_idx)
        entries.append(FaceEntry(gid, emb, face_crop.copy(), frame_idx, path))
        self._cache_dirty = True
        print(f"[FaceGallery] G{gid:04d} f{frame_idx}: new face keyframe  sim={max_sim:.3f}  total={len(entries)}", flush=True)
        return True, max_sim

    def has_entries(self, gid: int) -> bool:
        return bool(self._by_gid.get(gid))

    def _rebuild_cache(self) -> None:
        gids, embs = [], []
        for gid, entries in self._by_gid.items():
            for e in entries:
                gids.append(gid)
                embs.append(e.embedding)
        if embs:
            self._flat_embs = torch.stack(embs)
            self._flat_gids = gids
        else:
            self._flat_embs = None
            self._flat_gids = []
        self._cache_dirty = False

    def all_top1_sims(self, emb: torch.Tensor) -> dict[int, float]:
        """Returns {gid: max_cosine_sim} for all GIDs with face entries — one batched matmul."""
        if self._cache_dirty:
            self._rebuild_cache()
        if self._flat_embs is None:
            return {}
        sims = F.cosine_similarity(emb.unsqueeze(0), self._flat_embs)
        result: dict[int, float] = {}
        for gid, sim in zip(self._flat_gids, sims.tolist()):
            if sim > result.get(gid, -1.0):
                result[gid] = sim
        return result

    def top1_sim(self, emb: torch.Tensor, gid: int) -> float:
        """Max cosine similarity of emb against all entries for gid (0 if none)."""
        entries = self._by_gid.get(gid)
        if not entries:
            return 0.0
        stack = torch.stack([e.embedding for e in entries])
        return float(F.cosine_similarity(emb.unsqueeze(0), stack).max())

    def merge_gid(self, src_gid: int, dst_gid: int) -> None:
        """Move all face entries from src_gid into dst_gid, evict to cap, remove src."""
        src_entries = self._by_gid.pop(src_gid, [])
        if src_entries:
            dst = self._by_gid[dst_gid]
            dst.extend(src_entries)
            while len(dst) > self.max_per_gid:
                n = len(dst)
                stack    = torch.stack([e.embedding for e in dst])
                sim_mat  = (stack @ stack.T).numpy()
                mean_sims = [(float(sim_mat[i].sum()) - 1.0) / max(n - 1, 1) for i in range(n)]
                removed = dst.pop(int(np.argmax(mean_sims)))
                if removed.crop_path:
                    try:
                        Path(removed.crop_path).unlink(missing_ok=True)
                    except OSError:
                        pass
        keep_frame = max(self._last_seen.get(src_gid, 0), self._last_seen.get(dst_gid, 0))
        self._last_seen.pop(src_gid, None)
        if keep_frame:
            self._last_seen[dst_gid] = keep_frame
        self._cache_dirty = True


# ── RT-DETR detector ──────────────────────────────────────────────────────────
"""
class RTDetrDetector:

    def __init__(self, model_id: str, device: str, confidence: float):
        self.device = device
        self.confidence = confidence
        print(f"[RTDetrDetector] Loading {model_id} on {device} ...")
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.person_class_id = self._find_person_class()
        print(f"[RTDetrDetector] Ready. Person class id = {self.person_class_id}")

    def _find_person_class(self) -> int:
        for k, v in self.model.config.id2label.items():
            if v.lower() == "person":
                return int(k)
        return 0

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> sv.Detections:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]).to(self.device),
            threshold=self.confidence,
        )[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        mask = labels == self.person_class_id
        if not mask.any():
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
            )
        return sv.Detections(
            xyxy=boxes[mask],
            confidence=scores[mask],
            class_id=labels[mask].astype(int),
        )
"""
class RTDetrDetector:
    """Wraps an ONNX RT-DETR model; returns only person detections."""

    def __init__(self, model_id: str, device: str, confidence: float):
        """
        Args:
            model_id (str): Path to your local '.onnx' model file.
            device (str): Execution device string ('cuda' or 'cpu').
            confidence (float): Bounding box confidence threshold.
        """
        self.confidence = confidence
        
        # 1. Map PyTorch device string to ONNX Execution Providers
        if "cuda" in device.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"[RTDetrDetector] Loading ONNX model from {model_id} on {providers[0]} ...")
        self.session = ort.InferenceSession(model_id, providers=providers)

        # 2. Load the companion config.json (assumed to be near the ONNX file or in working dir)
        config_path = os.path.join(os.path.dirname(model_id), "config.json")
        if not os.path.exists(config_path):
            config_path = "config.json"  # Fallback to local directory

        with open(config_path, "r") as f:
            config_data = json.load(f)
        self.id2label = config_data.get("id2label", {})

        # 3. Find target tracking ID
        self.person_class_id = self._find_person_class()
        print(f"[RTDetrDetector] Ready. Person class id = {self.person_class_id}")

    def _find_person_class(self) -> int:
        for k, v in self.id2label.items():
            if v.lower() == "person":
                return int(k)
        return 0

    def detect(self, frame: np.ndarray) -> sv.Detections:
        orig_h, orig_w, _ = frame.shape

        # 1. Manual Preprocessing (replaces RTDetrImageProcessor)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Scale to [0, 1], transpose to CHW, and expand to batch dimension (1, 3, 640, 640)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 2. Run ONNX Inference
        logits, pred_boxes = self.session.run(["logits", "pred_boxes"], {"pixel_values": input_tensor})

        # Drop batch dimension -> Shape: (num_queries, classes), (num_queries, 4)
        logits = logits[0]
        pred_boxes = pred_boxes[0]

        # 3. Post-Processing Vectorized Logic (replaces post_process_object_detection)
        scores = 1 / (1 + np.exp(-logits))  # Sigmoid element-wise
        class_ids = np.argmax(scores, axis=-1)
        confidences = np.max(scores, axis=-1)

        # Mask logic: filter by confidence AND strict class selection matching 'person'
        mask = (confidences > self.confidence) & (class_ids == self.person_class_id)

        if not mask.any():
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
            )

        # Filter matrices natively with the mask
        filtered_boxes = pred_boxes[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # 4. Box transformation coordinates [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
        cx, cy, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
        
        xmin = (cx - 0.5 * w) * orig_w
        ymin = (cy - 0.5 * h) * orig_h
        xmax = (cx + 0.5 * w) * orig_w
        ymax = (cy + 0.5 * h) * orig_h

        # Clip arrays inside boundary limits
        xmin = np.clip(xmin, 0, orig_w)
        ymin = np.clip(ymin, 0, orig_h)
        xmax = np.clip(xmax, 0, orig_w)
        ymax = np.clip(ymax, 0, orig_h)

        # Re-stack columns into a structured (N, 4) array
        xyxy = np.stack([xmin, ymin, xmax, ymax], axis=-1)

        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=filtered_confidences.astype(np.float32),
            class_id=filtered_class_ids.astype(int),
        )

# ── ReID system ───────────────────────────────────────────────────────────────

class ReIDSystem:
    """
    Per-frame pipeline:
      1. Detect persons with RT-DETR
      2. Track with ByteTrack  →  short-lived tracker_id
      3. Embed each crop via the injected EmbedderBase strategy
      4. Map tracker_id → global_id via gallery cosine-similarity re-ID
    """

    def __init__(
        self,
        embedder:             EmbedderBase,
        rtdetr_model:         str   = RTDETR_MODEL,
        confidence:           float = DETECT_CONFIDENCE,
        similarity_threshold: float = REID_SIM_THRESHOLD,
        match_rate_threshold: float = MATCH_RATE_THRESHOLD,
        max_gallery_per_gid:  int   = MAX_GALLERY_PER_GID,
        match_strategy:       str   = MATCH_STRATEGY,
        gallery_dir:          Optional[str] = GALLERY_DUMP_DIR,
        lightglue_onnx:       Optional[str] = None,
        device:               Optional[str] = None,
        iou_gate_threshold:   float = IOU_GATE_THRESHOLD,
        stable_iou_threshold: float = STABLE_IOU_THRESHOLD,
        stable_min_frames:    int   = STABLE_MIN_FRAMES,
        pose_checker:           Optional[RTMOPoseChecker] = None,
        pose_min_quality:       str   = POSE_MIN_QUALITY,
        pending_confirm_frames: int   = PENDING_CONFIRM_FRAMES,
        face_embedder:          Optional[FaceEmbedder]    = None,
        face_match_strategy:      str   = FACE_MATCH_STRATEGY,
        max_face_gallery_per_gid: int   = MAX_FACE_GALLERY_PER_GID,
        face_keyframe_threshold:  float = FACE_KEYFRAME_THRESHOLD,
        gid_merge_threshold:      float = GID_MERGE_THRESHOLD,
        _reid_only:               bool  = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        if not _reid_only:
            self.detector = RTDetrDetector(rtdetr_model, self.device, confidence)
            self.tracker  = sv.ByteTrack()
        else:
            self.detector = None  # type: ignore[assignment]
            self.tracker  = None  # type: ignore[assignment]

        # Build LightGlue descriptor for gallery keyframe novelty detection.
        descriptor = None
        onnx_path = lightglue_onnx if lightglue_onnx is not None else LIGHTGLUE_ONNX
        try:
            descriptor = LightGlueDescriptor(onnx_path=onnx_path, device=self.device)
        except Exception as exc:
            print(f"[ReIDSystem] LightGlue load failed: {exc}\n"
                  "  → gallery trigger will use embedding novelty fallback.",
                  file=sys.stderr)

        self.gallery  = ReIDGallery(
            sim_threshold        = similarity_threshold,
            match_rate_threshold = match_rate_threshold,
            max_per_gid          = max_gallery_per_gid,
            match_strategy       = match_strategy,
            gallery_dir          = gallery_dir,
            descriptor           = descriptor,
            iou_gate_threshold   = iou_gate_threshold,
        )

        self.stable_iou_threshold = stable_iou_threshold
        self.stable_min_frames    = stable_min_frames

        self._pose_checker = pose_checker
        _qmap = {
            "full_body":  PoseQuality.FULL_BODY,
            "upper_body": PoseQuality.UPPER_BODY,
            "none":       PoseQuality.NONE,
        }
        self._pose_min_quality = _qmap.get(pose_min_quality, PoseQuality.UPPER_BODY)

        self.pending_confirm_frames = pending_confirm_frames

        self._face_embedder       = face_embedder
        self._face_match_strategy = face_match_strategy
        self.face_gallery         = FaceGallery(
            keyframe_threshold = face_keyframe_threshold,
            max_per_gid        = max_face_gallery_per_gid,
            gallery_dir        = gallery_dir,
        )

        self._gid_merge_threshold = gid_merge_threshold

        self._track_to_global:       dict[int, int]                      = {}
        self._track_stable_count:    dict[int, int]                      = {}
        self._track_prev_bbox:       dict[int, np.ndarray]               = {}
        self._pending_tracks:        dict[int, list]                     = {}
        self._pending_face_latest:   dict[int, tuple[np.ndarray, torch.Tensor]] = {}
        # tid → (confidence, match_type) where type is "B", "F", "F+B", or "NEW"
        self._track_match_info:      dict[int, tuple[float, str]]        = {}
        # tid → latest best cosine-sim of detected face vs face gallery (-1 = first entry)
        self._track_face_gallery_sim: dict[int, float]                   = {}
        # tid → cached pose quality (avoids RTMO on frames where all tracks are stable)
        self._track_pose_quality:     dict[int, PoseQuality]             = {}
        self._global_counter = 0
        self._frame_idx = 0

        self._box_ann   = sv.BoxAnnotator()
        self._label_ann = sv.LabelAnnotator()

    def _new_global_id(self) -> int:
        gid = self._global_counter
        self._global_counter += 1
        return gid

    def _apply_gid_merge(self, new_gid: int) -> Optional[int]:
        """If new_gid is a near-duplicate of an existing GID, merge and return the kept GID.
        Returns kept_gid if a merge happened, else None."""
        candidate = self.gallery.find_merge_candidate(new_gid, self._gid_merge_threshold)
        if candidate is None:
            return None
        src_gid, dst_gid, sim = candidate
        self.gallery.merge_gid(src_gid, dst_gid)
        self.face_gallery.merge_gid(src_gid, dst_gid)
        # Remap any active tracks still holding src_gid
        for tid in list(self._track_to_global):
            if self._track_to_global[tid] == src_gid:
                self._track_to_global[tid] = dst_gid
        print(
            f"[merge] gid{src_gid} → gid{dst_gid}  centroid_sim={sim:.3f}"
            f"  f{self._frame_idx}", flush=True
        )
        return dst_gid

    def _crop(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Run RT-DETR + ByteTrack. Meant to run in a dedicated detection thread."""
        detections = self.detector.detect(frame)
        return self.tracker.update_with_detections(detections)

    def process_detections(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Run the full ReID loop on pre-computed detections.
        Returns (annotated_frame, {tracker_id: global_id}).
        """

        tracker_to_global: dict[int, int] = {}
        if len(detections) == 0:
            self._frame_idx += 1
            return frame.copy(), tracker_to_global

        _t0 = time.perf_counter()

        # Pose gate — skip RTMO entirely when every detection is a known-stable track
        # with a cached quality value (the common case after warm-up).
        if self._pose_checker is not None and self._pose_min_quality > PoseQuality.NONE:
            need_rtmo = any(
                int(t) not in self._track_to_global
                or self._track_stable_count.get(int(t), 0) < self.stable_min_frames
                or int(t) not in self._track_pose_quality
                for t in detections.tracker_id
            )
            if need_rtmo:
                pose_qualities = self._pose_checker.check_frame(frame, detections.xyxy)
                for _i, _t in enumerate(detections.tracker_id):
                    self._track_pose_quality[int(_t)] = pose_qualities[_i]
            else:
                pose_qualities = [self._track_pose_quality[int(t)] for t in detections.tracker_id]
        else:
            pose_qualities = [PoseQuality.FULL_BODY] * len(detections)

        _t1 = time.perf_counter()

        # Face detection — run YuNet on the FULL frame once.
        frame_faces = (
            self._face_embedder.detect_frame(frame) if self._face_embedder else []
        )

        _t2 = time.perf_counter()

        # Pre-populate with GIDs already held by known tracks visible this frame.
        # Without this, a new track processed early in the loop can steal a GID
        # from a known track that appears later in the detection list.
        active_gids: set[int] = {
            self._track_to_global[int(t)]
            for t in detections.tracker_id
            if int(t) in self._track_to_global
        }
        face_detected:    dict[int, bool]              = {}
        face_bboxes_frame: dict[int, tuple[int, ...]]  = {}

        # ── Pre-scan: crop all detections + collect embedding needs ──────────
        crop_cache:  dict[int, np.ndarray]           = {}
        embed_tids:  list[int]                       = []
        embed_crops: list[np.ndarray]                = []

        for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            tid_p = int(tid)
            pq_p  = pose_qualities[i]
            crop  = self._crop(frame, xyxy)
            crop_cache[tid_p] = crop
            if pq_p >= self._pose_min_quality:
                if tid_p in self._track_to_global:
                    gid_p = self._track_to_global[tid_p]
                    if self.gallery.check_needs_keyframe(gid_p, crop, pose_quality=pq_p):
                        embed_tids.append(tid_p)
                        embed_crops.append(crop)
                else:
                    embed_tids.append(tid_p)
                    embed_crops.append(crop)

        _t3 = time.perf_counter()

        # Single batched DINOv3 forward for all crops that need embedding this frame.
        tid_to_emb: dict[int, Optional[torch.Tensor]] = (
            dict(zip(embed_tids, self.embedder.embed_batch(embed_crops)))
            if embed_crops else {}
        )

        _t4 = time.perf_counter()

        # ── Main loop: gallery + face updates using pre-computed embeddings ──
        # pending_completions: tracks whose buffer just filled — ranked before assignment
        # Each entry: (prescore, tid, mean_emb, latest_face, crop, pq, has_face, xyxy, match_type)
        pending_completions: list[tuple] = []
        for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            tid  = int(tid)
            pq   = pose_qualities[i]
            crop = crop_cache[tid]

            if tid in self._track_to_global:
                gid = self._track_to_global[tid]

                # Stability gate: count consecutive frames where bbox barely moved.
                prev_bbox = self._track_prev_bbox.get(tid)
                if prev_bbox is not None and bbox_iou(xyxy, prev_bbox) >= self.stable_iou_threshold:
                    self._track_stable_count[tid] = self._track_stable_count.get(tid, 0) + 1
                else:
                    self._track_stable_count[tid] = 0

                # Face update.
                if self._face_embedder:
                    face_res = self._face_embedder.pick_for_person(frame_faces, xyxy)
                    if face_res is not None:
                        fx, fy, fw, fh = face_res[2]
                        face_bboxes_frame[tid] = (fx, fy, fx+fw, fy+fh)
                        _, face_sim = self.face_gallery.update_or_add(
                            gid, face_res[1], face_res[0], self._frame_idx
                        )
                        self._track_face_gallery_sim[tid] = face_sim
                        face_detected[tid] = True
                    else:
                        face_detected[tid] = False
                else:
                    face_detected[tid] = False

                # Body gallery update: use pre-computed embedding (None → touch only).
                emb = tid_to_emb.get(tid)
                if emb is not None:
                    has_face = face_detected.get(tid, False)
                    self.gallery.update_or_add(
                        gid, emb, crop, self._frame_idx,
                        bbox=xyxy, pose_quality=pq, has_face=has_face,
                    )
                else:
                    self.gallery.touch(gid, self._frame_idx, bbox=xyxy)

                self._track_prev_bbox[tid] = xyxy.copy()
                tracker_to_global[tid] = gid

            else:
                # New tracker_id — record bbox, then run face+body detection.
                self._track_prev_bbox[tid] = xyxy.copy()

                face_result = self._face_embedder.pick_for_person(frame_faces, xyxy) if self._face_embedder else None
                has_face = face_result is not None
                face_detected[tid] = has_face
                if face_result is not None:
                    fx, fy, fw, fh = face_result[2]
                    face_bboxes_frame[tid] = (fx, fy, fx+fw, fy+fh)

                emb = tid_to_emb.get(tid)  # pre-computed, or None if quality_ok was False

                # Gate: need at least body quality or a detected face to proceed
                if emb is None and not has_face:
                    continue

                # ── Face-only case: no body skeleton but face detected ─────────
                if emb is None and has_face:
                    face_emb  = face_result[1]
                    face_crop = face_result[0]
                    result = self.gallery.match_with_face(
                        None, face_emb, self.face_gallery,
                        active_gids, "face_only_top1", query_bbox=xyxy,
                    )
                    if result is not None:
                        gid, score = result
                        self._track_match_info[tid] = (score, "F")
                    else:
                        gid = self._new_global_id()
                        self._track_match_info[tid] = (0.0, "NEW")
                    self._track_to_global[tid] = gid
                    _, face_sim_fo = self.face_gallery.update_or_add(gid, face_emb, face_crop, self._frame_idx)
                    self._track_face_gallery_sim[tid] = face_sim_fo
                    self.gallery.touch(gid, self._frame_idx, bbox=xyxy)
                    active_gids.add(gid)
                    tracker_to_global[tid] = gid
                    continue

                # ── Body (+ optional face) pending buffer ────────────────────
                buf = self._pending_tracks.setdefault(tid, [])
                if emb is not None:
                    buf.append(emb)
                if face_result is not None:
                    self._pending_face_latest[tid] = face_result  # (face_crop, face_emb)

                if len(buf) < self.pending_confirm_frames:
                    continue  # still accumulating

                mean_emb    = F.normalize(torch.stack(buf).mean(dim=0), dim=-1)
                latest_face = self._pending_face_latest.get(tid)
                face_emb_m  = latest_face[1] if latest_face is not None else None

                # Pre-score against current active_gids for ranking.
                # Multiple completions this frame are assigned in score-descending order
                # so the most-confident match wins when two tracks compete for the same GID.
                if face_emb_m is not None:
                    prescore_result = self.gallery.match_with_face(
                        mean_emb, face_emb_m, self.face_gallery,
                        active_gids, self._face_match_strategy, query_bbox=xyxy,
                    )
                    match_type = "F+B"
                else:
                    prescore_result = self.gallery.match(mean_emb, active_gids, query_bbox=xyxy)
                    match_type = "B"

                prescore = prescore_result[1] if prescore_result is not None else 0.0
                pending_completions.append(
                    (prescore, tid, mean_emb, latest_face, crop, pq, has_face, xyxy, match_type)
                )
                del self._pending_tracks[tid]
                self._pending_face_latest.pop(tid, None)

        # ── Rank pending completions by match score and assign greedily ──────
        # Highest-confidence re-IDs resolve first; ties become new GIDs.
        pending_completions.sort(key=lambda x: -x[0])
        for prescore, tid, mean_emb, latest_face, crop, pq, has_face, xyxy, match_type in pending_completions:
            face_emb_m = latest_face[1] if latest_face is not None else None
            # Re-run match with updated active_gids (may have grown since prescore).
            if face_emb_m is not None:
                result = self.gallery.match_with_face(
                    mean_emb, face_emb_m, self.face_gallery,
                    active_gids, self._face_match_strategy, query_bbox=xyxy,
                )
            else:
                result = self.gallery.match(mean_emb, active_gids, query_bbox=xyxy)

            is_new_gid = False
            if result is not None:
                gid, score = result
                self._track_match_info[tid] = (score, match_type)
            else:
                gid = self._new_global_id()
                self._track_match_info[tid] = (0.0, "NEW")
                is_new_gid = True

            self._track_to_global[tid] = gid
            self.gallery.update_or_add(
                gid, mean_emb, crop, self._frame_idx,
                bbox=xyxy, pose_quality=pq, has_face=has_face,
            )
            if latest_face is not None:
                _, face_sim_pb = self.face_gallery.update_or_add(
                    gid, face_emb_m, latest_face[0], self._frame_idx
                )
                self._track_face_gallery_sim[tid] = face_sim_pb

            # After adding the first gallery entry, check if this new GID is a
            # near-duplicate of an existing one (person reappeared after occlusion
            # and got a wrong new GID instead of re-IDing).
            if is_new_gid and self._gid_merge_threshold < 1.0:
                merged_into = self._apply_gid_merge(gid)
                if merged_into is not None:
                    gid = merged_into
                    self._track_to_global[tid] = gid
                    self._track_match_info[tid] = (self._gid_merge_threshold, "MERGE")

            active_gids.add(gid)
            tracker_to_global[tid] = gid

        # Drop stale pending buffers for tracks ByteTrack no longer reports.
        active_tids = {int(t) for t in detections.tracker_id}
        for t in [t for t in list(self._pending_tracks) if t not in active_tids]:
            del self._pending_tracks[t]
            self._pending_face_latest.pop(t, None)

        _t5 = time.perf_counter()

        self._frame_idx += 1

        _QL = {PoseQuality.FULL_BODY: "F", PoseQuality.UPPER_BODY: "U", PoseQuality.NONE: "?"}
        labels = []
        for i, tid in enumerate(detections.tracker_id):
            tid_i    = int(tid)
            face_tag = "f" if face_detected.get(tid_i, False) else ""
            if tid_i in tracker_to_global:
                info = self._track_match_info.get(tid_i)
                if info is not None:
                    conf, mtype = info
                    match_tag = f" [{mtype} {conf:.2f}]" if mtype != "NEW" else " [NEW]"
                else:
                    match_tag = ""
                face_sim_v = self._track_face_gallery_sim.get(tid_i, -2.0)
                face_sim_tag = f" fs:{face_sim_v:.2f}" if face_sim_v >= -1.0 else ""
                labels.append(
                    f"G{tracker_to_global[tid_i]} T{tid_i} {_QL[pose_qualities[i]]}{face_tag}{match_tag}{face_sim_tag}"
                )
            elif tid_i in self._pending_tracks:
                n = len(self._pending_tracks[tid_i])
                labels.append(f"buf{n}/{self.pending_confirm_frames} T{tid_i}{face_tag}")
            else:
                labels.append(f"? T{tid_i}")
        annotated = self._box_ann.annotate(frame.copy(), detections)
        annotated = self._label_ann.annotate(annotated, detections, labels=labels)
        for tid_i, (fx1, fy1, fx2, fy2) in face_bboxes_frame.items():
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            sim_v = self._track_face_gallery_sim.get(tid_i, -2.0)
            if sim_v >= -1.0:
                label_txt = f"fs:{sim_v:.2f}" if sim_v >= 0 else "fs:new"
                cv2.putText(annotated, label_txt, (fx1, max(0, fy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        _t6 = time.perf_counter()

        # ── Per-stage timing (printed every 30 frames) ──────────────────────
        if self._frame_idx % 30 == 0:
            n   = len(detections)
            emb = len(embed_crops)
            print(
                f"[timing f{self._frame_idx:06d} n={n} emb={emb}]"
                f"  rtmo={(_t1-_t0)*1e3:5.1f}ms"
                f"  yunet={(_t2-_t1)*1e3:5.1f}ms"
                f"  prescan={(_t3-_t2)*1e3:5.1f}ms"
                f"  dino={(_t4-_t3)*1e3:5.1f}ms"
                f"  loop={(_t5-_t4)*1e3:5.1f}ms"
                f"  annot={(_t6-_t5)*1e3:5.1f}ms"
                f"  total={(_t6-_t0)*1e3:5.1f}ms",
                flush=True,
            )

        return annotated, tracker_to_global

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
        """Single-threaded convenience wrapper (detect + reid in one call)."""
        return self.process_detections(frame, self.detect_and_track(frame))

    def find_reference(
        self,
        ref_crop: np.ndarray,
        active_tracker_to_global: dict[int, int],
    ) -> tuple[int, int, float, str]:
        """
        Match a reference image crop against the current gallery and active tracks.

        Returns (global_id, track_id, score, method):
          "track_gid"  — gallery matched a GID and an active track holds that GID
          "gid_only"   — gallery matched a GID but no active track has it → track_id=-1
          "direct"     — no gallery match; best active track by direct DINOv3+face score → global_id=-1
          "none"       — no gallery match and no active tracks → both -1

        Parameters
        ----------
        ref_crop                 : BGR image of the reference person (full crop)
        active_tracker_to_global : {tracker_id: global_id} from the latest processed frame
        """
        # ── Embed reference image ────────────────────────────────────────────
        ref_emb = self.embedder.embed(ref_crop)

        ref_face_emb: Optional[torch.Tensor] = None
        if self._face_embedder is not None:
            frame_faces = self._face_embedder.detect_frame(ref_crop)
            if frame_faces:
                h, w = ref_crop.shape[:2]
                face_res = self._face_embedder.pick_for_person(
                    frame_faces, np.array([0, 0, w, h], dtype=np.float32)
                )
                if face_res is not None:
                    ref_face_emb = face_res[1]

        # ── Step 1: Gallery matching (searches all known GIDs) ───────────────
        gid_result: Optional[tuple[int, float]] = None
        if ref_emb is not None:
            if ref_face_emb is not None:
                gid_result = self.gallery.match_with_face(
                    ref_emb, ref_face_emb, self.face_gallery,
                    set(), self._face_match_strategy,
                )
            else:
                gid_result = self.gallery.match(ref_emb, set())

        # ── Step 2: Resolve GID → active track ──────────────────────────────
        if gid_result is not None:
            best_gid, score = gid_result
            gid_to_tid = {gid: tid for tid, gid in active_tracker_to_global.items()}
            active_tid = gid_to_tid.get(best_gid, -1)
            if active_tid != -1:
                return (best_gid, active_tid, score, "track_gid")
            return (best_gid, -1, score, "gid_only")

        # ── Step 3: No gallery match — direct score against active tracks ────
        if ref_emb is None:
            return (-1, -1, 0.0, "none")

        if not active_tracker_to_global and not self._pending_tracks:
            return (-1, -1, 0.0, "none")

        best_tid, best_score = -1, -1.0

        # Confirmed tracks: score via their GID's gallery entries
        for tid, gid in active_tracker_to_global.items():
            entries = self.gallery._by_gid.get(gid, [])
            if not entries:
                continue
            stack = torch.stack([e.embedding for e in entries])
            body_sim = float(F.cosine_similarity(ref_emb.unsqueeze(0), stack).max())

            if ref_face_emb is not None and self.face_gallery.has_entries(gid):
                face_sim = self.face_gallery.top1_sim(ref_face_emb, gid)
                combined = (body_sim + face_sim) / 2.0
            else:
                combined = body_sim

            if combined > best_score:
                best_score, best_tid = combined, tid

        # Pending tracks: use mean of buffered embeddings (identity switch possible)
        for tid, buf in self._pending_tracks.items():
            if not buf:
                continue
            mean_emb = F.normalize(torch.stack(buf).mean(dim=0), dim=-1)
            sim = float(F.cosine_similarity(ref_emb.unsqueeze(0), mean_emb.unsqueeze(0)).item())
            if sim > best_score:
                best_score, best_tid = sim, tid

        return (-1, best_tid, best_score, "direct")


# ── video runner ──────────────────────────────────────────────────────────────

def run_video(
    source,
    embedder:    EmbedderBase,
    output:      Optional[str] = None,
    show:        bool = True,
    **reid_kwargs,
) -> None:
    system = ReIDSystem(embedder=embedder, **reid_kwargs)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    writer = None
    if output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))

    # Detection thread: RT-DETR + ByteTrack run independently of the ReID loop.
    # maxsize=2 keeps at most one frame buffered; if ReID is slower the detector waits.
    det_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def _detection_worker() -> None:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                det_queue.put(None)
                return
            dets = system.detect_and_track(frame)
            det_queue.put((frame, dets))

    det_thread = threading.Thread(target=_detection_worker, daemon=True)
    det_thread.start()

    try:
        while True:
            item = det_queue.get()
            if item is None:
                break
            frame, detections = item
            t0 = time.perf_counter()
            annotated, t2g = system.process_detections(frame, detections)
            fps_val = 1.0 / max(time.perf_counter() - t0, 1e-6)
            cv2.putText(
                annotated,
                f"{fps_val:.1f} FPS  persons: {len(t2g)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("annot", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        det_thread.join(timeout=2.0)
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ReID: RT-DETR + ByteTrack + DINOv3 appearance embedder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("source", nargs="?", default="0",
                    help="Video path or camera index")
    ap.add_argument("--output", "-o",
                    help="Save annotated video to this path")
    ap.add_argument("--no-show", action="store_true",
                    help="Skip display window")
    ap.add_argument("--device", default=None,
                    help="Torch device, e.g. cuda / cpu / cuda:1")

    # ── detection / tracking
    ap.add_argument("--rtdetr-model",  default=RTDETR_MODEL)
    ap.add_argument("--confidence",    type=float, default=DETECT_CONFIDENCE,
                    help="Detection confidence threshold")
    ap.add_argument("--similarity",      type=float, default=REID_SIM_THRESHOLD,
                    help="Cosine similarity threshold for re-ID match")
    ap.add_argument("--lightglue-onnx",        default=LIGHTGLUE_ONNX,
                    help="Path to superpoint_lightglue_pipeline.ort.onnx")
    ap.add_argument("--match-rate-threshold", type=float, default=MATCH_RATE_THRESHOLD,
                    help="LightGlue: max match rate below this → new gallery entry")
    ap.add_argument("--max-gallery-per-gid",  type=int,   default=MAX_GALLERY_PER_GID,
                    help="Max gallery entries per global_id; evicts most-central on overflow")
    ap.add_argument("--match-strategy",       choices=["top1", "centroid"], default=MATCH_STRATEGY,
                    help="top1: best single entry sim | centroid: best mean-embedding sim")
    ap.add_argument("--gallery-dir",          default=GALLERY_DUMP_DIR,
                    help="Folder for gallery crop image dumps (empty string to disable)")
    ap.add_argument("--iou-gate-threshold",   type=float, default=IOU_GATE_THRESHOLD,
                    help="Skip re-ID against gallery gids whose last bbox IoU < this (0 = disabled)")
    ap.add_argument("--stable-iou-threshold", type=float, default=STABLE_IOU_THRESHOLD,
                    help="Per-track bbox IoU above this counts as a stable frame")
    ap.add_argument("--stable-min-frames",      type=int,   default=STABLE_MIN_FRAMES,
                    help="Consecutive stable frames before skipping gallery update for a track")
    ap.add_argument("--pending-confirm-frames", type=int,   default=PENDING_CONFIRM_FRAMES,
                    help="Frames to buffer before committing to a new or matched GID")

    # ── face matching (YuNet + SFace — pure ONNX Runtime, no TensorFlow)
    ap.add_argument("--face",             action="store_true",
                    help="Enable face detection + matching (YuNet + SFace, pure ONNX)")
    ap.add_argument("--face-conf",        type=float, default=FACE_CONF_THRESHOLD,
                    help="YuNet minimum detection confidence for face matching")
    ap.add_argument("--face-match-strategy",
                    choices=["face_top1_dino_top1_avg", "face_body_all_gallery",
                             "face_body_face_gallery", "face_only_top1"],
                    default=FACE_MATCH_STRATEGY,
                    help="Matching strategy when a face is detected")
    ap.add_argument("--max-face-gallery-per-gid", type=int, default=MAX_FACE_GALLERY_PER_GID,
                    help="Max face gallery entries per global_id")
    ap.add_argument("--gid-merge-threshold",  type=float, default=GID_MERGE_THRESHOLD,
                        help="Centroid cosine threshold to merge a new GID into an existing one after occlusion re-entry (1.0 = disabled)")
    ap.add_argument("--face-keyframe-threshold", type=float, default=FACE_KEYFRAME_THRESHOLD,
                    help="Add new face keyframe when best gallery sim drops below this")
    ap.add_argument("--yunet-model", default=YUNET_MODEL,
                    help="Path to face_detection_yunet_2023mar.onnx")
    ap.add_argument("--sface-model", default=SFACE_MODEL,
                    help="Path to face_recognition_sface_2021dec.onnx")

    # ── pose quality gate
    ap.add_argument("--pose-model",       default=RTMO_MODEL,
                    help="Path to rtmo.onnx pose estimation model")
    ap.add_argument("--pose-min-quality", choices=["full_body", "upper_body", "none"],
                    default=POSE_MIN_QUALITY,
                    help="Minimum pose quality to assign a new GID (none = disabled)")
    ap.add_argument("--pose-conf",        type=float, default=POSE_CONF_THRESHOLD,
                    help="Keypoint confidence threshold for visibility check")
    ap.add_argument("--no-pose",          action="store_true",
                    help="Disable pose gating entirely (same as --pose-min-quality none)")

    # ── embedder selection
    ap.add_argument("--embedder", choices=["dinov3"], default="dinov3",
                    help="Appearance embedder strategy")

    # ── DINOv3 options
    ap.add_argument("--dinov3-repo",     default=DINOV3_REPO,
                    help="Path to local DINOv3 repo (torch.hub source)")
    ap.add_argument("--dinov3-weights",  default=DINOV3_WEIGHTS,
                    help="Path to DINOv3 .pth weights file")
    ap.add_argument("--dinov3-backbone", default=DINOV3_BACKBONE,
                    choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
                    help="DINOv3 backbone variant")

    args = ap.parse_args()

    embedder = make_embedder(
        name            = args.embedder,
        device          = args.device,
        dinov3_repo     = args.dinov3_repo,
        dinov3_weights  = args.dinov3_weights,
        dinov3_backbone = args.dinov3_backbone,
    )

    _dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise RTMO first so its ONNX Runtime CUDA context is established
    # before any other ONNX sessions (avoids potential multi-session init races).
    pose_min_quality = "none" if args.no_pose else args.pose_min_quality
    pose_checker = None
    if pose_min_quality != "none":
        model_path = args.pose_model
        if not Path(model_path).exists():
            print(f"[CLI] pose model not found at {model_path!r} — pose gating disabled",
                  file=sys.stderr)
            pose_min_quality = "none"
        else:
            pose_checker = RTMOPoseChecker(
                model_path     = model_path,
                device         = _dev,
                conf_threshold = args.pose_conf,
            )

    face_embedder = None
    if args.face:
        try:
            face_embedder = FaceEmbedder(
                conf_threshold = args.face_conf,
                yunet_path     = args.yunet_model,
                sface_path     = args.sface_model,
                device         = _dev,
            )
        except RuntimeError as exc:
            print(f"[CLI] Face matching disabled: {exc}", file=sys.stderr)

    src = int(args.source) if args.source.isdigit() else args.source
    run_video(
        src,
        embedder             = embedder,
        output               = args.output,
        show                 = not args.no_show,
        rtdetr_model         = args.rtdetr_model,
        confidence            = args.confidence,
        similarity_threshold  = args.similarity,
        match_rate_threshold  = args.match_rate_threshold,
        max_gallery_per_gid   = args.max_gallery_per_gid,
        match_strategy        = args.match_strategy,
        gallery_dir           = args.gallery_dir or None,
        lightglue_onnx        = args.lightglue_onnx or None,
        device                = args.device,
        iou_gate_threshold    = args.iou_gate_threshold,
        stable_iou_threshold  = args.stable_iou_threshold,
        stable_min_frames     = args.stable_min_frames,
        pose_checker            = pose_checker,
        pose_min_quality        = pose_min_quality,
        pending_confirm_frames  = args.pending_confirm_frames,
        face_embedder            = face_embedder,
        face_match_strategy      = args.face_match_strategy,
        max_face_gallery_per_gid = args.max_face_gallery_per_gid,
        face_keyframe_threshold  = args.face_keyframe_threshold,
        gid_merge_threshold      = args.gid_merge_threshold,
    )
