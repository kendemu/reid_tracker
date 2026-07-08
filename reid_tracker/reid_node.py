#!/usr/bin/env python3
"""
ROS2 node: ReID processor.

Subscribes to /bytetrack/tracks (from bytetrack_publisher.py), runs the full
ReID pipeline (DINOv3 embedding + LightGlue gallery + optional RTMO/SFace),
and publishes global-ID assignments as JSON.

Subscribed topic: /bytetrack/tracks    (std_msgs/String)
                  /reid/find_person   (std_msgs/String)  JSON query (see below)
Published topic:  /reid/tracks        (std_msgs/String)
                  /reid/debug_image   (sensor_msgs/Image)
                  /reid/find_result   (std_msgs/String)  JSON result (see below)

Output message schema (/reid/tracks):
  {
    "stamp":     float,     # original frame stamp from bytetrack_publisher
    "frame_idx": int,
    "tracks": [
      {
        "tracker_id":  int,
        "global_id":   int,
        "bbox":        [x1, y1, x2, y2],
        "confidence":  float,
        "match_score": float,
        "match_type":  str    # "B" | "F" | "F+B" | "NEW"
      },
      ...
    ]
  }

/reid/find_person request schema:
  { "request_id": str, "image_jpeg_b64": str }

/reid/find_result response schema:
  {
    "request_id": str,
    "global_id":  int,   # matched GID, or -1 if no gallery match
    "track_id":   int,   # active tracker_id for that GID, -1 if not currently tracked
    "score":      float,
    "method":     str    # "track_gid" | "gid_only" | "direct" | "none"
  }
  Cases:
    track_gid  — GID found in gallery AND active track has that GID
    gid_only   — GID found in gallery but no active track holds it  → track_id=-1
    direct     — no gallery match; best active track by DINOv3+face → global_id=-1
    none       — no gallery match and no active tracks              → both -1

ROS2 parameters:
  device               string  ""     torch device (empty → auto)
  similarity           double  0.82   cosine similarity threshold for re-ID match
  match_rate_threshold double  0.40   LightGlue: max match rate → new gallery entry
  lightglue_onnx       string  ""     path to pipeline ONNX (empty → default)
  pose                 bool    true   enable RTMO pose quality gating
  pose_model           string  ""     path to rtmo.onnx (empty → default)
  pose_min_quality     string  "upper_body"
  face                 bool    false  enable YuNet + SFace face matching
  yunet_model          string  ""     path to YuNet ONNX (empty → default)
  sface_model          string  ""     path to SFace ONNX (empty → default)
  dinov3_repo          string  ""     path to dinov3 repo (empty → default)
  dinov3_weights       string  ""     path to .pth weights (empty → default)
  gallery_dir          string  "gallery_dump"

Usage:
  ros2 run reid_tracker reid_node --ros-args \
      -p similarity:=0.82 -p pose:=true -p face:=false
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

sys.path.insert(0, str(Path(__file__).parent))
from fullreid import (
    ReIDSystem,
    RTMOPoseChecker,
    FaceEmbedder,
    make_embedder,
    LIGHTGLUE_ONNX,
    RTMO_MODEL,
    YUNET_MODEL,
    SFACE_MODEL,
    DINOV3_REPO,
    DINOV3_WEIGHTS,
    DINOV3_BACKBONE,
    REID_SIM_THRESHOLD,
    MATCH_RATE_THRESHOLD,
    GALLERY_DUMP_DIR,
    POSE_MIN_QUALITY,
)


class ReIDNode(Node):

    def __init__(self):
        super().__init__("reid_node")

        # Declare parameters
        self.declare_parameter("device",               "")
        self.declare_parameter("similarity",           REID_SIM_THRESHOLD)
        self.declare_parameter("match_rate_threshold", MATCH_RATE_THRESHOLD)
        self.declare_parameter("lightglue_onnx",       "")
        self.declare_parameter("pose",                 True)
        self.declare_parameter("pose_model",           "")
        self.declare_parameter("pose_min_quality",     POSE_MIN_QUALITY)
        self.declare_parameter("face",                 False)
        self.declare_parameter("yunet_model",          "")
        self.declare_parameter("sface_model",          "")
        self.declare_parameter("dinov3_repo",          "")
        self.declare_parameter("dinov3_weights",       "")
        self.declare_parameter("gallery_dir",          GALLERY_DUMP_DIR)

        import torch
        device_str = self.get_parameter("device").value
        device     = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")

        similarity           = float(self.get_parameter("similarity").value)
        match_rate_threshold = float(self.get_parameter("match_rate_threshold").value)
        lightglue_onnx       = self.get_parameter("lightglue_onnx").value or None
        enable_pose          = bool(self.get_parameter("pose").value)
        pose_model           = self.get_parameter("pose_model").value or RTMO_MODEL
        pose_min_quality     = self.get_parameter("pose_min_quality").value
        enable_face          = bool(self.get_parameter("face").value)
        yunet_model          = self.get_parameter("yunet_model").value or YUNET_MODEL
        sface_model          = self.get_parameter("sface_model").value or SFACE_MODEL
        dinov3_repo          = self.get_parameter("dinov3_repo").value or DINOV3_REPO
        dinov3_weights       = self.get_parameter("dinov3_weights").value or DINOV3_WEIGHTS
        gallery_dir          = self.get_parameter("gallery_dir").value or None

        # Build ReID components
        embedder = make_embedder(
            "dinov3", device=device,
            dinov3_repo=dinov3_repo, dinov3_weights=dinov3_weights,
            dinov3_backbone=DINOV3_BACKBONE,
        )

        pose_checker = None
        if enable_pose:
            try:
                pose_checker = RTMOPoseChecker(model_path=pose_model, device=device)
            except Exception as exc:
                self.get_logger().warning(f"RTMO load failed: {exc} — pose gating disabled")

        face_embedder = None
        if enable_face:
            try:
                face_embedder = FaceEmbedder(yunet_path=yunet_model, sface_path=sface_model, device=device)
            except Exception as exc:
                self.get_logger().warning(f"FaceEmbedder load failed: {exc} — face matching disabled")

        # ReIDSystem with _reid_only=True skips loading RT-DETR + ByteTrack
        self._system = ReIDSystem(
            embedder              = embedder,
            similarity_threshold  = similarity,
            match_rate_threshold  = match_rate_threshold,
            lightglue_onnx        = lightglue_onnx,
            device                = device,
            pose_checker          = pose_checker,
            pose_min_quality      = pose_min_quality,
            face_embedder         = face_embedder,
            gallery_dir           = gallery_dir,
            _reid_only            = True,
        )

        self.declare_parameter("debug_jpeg_quality", 70)
        self._debug_jpeg_q = int(self.get_parameter("debug_jpeg_quality").value)

        self._sub        = self.create_subscription(
            String, "/bytetrack/tracks", self._on_tracks, 10
        )
        self._sub_find   = self.create_subscription(
            String, "/reid/find_person", self._on_find_person, 10
        )
        self._pub        = self.create_publisher(String, "/reid/tracks",      10)
        self._pub_debug  = self.create_publisher(Image,  "/reid/debug_image", 10)
        self._pub_find   = self.create_publisher(String, "/reid/find_result", 10)

        # Latest confirmed active tracks from the most recent processed frame
        self._latest_active_tracks: dict[int, int] = {}  # tracker_id → global_id

        self.get_logger().info(
            f"ReIDNode ready — device={device} pose={enable_pose} face={enable_face} "
            f"→ /reid/tracks  /reid/find_person"
        )
        self.bridge = CvBridge()

    def _on_tracks(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"JSON decode error: {exc}")
            return

        stamp     = payload["stamp"]
        frame_idx = payload["frame_idx"]
        raw_tracks = payload.get("tracks", [])

        # Decode full frame (needed for RTMO pose gating and YuNet face detection)
        frame_b64 = payload.get("frame_jpeg_b64", "")
        if frame_b64:
            buf   = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        else:
            h = payload.get("frame_h", 1)
            w = payload.get("frame_w", 1)
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        if len(raw_tracks) == 0:
            self._publish(stamp, frame_idx, [])
            return

        # Reconstruct sv.Detections so process_detections can work unchanged
        xyxy        = np.array([t["bbox"] for t in raw_tracks], dtype=np.float32)
        confidences = np.array([t["confidence"] for t in raw_tracks], dtype=np.float32)
        tracker_ids = np.array([t["tracker_id"] for t in raw_tracks], dtype=int)

        detections = sv.Detections(
            xyxy       = xyxy,
            confidence = confidences,
            class_id   = np.zeros(len(raw_tracks), dtype=int),
        )
        detections.tracker_id = tracker_ids

        # Run full ReID pipeline
        annotated, tracker_to_global = self._system.process_detections(frame, detections)

        # Build output tracks
        out_tracks = []
        for t in raw_tracks:
            tid = t["tracker_id"]
            gid = tracker_to_global.get(tid, -1)
            match_score, match_type = self._system._track_match_info.get(tid, (0.0, "B"))
            out_tracks.append({
                "tracker_id":  tid,
                "global_id":   gid,
                "bbox":        t["bbox"],
                "confidence":  t["confidence"],
                "match_score": float(match_score),
                "match_type":  match_type,
            })

        # Keep a snapshot of confirmed active tracks for reference-image queries
        self._latest_active_tracks = {
            t["tracker_id"]: t["global_id"]
            for t in out_tracks
            if t["global_id"] >= 0
        }

        self._publish(stamp, frame_idx, out_tracks)
        self._publish_debug(annotated)

    def _publish(self, stamp: float, frame_idx: int, tracks: list):
        msg = String()
        msg.data = json.dumps({
            "stamp":     stamp,
            "frame_idx": frame_idx,
            "tracks":    tracks,
        })
        self._pub.publish(msg)

    def _publish_debug(self, annotated: np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(annotated)
        msg.encoding = "bgr8"
        self._pub_debug.publish(msg)

    def _on_find_person(self, msg: String):
        """
        Query: which active track best matches a reference image?

        Request JSON:  { "request_id": str, "image_jpeg_b64": str }
        Response JSON: { "request_id": str, "global_id": int, "track_id": int,
                         "score": float, "method": str }
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"find_person JSON decode error: {exc}")
            return

        request_id = payload.get("request_id", "")
        image_b64  = payload.get("image_jpeg_b64", "")
        if not image_b64:
            self.get_logger().error("find_person: missing image_jpeg_b64")
            return

        buf = base64.b64decode(image_b64)
        ref_crop = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if ref_crop is None or ref_crop.size == 0:
            self.get_logger().error("find_person: could not decode reference image")
            return

        global_id, track_id, score, method = self._system.find_reference(
            ref_crop, self._latest_active_tracks
        )

        result = String()
        result.data = json.dumps({
            "request_id": request_id,
            "global_id":  global_id,
            "track_id":   track_id,
            "score":      float(score),
            "method":     method,
        })
        self._pub_find.publish(result)
        self.get_logger().info(
            f"find_person [{request_id}] → gid={global_id} tid={track_id} "
            f"score={score:.3f} method={method}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ReIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
