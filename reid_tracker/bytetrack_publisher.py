#!/usr/bin/env python3
"""
ROS2 node: RT-DETR + ByteTrack publisher.

Reads from a video source (webcam or video file), runs RT-DETR person detection
+ ByteTrack temporal tracking, and publishes each frame's tracks as a JSON string.

Press S in the display window to capture the largest detected person as the
reference image.  On every subsequent frame a /reid/find_person query is sent
and the result is overlaid on the display.  Press S again to replace the
reference.  Press Q to quit.

Published topic:  /bytetrack/tracks    (std_msgs/String)
                  /reid/find_person     (std_msgs/String)  reference query
Subscribed topic: /reid/find_result    (std_msgs/String)  query result

ROS2 parameters:
  source        string  "0"        Camera index or video file path
  rtdetr_model  string  <default>  RT-DETR ONNX model path
  confidence    double  0.5        Detection confidence threshold
  device        string  ""         torch device (empty → auto)
  jpeg_quality  int     70         JPEG compression quality for the frame
  show          bool    true       Show OpenCV display window
  ref_jpeg_quality int  90         JPEG quality for the reference image encoding

Usage:
  ros2 run <pkg> bytetrack_publisher --ros-args \
      -p source:=0 -p confidence:=0.5
"""

import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).parent))
from fullreid import RTDetrDetector, RTDETR_MODEL, DETECT_CONFIDENCE


class ByteTrackPublisher(Node):

    def __init__(self):
        super().__init__("bytetrack_publisher")

        self.declare_parameter("source",           "0")
        self.declare_parameter("rtdetr_model",     RTDETR_MODEL)
        self.declare_parameter("confidence",       DETECT_CONFIDENCE)
        self.declare_parameter("device",           "")
        self.declare_parameter("jpeg_quality",     70)
        self.declare_parameter("show",             True)
        self.declare_parameter("ref_jpeg_quality", 90)

        source_str    = self.get_parameter("source").value
        rtdetr_model  = self.get_parameter("rtdetr_model").value
        confidence    = self.get_parameter("confidence").value
        device_str    = self.get_parameter("device").value
        self._jpeg_q  = int(self.get_parameter("jpeg_quality").value)
        self._show    = bool(self.get_parameter("show").value)
        self._ref_q   = int(self.get_parameter("ref_jpeg_quality").value)

        self._pub      = self.create_publisher(String, "/bytetrack/tracks",  10)
        self._find_pub = self.create_publisher(String, "/reid/find_person",  10)
        self._find_sub = self.create_subscription(
            String, "/reid/find_result", self._on_find_result, 10
        )

        import torch
        device = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = RTDetrDetector(rtdetr_model, device, confidence)
        self._tracker  = sv.ByteTrack()

        src = int(source_str) if source_str.isdigit() else source_str
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open source: {source_str!r}")
            raise RuntimeError(f"Cannot open source: {source_str!r}")

        # Reference image state
        self._reference_crop: Optional[np.ndarray] = None
        # Pre-encoded base64 so we don't re-encode every frame
        self._reference_b64:  Optional[str]        = None
        # Latest result from /reid/find_result
        self._latest_find_result: Optional[dict]   = None

        # Snapshot of last frame / detections for use in _capture_reference
        self._last_frame:      Optional[np.ndarray]      = None
        self._last_detections: Optional[sv.Detections]   = None

        self._frame_idx = 0

        if self._show:
            cv2.namedWindow("ByteTrack", cv2.WINDOW_NORMAL)

        # Run as fast as the camera delivers frames; 1 ms timer just means "next tick"
        self._timer = self.create_timer(0.001, self._tick)

        self.get_logger().info(
            f"ByteTrackPublisher ready — source={source_str!r} device={device} "
            f"→ /bytetrack/tracks  (S: capture reference, Q: quit)"
        )

    # ── main loop ────────────────────────────────────────────────────────────

    def _tick(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().info("Video stream ended.")
            self._timer.cancel()
            return

        detections = self._detector.detect(frame)
        detections = self._tracker.update_with_detections(detections)

        self._last_frame      = frame
        self._last_detections = detections

        # Keyboard events (non-blocking, 1 ms)
        if self._show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self._capture_reference()
            elif key == ord('q'):
                self.get_logger().info("Q pressed — shutting down.")
                self._timer.cancel()
                rclpy.shutdown()
                return

        # Publish tracks
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q]
        )
        frame_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

        tracks = []
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int).tolist()
                tracks.append({
                    "tracker_id": int(tid),
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": float(detections.confidence[i]),
                })

        msg = String()
        msg.data = json.dumps({
            "stamp":          time.time(),
            "frame_idx":      self._frame_idx,
            "frame_h":        frame.shape[0],
            "frame_w":        frame.shape[1],
            "frame_jpeg_b64": frame_b64,
            "tracks":         tracks,
        })
        self._pub.publish(msg)

        # Reference-image query — send every frame once a reference is set
        if self._reference_b64 is not None:
            self._publish_find_person()

        # Display
        if self._show:
            cv2.imshow("ByteTrack", self._make_display(frame, detections))

        self._frame_idx += 1

    # ── reference capture ────────────────────────────────────────────────────

    def _capture_reference(self):
        frame      = self._last_frame
        detections = self._last_detections
        if frame is None:
            return

        crop: Optional[np.ndarray] = None
        if detections is not None and detections.tracker_id is not None and len(detections) > 0:
            areas    = ((detections.xyxy[:, 2] - detections.xyxy[:, 0]) *
                        (detections.xyxy[:, 3] - detections.xyxy[:, 1]))
            best_idx = int(np.argmax(areas))
            x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

        if crop is None or crop.size == 0:
            self.get_logger().warning("No person detected — using full frame as reference")
            crop = frame.copy()

        self._reference_crop    = crop
        self._latest_find_result = None  # reset stale result

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self._ref_q])
        if ok:
            self._reference_b64 = base64.b64encode(buf.tobytes()).decode()
            cv2.imwrite("reference.jpg", crop)
            cv2.imshow("reference", crop)
            cv2.waitKey(1)
            self.get_logger().info(
                f"Reference captured ({crop.shape[1]}×{crop.shape[0]}) → reference.jpg"
            )
        else:
            self._reference_b64 = None
            self.get_logger().error("Failed to encode reference image")

    # ── find_person publish / result callback ────────────────────────────────

    def _publish_find_person(self):
        msg = String()
        msg.data = json.dumps({
            "request_id":     str(self._frame_idx),
            "image_jpeg_b64": self._reference_b64,
        })
        self._find_pub.publish(msg)

    def _on_find_result(self, msg: String):
        try:
            self._latest_find_result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"find_result decode error: {exc}")

    # ── display ──────────────────────────────────────────────────────────────

    def _make_display(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        display = frame.copy()
        result  = self._latest_find_result
        matched_tid = result.get("track_id", -1) if result else -1

        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                tid_i = int(tid)
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                if tid_i == matched_tid and matched_tid != -1:
                    color = (0, 64, 255)   # orange-red: matched
                    gid   = result.get("global_id", -1)
                    score = result.get("score", 0.0)
                    meth  = result.get("method", "")
                    label = f"T{tid_i} G{gid} {score:.2f} [{meth}]"
                else:
                    color = (0, 200, 0)    # green: normal
                    label = f"T{tid_i}"
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Reference thumbnail (top-left corner)
        if self._reference_crop is not None:
            thumb_h = 80
            scale   = thumb_h / self._reference_crop.shape[0]
            thumb_w = max(1, int(self._reference_crop.shape[1] * scale))
            thumb   = cv2.resize(self._reference_crop, (thumb_w, thumb_h))
            dh, dw  = display.shape[:2]
            if thumb_w + 10 <= dw and thumb_h + 10 <= dh:
                display[5:5 + thumb_h, 5:5 + thumb_w] = thumb
                cv2.rectangle(display, (5, 5), (5 + thumb_w, 5 + thumb_h), (0, 220, 255), 2)
                cv2.putText(display, "REF", (5, 5 + thumb_h + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

        # Status bar
        if self._reference_crop is None:
            status = "S: capture reference"
        elif result is None:
            status = "REF set | waiting for reid..."
        else:
            tid_r = result.get("track_id", -1)
            gid_r = result.get("global_id", -1)
            meth  = result.get("method", "?")
            status = f"REF active | T{tid_r} G{gid_r} [{meth}]  (S: new ref)"
        cv2.putText(display, status, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return display

    # ── cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._show:
            cv2.destroyAllWindows()
        self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ByteTrackPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
