<<<<<<< HEAD
"""
Simple MP4 writer.
"""

import cv2
import numpy as np
from typing import Optional


class VideoRecorder:
    def __init__(self, output_path: str, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray):
        if frame is None:
            return

        if self.writer is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        # Convert RGB → BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[VIDEO] Saved training video to: {self.output_path}")
            self.writer = None
=======
"""
Simple MP4 writer.
"""

import cv2
import numpy as np
from typing import Optional


class VideoRecorder:
    def __init__(self, output_path: str, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray):
        if frame is None:
            return

        if self.writer is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        # Convert RGB → BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[VIDEO] Saved training video to: {self.output_path}")
            self.writer = None
>>>>>>> 00debab322f4b7ccc48e1f27f50b6ef94360d3dc
