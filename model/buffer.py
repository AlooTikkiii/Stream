import threading
from collections import OrderedDict
from typing import Callable, Dict, Tuple
import time


class FrameBuffer:
    def __init__(
            self,
            on_stereo_pair : Callable,
            max_buffer_size : int = 60,
            max_age_seconds : float = 2.0,
        ) -> None:
        """
        Args:
            on_stereo_pair: function called when both L/R frames for a timestamp exist.
                            Signature: on_stereo_pair(left_frame, right_frame, timestamp)
            max_entries: Maximum different timestamps stored
            max_age_sec: Drop unmatched timestamps older than this time window
        """
        self.on_stereo_pair = on_stereo_pair
        self.max_buffer_size = max_buffer_size
        self.max_age_seconds = max_age_seconds
        self.buffer : Dict[int, Dict] = OrderedDict()
        self.lock = threading.Lock()

    def add_frame(
            self,
            frame,
            meta
        ) -> None:
        """Insert frame and check if stereo pair can be processed."""

        ts = meta.timestamp
        with self.lock:
            if ts not in self.buffer:
                self.buffer[ts] = {"left": None, "right": None, "arrival_time": time.time()}

            if meta.eye_id == 0:
                self.buffer[ts]["left"] = frame
            else:
                self.buffer[ts]["right"] = frame

            slot = self.buffer[ts]
            # Check if stereo pair ready
            if slot["left"] is not None and slot["right"] is not None:
                left = slot["left"]
                right = slot["right"]

                # Trigger callback
                self.on_stereo_pair(left, right, ts)

                # Evict immediately
                del self.buffer[ts]

            else:
                # Clean old & excess entries
                self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove old or excess entries from the buffer."""
        now = time.time()
        to_delete = []

        for ts, slot in list(self.buffer.items()):
            age = now - slot["arrival_time"]

            if age > self.max_age_seconds or len(self.buffer) > self.max_buffer_size:
                to_delete.append(ts)

        for ts in to_delete:
            del self.buffer[ts]