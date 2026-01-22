import threading
import socket
import struct
import time
from queue import Empty, Queue
from typing import Tuple, Sequence, Optional, List

from .logger import logger

Det = Tuple[float, float, float, float, int, float]        # (x1,y1,x2,y2, track_id, score)
MatchItem = Tuple[Det, Det, int, float]                    # (detA, detB, cls_id, match_score)
Message = Tuple[int, Sequence[MatchItem]]                  # (timestamp, matches)

HEADER_FMT = "<QI"        # uint64 timestamp, uint32 count
DATA_FMT = "<iffffffff"   # int32 cls_id + 8 floats (two boxes)

HEADER_SIZE = struct.calcsize(HEADER_FMT)
ITEM_SIZE = struct.calcsize(DATA_FMT)


class Sender:
    """
    TCP sender for (timestamp, matches).

    - Thread-safe send_data() (serializes socket writes with a lock)
    - Optional reconnect
    - Clean close()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5002,
        timeout: float = 5.0,
        reconnect: bool = True,
        connect_backoff_s: float = 0.5,
        connect_backoff_max_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect = reconnect
        self.connect_backoff_s = connect_backoff_s
        self.connect_backoff_max_s = connect_backoff_max_s

        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()

    def _connect_once(self) -> bool:
        """Try connecting once. Returns True on success, False on failure."""
        self.close()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            self.sock = s
            logger.info(f"[Sender] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.info(f"[Sender] Connection failed to {self.host}:{self.port}: {e}")
            self.sock = None
            return False

    def ensure_connected(self, stop_event: threading.Event) -> bool:
        """
        Ensure we have a connected socket.
        If reconnect=True, keep trying with backoff until stop_event is set.
        """
        if self.sock is not None:
            return True

        if not self.reconnect:
            return self._connect_once()

        backoff = self.connect_backoff_s
        while not stop_event.is_set():
            if self._connect_once():
                return True
            time.sleep(backoff)
            backoff = min(self.connect_backoff_max_s, backoff * 1.5)

        return False

    def close(self) -> None:
        s = self.sock
        self.sock = None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    def _build_payload(self, ts: int, matches: Sequence[MatchItem]) -> bytes:
        """
        Payload layout:
        [HEADER: uint64 ts, uint32 n]
        then n * [int32 cls_id, 8 floats boxes]
        """
        n = len(matches)
        chunks: List[bytes] = [struct.pack(HEADER_FMT, int(ts), int(n))]

        for det1, det2, cls_id, _match_score in matches:
            x1a, y1a, x2a, y2a, _track_a, _score_a = det1
            x1b, y1b, x2b, y2b, _track_b, _score_b = det2

            chunks.append(
                struct.pack(
                    DATA_FMT,
                    int(cls_id),
                    float(x1a), float(y1a), float(x2a), float(y2a),
                    float(x1b), float(y1b), float(x2b), float(y2b),
                )
            )

        return b"".join(chunks)

    def send_data(self, msg: Message) -> bool:
        """
        Sends one message if currently connected.
        Does NOT loop/retry forever here. Worker handles reconnect/backoff.
        """
        ts, matches = msg
        payload = self._build_payload(ts, matches)

        with self.lock:
            if self.sock is None:
                return False

            try:
                self.sock.sendall(payload)
                return True
            except Exception as e:
                logger.warning(f"[Sender] Send failed: {e}")
                self.close()
                return False
            
def sender_worker(
    sender: Sender,
    det_queue: "Queue[Message]",
    stop_event: threading.Event,
    idle_timeout_s: float = 0.2,
) -> None:
    """
    Dedicated thread:
    - Pop messages from det_queue
    - Ensure connection (with backoff)
    - Send
    - Exit when stop_event is set

    NOTE: No extra dropping/draining here.
    If you want "latest-wins", do it only in the producer (YOLO worker),
    where you already handle det_queue MAX_SIZE drops.
    """
    logger.info("[SenderWorker] Started")

    try:
        # Ensure we're connected early (optional, but nice)
        sender.ensure_connected(stop_event)

        while not stop_event.is_set():
            # 1) Wait for a message (wake periodically to check stop_event)
            try:
                msg = det_queue.get(timeout=idle_timeout_s)
            except Empty:
                continue

            # 2) Ensure connected (if Unity restarted receiver etc.)
            if not sender.ensure_connected(stop_event):
                # If reconnect disabled or stop_event set; drop msg (real-time)
                continue

            # 3) Send once; if it fails, sender closes socket and we retry next loop
            ok = sender.send_data(msg)
            if not ok:
                # send_data already closed on failure
                continue

    except Exception as e:
        logger.exception(f"[SenderWorker] Crashed: {e}")
        stop_event.set()
    finally:
        sender.close()
        logger.info("[SenderWorker] Stopped")