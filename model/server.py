import socket
import time
import threading
from typing import Optional, Callable

from .protocol import unpack_header
from .decoder import decode_frame
from .logger import logger


class StreamServer:
    """TCP server to receive Varjo stereo frames from Unity.

    Design goals:
    - No daemon threads for core pipeline.
    - Cooperative shutdown via a shared stop_event.
    - start() is non-blocking (spawns an accept/handler thread).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 5.0,
        on_frame: Optional[Callable] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.on_frame = on_frame

        self.stop_event = stop_event or threading.Event()

        self.socket: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr = None

        self.running = False
        self._thread: Optional[threading.Thread] = None

        # FPS stats
        self.left_count = 0
        self.right_count = 0
        self.start_time = 0.0

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self) -> None:
        """Start listening in a background (non-daemon) thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, name="StreamServer", daemon=False)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def stop(self) -> None:
        """Signal shutdown and close sockets."""
        self.running = False
        self.stop_event.set()

        # Close conn first, then listening socket
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            logger.info("[StreamServer] Connection closed")

        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            logger.info("[StreamServer] Socket closed")

    # -------------------------
    # Internal server loop
    # -------------------------
    def _run(self) -> None:
        """Accept one connection and handle it until disconnect/stop."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(0.5)  # short timeout so we can react to stop_event

            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            logger.info(f"[StreamServer] Listening on {self.host}:{self.port}")

            # Accept loop (allows stop while waiting)
            while self.running and not self.stop_event.is_set():
                try:
                    conn, addr = self.socket.accept()
                    self.conn = conn
                    self.addr = addr
                    self.conn.settimeout(self.timeout)
                    logger.info(f"[StreamServer] Connection accepted from {self.addr}")
                    break
                except socket.timeout:
                    continue

            # If we exited accept without a connection
            if self.conn is None or self.stop_event.is_set() or not self.running:
                return

            self._handle_client()

        except Exception as e:
            logger.exception(f"[StreamServer] Fatal server error: {e}")
        finally:
            # Ensure resources are closed
            self.stop()

    def recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from the connection."""
        if self.conn is None:
            raise ConnectionError("No active connection")

        data = bytearray()
        while len(data) < n and self.running and not self.stop_event.is_set():
            try:
                chunk = self.conn.recv(n - len(data))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data.extend(chunk)

        if len(data) < n:
            raise ConnectionError("Stopped while receiving")

        return bytes(data)

    def _handle_client(self) -> None:
        self.start_time = time.time()

        try:
            while self.running and not self.stop_event.is_set():
                header_bytes = self.recv_exact(32)
                meta = unpack_header(header_bytes)

                img_bytes = self.recv_exact(meta.size)
                frame = decode_frame(meta, img_bytes)

                if self.on_frame:
                    self.on_frame(frame, meta)

                if meta.eye_id == 0:
                    self.left_count += 1
                else:
                    self.right_count += 1
                self._print_fps_stats()

        except ConnectionError as e:
            logger.error(f"[StreamServer] Connection error: {e}")
        except Exception as e:
            logger.exception(f"[StreamServer] Client handler error: {e}")

    def _print_fps_stats(self) -> None:
        total_frames = self.left_count + self.right_count
        if total_frames == 0 or total_frames % 60 != 0:
            return
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return
        fps_left = self.left_count / elapsed
        fps_right = self.right_count / elapsed
        logger.critical(f"[StreamServer] FPS - Left: {fps_left:.2f}, Right: {fps_right:.2f}")

        if abs(fps_left - fps_right) > 0.5:
            logger.warning("[StreamServer] Significant FPS difference between eyes detected.")
