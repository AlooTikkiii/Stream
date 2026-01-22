import socket
import struct
from typing import Optional, Callable
import time
from .protocol import unpack_header
from .decoder import decode_frame
from .logger import logger
import threading


class StreamServer:
    def __init__(
            self,
            host : str = "127.0.0.1",
            port : int = 5001,
            timeout : float = 5.0,
            on_frame : Optional[Callable] = None,
        ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.on_frame = on_frame
        self.socket = None
        self.conn = None
        self.running = False  

        self.left_count = 0
        self.right_count = 0
        self.start_time = 0.0  

    def start(self) -> None:
        """Start the server and listen for incoming connections."""

        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)

        # Bind and listen
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info(f"Server listening on {self.host}:{self.port}")

        # Accept a connection
        self.conn, self.addr = self.socket.accept()
        self.conn.settimeout(self.timeout)
        logger.info(f"Connection accepted from {self.addr}")

        #Run the client handler in a separate thread
        threading.Thread(target=self._handle_client, daemon=True).start()

    def stop(self) -> None:
        """Stop the server and close the connection."""
        self.running = False
        if self.conn:
            self.conn.close()
            logger.info("Connection closed")
        if self.socket:
            self.socket.close()
            logger.info("Socket closed")

    def recv_exact(self, n : int) -> bytes:
        """Receive exactly n bytes from the connection."""
        data = b''
        while len(data) < n:
            chunk = self.conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data
    
    def _handle_client(self) -> None:
        """Handle incoming data from the client."""
        self.start_time = time.time()
        try:
            while self.running:
                # Receive header 
                header_bytes = self.recv_exact(32)
                meta = unpack_header(header_bytes)
                
                # Receive image payload and decode
                img_bytes = self.recv_exact(meta.size)
                frame = decode_frame(meta, img_bytes)

                if self.on_frame:
                    self.on_frame(frame, meta)

                #Update FPS counters and log
                if meta.eye_id == 0:
                    self.left_count += 1
                else:  
                    self.right_count += 1
                self._print_fps_stats()
        
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")

        finally:
            logger.info("Shutting down server.")
            self.stop()
    
    def _print_fps_stats(self) -> None:
        total_frames = self.left_count + self.right_count
        if total_frames == 0 or total_frames % 60 != 0:
            return
        elapsed = time.time() - self.start_time
        fps_left = self.left_count / elapsed
        fps_right = self.right_count / elapsed
        logger.critical(f"FPS - Left: {fps_left:.2f}, Right: {fps_right:.2f}")

        if abs(fps_left - fps_right) > 0.5:
            logger.warning("Significant FPS difference between eyes detected.")


