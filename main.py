from model.server import StreamServer
from model.buffer import FrameBuffer
import threading
import time
from queue import Queue, Full, Empty
from typing import Tuple, Sequence

import cv2
import numpy as np
from ultralytics import YOLOE

from model.server import StreamServer
from model.buffer import FrameBuffer
from model.logger import logger
from model.sender import Sender, sender_worker  # <-- create model/sender.py from the Sender we wrote
from utils import extract_topk_detections, postprocess_detections


Det = Tuple[float, float, float, float, int, float] 
Match = Tuple[Det, Det, int, float]
# ----------------------------
# Config
# ----------------------------
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5001

DET_HOST = "127.0.0.1"
DET_PORT = 5002

YOLO_MODEL_PATH = "yoloe-11l-seg-pf.pt"  # change to your file
DEVICE = "cuda"  # or "cpu"

YOLO_QUEUE_MAX = 50
DET_QUEUE_MAX = 50

MAX_DETS_PER_FRAME = 100
SHOW = True  # set True if you want to visualize the results

# ----------------------------
# Queues + stop flag
# ----------------------------
yolo_queue = Queue(maxsize=YOLO_QUEUE_MAX)   # stereo pairs -> yolo
det_queue = Queue(maxsize=DET_QUEUE_MAX)     # detections -> server


def on_stereo_pair(left_frame, right_frame, timestamp, stop_event: threading.Event):
    """Callback from FrameBuffer when a stereo pair is ready."""
    if stop_event.is_set():
        return
    try:
        yolo_queue.put_nowait((left_frame, right_frame, timestamp))
    except Full:
        logger.warning("[Main] YOLO queue full, dropping stereo pair")


def yolo_worker(server: StreamServer, stop_event: threading.Event):
    """Consumes stereo frames -> runs YOLO -> postprocess -> pushes to det_queue."""
    read_time, preprocess_time, inference_time, postprocess_time, queue_time, show_time = [], [], [], [], [], []

    logger.info("[YOLO] Loading model...")
    model = YOLOE(YOLO_MODEL_PATH)
    try:
        model.to(DEVICE)
    except Exception as e:
        logger.error(f"[YOLO] Failed to move model to {DEVICE}: {e}")
        stop_event.set()
        return

    if SHOW:
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    last_total_time = time.perf_counter()

    try:
        while server.running and not stop_event.is_set():
            # 1) Read stereo pair
            r1 = time.perf_counter()
            try:
                left_frame, right_frame, timestamp = yolo_queue.get(timeout=0.1)
            except Empty:
                continue
            r2 = time.perf_counter()

            if left_frame is None or right_frame is None:
                stop_event.set()
                break

            # 2) Preprocess
            i1 = time.perf_counter()
            left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

            left_bgr = cv2.resize(left_bgr, (224, 224))
            right_bgr = cv2.resize(right_bgr, (224, 224))

            H, W = left_bgr.shape[:2]
            combined_bgr = cv2.hconcat([left_bgr, right_bgr])
            i2 = time.perf_counter()

            # 3) Inference
            results = model.predict(source=combined_bgr, verbose=False)
            i3 = time.perf_counter()

            # 4) Postprocess
            p1 = time.perf_counter()
            detections = extract_topk_detections(results[0], MAX_DETS_PER_FRAME)
            processed_results = postprocess_detections(
                detections, H, W,
                iou_threshold=0.5,
                area_ratio_thresh=0.90,
                edge_tol=3.0,
                classes=None,
            )
            p2 = time.perf_counter()

            # 5) Push to detection queue (drop oldest if full)
            q1 = time.perf_counter()
            try:
                det_queue.put_nowait((timestamp, processed_results))
            except Full:
                logger.info("[YOLO] det_queue full, dropping oldest")
                try:
                    det_queue.get_nowait()
                except Empty:
                    pass
                try:
                    det_queue.put_nowait((timestamp, processed_results))
                except Full:
                    pass
            q2 = time.perf_counter()

            # 6) Visualization + ESC to stop
            if SHOW:
                vis = combined_bgr.copy()
                s1 = time.perf_counter()
                for (L, R, cls_id, _iou) in processed_results:
                    lx1, ly1, lx2, ly2, _, _lconf = L
                    rx1, ry1, rx2, ry2, _, _rconf = R

                    cv2.rectangle(vis, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 0), 2)
                    cv2.rectangle(vis, (int(rx1 + W), int(ry1)), (int(rx2 + W), int(ry2)), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"ID:{cls_id}",
                        (int(lx1), int(ly1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                cv2.imshow("Detections", vis)
                s2 = time.perf_counter()

                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("[YOLO] ESC pressed, stopping...")
                    stop_event.set()
                    server.stop()
                    break

            # 7) Timing logs
            read_time.append((r2 - r1) * 1000)
            preprocess_time.append((i2 - i1) * 1000)
            inference_time.append((i3 - i2) * 1000)
            postprocess_time.append((p2 - p1) * 1000)
            queue_time.append((q2 - q1) * 1000)
            if SHOW:
                show_time.append((s2 - s1) * 1000)

            current_time = time.perf_counter()
            if current_time - last_total_time >= 10.0:
                if SHOW and show_time:
                    avg_show = sum(show_time) / len(show_time)
                    logger.critical(
                        f"[YOLO] Avg Timing over 10s (ms): "
                        f"Read:{sum(read_time):.2f}, Pre:{sum(preprocess_time):.2f}, "
                        f"Inf:{sum(inference_time):.2f}, Post:{sum(postprocess_time):.2f}, "
                        f"Queue:{sum(queue_time):.2f}, Show:{avg_show:.2f}"
                    )
                    show_time.clear()
                else:
                    logger.critical(
                        f"[YOLO] Avg Timing over 10s (ms): "
                        f"Read:{sum(read_time):.2f}, Pre:{sum(preprocess_time):.2f}, "
                        f"Inf:{sum(inference_time):.2f}, Post:{sum(postprocess_time):.2f}, "
                        f"Queue:{sum(queue_time):.2f}"
                    )

                read_time.clear()
                preprocess_time.clear()
                inference_time.clear()
                postprocess_time.clear()
                queue_time.clear()
                last_total_time = current_time

    except Exception as e:
        logger.exception(f"[YOLO] Worker crashed: {e}")
        stop_event.set()
    finally:
        if SHOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("[YOLO] Worker stopped")


def main():
    stop_event = threading.Event()

    # Frame buffer -> pushes stereo pairs into yolo_queue
    buffer = FrameBuffer(on_stereo_pair=lambda l, r, ts: on_stereo_pair(l, r, ts, stop_event))

    # Stream server (Unity -> Python)
    server = StreamServer(
        host=STREAM_HOST,
        port=STREAM_PORT,
        timeout=5.0,
        on_frame=buffer.add_frame,
        stop_event=stop_event,  # <-- requires updated server.py
    )
    server.start()

    # Sender (Python -> Unity detections)
    sender = Sender(host=DET_HOST, port=DET_PORT, timeout=5.0, reconnect=True)

    # Threads (NON-daemon)
    yolo_thread = threading.Thread(target=yolo_worker, args=(server, stop_event), name="YOLOWorker", daemon=False)
    send_thread = threading.Thread(target=sender_worker, args=(sender, det_queue, stop_event), name="DetSender", daemon=False)

    yolo_thread.start()
    send_thread.start()

    try:
        # Main thread: supervise
        while not stop_event.is_set():
            # If server stopped (Unity disconnected), stop everything
            if not server.running:
                logger.warning("[Main] Stream server stopped (Unity disconnect?). Shutting down.")
                stop_event.set()
                break
            time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("[Main] Ctrl+C received. Shutting down.")
        stop_event.set()

    finally:
        # Signal shutdown & close resources
        stop_event.set()
        server.stop()
        sender.close()

        # Join workers
        yolo_thread.join(timeout=3.0)
        send_thread.join(timeout=3.0)

        # If your StreamServer has join()
        try:
            server.join(timeout=3.0)
        except Exception:
            pass

        logger.info("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()