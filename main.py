from model.server import StreamServer
from model.buffer import FrameBuffer
from model.logger import logger
import cv2
import time
import threading
import numpy as np
from queue import Queue, Full, Empty
from ultralytics import YOLOE
from typing import Any, List, Optional, Iterable, Set, Tuple
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


def on_stereo_pair(left_frame, right_frame, timestamp):
    try:
        yolo_queue.put_nowait((left_frame, right_frame, timestamp))
    except Full:
        logger.warning("YOLO queue full, dropping frame")

def yolo_worker(server : StreamServer):
    read_time, preprocess_time, inference_time, postprocess_time, queue_time, show_time = [], [], [], [], [], []
    #LOADING THE YOLO MODEL
    logger.info("Loading YOLO model...")
    # model = YOLO(YOLO_MODEL_PATH)
    model = YOLOE("yoloe-11l-seg-pf.pt")
    try :
        model.to(DEVICE)
    except Exception as e:
        logger.error(f"Failed to load model on device {DEVICE}: {e}")
        return
    if SHOW:
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    last_total_time = time.perf_counter()

    while server.running:
        #READING FRAMES FROM THE YOLO QUEUE
        r1 = time.perf_counter()
        try:   
            left_frame, right_frame, timestamp = yolo_queue.get(timeout=0.1)
        except Empty:
            continue
        if left_frame is None or right_frame is None:
            break
        r2 = time.perf_counter()
        

        #PREPROCESSING FRAMES
        i1 = time.perf_counter()
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
        left_bgr = cv2.resize(left_bgr, (224, 224))
        right_bgr = cv2.resize(right_bgr, (224, 224))
        H, W = left_bgr.shape[:2]
        combined_bgr = cv2.hconcat([left_bgr, right_bgr])
        i2 = time.perf_counter()
        
        
        #INFERENCE ON THE COMBINED IMAGE
        # logger.info("YOLOE Detection Started")
        results = model.predict(source=combined_bgr, verbose=False)
        i3 = time.perf_counter()

        
        
        #POST-PROCESSING RESULTS
        p1 = time.perf_counter()
        detections = extract_topk_detections(results[0], MAX_DETS_PER_FRAME)
        # logger.info(f"Detections extracted: {len(detections)}")
        processed_results = postprocess_detections(detections, H, W, iou_threshold=0.5, area_ratio_thresh=0.90, edge_tol=3.0, classes=None)
        p2 = time.perf_counter()


        
        #PUTTING DETECTIONS INTO THE DETECTION QUEUE
        q1 = time.perf_counter()
        try:
            det_queue.put_nowait((timestamp, processed_results))
        except Full:
            logger.info("Detection queue full, dropping detections")
            # Remove the oldest entry and add the new one
            det_queue.get_nowait()
            det_queue.put_nowait((timestamp, processed_results))
        q2 = time.perf_counter()

       
       
        #VISUALIZATION
        if SHOW:
            try:
                vis = combined_bgr.copy()
                s1 = time.perf_counter()
                for (L, R, cls_id, iou) in processed_results:
                    lx1, ly1, lx2, ly2, _, lconf = L
                    rx1, ry1, rx2, ry2, _, rconf = R

                    cv2.rectangle(vis, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 0), 2)
                    cv2.rectangle(vis, (int(rx1 + W), int(ry1)), (int(rx2 + W), int(ry2)), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID:{cls_id}", (int(lx1), int(ly1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

                cv2.imshow("Detections", vis)
                s2 = time.perf_counter()
            except Empty:
                pass
            #Destroy window on 'esc' key press
            if cv2.waitKey(1) & 0xFF == 27:
                server.stop()
                break
        
        #LOGGING
        read_time.append((r2 - r1)*1000)
        preprocess_time.append((i2 - i1)*1000)
        inference_time.append((i3 - i2)*1000)
        postprocess_time.append((p2 - p1)*1000)
        queue_time.append((q2 - q1)*1000)
        if SHOW:
            show_time.append((s2 - s1)*1000)
        current_time = time.perf_counter()
        if current_time - last_total_time >= 10.0:
            if SHOW:
                avg_show = sum(show_time)/len(show_time)
                logger.critical(f"YOLO Worker Avg Timing over 10s (ms): Readframe: {sum(read_time):.2f}, Preprocess:{sum(preprocess_time):.2f}, Inference: {sum(inference_time):.2f}, Postprocess: {sum(postprocess_time):.2f}, Send: {sum(queue_time):.2f}, Show: {avg_show:.2f}")
                show_time.clear()
            else:
                logger.critical(f"YOLO Worker Avg Timing over 10s (ms): Readframe: {sum(read_time):.2f}, Preprocess:{sum(preprocess_time):.2f}, Inference: {sum(inference_time):.2f}, Postprocess: {sum(postprocess_time):.2f}, Send: {sum(queue_time):.2f}")
            read_time.clear()
            preprocess_time.clear()
            inference_time.clear()
            postprocess_time.clear()
            queue_time.clear()
            last_total_time = current_time
    if SHOW:
            cv2.destroyAllWindows()




def main():
    buffer = FrameBuffer(on_stereo_pair=on_stereo_pair)
    server = StreamServer(STREAM_HOST, STREAM_PORT, 100, buffer.add_frame)
    server.start()

    yolo_thread = threading.Thread(target=yolo_worker, args=(server,), daemon=True)
    yolo_thread.start()

    try:
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping server...")
        server.stop()

    yolo_thread.join()
    logger.info("Server stopped.")

if __name__ == "__main__":
    main()