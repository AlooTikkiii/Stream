import argparse
import socket
import struct
import time

import cv2
import numpy as np


HEADER_FORMAT = "<iiiiiiq"


def dtype_to_code(dtype: np.dtype) -> int:
    if dtype == np.uint8:
        return 0
    if dtype == np.float32:
        return 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def pack_header(frame: np.ndarray, eye_id: int, meta_value: int) -> bytes:
    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    dtype_code = dtype_to_code(frame.dtype)
    size = frame.nbytes
    return struct.pack(
        HEADER_FORMAT,
        width,
        height,
        channels,
        dtype_code,
        size,
        eye_id,
        int(meta_value),
    )


def send_frame(sock: socket.socket, frame: np.ndarray, eye_id: int, meta_value: int) -> None:
    header = pack_header(frame, eye_id, meta_value)
    sock.sendall(header)
    sock.sendall(frame.tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send video frames over TCP.")
    parser.add_argument("video", help="Path to a video file.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=5001, help="Server port.")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override FPS (0 uses source FPS, default 30 if unavailable).",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Send each frame twice (eye_id 0 and 1).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the video when it ends.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Downscale frames to this resolution before sending (e.g. 1280 1210)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 30.0)
    frame_interval = 1.0 / fps

    read_time = []
    send_time = []
    sleep_time = []
    loop_time = []
    last_report = time.perf_counter()

    with socket.create_connection((args.host, args.port)) as sock:
        frame_idx = 0
        next_send = time.time()
        print(f"Resolution of an image : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        while True:
            r0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            r1 = time.perf_counter()
            read_time.append(r1 - r0)

            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # meta_value = frame_idx

            # s0 = time.perf_counter()
            # send_frame(sock, frame, 0, meta_value)
            # if args.stereo:
            #     send_frame(sock, frame, 1, meta_value)
            # s1 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]

            if args.downscale:
                target_w, target_h = args.downscale
                frame_ds = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA
                )

                sx = orig_w / target_w
                sy = orig_h / target_h
            else:
                frame_ds = frame
                sx = sy = 1.0

            # Pack scale info into meta if you want (optional)
            meta_value = frame_idx  # keep frame id simple

            s0 = time.perf_counter()
            send_frame(sock, frame_ds, 0, meta_value)
            if args.stereo:
                send_frame(sock, frame_ds, 1, meta_value)
            s1 = time.perf_counter()
            send_time.append(s1 - s0)

            frame_idx += 1

            next_send += frame_interval
            sleep_for = next_send - time.time()
            sleep_time.append(max(0.0, sleep_for))
            if sleep_for > 0:
                time.sleep(sleep_for)
            
            loop1 = time.perf_counter()
            #Append loop timing for reporting
            loop_time.append(loop1 - r0)
            

            # Report stats every 2 seconds
            if (loop1 - last_report) >= 2.0:
                avg_read = sum(read_time) / len(read_time) if read_time else 0.0
                avg_send = sum(send_time) / len(send_time) if send_time else 0.0
                avg_sleep = sum(sleep_time) / len(sleep_time) if sleep_time else 0.0
                avg_loop = sum(loop_time) / len(loop_time) if loop_time else 0.0
                total_frames = frame_idx
                print(
                    f"Frames sent: {total_frames}, "
                    f"Avg read time: {avg_read*1000:.2f} ms, "
                    f"Avg send time: {avg_send*1000:.2f} ms, "
                    f"Avg sleep time: {avg_sleep*1000:.2f} ms, "
                    f"Avg loop time: {avg_loop*1000:.2f} ms"
                )
                read_time.clear()
                send_time.clear()
                sleep_time.clear()
                loop_time.clear()
                last_report = loop1


if __name__ == "__main__":
    main()
