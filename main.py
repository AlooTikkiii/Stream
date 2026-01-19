from stream.server import StreamServer
from stream.buffer import FrameBuffer
import cv2
import time


def handle_stereo_pair(left_frame, right_frame, timestamp):
    # Display two frames side by side (convert RGB → BGR for display)
    left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
    combined = cv2.hconcat([left_bgr, right_bgr])
    
    cv2.imshow(f"Stereo {timestamp}", combined)

    if cv2.waitKey(1) == 27:  # ESC to exit
        print("ESC pressed. Stopping server...")
        server.stop()
        cv2.destroyAllWindows()


# Create buffer first, then server so callback can reference it
buffer = FrameBuffer(on_stereo_pair=handle_stereo_pair)
server = StreamServer(on_frame=buffer.add_frame)

server.start()

try:
    print("Server running... Press CTRL+C to stop.")
    while server.running:
        time.sleep(0.01)  # avoid 100% CPU
except KeyboardInterrupt:
    print("CTRL+C detected. Stopping server...")
finally:
    server.stop()
    cv2.destroyAllWindows()
