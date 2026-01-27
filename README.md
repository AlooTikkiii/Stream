# Stream
Python service that receives stereo frames from a Varjo Unity app, runs YOLO on the paired frames, and sends detections back to Unity using the shared timestamp (`meta_value`) for matching.

## Setup
### Install environment
This repo ships a minimal Conda environment and expects a few extra Python packages.

1) Create and activate the Conda env:
```bash
conda env create -f environment.yaml
conda activate stream
```

2) Install runtime dependencies not listed in `environment.yaml`:
```bash
pip install ultralytics torch
```

Notes:
- If you want CUDA, install the appropriate PyTorch build for your GPU/driver.
- The YOLO model file is expected at the path set in `main.py` (`YOLO_MODEL_PATH`).

### How to run the code
1) Start your Unity app so it connects to the Python server and streams frames.
2) Run the Python pipeline:
```bash
python main.py
```

Optional: stream a local video into the server (useful for testing without Unity):
```bash
python sender.py /path/to/video.mp4 --host 127.0.0.1 --port 5001 --stereo --downscale 1280 1210
```

Key runtime settings live at the top of `main.py`:
- `STREAM_HOST` / `STREAM_PORT`: where Unity connects to send frames.
- `DET_HOST` / `DET_PORT`: where Unity listens for detections.
- `YOLO_MODEL_PATH` and `DEVICE`: model file + `cuda`/`cpu`.
- `SHOW`: set `True` to visualize detections.

## Structure
Concise map of the codebase:

- `main.py`: Orchestrates the pipeline. Receives stereo frames, runs YOLO, postprocesses matches, and pushes results to the sender queue.
- `sender.py`: Simple TCP video streamer for local testing (sends frames to the server).
- `utils.py`: YOLO post-processing helpers (top‑k selection, stereo matching).
- `model/server.py`: TCP server receiving frames from Unity.
- `model/buffer.py`: Timestamp-based stereo pairing buffer.
- `model/protocol.py`: Frame header structure and validation.
- `model/decoder.py`: Decodes raw bytes into numpy frames.
- `model/sender.py`: Sends detection packets back to Unity (timestamp + matched boxes).
- `model/logger.py`: Colored logging setup.
