import numpy as np
from .protocol import FrameHeader

def decode_frame(
        header: FrameHeader,
        img_bytes: bytes
    ) -> np.ndarray
    """
    Convert raw payload to a decoded RGB image frame.

    Parameters:
        img_bytes: Raw memory buffer received from TCP
        meta: FrameHeader object (shape + dtype info)

    Returns:
        np.ndarray: (H, W, C) RGB frame
    """
    frame = np.frombuffer(img_bytes, dtype=header.dtype)
    try:
        frame = frame.reshape(header.shape)
    except ValueError as e:
        raise ValueError(
            f"Payload size mismatch: cannot reshape to {header.shape} "
            f"(received {len(img_bytes)} bytes)"
        )
    return frame
b