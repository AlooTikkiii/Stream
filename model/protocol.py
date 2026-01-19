import struct
from dataclasses import dataclass

from numpy import size


HEADER_FORMAT = "<iiiiiiq"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)   

@dataclass
class FrameHeader:
    """Data class to represent the frame header structure."""
    width: int
    height: int
    channels: int
    dtype_code: int
    size: int
    eye_id: int
    meta_value: int

    @property
    def shape(self):
        """Return the shape of the image as (height, width, channels)."""
        return (self.height, self.width, self.channels)
    
    @property
    def dtype(self):
        """Return the numpy dtype corresponding to the dtype_code."""
        dtype_map = {
            0: 'uint8',
            1: 'float32',
            # Add more mappings as needed
        }
        if self.dtype_code not in dtype_map:
            raise ValueError(f"Unsupported dtype code: {self.dtype_code}")
        
        return dtype_map[self.dtype_code]
    
    
def unpack_header(header_bytes: bytes) -> FrameHeader:
    """Unpack header bytes into a FrameHeader instance."""
    fields = struct.unpack(HEADER_FORMAT, header_bytes)
    width, height, channels, dtype_code, size, eye_id, meta_value = fields  
    #Basic validation
    if width <= 0 or height <= 0 or channels <= 0 or size <= 0:
        raise ValueError(f"Invalid header values received : {width}x{height}x{channels}, size={size}")
    
    if eye_id not in (0, 1):
        raise ValueError(f"Invalid eye_id received: {eye_id}")
    
    return FrameHeader(*fields)