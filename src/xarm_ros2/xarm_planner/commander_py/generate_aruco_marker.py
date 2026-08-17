#!/usr/bin/env python3
"""Generate an ArUco texture compatible with the bundled aruco_ros library."""
import cv2
import numpy as np
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTURE_DIR = os.path.join(PKG_DIR, "models", "aruco_marker_texture")
os.makedirs(TEXTURE_DIR, exist_ok=True)

MARKER_ID = 0
CELL_SIZE = 30
CODE_BITS = 5
EXTERNAL_BORDER = 20

# aruco_ros::Dictionary::ARUCO, marker ID 0. The library uses a 5x5 code
# with one black cell around it; this is the exact code from dictionary.cpp.
ARUCO_ID0_CODE = 0x1084210
marker = np.zeros(
    ((CODE_BITS + 2) * CELL_SIZE, (CODE_BITS + 2) * CELL_SIZE),
    dtype=np.uint8,
)
bit_index = 0
for y in range(CODE_BITS - 1, -1, -1):
    for x in range(CODE_BITS - 1, -1, -1):
        if (ARUCO_ID0_CODE >> bit_index) & 1:
            y0 = (1 + y) * CELL_SIZE
            x0 = (1 + x) * CELL_SIZE
            marker[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE] = 255
        bit_index += 1

padded = cv2.copyMakeBorder(marker, EXTERNAL_BORDER, EXTERNAL_BORDER,
                            EXTERNAL_BORDER, EXTERNAL_BORDER,
                            cv2.BORDER_CONSTANT, value=255)

path = os.path.join(TEXTURE_DIR, f"aruco_id{MARKER_ID}.png")
cv2.imwrite(path, padded)
print(f"Marker texture saved: {path}")
print(f"Dimensions: {padded.shape[1]}x{padded.shape[0]}")
