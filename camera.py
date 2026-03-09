import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import pyrealsense2 as rs
import numpy as np
import cv2
import time

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("no devices found")
    exit(1)

pipeline = rs.pipeline()
config = rs.config()
# Use 640x480 for both depth and color, then apply a digital zoom in the preview
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30
COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 30
ZOOM_FACTOR = 1.2  # 1.0 = no zoom, >1.0 = zoom in

config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
pipeline.start(config)
time.sleep(1)

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Ensure sizes match, then apply digital zoom by center-cropping and resizing back
        if depth_colormap.shape[:2] != color_image.shape[:2]:
            depth_colormap = cv2.resize(
                depth_colormap,
                (color_image.shape[1], color_image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        h, w = color_image.shape[:2]
        if ZOOM_FACTOR > 1.0:
            new_w = int(w / ZOOM_FACTOR)
            new_h = int(h / ZOOM_FACTOR)
            new_w = max(1, min(w, new_w))
            new_h = max(1, min(h, new_h))

            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            x2 = x1 + new_w
            y2 = y1 + new_h

            color_cropped = color_image[y1:y2, x1:x2]
            depth_cropped = depth_colormap[y1:y2, x1:x2]

            color_image = cv2.resize(color_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_colormap = cv2.resize(depth_cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        display = np.hstack((color_image, depth_colormap))
        cv2.imshow('q to quit', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
