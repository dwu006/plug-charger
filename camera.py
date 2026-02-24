import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import pyrealsense2 as rs
import numpy as np
import cv2
import time

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense devices found")
    exit(1)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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
        display = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Preview - Press Q to quit', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
