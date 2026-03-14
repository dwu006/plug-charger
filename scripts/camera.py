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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
time.sleep(1)

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()
        if not depthFrame or not colorFrame:
            continue
        depthImage = np.asanyarray(depthFrame.get_data())
        colorImage = np.asanyarray(colorFrame.get_data())
        depthColormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)
        display = np.hstack((colorImage, depthColormap))
        cv2.imshow('q to quit', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
