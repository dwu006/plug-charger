import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense devices found")
    exit(1)

device = devices[0]
serial = device.get_info(rs.camera_info.serial_number)
print(f"Using device: {device.get_info(rs.camera_info.name)} (Serial: {serial})")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
time.sleep(3)

output_dir = "data_collection"
episode_name = datetime.now().strftime("%Y%m%d_%H%M%S")
episode_path = os.path.join(output_dir, episode_name)
os.makedirs(episode_path, exist_ok=True)

frame_count = 0
save_count = 0
fps = 30
save_interval = fps

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
        
        if frame_count % save_interval == 0:
            depth_normalized = cv2.convertScaleAbs(depth_image, alpha=255.0/5000.0)
            rgba_image = np.dstack((color_image, depth_normalized))
            cv2.imwrite(os.path.join(episode_path, f"{save_count}.png"), rgba_image)
            save_count += 1
        
        frame_count += 1
        cv2.imshow('RealSense - Press Q to quit', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
