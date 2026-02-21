import cv2
import os

TOP_VIDEO_PATH = os.path.join(os.getcwd(), "test/videos/observation.images.top/chunk-000")
WRIST_VIDEO_PATH = os.path.join(os.getcwd(), "test/videos/observation.images.wrist/chunk-000")
FPS = 30

for file in os.listdir(TOP_VIDEO_PATH):
    INDEX = 0
    top_path = os.path.join(TOP_VIDEO_PATH, file)
    wrist_path = os.path.join(WRIST_VIDEO_PATH, file)

    top_img = cv2.VideoCapture(top_path)
    wrist_img = cv2.VideoCapture(wrist_path)

    if not top_img.isOpened():
        wrist_img.release()
        continue
    if not wrist_img.isOpened():
        top_img.release()
        continue

    video_name, _ = os.path.splitext(file)
    output_top = os.path.join(os.getcwd(), "output/frames/observation.image.top", video_name)
    output_wrist = os.path.join(os.getcwd(), "output/frames/observation.image.wrist", video_name)
    os.makedirs(output_top, exist_ok=True)
    os.makedirs(output_wrist, exist_ok=True)

    frame_num = 0 

    while True:
        ret_top, frame_top = top_img.read()
        ret_wrist, frame_wrist = wrist_img.read()
        if not ret_top or not ret_wrist:
            break

        if frame_num % FPS == 0:
            cv2.imwrite(os.path.join(output_top, f"{INDEX}.jpg"), frame_top)
            cv2.imwrite(os.path.join(output_wrist, f"{INDEX}.jpg"), frame_wrist)
            INDEX += 1
        frame_num += 1
    print({file}, "done")

    top_img.release()
    wrist_img.release()

print("all done")