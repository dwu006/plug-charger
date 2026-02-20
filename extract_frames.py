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

    video_name, _ = os.path.splitext(file)
    output_top = os.path.join(os.getcwd() + "/output/frames/observation.image.top", video_name)
    output_wrist = os.path.join(os.getcwd() + "/output/frames/observation.image.wrist", video_name)
    os.makedirs(output_top, exist_ok=True)
    os.makedirs(output_wrist, exist_ok=True)

    frame_num = 0 

    while True:
        ret_top, frame_top = top_img.read()
        ret_wrist, frame_wrist = wrist_img.read()
        if not ret_top or not ret_wrist:
            print("no frames")
            break

        if frame_num % FPS == 0:
            cv2.imwrite(output_top + "/" + str(INDEX)+".jpg", frame_top)
            cv2.imwrite(output_wrist + "/" + str(INDEX)+ ".jpg", frame_wrist)
            INDEX += 1
        frame_num += 1

    top_img.release()
    wrist_img.release()

