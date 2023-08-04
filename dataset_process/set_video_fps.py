import os
import shutil

import cv2

from dataset_process.dataset_processing import DatasetProcessing
from parameters import *
from utils import get_name_from_link

# Link to video (from repository/content root)
folder = os.path.join(ROOT_DIR, 'videos/init')
vid = os.listdir(folder)
print("Folder content =", vid)
TARGET_FPS = 25

for v in vid:
    v = os.path.join(folder, v)
    video_capture = cv2.VideoCapture()
    video_capture.open(v)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    vn = get_name_from_link(v)
    if fps == 25:
        print(f"Video {vn} already has fps = 25")
    else:
        print(f"Video {vn} has fps = {fps}. Setting fps to 25")
        shutil.move(v, 'test.mp4')
        DatasetProcessing.change_fps(
            video_path='test.mp4',
            save_path=v,
            set_fps=TARGET_FPS
        )
        os.remove('test.mp4')
