import os
import shutil

from dataset_processing import DatasetProcessing
from parameters import ROOT_DIR

# Link to video (from repository/content root)
vid = [
    # 'videos/test 1_cam 1.mp4', 'videos/test 1_cam 2.mp4',
    # 'videos/test 2_cam 1.mp4', 'videos/test 2_cam 2.mp4',
    # 'videos/test 3_cam 1.mp4', 'videos/test 3_cam 2.mp4',
    # 'videos/test 4_cam 1.mp4', 'videos/test 4_cam 2.mp4',
    # 'videos/test 5_cam 1.mp4', 'videos/test 5_cam 2.mp4',
    # 'videos/classification_videos/13_05 ВО.mp4', 'videos/classification_videos/13_05 ВО_2.mp4',
    # 'videos/classification_videos/16-10 ЦП.mp4', 'videos/classification_videos/16-10 ЦП_2.mp4',
    # 'videos/classification_videos/МОС,19-40.mp4', 'videos/classification_videos/МОС,19-40_2.mp4',
    # 'videos/classification_videos/НОЧЬ,20-11.mp4', 'videos/classification_videos/НОЧЬ,20-11_2.mp4',
    # 'videos/test 6_cam 1.mp4', 'videos/test 6_cam 2.mp4',
    'videos/test 21_cam 1.mp4', 'videos/test 21_cam 2.mp4',
]

TARGET_FPS = 25

for v in vid:
    v = os.path.join(ROOT_DIR, v)
    shutil.move(v, 'test.mp4')
    DatasetProcessing.change_fps(
        video_path='test.mp4',
        save_path=v,
        set_fps=TARGET_FPS
    )
    os.remove('test.mp4')
