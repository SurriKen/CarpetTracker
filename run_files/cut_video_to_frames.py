import os

from dataset_processing import DatasetProcessing
from parameters import ROOT_DIR

# Link to video (from repository/content root)
vid = [
    # 'videos/test 1_cam 1.mp4', 'videos/test 1_cam 2.mp4',
    # 'videos/test 2_cam 1.mp4', 'videos/test 2_cam 2.mp4',
    # 'videos/test 3_cam 1.mp4', 'videos/test 3_cam 2.mp4',
    # 'videos/test 4_cam 1.mp4', 'videos/test 4_cam 2.mp4',
    # 'videos/test 5_cam 1.mp4', 'videos/test 5_cam 2.mp4',
    # 'videos/classification_videos/13-05 ВО_cam1.mp4', 'videos/classification_videos/13-05 ВО_cam2.mp4',
    # 'videos/classification_videos/16-10 ЦП_cam1.mp4', 'videos/classification_videos/16-10 ЦП_cam2.mp4',
    # 'videos/classification_videos/МОС 19-40_cam1.mp4', 'videos/classification_videos/МОС 19-40_cam2.mp4',
    # 'videos/classification_videos/Ночь 20-11_cam1.mp4', 'videos/classification_videos/Ночь 20-11_cam2.mp4',
    # 'videos/classification_videos/13_05 ВО_sync.mp4', 'videos/classification_videos/13_05 ВО_2_sync.mp4',
    # 'videos/classification_videos/16-10 ЦП_sync.mp4', 'videos/classification_videos/16-10 ЦП_2_sync.mp4',
    # 'videos/classification_videos/МОС,19-40_sync.mp4', 'videos/classification_videos/МОС,19-40_2_sync.mp4',
    # 'videos/classification_videos/НОЧЬ,20-11_sync.mp4', 'videos/classification_videos/НОЧЬ,20-11_2_sync.mp4',
    # 'videos/sync_test/test 5_cam 1_sync.mp4', 'videos/sync_test/test 5_cam 2_sync.mp4',
    # 'videos/test 6_cam 1.mp4', 'videos/test 6_cam 2.mp4',
    # 'videos/test 21_cam 1.mp4', 'videos/test 21_cam 2.mp4',
    # 'videos/init/test 22_cam 1.mp4', 'videos/init/test 22_cam 2.mp4',
    'videos/init/test 23_cam 1.mp4', 'videos/init/test 23_cam 2.mp4',
    'videos/init/test 24_cam 1.mp4', 'videos/init/test 24_cam 2.mp4',
    'videos/init/test 25_cam 1.mp4', 'videos/init/test 25_cam 2.mp4',
    'videos/init/test 26_cam 1.mp4', 'videos/init/test 26_cam 2.mp4',
    'videos/init/test 27_cam 1.mp4', 'videos/init/test 27_cam 2.mp4',
    'videos/init/test 28_cam 1.mp4', 'videos/init/test 28_cam 2.mp4',
    'videos/init/test 29_cam 1.mp4', 'videos/init/test 29_cam 2.mp4',
    'videos/init/test 30_cam 1.mp4', 'videos/init/test 30_cam 2.mp4',
]

FOLDER_FOR_FRAMES = 'datasets'

# from_time - time in video to start cutting, sec (default - 0)
# to_time - time in video to end cutting, sec (default - 10000), if to_time > frame count -> to_time = frame count
for v in vid:
    v = os.path.join(ROOT_DIR, v)
    DatasetProcessing.video2frames(
        video_path=v,
        save_path=os.path.join(ROOT_DIR, FOLDER_FOR_FRAMES),
        from_time=0,
        to_time=100,
        size=(640, 360)
    )