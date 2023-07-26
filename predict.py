from datetime import datetime
import os

from parameters import *
from predict_sync_videos import predict
from utils import save_data


for i in range(33, 53):
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_folder = os.path.join(ROOT_DIR, f'temp/test {i}')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    video_paths = {
        'model_1': os.path.join(DATASET_DIR, f'videos/classification_videos/video_sync/test {i}_cam 1_sync.mp4'),
        'model_2': os.path.join(DATASET_DIR, f'videos/classification_videos/video_sync/test {i}_cam 2_sync.mp4'),
        'save_path': save_folder,
    }

    # video_paths = {
    #     'model_1': "/dev/video0",
    #     'model_2': "rtsp://zephyr.rtsp.stream/movie?streamKey=92d5277be96274d35ef49d8f94f8177a",
    #     'save_path': os.path.join(ROOT_DIR, f'temp/{dt}.mp4'),
    # }

    result = predict(video_paths=video_paths, stream=False, save_predict_video=True)
    save_data(data=result, folder_path=os.path.join(ROOT_DIR, 'temp'), filename=f"{dt}")
