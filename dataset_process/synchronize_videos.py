import os.path

from dataset_process.dataset_processing import DatasetProcessing
from parameters import ROOT_DIR


# List of dicts, each dict in formate {'camera 1': 'link to video from camera 1', 'camera 2': 'link to video from camera 2'}
# all videos must have fps=25
sync_videos = [
    {'camera 1': [os.path.join(ROOT_DIR, 'videos/init/test 53_cam 1.mp4'), 622],
     'camera 2': [os.path.join(ROOT_DIR, 'videos/init/test 53_cam 2.mp4'), 80]},
]
save_folder = os.path.join(ROOT_DIR, 'videos/sync_test')

for pair in sync_videos:
    print(pair)
    # save_name_1 = f"{pair.get('camera 1')[0].split('/')[-1].split('.')[0]}_sync.mp4"
    vn = pair.get('camera 1')[0].split('/')[-1].split('.')[:-1]
    save_name_1 = ''
    for v in vn:
        save_name_1 = f"{save_name_1}.{v}"
    save_name_1 = f"{save_name_1[1:]}_sync.mp4"
    # save_name_1 = f"test 52_cam 1_sync.mp4"
    print(save_name_1)
    DatasetProcessing.synchronize_video(
        video_path=pair.get('camera 1')[0],
        save_path=os.path.join(ROOT_DIR, f"{save_folder}/{save_name_1}"),
        from_frame=pair.get('camera 1')[1]
    )
    vn = pair.get('camera 2')[0].split('/')[-1].split('.')[:-1]
    save_name_2 = ''
    for v in vn:
        save_name_2 = f"{save_name_2}.{v}"
    save_name_2 = f"{save_name_2[1:]}_sync.mp4"
    # save_name_2 = f"test 52_cam 2_sync.mp4"
    print(save_name_2)
    DatasetProcessing.synchronize_video(
        video_path=pair.get('camera 2')[0],
        save_path=os.path.join(ROOT_DIR, f"{save_folder}/{save_name_2}"),
        from_frame=pair.get('camera 2')[1]
    )
