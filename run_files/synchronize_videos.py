import os.path

from dataset_processing import DatasetProcessing
from parameters import ROOT_DIR, DATASET_DIR

# dict with info about video, where key - link to video, value - list of info,
# index 0 - frame to start synch, index 1 - timestamp in video for this frame
# Must have info for videos from both cameras
sync_data = {
    # 'videos/test 1_cam 1.mp4': [519, '18:00:10'],
    # 'videos/test 1_cam 2.mp4': [42, '18:00:10'],
    # 'videos/test 2_cam 1.mp4': [94, '16:28:08'],
    # 'videos/test 2_cam 2.mp4': [45, '16:28:08'],
    # 'videos/test 3_cam 1.mp4': [693, '18:00:27'],
    # 'videos/test 3_cam 2.mp4': [297, '18:00:27'],
    # 'videos/test 4_cam 1.mp4': [22, '13:05:21'],
    # 'videos/test 4_cam 2.mp4': [976, '13:05:21'],
    # 'videos/test 5_cam 1.mp4': [376, '13:41:57'],
    # 'videos/test 5_cam 2.mp4': [400, '13:41:56'],
    # 'videos/classification_videos/13-05 ВО_cam1.mp4': [32, '13:05:21'],
    # 'videos/classification_videos/13-05 ВО_cam2.mp4': [986, '13:05:21'],
    # 'videos/classification_videos/16-10 ЦП_cam1.mp4': [476, '16:08:55'],
    # 'videos/classification_videos/16-10 ЦП_cam2.mp4': [8, '16:08:56'],
    # 'videos/classification_videos/МОС 19-40_cam1.mp4': [110, '19:40:52'],
    # 'videos/classification_videos/МОС 19-40_cam2.mp4': [16, '19:40:52'],
    # 'videos/classification_videos/Ночь 20-11_cam1.mp4': [193, '20:11:13'],
    # 'videos/classification_videos/Ночь 20-11_cam2.mp4': [8, '20:11:13'],
    # 'videos/test 6_cam 1.mp4': [31, '13:30:01'],
    # 'videos/test 6_cam 2.mp4': [2159, '13:30:02'],
    # 'videos/test 21_cam 1.mp4': [195, '18:59:35'],
    # 'videos/test 21_cam 2.mp4': [931, '18:59:38'],
    # 'videos/init/test 22_cam 1.mp4': [715, '19:45:35'],
    # 'videos/init/test 22_cam 2.mp4': [414, '19:45:35'],
    # 'videos/init/test 23_cam 1.mp4': [934, ''],
    # 'videos/init/test 23_cam 2.mp4': [179, ''],
    # 'videos/init/test 24_cam 1.mp4': [772, ''],
    # 'videos/init/test 24_cam 2.mp4': [13, ''],
    # 'videos/init/test 25_cam 1.mp4': [27, ''],
    # 'videos/init/test 25_cam 2.mp4': [482, ''],
    # 'videos/init/test 26_cam 1.mp4': [27, ''],
    # 'videos/init/test 26_cam 2.mp4': [69, ''],
    # 'videos/init/test 27_cam 1.mp4': [192, ''],
    # 'videos/init/test 27_cam 2.mp4': [189, ''],
    # 'videos/init/test 28_cam 1.mp4': [72, ''],
    # 'videos/init/test 28_cam 2.mp4': [368, ''],
    # 'videos/init/test 29_cam 1.mp4': [244, ''],
    # 'videos/init/test 29_cam 2.mp4': [122, ''],
    # 'videos/init/test 30_cam 1.mp4': [48, ''],
    # 'videos/init/test 30_cam 2.mp4': [351, ''],
    # 'videos/classification_videos/05.06.23_cam 1.mp4': [92, ''],
    # 'videos/classification_videos/05.06.23_cam 2.mp4': [519, ''],
    # 'videos/classification_videos/05.06.23 вечер_cam 1.mp4': [51, ''],
    # 'videos/classification_videos/05.06.23 вечер_cam 2.mp4': [27, ''],
    # 'videos/classification_videos/19.06 в 13.40_cam 1.mp4': [974, ''],
    # 'videos/classification_videos/19.06 в 13.40_cam 2.mp4': [13, ''],
    # 'videos/classification_videos/20.06 в 14.02_cam 1.mp4': [34, ''],
    # 'videos/classification_videos/20.06 в 14.02_cam 2.mp4': [69, ''],
    # 'videos/classification_videos/21.06 в 14.40_cam 1.mp4': [94, ''],
    # 'videos/classification_videos/21.06 в 14.40_cam 2.mp4': [367, ''],
    # 'videos/classification_videos/21.06 в 16.44_cam 1.mp4': [313, ''],
    # 'videos/classification_videos/21.06 в 16.44_cam 2.mp4': [123, ''],
    # 'videos/classification_videos/video/test 34_cam 1.mp4': [17, ''],
    # 'videos/classification_videos/video/test 34_cam 2.mp4': [33, ''],
    # 'videos/classification_videos/video/test 35_cam 1.mp4': [91, ''],
    # 'videos/classification_videos/video/test 35_cam 2.mp4': [40, ''],
    # 'videos/init/test 31_cam 1.mp4': [189, ''],
    # 'videos/init/test 31_cam 2.mp4': [39, ''],
    # 'videos/init/test 32_cam 1.mp4': [8, ''],
    # 'videos/init/test 32_cam 2.mp4': [73, ''],
    # 'videos/init/test 33_cam 1.mp4': [19, ''],
    # 'videos/init/test 33_cam 2.mp4': [531, ''],
    # 'videos/classification_videos/video/test 36_cam 1.mp4': [11, ''],
    # 'videos/classification_videos/video/test 36_cam 2.mp4': [75, ''],
}

# List of dicts, each dict in formate {'camera 1': 'link to video from camera 1', 'camera 2': 'link to video from camera 2'}
# all videos must g=have fps=25
sync_videos = [
    # {'camera 1': ['videos/classification_videos/video/test 37_cam 1.mp4', 120],
    #  'camera 2': ['videos/classification_videos/video/test 37_cam 2.mp4', 54]},
    # {'camera 1': ['videos/classification_videos/video/test 38_cam 1.mp4', 80],
    #  'camera 2': ['videos/classification_videos/video/test 38_cam 2.mp4', 169]},
    # {'camera 1': ['videos/classification_videos/video/test 39_cam 1.mp4', 92],
    #  'camera 2': ['videos/classification_videos/video/test 39_cam 2.mp4', 165]},
    # {'camera 1': ['videos/classification_videos/video/test 40_cam 1.mp4', 55],
    #  'camera 2': ['videos/classification_videos/video/test 40_cam 2.mp4', 156]},
    # {'camera 1': ['videos/classification_videos/video/test 41_cam 1.mp4', 97],
    #  'camera 2': ['videos/classification_videos/video/test 41_cam 2.mp4', 305]},
    # {'camera 1': ['videos/classification_videos/video/test 42_cam 1.mp4', 178],
    #  'camera 2': ['videos/classification_videos/video/test 42_cam 2.mp4', 460]},
    # {'camera 1': ['videos/classification_videos/video/test 43_cam 1.mp4', 26],
    #  'camera 2': ['videos/classification_videos/video/test 43_cam 2.mp4', 335]},
    # {'camera 1': ['videos/classification_videos/video/test 44_cam 1.mp4', 164],
    #  'camera 2': ['videos/classification_videos/video/test 44_cam 2.mp4', 111]},
    # {'camera 1': ['videos/classification_videos/video/test 45_cam 1.mp4', 35],
    #  'camera 2': ['videos/classification_videos/video/test 45_cam 2.mp4', 482]},
    # {'camera 1': ['videos/classification_videos/video/test 46_cam 1.mp4', 112],
    #  'camera 2': ['videos/classification_videos/video/test 46_cam 2.mp4', 2240]},
    # {'camera 1': ['videos/classification_videos/video/test 47_cam 1.mp4', 1380],
    #  'camera 2': ['videos/classification_videos/video/test 47_cam 2.mp4', 573]},
    # {'camera 1': ['videos/classification_videos/video/test 48_cam 1.mp4', 2705],
    #  'camera 2': ['videos/classification_videos/video/test 48_cam 2.mp4', 482]},
    # {'camera 1': ['videos/classification_videos/video/test 49_cam 1.mp4', 388],
    #  'camera 2': ['videos/classification_videos/video/test 49_cam 2.mp4', 561]},
    # {'camera 1': ['videos/classification_videos/video/test 50_cam 1.mp4', 2546],
    #  'camera 2': ['videos/classification_videos/video/test 50_cam 2.mp4', 2743]},
    # {'camera 1': ['videos/classification_videos/video/test 51_cam 1.mp4', 2092],
    #  'camera 2': ['videos/classification_videos/video/test 51_cam 2.mp4', 2669]},
    {'camera 1': ['videos/classification_videos/video/СЕРГЕЕВ КАМ1.mp4', 109],
     'camera 2': ['videos/classification_videos/video/СЕРГЕЕВ КАМ2.mp4', 145]},
]
save_folder = 'videos/classification_videos/video_sync'

for pair in sync_videos:
    # save_name_1 = f"{pair.get('camera 1').split('/')[-1].split('.')[0]}_sync.mp4"
    # vn = pair.get('camera 1')[0].split('/')[-1].split('.')[:-1]
    # save_name_1 = ''
    # for v in vn:
    #     save_name_1 = f"{save_name_1}.{v}"
    # save_name_1 = f"{save_name_1[1:]}_sync.mp4"
    save_name_1 = f"test 52_cam 1_sync.mp4"
    print(save_name_1)
    DatasetProcessing.synchronize_video(
        video_path=os.path.join(DATASET_DIR, pair.get('camera 1')[0]),
        save_path=os.path.join(DATASET_DIR, f"{save_folder}/{save_name_1}"),
        from_frame=pair.get('camera 1')[1]
    )
    # vn = pair.get('camera 2')[0].split('/')[-1].split('.')[:-1]
    # save_name_2 = ''
    # for v in vn:
    #     save_name_2 = f"{save_name_2}.{v}"
    # save_name_2 = f"{save_name_2[1:]}_sync.mp4"
    save_name_2 = f"test 52_cam 2_sync.mp4"
    print(save_name_2)
    DatasetProcessing.synchronize_video(
        video_path=os.path.join(DATASET_DIR, pair.get('camera 2')[0]),
        save_path=os.path.join(DATASET_DIR, f"{save_folder}/{save_name_2}"),
        from_frame=pair.get('camera 2')[1]
    )
