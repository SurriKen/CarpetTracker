from dataset_processing import DatasetProcessing

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
    'videos/test 6_cam 1.mp4': [31, '13:30:01'],
    'videos/test 6_cam 2.mp4': [2159, '13:30:02'],
}

# List of dicts, each dict in formate {'camera 1': 'link to video from camera 1', 'camera 2': 'link to video from camera 2'}
# all videos must g=have fps=25
sync_videos = [
    # {'camera 1': 'videos/test 1_cam 1.mp4', 'camera 2': 'videos/test 1_cam 2.mp4'},
    # {'camera 1': 'videos/test 2_cam 1.mp4', 'camera 2': 'videos/test 2_cam 2.mp4'},
    # {'camera 1': 'videos/test 3_cam 1.mp4', 'camera 2': 'videos/test 3_cam 2.mp4'},
    # {'camera 1': 'videos/test 4_cam 1.mp4', 'camera 2': 'videos/test 4_cam 2.mp4'},
    # {'camera 1': 'videos/test 5_cam 1.mp4', 'camera 2': 'videos/test 5_cam 2.mp4'},
    # {'camera 1': 'videos/classification_videos/13-05 ВО_cam1.mp4', 'camera 2': 'videos/classification_videos/13-05 ВО_cam2.mp4'},
    # {'camera 1': 'videos/classification_videos/16-10 ЦП_cam1.mp4', 'camera 2': 'videos/classification_videos/16-10 ЦП_cam2.mp4'},
    # {'camera 1': 'videos/classification_videos/МОС 19-40_cam1.mp4', 'camera 2': 'videos/classification_videos/МОС 19-40_cam2.mp4'},
    # {'camera 1': 'videos/classification_videos/Ночь 20-11_cam1.mp4', 'camera 2': 'videos/classification_videos/Ночь 20-11_cam2.mp4'},
    {'camera 1': 'videos/test 6_cam 1.mp4', 'camera 2': 'videos/test 6_cam 2.mp4'},
]

for pair in sync_videos:
    save_name_1 = f"{pair.get('camera 1').split('/')[-1].split('.')[0]}_sync.mp4"
    DatasetProcessing.synchronize_video(
        video_path=pair.get('camera 1'),
        save_path=f"videos/sync_test/{save_name_1}",
        from_frame=sync_data.get(pair.get('camera 1'))[0]
    )
    save_name_2 = f"{pair.get('camera 2').split('/')[-1].split('.')[0]}_sync.mp4"
    DatasetProcessing.synchronize_video(
        video_path=pair.get('camera 2'),
        save_path=f"videos/sync_test/{save_name_2}",
        from_frame=sync_data.get(pair.get('camera 2'))[0]
    )
