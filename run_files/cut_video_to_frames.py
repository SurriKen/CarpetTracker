import osfrom dataset_processing import DatasetProcessingfrom parameters import ROOT_DIRfrom utils import get_name_from_link# Link to video (from repository/content root)# folder = 'videos/classification_videos/video'folder = 'videos/classification_videos/video_sync'vid_num_range = [46, 48]vid = []for i in range(vid_num_range[0], vid_num_range[1]+1):    # vid.extend([f'{folder}/test {i}_cam 1.mp4', f'{folder}/test {i}_cam 2.mp4'])    vid.extend([f'{folder}/test {i}_cam 1_sync.mp4'])# vid = [#     'videos/classification_videos/video/test 37_cam 1.mp4', 'videos/classification_videos/video/test 37_cam 2.mp4',# ]print(vid)# FOLDER_FOR_FRAMES = 'datasets'FOLDER_FOR_FRAMES = os.path.join(ROOT_DIR, '/home/deny/Рабочий стол/1')# from_time - time in video to start cutting, sec (default - 0)# to_time - time in video to end cutting, sec (default - 10000), if to_time > frame count -> to_time = frame countfor v in vid:    v = os.path.join(ROOT_DIR, v)    print(v, os.path.isfile(v))    DatasetProcessing.video2frames(        video_path=v,        save_path=FOLDER_FOR_FRAMES,        from_time=0,        # to_time=120,        size=(640, 360)    )