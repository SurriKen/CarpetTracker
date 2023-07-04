from dataset_processing import DatasetProcessing

# List of links for both cameras and csv file with classes
# all videos must have fps=25
video_links = [
    ['videos/classification_videos/video_sync/13-05 ВО_cam1_sync.mp4',
     'videos/classification_videos/video_sync/13-05 ВО_cam2_sync.mp4',
     'videos/classification_videos/csv/13-05 ВО.csv'],
    ['videos/classification_videos/video_sync/16-10 ЦП_cam1_sync.mp4',
     'videos/classification_videos/video_sync/16-10 ЦП_cam2_sync.mp4',
     'videos/classification_videos/csv/16-10 ЦП.csv'],
    ['videos/classification_videos/video_sync/МОС 19-40_cam1_sync.mp4',
     'videos/classification_videos/video_sync/МОС 19-40_cam2_sync.mp4',
     'videos/classification_videos/csv/МОС 19-40.csv'],
    ['videos/classification_videos/video_sync/Ночь 20-11_cam1_sync.mp4',
     'videos/classification_videos/video_sync/Ночь 20-11_cam2_sync.mp4',
     'videos/classification_videos/csv/Ночь 20-11.csv'],
    ['videos/classification_videos/video_sync/05.06.23_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/05.06.23_cam 2_sync.mp4',
     'videos/classification_videos/csv/05.06.23.csv'],
    ['videos/classification_videos/video_sync/05.06.23 вечер_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/05.06.23 вечер_cam 2_sync.mp4',
     'videos/classification_videos/csv/05.06.23 вечер.csv'],
    ['videos/classification_videos/video_sync/19.06 в 13.40_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/19.06 в 13.40_cam 2_sync.mp4',
     'videos/classification_videos/csv/19.06.23 в 13.40.csv'],
    ['videos/classification_videos/video_sync/20.06 в 14.02_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/20.06 в 14.02_cam 2_sync.mp4',
     'videos/classification_videos/csv/20.06.23 в 14.02.csv'],
    ['videos/classification_videos/video_sync/21.06 в 14.40_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/21.06 в 14.40_cam 2_sync.mp4',
     'videos/classification_videos/csv/21.06.23 в 14.40.csv'],
    ['videos/classification_videos/video_sync/21.06 в 16.44_cam 1_sync.mp4',
     'videos/classification_videos/video_sync/21.06 в 16.44_cam 2_sync.mp4',
     'videos/classification_videos/csv/21.06.23 в 16.44.csv'],
]
save_folder = f'datasets/class_videos_{len(video_links)}'

DatasetProcessing.video_class_dataset(
    video_links=video_links,
    save_folder=save_folder,
    separate=True,
)
