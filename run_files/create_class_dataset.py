from dataset_processing import DatasetProcessing

# List of lisks to csv file with info about class/carpet size, frame to start for this carpet and frame to end
csv_files = [
    'videos/classification_videos/13-05 ВО.csv',
    'videos/classification_videos/16-10 ЦП.csv',
    'videos/classification_videos/МОС 19-40.csv',
    'videos/classification_videos/Ночь 20-11.csv',
]

# List of pairs links for both cameras
# all videos must g=have fps=25
video_links = [
    ['videos/sync_test/13-05 ВО_cam1_sync.mp4', 'videos/sync_test/13-05 ВО_cam2_sync.mp4'],
    ['videos/sync_test/16-10 ЦП_cam1_sync.mp4', 'videos/sync_test/16-10 ЦП_cam2_sync.mp4'],
    ['videos/sync_test/МОС 19-40_cam1_sync.mp4', 'videos/sync_test/МОС 19-40_cam2_sync.mp4'],
    ['videos/sync_test/Ночь 20-11_cam1_sync.mp4', 'videos/sync_test/Ночь 20-11_cam2_sync.mp4'],
]
save_folder = 'datasets/class_videos_join'

DatasetProcessing.video_class_dataset(
    csv_files=csv_files,
    video_links=video_links,
    save_folder=save_folder,
    separate=False,
)