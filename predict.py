from parameters import *
# from predict_sync_videos import predict

i = 54
save_folder = os.path.join(ROOT_DIR, f'temp/test {i}')
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
video_paths = {
    'model_1': os.path.join(ROOT_DIR, f'videos/sync_test/test {i}_cam 1_sync.mp4'),
    'model_2': os.path.join(ROOT_DIR, f'videos/sync_test/test {i}_cam 2_sync.mp4'),
    'save_path': save_folder,
}
# video_paths = {
#     'model_1': "/dev/video0",
#     'model_2': "rtsp://zephyr.rtsp.stream/movie?streamKey=92d5277be96274d35ef49d8f94f8177a",
#     'save_path': save_folder,
# }

# predict is run from main.py in background tasks
# result = predict(video_paths=video_paths, stream=False, save_predict_video=True)
