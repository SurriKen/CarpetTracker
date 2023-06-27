import os.path

from dataset_processing import DatasetProcessing
from parameters import ROOT_DIR

# List of links:
# index 0 - link to image folder,
# index 1 - link to txt label folder,
# index 2 - weight coefficien for this folder, float from 0. to 1.
data = [
    # ['datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01',
    #  'datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01_', 1.0],
    # ['datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02',
    #  'datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02_', 1.0],
    # ['datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03',
    #  'datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03_', 1.0],
    # ['datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04',
    #  'datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04_', 1.0],
    # ['datasets/От разметчиков/batch_05_#147536/batch_05',
    #  'datasets/От разметчиков/batch_05_#147536/batch_05_', 1.0],
    # ['datasets/От разметчиков/batch_mine/obj_train_data/batch_01',
    #  'datasets/От разметчиков/batch_mine/obj_train_data/batch_01_', 1.0],
    [os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/red'),
     os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_1/boxes'), 1.0],
    # ['datasets/diff/diff/cam_2/init', 'datasets/diff/diff/cam_2/boxes', 1.0]
]

split = 0.9
save_path_1 = os.path.join(ROOT_DIR, 'datasets/yolov8_red_camera_1')
save_path_2 = os.path.join(ROOT_DIR, 'datasets/yolov8_red_camera_2')

# form dataset for camera 1 with frame shape 1920, 1080
DatasetProcessing.form_dataset_for_train(
    data=data,
    split=split,
    save_path=save_path_1,
    condition={'orig_shape': (1920, 1080)}
)
# put box on image to check quality of dataset
# for l in ['train', 'val']:
#     DatasetProcessing.put_box_on_image(
#         images=f'{save_path_1}/{l}/images',
#         labels=f'{save_path_1}/{l}/labels',
#         save_path=f'{save_path_1}/{l}/img+lbl'
#     )
#
data = [
    [os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/red'),
     os.path.join(ROOT_DIR, 'datasets/diff/diff/cam_2/boxes'), 1.0]
]

# form dataset for camera 2 with frame shape 640, 360
DatasetProcessing.form_dataset_for_train(
    data=data,
    split=split,
    save_path=save_path_2,
    condition={'orig_shape': (640, 360)}
)
# # put box on image to check quality of dataset
# for l in ['train', 'val']:
#     DatasetProcessing.put_box_on_image(
#         images=f'{save_path_2}/{l}/images',
#         labels=f'{save_path_2}/{l}/labels',
#         save_path=f'{save_path_2}/{l}/img+lbl'
#     )
