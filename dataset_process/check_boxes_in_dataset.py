from dataset_process.dataset_processing import DatasetProcessing

# List of links: index 0 - link to image folder, index 1 - link to txt label folder
data = [
    # ['datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01',
    #  'datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01_'],
    # ['datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02',
    #  'datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02_'],
    # ['datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03',
    #  'datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03_'],
    # ['datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04',
    #  'datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04_'],
    ['datasets/От разметчиков/batch_05_#147536/batch_05',
     'datasets/От разметчиков/batch_05_#147536/batch_05_'],
]

# List of links where to save images with boxes
save = [
    # 'datasets/От разметчиков/batch_01_#108664/obj_train_data/img+lbl',
    # 'datasets/От разметчиков/batch_02_#110902/obj_train_data/img+lbl',
    # 'datasets/От разметчиков/batch_03_#112497/obj_train_data/img+lbl',
    # 'datasets/От разметчиков/batch_04_#119178/obj_train_data/img+lbl',
    'datasets/От разметчиков/batch_05_#147536/img+lbl',
]

for i, p in enumerate(data):
    DatasetProcessing.put_box_on_image(
        images=p[0],
        labels=p[1],
        save_path=save[i]
    )