from dataset_processing import DatasetProcessing
from tests.test import VideoClassifier
from utils import logger


logger.info("\n    --- Running test2.py ---    \n")


videos = 'datasets/class_videos_2'
dataset = DatasetProcessing.create_video_class_dataset_generator(
    folder_path=videos, split=0.8
)
logger.info(f'Dataset generator was formed\nclasses {dataset.classes}\ntrain_stat {dataset.train_stat}\n'
            f'val_stat {dataset.val_stat}\nparameters: {dataset.params}\n')
# frame_size=(640, 720)
vc = VideoClassifier(num_classes=len(dataset.classes), weights='', frame_size=(128, 128))
vc.train(
    dataset=dataset,
    epochs=3,
    batch_size=4,
    lr=0.0001
)
