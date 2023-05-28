from dataset_processing import DatasetProcessing
from tests.test import VideoClassifier
from utils import logger


logger.info("\n    --- Running test2.py ---    \n")


videos = 'datasets/class_videos'
dataset = DatasetProcessing.create_video_class_dataset_generator(
    folder_path=videos, split=0.8
)
logger.info(f'Dataset generator was formed\nclasses {dataset.classes}\ntrain_stat {dataset.train_stat}\n'
            f'val_stat {dataset.val_stat}\nparameters: {dataset.params}\n')

vc = VideoClassifier(num_classes=len(dataset.classes), weights='resnet3d')
vc.train(
    dataset=dataset,
    epochs=20,
    lr=0.005
)
