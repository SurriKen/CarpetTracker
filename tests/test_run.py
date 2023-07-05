from dataset_processing import DatasetProcessing
from tests.classif_models import VideoClassifier
from utils import logger


logger.info("\n    --- Running test_run.py ---    \n")


# videos = 'datasets/class_videos_2'
# videos = 'tests/mnist_videos'
# name = 'model'
# dataset = DatasetProcessing.create_video_class_dataset_generator(
#     folder_path=videos, split=0.8
# )
# logger.info(f'Dataset generator was formed\nclasses {dataset.classes}\ntrain_stat {dataset.train_stat}\n'
#             f'val_stat {dataset.val_stat}\nparameters: {dataset.params}\n')
# # frame_size=(640, 720)
# vc = VideoClassifier(num_classes=len(dataset.classes), weights='', frame_size=(32, 32), name=name)
# vc.train(
#     dataset=dataset,
#     epochs=5,
#     batch_size=8,
#     lr=0.002
# )

print([6, *[1,2,3]])
