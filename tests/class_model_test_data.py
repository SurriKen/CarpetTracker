import os

from nn_classificator import VideoClassifier
from parameters import ROOT_DIR

test_dataset = os.path.join(ROOT_DIR, 'tests/test_class_boxes_model5_Pex.dict')
model_weights = os.path.join(ROOT_DIR, 'video_class_train/model5_16f_(64, 64)_ca2_1/best.pt')

predict = VideoClassifier.evaluate_on_test_data(
    test_dataset=test_dataset,
    weights=model_weights,
)

print(predict)
