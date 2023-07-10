import os

import numpy as np

from parameters import ROOT_DIR
from tests.test_train_class import VideoClassifier


classes = ['115x200', '115x400', '150x300', '60x90', '85x150']
dataset = VideoClassifier.create_box_video_dataset(
        box_path=os.path.join(ROOT_DIR, 'tests/class_boxes_26_model3_full.dict'),
        split=0.9,
        frame_size=(128, 128),
)
# x = dataset.asdict()
# x = np.array(dataset.x_train)
# print(len(x))
# print(x[0][0].shape, x[0][1].shape)
# print(dataset.y_val)

# av = {}
# for cl in classes:
#     av[cl] = []
# for i in range(len(x)):
#     av[classes[dataset.y_train[i]]].append(len(x[i][0]))
# x = np.array(dataset.x_val)
# for i in range(len(x)):
#     av[classes[dataset.y_val[i]]].append(len(x[i][0]))
# print(av)
# for k, v in av.items():
#     print(f"{k} - Len={len(v)}, average={int(np.mean(v))}, max={np.max(v)}, min={np.min(v)}")
