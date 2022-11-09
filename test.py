import importlib
import json

import numpy as np
import tensorflow
from PIL import Image

from parameters import BOX_CLASSIFICATION_MODEL_PATH
from predict import Predict

xxx = [[105., 107., 142., 155., 0.17594504, 0.17594495]]


def get_json_data(model_json, custom_obj_json):
    with open(model_json) as json_file:
        data = json.load(json_file)

    with open(custom_obj_json) as json_file:
        custom_dict = json.load(json_file)

    return data, custom_dict


def set_custom_objects(custom_dict):
    custom_object = {}
    for k, v in custom_dict.items():
        try:
            custom_object[k] = getattr(importlib.import_module(f".{v}", package="custom_objects"), k)
        except:
            continue
    return custom_object


model_json = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_json.trm"
custom_obj_json = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_custom_obj_json.trm"
model_best_weights = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_best_weights"
model_data, custom_dict = get_json_data(model_json, custom_obj_json)
custom_object = set_custom_objects(custom_dict)
model = tensorflow.keras.models.model_from_json(model_data, custom_objects=custom_object)
model.load_weights(model_best_weights)
img = Image.open('datasets/Train_0_0s-300s/frames/00000.png')
# img.show()
new_boxes = []
for b in xxx:
    w, h = img.size
    print(w, h)
    target_size = (384, 216)
    box_center = ((b[0] + int((b[2]-b[0])/2)) * h / 416, (b[1] + int((b[3]-b[1])/2)) * w / 416)
    print(box_center)
    if box_center[1] < w / 2:
        left = int(box_center[1] - 0.5 * target_size[0]) if int(box_center[1] - 0.5 * target_size[0]) > 0 else 0
        right = int(box_center[1] + 0.5 * target_size[0]) if left > 0 else int(target_size[0])
    else:
        right = int(box_center[1] + 0.5 * target_size[0]) if int(box_center[1] + 0.5 * target_size[0]) < w else target_size[0]
        left = int(box_center[1] - 0.5 * target_size[0]) if right < w else int(w - target_size[0])
    if box_center[0] < h / 2:
        top = int(box_center[0] - 0.5 * target_size[1]) if int(box_center[0] - 0.5 * target_size[1]) > 0 else 0
        bottom = int(box_center[0] + 0.5 * target_size[1]) if top > 0 else int(0.5 * target_size[1])
    else:
        bottom = int(box_center[0] + 0.5 * target_size[1]) if int(box_center[0] + 0.5 * target_size[1]) < h else target_size[0]
        top = int(box_center[0] - 0.5 * target_size[1]) if bottom < h else int(h - 0.5 * target_size[1])
    print((left, top, right, bottom))
    crop_img = np.array(img.crop((left, top, right, bottom)))
    print(crop_img.shape)
    res = model.predict(np.expand_dims(crop_img, axis=0))
    print(res)
    if np.argmax(res[0]):
        new_boxes.append(b)



