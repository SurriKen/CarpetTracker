import importlib
import json
import os
import shutil
import cv2

import numpy as np
import tensorflow as tf
import torch
import torchvision
from PIL import Image, ImageFont, ImageDraw
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from VideoProcessing import VideoProcessing
from prepare_dataset import PrepareDataset
from utils import get_colors


class Predict:

    def __init__(self, video_path, yolo_model_path, class_model_path, save_path, data_dict_path, yolo_version='v3'):
        self.video_path = video_path
        self.yolo_model_path = yolo_model_path
        self.class_model_path = class_model_path
        self.save_path = save_path
        self.yolo_version = yolo_version
        self.data_dict_path = data_dict_path
        f = open(f'{self.yolo_model_path}/instructions/parameters/2_object_detection.json')
        self.classes_names = json.load(f)['classes_names']
        f.close()
        f2 = open(f'{self.yolo_model_path}/instructions/parameters/1_image.json')
        dict_ = json.load(f2)
        self.target_yolo_size = (dict_["width"], dict_["height"])
        self.image_yolo_scaler = dict_["scaler"]
        f2.close()
        self.tmp_folder = f"{save_path}/tmp2"
        try:
            os.mkdir(self.tmp_folder)
        except:
            shutil.rmtree(self.tmp_folder, ignore_errors=True)
            os.mkdir(self.tmp_folder)
        os.mkdir(f"{self.tmp_folder}/init")
        os.mkdir(f"{self.tmp_folder}/pred")

        VideoProcessing.video2frames(
            video_path=self.video_path,
            save_path=f"{self.tmp_folder}",
            max_time=None,
            predict_mode=True
        )
        if self.class_model_path:
            self.classification_classes = ["no", "yes"]
            self.class_model = self.set_model(model_type='normal')
            f3 = open(f'{self.class_model_path}/instructions/parameters/1_image.json')
            dict_ = json.load(f3)
            self.target_class_size = (dict_["width"], dict_["height"])
            self.classification_scaler = dict_["scaler"]
            f3.close()
        self.yolo_model = self.set_model(model_type='yolo')
        self.box_colors = get_colors(self.classes_names)

    def set_model(self, model_type='normal'):

        if model_type == 'normal':
            model_json = f"{self.class_model_path}/trained_model_json.trm"
            custom_obj_json = f"{self.class_model_path}/trained_model_custom_obj_json.trm"
            model_best_weights = f"{self.class_model_path}/trained_model_best_weights"
            model_data, custom_dict = Predict.__get_json_data(model_json, custom_obj_json)
            custom_object = self.__set_custom_objects(custom_dict)
            model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)
            model.load_weights(model_best_weights)
            return model
        elif model_type == 'yolo':
            model_json = f"{self.yolo_model_path}/trained_model_json.trm"
            custom_obj_json = f"{self.yolo_model_path}/trained_model_custom_obj_json.trm"
            model_best_weights = f"{self.yolo_model_path}/trained_model_best_weights"
            model_data, custom_dict = Predict.__get_json_data(model_json, custom_obj_json)
            custom_object = self.__set_custom_objects(custom_dict)
            model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)
            model.load_weights(model_best_weights)
            yolo_model = Predict.__create_yolo(
                model=model,
                classes=self.classes_names,
                version=self.yolo_version
            )
            return yolo_model
        else:
            print("Incorrect model type")
            return None

    @staticmethod
    def __get_json_data(model_json, custom_obj_json):
        with open(model_json) as json_file:
            data = json.load(json_file)

        with open(custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return data, custom_dict

    @staticmethod
    def __set_custom_objects(custom_dict):
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                custom_object[k] = getattr(importlib.import_module(f".{v}", package="custom_objects"), k)
            except:
                continue
        return custom_object

    @staticmethod
    def __create_yolo(model: tf.keras.Model, classes=None, version='v3') -> tf.keras.Model:
        if classes is None:
            classes = []
        num_class = len(classes)
        conv_tensors = model.outputs
        if conv_tensors[0].shape[1] == 13:
            conv_tensors.reverse()
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = Predict.decode(conv_tensor, num_class, i, version)
            output_tensors.append(pred_tensor)
        yolo = tf.keras.Model(model.inputs, output_tensors)
        return yolo

    @staticmethod
    def decode(conv_output, NUM_CLASS, i=0, YOLO_TYPE="v3", STRIDES=None):
        if STRIDES is None:
            STRIDES = [8, 16, 32]
        if (YOLO_TYPE == "v4") or (YOLO_TYPE == "v5"):
            ANCHORS = [[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]]
        elif YOLO_TYPE == "v3":
            ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]
        # Train options
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = \
            tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def predict(self, headline=False):
        image_list = []
        with os.scandir(f"{self.tmp_folder}/init") as folder:
            for f in folder:
                image_list.append(f.name)
        class_counter = []
        count = 0
        for img in image_list:
            classification_pass = 'yes'
            if self.class_model_path:
                cl_array, _ = PrepareDataset.image2array(
                    image_path=f"{self.tmp_folder}/init/{img}",
                    target_size=self.target_class_size,
                    scaler=self.classification_scaler
                )
                pred = self.class_model(cl_array, training=False)
                pred = int(np.argmax(pred, -1)[0])
                if self.classification_classes[pred] == 'no':
                    classification_pass = 'no'
                    shutil.copy2(f"{self.tmp_folder}/init/{img}", f"{self.tmp_folder}/pred/{img}")
                    class_counter.append(0)
                else:
                    class_counter.append(1)

            if classification_pass == 'yes':
                img_array, init_size = PrepareDataset.image2array(
                    image_path=f"{self.tmp_folder}/init/{img}",
                    target_size=self.target_yolo_size,
                    scaler=self.image_yolo_scaler
                )
                predict = self.yolo_model(img_array, training=False)
                predict = self.get_yolo_y_pred(predict)
                bb = Predict.get_optimal_box_channel(predict)
                # print(img, bb, predict[bb])
                Predict.plot_boxes(
                    pred_bb=predict[bb][0],
                    img_path=f"{self.tmp_folder}/init/{img}",
                    name_classes=self.classes_names,
                    colors=self.box_colors,
                    image_size=self.target_yolo_size,
                    save_path=f"{self.tmp_folder}/pred/{img}"
                )
            count += 1
            if count % int(len(image_list) * 0.1) == 0:
                print(f"{round(count *100 / len(image_list), 0)}% images was processes by YOLO...")
        if headline:
            # _, headline_list = Predict.object_counter(class_counter, generate_headline=True)
            for i, img in enumerate(image_list):
                Predict.add_headline_to_image(
                    image_path=f"{self.tmp_folder}/pred/{img}",
                    headline=f"Обнаружено фреймов: {i}", #headline,
                    save_path=f"{self.tmp_folder}/pred/{img}",
                )
        VideoProcessing.frames2video(
            frames_path=f"{self.tmp_folder}/pred",
            save_path=self.save_path,
            video_name='predict_tracker',
            params=self.data_dict_path,
        )
        shutil.rmtree(self.tmp_folder, ignore_errors=True)

    def get_yolo_y_pred(self, array, sensitivity: float = 0.15, threashold: float = 0.1):
        y_pred = {}
        for i, box_array in enumerate(array):
            channel_boxes = []
            for ex in box_array:
                boxes = Predict.get_predict_boxes(
                    array=np.expand_dims(ex, axis=0),
                    name_classes=self.classes_names,
                    sensitivity=sensitivity,
                    threashold=threashold
                )
                channel_boxes.append(boxes)
            y_pred[i] = channel_boxes
        return y_pred

    @staticmethod
    def get_predict_boxes(array, name_classes: list, sensitivity: float = 0.15, threashold: float = 0.1):
        num_classes = len(name_classes)
        num_anchors = 3
        feats = np.reshape(array, (-1, array.shape[1], array.shape[2], num_anchors, num_classes + 5))
        xy_param = feats[..., :2]
        wh_param = feats[..., 2:4]
        conf_param = feats[..., 4:5]
        class_param = feats[..., 5:]
        box_yx = xy_param[..., ::-1].copy()
        box_hw = wh_param[..., ::-1].copy()
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        _boxes = np.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        _boxes_reshape = np.reshape(_boxes, (-1, 4))
        _box_scores = conf_param * class_param
        _box_scores_reshape = np.reshape(_box_scores, (-1, num_classes))
        _class_param_reshape = np.reshape(class_param, (-1, num_classes))
        mask = _box_scores_reshape >= threashold
        _boxes_out = np.zeros_like(_boxes_reshape[0:1])
        _scores_out = np.zeros_like(_box_scores_reshape[0:1])
        _class_param_out = np.zeros_like(_class_param_reshape[0:1])
        for cl in range(num_classes):
            if np.sum(mask[:, cl]):
                _boxes_out = np.concatenate((_boxes_out, _boxes_reshape[mask[:, cl]]), axis=0)
                _scores_out = np.concatenate((_scores_out, _box_scores_reshape[mask[:, cl]]), axis=0)
                _class_param_out = np.concatenate((_class_param_out, _class_param_reshape[mask[:, cl]]), axis=0)
        _boxes_out = _boxes_out[1:].astype('int')
        _scores_out = _scores_out[1:]
        _class_param_out = _class_param_out[1:]
        _conf_param = (_scores_out / _class_param_out)[:, :1]
        pick, _ = Predict.non_max_suppression_fast(_boxes_out, _scores_out, sensitivity)
        return np.concatenate([_boxes_out[pick], _conf_param[pick], _scores_out[pick]], axis=-1)

    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, sensitivity: float = 0.15):
        if len(boxes) == 0:
            return [], []
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        classes = np.argmax(scores, axis=-1)
        idxs = np.argsort(classes)[..., ::-1]
        mean_iou = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            mean_iou.append(overlap)
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > sensitivity)[0])))
        return pick, mean_iou

    @staticmethod
    def plot_boxes(pred_bb, img_path, name_classes, colors, image_size=(416, 416), save_path=''):
        image = Image.open(img_path)
        real_size = image.size
        scale_w = real_size[0] / image_size[0]
        scale_h = real_size[1] / image_size[1]

        coord = pred_bb[:, :4].astype('float')
        coord = np.where(coord < 0, 0, coord)
        coord = np.where(coord > 416, 416, coord)
        resized_coord = np.concatenate(
            [coord[:, 0:1] * scale_h, coord[:, 1:2] * scale_w,
             coord[:, 2:3] * scale_h, coord[:, 3:4] * scale_w], axis=-1).astype('int')
        resized_coord = np.concatenate([resized_coord, pred_bb[:, 4:]], axis=-1)
        pred_bb = resized_coord
        classes = np.argmax(pred_bb[:, 5:], axis=-1)
        # bounding box in (xmin, ymin, xmax, ymax) format
        box = np.concatenate([
            pred_bb[..., 1:2],
            pred_bb[..., 0:1],
            pred_bb[..., 3:4],
            pred_bb[..., 2:3]
        ], axis=-1)
        bbox = torch.tensor(box, dtype=torch.int)
        predicted_class = ['{}'.format(name_classes[classes[i]]) for i in range(len(pred_bb))]
        score = [pred_bb[:, 5:][i][classes[i]] * 100 for i in range(len(pred_bb))]
        label = ['{} {:.0f}% '.format(predicted_class[i], score[i]) for i in range(len(pred_bb))]
        cols = [colors[classes[i]] for i in range(len(pred_bb))]

        # draw bounding boxes with fill color
        image_pred = read_image(img_path)
        image_pred = draw_bounding_boxes(image_pred, bbox, width=3, labels=label, colors=cols, fill=True)
        image_pred = torchvision.transforms.ToPILImage()(image_pred)
        image_pred.save(save_path)

    @staticmethod
    def add_headline_to_image(image_path, headline, save_path):
        image = Image.open(image_path)
        font_size = int(image.size[1] * 0.05)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text((10, 25), "world", font=font)

        image = Image.open(image_path)
        font_size = int(image.size[1] * 0.05)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        label_size = draw.textsize(headline, font)
        text_origin = np.array([int(image.size[0] * 0.01), int(image.size[1] * 0.01)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(255, 255, 255)
        )
        draw.text((int(image.size[0] * 0.01), int(image.size[1] * 0.01)), headline, font=font, fill=(0, 0, 0))
        image.save(save_path)

    @staticmethod
    def object_counter(self, class_counter, generate_headline=True):
        return None, None

    @staticmethod
    def get_optimal_box_channel(array):
        best_val = 0
        best_channel = 0
        for i in array.keys():
            total = 0
            for b in array[i]:
                if np.sum(b) > 0:
                    for bb in b:
                        conf = bb[4]
                        class_conf = bb[5]
                        total += conf * class_conf
            if total > best_val:
                best_val = total
                best_channel = i
        return best_channel


if __name__ == '__main__':
    video_path = 'init_video/Air_1.mp4'
    path = 'init_frames/Air_1_24s'
    # VideoProcessing.video2frames(
    #     video_path=video_path,
    #     save_path=f"init_frames",
    #     max_time=None,
    #     predict_mode=True
    # )

    pred = Predict(
        video_path=f"{path}/init_video/Air_1.mp4",
        yolo_model_path=f"{path}/yolo_model",
        class_model_path="",
        save_path=path,
        data_dict_path=f"{path}/data.dict",
        yolo_version='v3'
    )
    pred.predict(headline=True)
    # video_capture = cv2.VideoCapture()
    # video_capture.open(f"{path}/init_video/Air_1.mp4")
    # fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    # frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(fps, frame_count)

