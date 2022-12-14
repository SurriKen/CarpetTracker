import importlib
import json
import os
import shutil
import time
import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision
from PIL import Image, ImageFont, ImageDraw
from torchvision.utils import draw_bounding_boxes
from VideoProcessing import VideoProcessing
from parameters import YOLO_MODEL_PATH, CLASSIFICATION_MODEL_PATH, PREDICT_PATH, IMAGE_IRRELEVANT_SPACE_PERCENT, \
    BOX_CLASSIFICATION_MODEL_PATH
from utils import get_colors


class Predict:

    def __init__(self, video_path, yolo_version='v3', mode="best", cut_video=None):
        self.video_path = video_path
        self.mode = mode
        self.predict_video_name = f"predict_{self.video_path.split('/')[-1].split('.')[0]}"
        if cut_video:
            tmp_folder = f"{PREDICT_PATH}/tmp_{self.video_path.split('/')[-1].split('.')[0]}"
            try:
                os.mkdir(tmp_folder)
            except:
                shutil.rmtree(tmp_folder, ignore_errors=True)
                os.mkdir(tmp_folder)
            VideoProcessing.cut_video(
                video_path=video_path,
                save_path=tmp_folder,
                from_time=0,
                to_time=cut_video)
            self.video_path = f"{tmp_folder}/{video_path.split('/')[-1]}"
        self.save_path = f"{PREDICT_PATH}/{self.predict_video_name}.mp4"
        self.yolo_version = yolo_version
        f = open(f'{YOLO_MODEL_PATH}/instructions/parameters/2_object_detection.json')
        self.classes_names = json.load(f)['classes_names']
        f.close()
        f2 = open(f'{YOLO_MODEL_PATH}/instructions/parameters/1_image.json')
        dict_ = json.load(f2)
        self.target_yolo_size = (dict_["width"], dict_["height"])
        self.image_yolo_scaler = dict_["scaler"]
        f2.close()
        if CLASSIFICATION_MODEL_PATH:
            self.classification_classes = ["no", "yes"]
            self.class_model = self.set_model(model_type='normal')
            f3 = open(f'{CLASSIFICATION_MODEL_PATH}/instructions/parameters/1_image.json')
            dict_ = json.load(f3)
            self.target_class_size = (dict_["width"], dict_["height"])
            self.classification_scaler = dict_["scaler"]
            f3.close()
        if BOX_CLASSIFICATION_MODEL_PATH:
            self.box_classification_classes = ["no", "yes"]
            self.box_class_model = self.set_model(model_type='box_class')
            f4 = open(f'{BOX_CLASSIFICATION_MODEL_PATH}/instructions/parameters/1_image.json')
            dict_ = json.load(f4)
            self.target_box_class_size = (dict_["width"], dict_["height"])
            self.box_classification_scaler = dict_["scaler"]
            f4.close()
        self.yolo_model = self.set_model(model_type='yolo')
        self.box_colors = get_colors(self.classes_names)

    def set_model(self, model_type='normal'):

        if model_type == 'normal':
            model_json = f"{CLASSIFICATION_MODEL_PATH}/trained_model_json.trm"
            custom_obj_json = f"{CLASSIFICATION_MODEL_PATH}/trained_model_custom_obj_json.trm"
            model_best_weights = f"{CLASSIFICATION_MODEL_PATH}/trained_model_{self.mode}_weights" \
                if self.mode == "best" else f"{CLASSIFICATION_MODEL_PATH}/trained_model_weights"
            model_data, custom_dict = Predict.__get_json_data(model_json, custom_obj_json)
            custom_object = self.__set_custom_objects(custom_dict)
            model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)
            model.load_weights(model_best_weights)
            return model
        elif model_type == 'box_class':
            model_json = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_json.trm"
            custom_obj_json = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_custom_obj_json.trm"
            model_best_weights = f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_{self.mode}_weights" \
                if self.mode == "best" else f"{BOX_CLASSIFICATION_MODEL_PATH}/trained_model_weights"
            model_data, custom_dict = Predict.__get_json_data(model_json, custom_obj_json)
            custom_object = self.__set_custom_objects(custom_dict)
            model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)
            model.load_weights(model_best_weights)
            return model
        elif model_type == 'yolo':
            model_json = f"{YOLO_MODEL_PATH}/trained_model_json.trm"
            custom_obj_json = f"{YOLO_MODEL_PATH}/trained_model_custom_obj_json.trm"
            model_best_weights = f"{YOLO_MODEL_PATH}/trained_model_{self.mode}_weights" \
                if self.mode == "best" else f"{YOLO_MODEL_PATH}/trained_model_weights"
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

    def predict(self, obj_range=4, headline=False, classification=True):
        st = time.time()
        video_capture = cv2.VideoCapture()
        video_capture.open(self.video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        size = (int(width), int(height))
        # print(fps, frame_count, size, self.video_path)
        out = cv2.VideoWriter(
            f'{self.save_path}', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        obj_seq, box_seq = [], []
        total_obj, count = 0, 0
        emp, obj = False, False
        cur_obj = 0
        for i in range(frame_count - 1):
            ret, frame = video_capture.read()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if CLASSIFICATION_MODEL_PATH and classification:
                if self.classification_scaler == 'no_scaler':
                    class_array = np.expand_dims(np.array(pil_img.resize(self.target_class_size)), 0)
                else:
                    class_array = np.expand_dims(np.array(pil_img.resize(self.target_class_size)), 0) / 255
                result = self.class_model(class_array, training=False)
                result = int(np.argmax(result, -1)[0])
                obj_seq.append(result)
            else:
                obj_seq.append(1)

            if obj_seq[-1] == 0:
                total_obj, emp, obj = Predict.object_counter(obj_seq, emp, obj, obj_range, total_obj)
                if headline:
                    headline_str = f"???????????????????? ????????????????: {total_obj}"
                else:
                    headline_str = ""
                pil_img = Predict.add_headline_to_image(
                    image=pil_img,
                    headline=headline_str
                )
                box_seq.append([])
            else:
                img_array = pil_img.resize(self.target_yolo_size)
                if self.image_yolo_scaler == 'no_scaler':
                    img_array = np.expand_dims(np.array(img_array), 0)
                else:
                    img_array = np.expand_dims(np.array(img_array), 0) / 255
                predict = self.yolo_model(img_array, training=False)
                predict = self.get_yolo_y_pred(predict)
                predict = Predict.remove_irrelevant_box(predict)
                bb = Predict.get_optimal_box_channel(predict)
                predict = self.drop_wrong_boxes(predict[bb][0], pil_img)
                if not predict.any(): # obj_seq[-1] != 0 and not predict.any():
                    obj_seq[-1] = predict.shape[0]
                    box_seq.append(predict.tolist())
                else:
                    box_seq.append([])
                total_obj, emp, obj, cur_obj = Predict.object_counter(obj_seq, emp, obj, cur_obj, obj_range, total_obj)
                if headline:
                    headline_str = f"???????????????????? ????????????????: {total_obj}"
                else:
                    headline_str = ""
                pil_img = Predict.plot_boxes(
                    pred_bb=predict,
                    image=pil_img,
                    name_classes=self.classes_names,
                    colors=self.box_colors,
                    image_size=self.target_yolo_size,
                    headline=headline_str
                )
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(img)
            count += 1
            if count % round(frame_count * 0.01, 0) == 0:
                print(
                    f"{int(count * 100 / frame_count)}% images were processed by tracker "
                    f"(frame: {i + 1}/{frame_count}, "
                    f"time: {int((time.time() - st) // 60)}m {int((time.time() - st) % 60)}s)..."
                )
        print(obj_seq)
        print(box_seq)
        out.release()
        print(f"\nPrediction time={round(time.time() - st, 1)}s")

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
    def plot_boxes(pred_bb, image, name_classes, colors, image_size=(416, 416), headline=""):
        real_size = image.size
        scale_w = real_size[0] / image_size[0]
        scale_h = real_size[1] / image_size[1]

        if list(pred_bb):
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

            image_pred = torchvision.transforms.ToTensor()(image)
            image_pred = torch.tensor(image_pred * 255, dtype=torch.uint8)
            font_size = int(real_size[1] * 0.03) if int(real_size[1] * 0.03) > 20 else 20
            image_pred = draw_bounding_boxes(image_pred, bbox, width=3, labels=label, colors=cols, fill=True,
                                             font='arial.ttf', font_size=font_size)
            image = torchvision.transforms.ToPILImage()(image_pred)
        if headline:
            return Predict.add_headline_to_image(image, headline)
        else:
            return image

    @staticmethod
    def add_headline_to_image(image, headline):
        if headline:
            font_size = int(image.size[1] * 0.03)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", font_size)
            label_size = draw.textsize(headline, font)
            text_origin = np.array([int(image.size[0] * 0.01), int(image.size[1] * 0.01)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(255, 255, 255)
            )
            draw.text((int(image.size[0] * 0.01), int(image.size[1] * 0.01)), headline, font=font, fill=(0, 0, 0))
        return image

    @staticmethod
    def object_counter(class_counter, emp: bool, obj: bool, cur_obj: int, obj_range, total_obj):
        if len(class_counter) < obj_range:
            emp = True
            obj = False
            total_obj = 0
            cur_obj = 0
        else:
            found_obj = class_counter[-1]
            objects = 0
            for j in range(obj_range):
                if class_counter[-(j + 1)] > 0:
                    objects += 1
            if emp and objects < obj_range:
                emp = True
                obj = False
            elif emp and objects == obj_range:
                emp = False
                obj = True
                if found_obj < cur_obj:
                    pass
                elif found_obj == cur_obj:
                    total_obj += cur_obj
                elif found_obj > cur_obj and np.sum(class_counter[-obj_range:]) == found_obj * obj_range:
                    total_obj += found_obj - cur_obj
                    cur_obj = found_obj
            elif obj and objects > 0:
                emp = False
                obj = True
                if found_obj > cur_obj and np.sum(class_counter[-obj_range:]) == found_obj * obj_range:
                    total_obj += found_obj - cur_obj
                    cur_obj = found_obj
            elif obj and objects == 0:
                emp = True
                obj = False
                cur_obj = 0
            else:
                emp = False
                obj = True
        return total_obj, emp, obj, cur_obj

    # def object_counter(class_counter, emp: bool, obj: bool, obj_range, total_obj):
    #     if len(class_counter) < obj_range:
    #         emp = True
    #         obj = False
    #         total_obj = 0
    #     else:
    #         if emp and np.sum(class_counter[-obj_range:]) < obj_range:
    #             emp = True
    #             obj = False
    #         elif emp and np.sum(class_counter[-obj_range:]) == obj_range:
    #             emp = False
    #             obj = True
    #             total_obj += 1
    #         elif obj and np.sum(class_counter[-obj_range:]) > 0:
    #             emp = False
    #             obj = True
    #         elif obj and np.sum(class_counter[-obj_range:]) == 0:
    #             emp = True
    #             obj = False
    #         else:
    #             emp = False
    #             obj = True
    #     return total_obj, emp, obj

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

    @staticmethod
    def remove_irrelevant_box(boxes: dict):
        # bounding box in (ymin, xmin, ymax, xmax) format
        x_min = int(416 * IMAGE_IRRELEVANT_SPACE_PERCENT)
        x_max = int(416 - 416 * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_min = int(416 * IMAGE_IRRELEVANT_SPACE_PERCENT)
        y_max = int(416 - 416 * IMAGE_IRRELEVANT_SPACE_PERCENT)
        new_boxes = {}
        for i, ch in boxes.items():
            if list(ch[0]):
                bb = []
                for b in ch[0]:
                    if b[0] > y_min and b[1] > x_min and b[2] < y_max and b[3] < x_max:
                        bb.append(b)
                new_boxes[i] = [np.array(bb, dtype='float64')]
            else:
                new_boxes[i] = ch
        return new_boxes

    def drop_wrong_boxes(self, boxes: np.ndarray, img: Image):
        if not boxes.any() or not BOX_CLASSIFICATION_MODEL_PATH:
            return boxes
        new_boxes = []
        for b in boxes:
            w, h = img.size
            target_size = self.target_box_class_size
            box_center = ((b[0] + int((b[2] - b[0]) / 2)) * h / 416, (b[1] + int((b[3] - b[1]) / 2)) * w / 416)
            if box_center[1] < w / 2:
                left = int(box_center[1] - 0.5 * target_size[0]) if int(
                    box_center[1] - 0.5 * target_size[0]) > 0 else 0
                right = int(box_center[1] + 0.5 * target_size[0]) if left > 0 else int(target_size[0])
            else:
                right = int(box_center[1] + 0.5 * target_size[0]) if int(
                    box_center[1] + 0.5 * target_size[0]) < w else target_size[0]
                left = int(box_center[1] - 0.5 * target_size[0]) if right < w else int(w - target_size[0])
            if box_center[0] < h / 2:
                top = int(box_center[0] - 0.5 * target_size[1]) if int(
                    box_center[0] - 0.5 * target_size[1]) > 0 else 0
                bottom = int(box_center[0] + 0.5 * target_size[1]) if top > 0 else int(0.5 * target_size[1])
            else:
                bottom = int(box_center[0] + 0.5 * target_size[1]) if int(
                    box_center[0] + 0.5 * target_size[1]) < h else target_size[0]
                top = int(box_center[0] - 0.5 * target_size[1]) if bottom < h else int(h - 0.5 * target_size[1])
            crop_img = np.array(img.crop((left, top, right, bottom)))
            res = self.box_class_model.predict(np.expand_dims(crop_img, axis=0))
            if self.box_classification_classes[np.argmax(res[0])] == "yes":
                new_boxes.append(b)
        # print(list(boxes), np.array(new_boxes))
        return np.array(new_boxes)


if __name__ == '__main__':
    pass
    # for i in range(5):
    #     pred = Predict(
    #         video_path=f"videos/Train_{i}.mp4",
    #         yolo_version='v4',
    #         cut_video=300,
    #         mode="best",
    #     )
    #     pred.predict(obj_range=6, headline=True, classification=False)

    # pred = Predict(
    #     video_path=f"predict/tmp_Test_0/Test_0.mp4",
    #     yolo_version='v4',
    #     # cut_video=300,
    #     mode="",
    # )
    # pred.predict(obj_range=6, headline=True, classification=False)
