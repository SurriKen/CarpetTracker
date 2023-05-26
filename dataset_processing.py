import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
import pandas as pd
import skvideo.io
import torch
import torchvision
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image

from parameters import ROOT_DIR
from utils import *


@dataclass
class VideoClass:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.classes = []
        self.train_stat = []
        self.val_stat = []



class DatasetProcessing:

    def __init__(self):
        pass

    @staticmethod
    def cut_video(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000) -> str:
        """
        Cut video in given time range.

        Args:
            video_path: path to video file
            save_path: path to save folder
            from_time: time to start cut in seconds
            to_time: time to finish cut in seconds

        Returns: path to saved video file
        """
        try:
            os.mkdir(save_path)
        except:
            pass
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = int(frame_count / fps)
        saved_video_path = f"{save_path}/{video_path.split('/')[-1]}"
        if to_time and to_time < duration:
            clip = VideoFileClip(video_path).subclip(from_time, to_time)
            clip.write_videofile(saved_video_path)
        elif from_time:
            clip = VideoFileClip(video_path).subclip(from_time, duration)
            clip.write_videofile(saved_video_path)
        else:
            shutil.copy2(video_path, saved_video_path)
        print("Video was cut and save")
        return saved_video_path

    @staticmethod
    def synchronize_video(video_path: str, save_path: str = 'datasets', from_frame=None, to_frame=None):
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        vc = cv2.VideoCapture()
        vc.open(video_path)
        frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        size = None
        for i in range(frames):
            ret, frame = video_capture.read()
            size = (frame.shape[1], frame.shape[0])
            break

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(frame_count):
            if (i + 1) % 200 == 0:
                print(f"{i + 1} frames are ready")
            ret, frame = video_capture.read()
            if from_frame and i >= from_frame:
                out.write(frame)
            if to_frame and i >= to_frame:
                break

        out.release()

    @staticmethod
    def video_class_dataset(csv_files: list, video_links: list, save_folder: str):
        try:
            os.mkdir(save_folder)
        except:
            shutil.rmtree(save_folder)
            os.mkdir(save_folder)

        count = 0
        for i, csv in enumerate(csv_files):
            obj = 0
            data = pd.read_csv(csv)
            carpet_size = data['Размер']
            start_frame = data['Кадр начала']
            end_frame = data['Кадр конца']
            # print(len(carpet_size), len(start_frame), len(end_frame))

            cls = carpet_size.unique()
            for cl in cls:
                cl = cl.replace('*', 'x')
                if not os.path.isdir(os.path.join(save_folder, cl)):
                    os.mkdir(os.path.join(save_folder, cl))

            cam_1, cam_2 = video_links[i]
            vc1 = cv2.VideoCapture()
            vc1.open(cam_1)
            w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))
            h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames1 = int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))

            vc2 = cv2.VideoCapture()
            vc2.open(cam_2)
            frames2 = int(vc2.get(cv2.CAP_PROP_FRAME_COUNT))
            w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))
            h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))

            w = min([w1, w2])
            h = min([h1, h2])

            for j in range(min([frames1, frames2])):
                _, frame1 = vc1.read()
                _, frame2 = vc2.read()

                if j == start_frame[obj]:
                    cs = carpet_size[obj].replace('*', 'x')

                    out1 = cv2.VideoWriter(
                        os.path.join(save_folder, cs, f"{count}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h * 2)
                    )

                if start_frame[obj] <= j <= end_frame[obj]:
                    size1 = (frame1.shape[1], frame1.shape[0])
                    if size1 != (w, h):
                        frame1 = cv2.resize(frame1, (w, h))
                    size2 = (frame2.shape[1], frame2.shape[0])
                    if size2 != (w, h):
                        frame2 = cv2.resize(frame2, (w, h))
                    frame = np.concatenate((frame1, frame2), axis=0)
                    out1.write(frame)

                if j == end_frame[obj]:
                    out1.release()
                    obj += 1
                    count += 1

                if obj == len(carpet_size):
                    break

    @staticmethod
    def change_fps(video_path: str, save_path: str, set_fps=25):

        clip = VideoFileClip(video_path)
        clip.write_videofile(save_path, fps=set_fps)

    @staticmethod
    def video2frames(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000):
        video_name = video_path.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"fps = {fps}, frame_count = {frame_count}")
        duration = int(frame_count / fps)
        if to_time:
            if to_time > int(duration):
                to_time = int(duration)
            to_path = f"{save_path}/{video_name}_{from_time}s-{to_time}s"
        elif from_time:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        else:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        try:
            os.mkdir(to_path)
        except:
            shutil.rmtree(to_path, ignore_errors=True)
            os.mkdir(to_path)
        os.mkdir(f"{to_path}/frames")
        os.mkdir(f"{to_path}/video")
        os.mkdir(f"{to_path}/xml_labels")
        saved_video_path = DatasetProcessing.cut_video(
            video_path=video_path,
            save_path=f"{to_path}/video",
            from_time=from_time,
            to_time=to_time
        )
        video_capture = cv2.VideoCapture()
        video_capture.open(saved_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Getting frames ({int(frames)} in total)...")
        size = ()
        for i in range(int(frames)):
            if (i + 1) % 200 == 0:
                print(f"{i + 1} frames are ready")
            ret, frame = video_capture.read()
            size = (frame.shape[1], frame.shape[0])
            cv2.imwrite(f"{to_path}/frames/%05d.png" % i, frame)
        video_data = {
            "fps": int(fps), "frames": int(frames), 'size': size
        }
        print(f"frames were got: fps - {int(fps)}, total frames - {int(frames)}, frame size - {size}")
        save_dict(video_data, to_path, 'data')

    @staticmethod
    def frames2video(frames_path: str, save_path: str, video_name: str,
                     params: str, box_path: str = None, resize=False, box_type='xml'):
        parameters = load_dict(params)
        image_list, box_list, lbl_list, color_list = [], [], [], []
        with os.scandir(frames_path) as folder:
            for f in folder:
                image_list.append(f.name)
        image_list = sorted(image_list)
        if box_path:
            remove_empty_xml(box_path)
            with os.scandir(box_path) as folder:
                for f in folder:
                    box_list.append(f.name)
            for box in box_list:
                box_info = read_xml(f"{box_path}/{f'{box}'}")
                if box_info["coords"][-1] not in lbl_list:
                    lbl_list.append(box_info["coords"][-1])
            lbl_list = sorted(lbl_list)
            color_list = get_colors(lbl_list)

        st = time.time()
        count = 0
        print(f"Start processing...\n")
        out = cv2.VideoWriter(
            f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
        for img in image_list:
            if (count + 1) % int(len(image_list) * 0.1) == 0:
                print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were processed...")
            name = img.split(".")[0]
            if box_path:
                if box_type == 'xml':
                    if resize:
                        box_info = read_xml(
                            xml_path=f"{box_path}/{f'{name}.xml'}",
                            shrink=True,
                            new_width=parameters['size'][0],
                            new_height=parameters['size'][1]
                        )
                    else:
                        box_info = read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
                    boxes, labels = [], []
                    for b in box_info["coords"]:
                        boxes.append(b[:-1])
                        labels.append(b[-1])
                    bbox = torch.tensor(boxes, dtype=torch.int)
                    image = read_image(f"{frames_path}/{img}")
                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
                    image = torchvision.transforms.ToPILImage()(image_true)
                if box_type == 'txt':
                    box_info = read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
                    boxes, labels = [], []
                    for b in box_info["coords"]:
                        boxes.append(b[:-1])
                        labels.append(b[-1])
                    bbox = torch.tensor(boxes, dtype=torch.int)
                    image = read_image(f"{frames_path}/{img}")
                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
                    image = torchvision.transforms.ToPILImage()(image_true)
                elif box_type == 'terra':
                    image = Image.open(f"{frames_path}/{img}")
                else:
                    image = Image.open(f"{frames_path}/{img}")
            else:
                if resize:
                    image = Image.open(f"{frames_path}/{img}")
                    image = image.resize(parameters['size'])
                else:
                    image = Image.open(f"{frames_path}/{img}")
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            count += 1
        out.release()
        print(f"\nProcessing is finished! Processing time = {round(time.time() - st, 1)}s\n")

    @staticmethod
    def put_box_on_image(images: str, labels: str, save_path: str):
        try:
            os.mkdir(save_path)
        except:
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        lbl = ['carpet']
        color_list = get_colors(lbl)
        img_names = []
        empty_box, fill_box = 0, 0
        with os.scandir(images) as folder:
            for f in folder:
                img_names.append(f.name[:-4])
        for name in img_names:
            img_path = f'{images}/{name}.jpg'
            box_path = f'{labels}/{name}.txt'
            img = Image.open(img_path)
            with open(box_path, 'r') as handle:
                box_info = handle.readlines()
                if box_info:
                    fill_box += 1
                    box_info = [re.sub(f'\n', ' ', b) for b in box_info]
                else:
                    empty_box += 1

            coord = []
            if box_info:
                for box in box_info:
                    if box:
                        box = box.split(" ")
                        coord.append([
                            int((float(box[1]) - float(box[3]) / 2) * img.size[0]),
                            int((float(box[2]) - float(box[4]) / 2) * img.size[1]),
                            int((float(box[1]) + float(box[3]) / 2) * img.size[0]),
                            int((float(box[2]) + float(box[4]) / 2) * img.size[1]),
                        ])
                bbox = torch.tensor(coord, dtype=torch.int)
                image = read_image(img_path)
                lbl2 = lbl * len(coord)
                color_list2 = color_list * len(coord)
                image_true = draw_bounding_boxes(image, bbox, width=3, labels=lbl2, colors=color_list2, fill=True)
                image = torchvision.transforms.ToPILImage()(image_true)
                image.save(f'{save_path}/{name}.png')
            else:
                image = read_image(img_path)
                image = torchvision.transforms.ToPILImage()(image)
                image.save(f'{save_path}/{name}.png')
            print('fill_box=', fill_box, 'empty_box=', empty_box)

    @staticmethod
    def fill_empty_box(dataset: str):
        empty_text = ""
        img_names, lbl_names = [], []

        with os.scandir(f"{dataset}/images") as folder:
            for f in folder:
                img_names.append(f.name[:-4])

        with os.scandir(f"{dataset}/labels") as folder:
            for f in folder:
                lbl_names.append(f.name[:-4])

        for name in img_names:
            if name not in lbl_names:
                save_txt(
                    txt=empty_text,
                    txt_path=f"{dataset}/labels/{name}.txt"
                )

    @staticmethod
    def form_dataset_for_train(data: list, split: float, save_path: str, condition=None):
        """
        :param data: list of lists of 2 str and 1 float [[image_folder, corresponding_labels_folder, 0.5], ...]
        :param split: float between 0 and 1
        :param save_path: str
        :param condition: dict
        """
        if condition is None:
            condition = {}
        try:
            os.mkdir(save_path)
            os.mkdir(f"{save_path}/train")
            os.mkdir(f"{save_path}/train/images")
            os.mkdir(f"{save_path}/train/labels")
            os.mkdir(f"{save_path}/val")
            os.mkdir(f"{save_path}/val/images")
            os.mkdir(f"{save_path}/val/labels")
        except:
            shutil.rmtree(save_path)
            os.mkdir(save_path)
            os.mkdir(f"{save_path}/train")
            os.mkdir(f"{save_path}/train/images")
            os.mkdir(f"{save_path}/train/labels")
            os.mkdir(f"{save_path}/val")
            os.mkdir(f"{save_path}/val/images")
            os.mkdir(f"{save_path}/val/labels")

        count = 0
        for folders in data:
            img_list = []
            lbl_list = []

            with os.scandir(folders[0]) as fold:
                for f in fold:
                    if f.name[-3:] in ['png', 'jpg']:
                        if condition.get('orig_shape'):
                            img = Image.open(f"{folders[0]}/{f.name}")
                            if img.size == condition.get('orig_shape'):
                                img_list.append(f.name)
                        else:
                            img_list.append(f.name)

            with os.scandir(folders[1]) as fold:
                for f in fold:
                    if f.name[-3:] in ['txt']:
                        lbl_list.append(f.name)

            try:
                if 0 < float(folders[2]) <= 1:
                    take_num = int(len(img_list) * float(folders[2]))
                else:
                    take_num = len(img_list)
            except:
                take_num = len(img_list)

            ids = list(range(len(img_list)))
            z = np.random.choice(ids, take_num, replace=False)
            img_list = [img_list[i] for i in z]
            logger.info(f'\n- img_list: {len(img_list)}\n- lbl_list: {len(lbl_list)}\n')

            random.shuffle(img_list)
            delimiter = int(len(img_list) * split)

            for i, img in enumerate(img_list):
                if i <= delimiter:
                    shutil.copy2(f"{folders[0]}/{img}", f"{save_path}/train/images/{count}.jpg")
                    if f"{img[:-3]}txt" in lbl_list:
                        shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/train/labels/{count}.txt")
                    else:
                        save_txt(txt='', txt_path=f"{save_path}/train/labels/{count}.txt")
                else:
                    shutil.copy2(f"{folders[0]}/{img}", f"{save_path}/val/images/{count}.jpg")
                    if f"{img[:-3]}txt" in lbl_list:
                        shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/val/labels/{count}.txt")
                    else:
                        save_txt(txt='', txt_path=f"{save_path}/val/labels/{count}.txt")

                if (count + 1) % 200 == 0:
                    logger.info(f"-- prepared {i + 1} images")
                count += 1

    @staticmethod
    def video_to_array(video_path: str) -> np.ndarray:
        """
        Transform video to numpy array
        """
        return skvideo.io.vread(os.path.join(ROOT_DIR, video_path))

    @staticmethod
    def ohe_from_list(data: list[int], num: int) -> np.ndarray:
        """Transform list of labels to one hot encoding array"""
        targets = np.array([data]).reshape(-1)
        return np.eye(num)[targets]

    @staticmethod
    def create_video_class_dataset_generator(folder_path: str, split: float) -> VideoClass:
        # st = time.time()
        vc = VideoClass()
        classes = os.listdir(os.path.join(ROOT_DIR, folder_path))
        classes = sorted(classes)
        vc.classes = classes
        data, lbl, stat_lbl = [], [], []
        # stat = dict(train={}, val={}, classes=classes)
        for cl in classes:
            content = os.listdir(os.path.join(ROOT_DIR, folder_path, cl))
            content = sorted(content)
            lbl.extend([classes.index(cl)] * len(content))
            for file in content:
                # vid = DatasetProcessing.video_to_array(os.path.join(folder_path, cl, file))
                # data.append(vid / 255)
                data.append(os.path.join(folder_path, cl, file))
            logging.info(f"-- Class {cl}, processed {len(content)} videos")

        zip_data = list(zip(data, lbl))
        random.shuffle(zip_data)
        train, val = zip_data[:int(split * len(lbl))], zip_data[int(split * len(lbl)):]
        vc.x_train, vc.y_train = list(zip(*train))
        vc.x_val, vc.y_val = list(zip(*val))

        ytr = dict(Counter(vc.y_train))
        stat_ytr = {}
        for k, v in ytr.items():
            stat_ytr[classes[k]] = v
        vc.train_stat = stat_ytr
        # stat['train'] = stat_ytr

        yv = dict(Counter(vc.y_val))
        stat_yv = {}
        for k, v in yv.items():
            stat_yv[classes[k]] = v
        vc.val_stat = stat_yv
        # stat['val'] = stat_yv

        # y_train = DatasetProcessing.ohe_from_list(y_train, len(classes))
        # y_val = DatasetProcessing.ohe_from_list(y_val, len(classes))
        # logger.info(f"Total dataset processing time = {round(time.time() - st, 1)} sec")
        return vc

    @staticmethod
    def generate_video_class_batch(generator_dict: dict, iteration: int, mode: str = 'train'
                                   ) -> tuple[np.ndarray, np.ndarray]:
        num_classes = len(generator_dict.get('stat').get('classes'))
        x_array = DatasetProcessing.video_to_array(generator_dict.get(f'x_{mode}')[iteration])
        y_array = DatasetProcessing.ohe_from_list([generator_dict.get(f'y_{mode}')[iteration]], num=num_classes)
        return np.expand_dims(np.array(x_array), axis=0), np.array(y_array)


if __name__ == '__main__':
    CHECK_BOXES_IN_DATASET = True
    # data = [
    #     # ['datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01',
    #     #  'datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01_'],
    #     # ['datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02',
    #     #  'datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02_'],
    #     # ['datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03',
    #     #  'datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03_'],
    #     # ['datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04',
    #     #  'datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04_'],
    #     ['datasets/От разметчиков/batch_05_#147536/batch_05',
    #      'datasets/От разметчиков/batch_05_#147536/batch_05_'],
    # ]
    # save = [
    #     # 'datasets/От разметчиков/batch_01_#108664/obj_train_data/img+lbl',
    #     # 'datasets/От разметчиков/batch_02_#110902/obj_train_data/img+lbl',
    #     # 'datasets/От разметчиков/batch_03_#112497/obj_train_data/img+lbl',
    #     # 'datasets/От разметчиков/batch_04_#119178/obj_train_data/img+lbl',
    #     'datasets/От разметчиков/batch_05_#147536/img+lbl',
    # ]
    # for i, p in enumerate(data):
    #     DatasetProcessing.put_box_on_image(
    #         images=p[0],
    #         labels=p[1],
    #         save_path=save[i]
    #     )

    PREPARE_DATASET = True
    # data = [
    #         ['datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01',
    #          'datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01_', 1.0],
    #         ['datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02',
    #          'datasets/От разметчиков/batch_02_#110902/obj_train_data/batch_02_', 1.0],
    #         ['datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03',
    #          'datasets/От разметчиков/batch_03_#112497/obj_train_data/batch_03_', 1.0],
    #         ['datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04',
    #          'datasets/От разметчиков/batch_04_#119178/obj_train_data/batch_04_', 1.0],
    #         ['datasets/От разметчиков/batch_05_#147536/batch_05',
    #          'datasets/От разметчиков/batch_05_#147536/batch_05_', 1.0],
    #         ['datasets/От разметчиков/batch_mine/obj_train_data/batch_01',
    #          'datasets/От разметчиков/batch_mine/obj_train_data/batch_01_', 1.0],
    #     ]
    # split = 0.9
    # save_path_1 = 'datasets/yolov8_camera_1'
    # save_path_2 = 'datasets/yolov8_camera_2'
    #
    # DatasetProcessing.form_dataset_for_train(
    #     data=data,
    #     split=split,
    #     save_path=save_path_1,
    #     condition={'orig_shape': (1920, 1080)}
    # )
    # for l in ['train', 'val']:
    #     DatasetProcessing.put_box_on_image(
    #         images=f'datasets/yolov8_camera_1/{l}/images',
    #         labels=f'datasets/yolov8_camera_1/{l}/labels',
    #         save_path=f'datasets/yolov8_camera_1/{l}/img+lbl'
    #     )

    # DatasetProcessing.form_dataset_for_train(
    #     data=data,
    #     split=split,
    #     save_path=save_path_2,
    #     condition={'orig_shape': (640, 360)}
    # )
    # for l in ['train', 'val']:
    #     DatasetProcessing.put_box_on_image(
    #         images=f'datasets/yolov8_camera_2/{l}/images',
    #         labels=f'datasets/yolov8_camera_2/{l}/labels',
    #         save_path=f'datasets/yolov8_camera_2/{l}/img+lbl'
    #     )

    CHANGE_FPS = True
    # vid = [
    #     # 'videos/test 1_cam 1.mp4', 'videos/test 1_cam 2.mp4',
    #     # 'videos/test 2_cam 1.mp4', 'videos/test 2_cam 2.mp4',
    #     # 'videos/test 3_cam 1.mp4', 'videos/test 3_cam 2.mp4',
    #     # 'videos/test 4_cam 1.mp4', 'videos/test 4_cam 2.mp4',
    #     # 'videos/test 5_cam 1.mp4', 'videos/test 5_cam 2.mp4',
    #     # 'videos/classification_videos/13_05 ВО.mp4', 'videos/classification_videos/13_05 ВО_2.mp4',
    #     # 'videos/classification_videos/16-10 ЦП.mp4', 'videos/classification_videos/16-10 ЦП_2.mp4',
    #     # 'videos/classification_videos/МОС,19-40.mp4', 'videos/classification_videos/МОС,19-40_2.mp4',
    #     # 'videos/classification_videos/НОЧЬ,20-11.mp4', 'videos/classification_videos/НОЧЬ,20-11_2.mp4',
    #     'videos/test 6_cam 1.mp4', 'videos/test 6_cam 2.mp4',
    # ]
    # for v in vid:
    #     shutil.move(v, 'test.mp4')
    #     DatasetProcessing.change_fps(
    #         video_path='test.mp4',
    #         save_path=v,
    #         set_fps=25
    #     )
    #     os.remove('test.mp4')

    CUT_VIDEOS_TO_FRAMES = True
    # vid = [
    #     # 'videos/test 1_cam 1.mp4', 'videos/test 1_cam 2.mp4',
    #     # 'videos/test 2_cam 1.mp4', 'videos/test 2_cam 2.mp4',
    #     # 'videos/test 3_cam 1.mp4', 'videos/test 3_cam 2.mp4',
    #     # 'videos/test 4_cam 1.mp4', 'videos/test 4_cam 2.mp4',
    #     # 'videos/test 5_cam 1.mp4', 'videos/test 5_cam 2.mp4',
    #     # 'videos/classification_videos/13-05 ВО_cam1.mp4', 'videos/classification_videos/13-05 ВО_cam2.mp4',
    #     # 'videos/classification_videos/16-10 ЦП_cam1.mp4', 'videos/classification_videos/16-10 ЦП_cam2.mp4',
    #     # 'videos/classification_videos/МОС 19-40_cam1.mp4', 'videos/classification_videos/МОС 19-40_cam2.mp4',
    #     # 'videos/classification_videos/Ночь 20-11_cam1.mp4', 'videos/classification_videos/Ночь 20-11_cam2.mp4',
    #     # 'videos/classification_videos/13_05 ВО_sync.mp4', 'videos/classification_videos/13_05 ВО_2_sync.mp4',
    #     # 'videos/classification_videos/16-10 ЦП_sync.mp4', 'videos/classification_videos/16-10 ЦП_2_sync.mp4',
    #     # 'videos/classification_videos/МОС,19-40_sync.mp4', 'videos/classification_videos/МОС,19-40_2_sync.mp4',
    #     # 'videos/classification_videos/НОЧЬ,20-11_sync.mp4', 'videos/classification_videos/НОЧЬ,20-11_2_sync.mp4',
    #     # 'videos/sync_test/test 5_cam 1_sync.mp4', 'videos/sync_test/test 5_cam 2_sync.mp4',
    #     'videos/test 6_cam 1.mp4', 'videos/test 6_cam 2.mp4',
    # ]

    # for v in vid:
    #     DatasetProcessing.video2frames(
    #         video_path=v,
    #         save_path=f"datasets",
    #         from_time=0,
    #         to_time=120
    #     )

    SYNCHRONIZE_VIDEOS = True
    # sync_data = {
    #     # 'videos/test 1_cam 1.mp4': [519, '18:00:10'],
    #     # 'videos/test 1_cam 2.mp4': [42, '18:00:10'],
    #     # 'videos/test 2_cam 1.mp4': [94, '16:28:08'],
    #     # 'videos/test 2_cam 2.mp4': [45, '16:28:08'],
    #     # 'videos/test 3_cam 1.mp4': [693, '18:00:27'],
    #     # 'videos/test 3_cam 2.mp4': [297, '18:00:27'],
    #     # 'videos/test 4_cam 1.mp4': [22, '13:05:21'],
    #     # 'videos/test 4_cam 2.mp4': [976, '13:05:21'],
    #     # 'videos/test 5_cam 1.mp4': [376, '13:41:57'],
    #     # 'videos/test 5_cam 2.mp4': [400, '13:41:56'],
    #     # 'videos/classification_videos/13-05 ВО_cam1.mp4': [32, '13:05:21'],
    #     # 'videos/classification_videos/13-05 ВО_cam2.mp4': [986, '13:05:21'],
    #     # 'videos/classification_videos/16-10 ЦП_cam1.mp4': [476, '16:08:55'],
    #     # 'videos/classification_videos/16-10 ЦП_cam2.mp4': [8, '16:08:56'],
    #     # 'videos/classification_videos/МОС 19-40_cam1.mp4': [110, '19:40:52'],
    #     # 'videos/classification_videos/МОС 19-40_cam2.mp4': [16, '19:40:52'],
    #     # 'videos/classification_videos/Ночь 20-11_cam1.mp4': [193, '20:11:13'],
    #     # 'videos/classification_videos/Ночь 20-11_cam2.mp4': [8, '20:11:13'],
    #     'videos/test 6_cam 1.mp4': [31, '13:30:01'],
    #     'videos/test 6_cam 2.mp4': [2159, '13:30:02'],
    # }
    # sync_videos = [
    #     # {'camera 1': 'videos/test 1_cam 1.mp4', 'camera 2': 'videos/test 1_cam 2.mp4'},
    #     # {'camera 1': 'videos/test 2_cam 1.mp4', 'camera 2': 'videos/test 2_cam 2.mp4'},
    #     # {'camera 1': 'videos/test 3_cam 1.mp4', 'camera 2': 'videos/test 3_cam 2.mp4'},
    #     # {'camera 1': 'videos/test 4_cam 1.mp4', 'camera 2': 'videos/test 4_cam 2.mp4'},
    #     # {'camera 1': 'videos/test 5_cam 1.mp4', 'camera 2': 'videos/test 5_cam 2.mp4'},
    #     # {'camera 1': 'videos/classification_videos/13-05 ВО_cam1.mp4', 'camera 2': 'videos/classification_videos/13-05 ВО_cam2.mp4'},
    #     # {'camera 1': 'videos/classification_videos/16-10 ЦП_cam1.mp4', 'camera 2': 'videos/classification_videos/16-10 ЦП_cam2.mp4'},
    #     # {'camera 1': 'videos/classification_videos/МОС 19-40_cam1.mp4', 'camera 2': 'videos/classification_videos/МОС 19-40_cam2.mp4'},
    #     # {'camera 1': 'videos/classification_videos/Ночь 20-11_cam1.mp4', 'camera 2': 'videos/classification_videos/Ночь 20-11_cam2.mp4'},
    #     {'camera 1': 'videos/test 6_cam 1.mp4', 'camera 2': 'videos/test 6_cam 2.mp4'},
    # ]
    # for pair in sync_videos:
    #     save_name_1 = f"{pair.get('camera 1').split('/')[-1].split('.')[0]}_sync.mp4"
    #     DatasetProcessing.synchronize_video(
    #         video_path=pair.get('camera 1'),
    #         save_path=f"videos/sync_test/{save_name_1}",
    #         from_frame=sync_data.get(pair.get('camera 1'))[0]
    #     )
    #     save_name_2 = f"{pair.get('camera 2').split('/')[-1].split('.')[0]}_sync.mp4"
    #     DatasetProcessing.synchronize_video(
    #         video_path=pair.get('camera 2'),
    #         save_path=f"videos/sync_test/{save_name_2}",
    #         from_frame=sync_data.get(pair.get('camera 2'))[0]
    #     )

    CREATE_CLASSIFICATION_VIDEO = True
    # csv_files = [
    #     'videos/classification_videos/13-05 ВО.csv',
    #     'videos/classification_videos/16-10 ЦП.csv',
    #     'videos/classification_videos/МОС 19-40.csv',
    #     'videos/classification_videos/Ночь 20-11.csv',
    # ]
    # video_links = [
    #     ['videos/sync_test/13-05 ВО_cam1_sync.mp4', 'videos/sync_test/13-05 ВО_cam2_sync.mp4'],
    #     ['videos/sync_test/16-10 ЦП_cam1_sync.mp4', 'videos/sync_test/16-10 ЦП_cam2_sync.mp4'],
    #     ['videos/sync_test/МОС 19-40_cam1_sync.mp4', 'videos/sync_test/МОС 19-40_cam2_sync.mp4'],
    #     ['videos/sync_test/Ночь 20-11_cam1_sync.mp4', 'videos/sync_test/Ночь 20-11_cam2_sync.mp4'],
    # ]
    # save_folder = 'datasets/class_videos'
    #
    # DatasetProcessing.video_class_dataset(
    #     csv_files=csv_files,
    #     video_links=video_links,
    #     save_folder=save_folder
    # )

    pass