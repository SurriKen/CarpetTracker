import os
import pickle
import cv2
import numpy as np
import wget
from PIL import Image
from ultralytics import YOLO
from tracker import Tracker
from utils import get_colors, load_dict, add_headline_to_cv_image

yolov8_types = {
    "yolov8n": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"},
    "yolov8s": {"Test Size": 640, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"},
    "yolov8m": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"},
    "yolov8I": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"},
    "yolov8x": {"Test Size": 1280, "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"},
}


def load_yolo_v8(v8_mode="yolov8n"):
    # if not os.path.isdir("yolov7"):
    #     os.system("git clone https://github.com/WongKinYiu/yolov7.git")

    if not os.path.isdir(f"yolov8/{v8_mode}.pt"):
        url = yolov8_types[v8_mode]["link"]
        wget.download(url, f"yolov8/{v8_mode}.pt")


def load_kmeans_model(path, name, dict_=False):
    with open(f"{path}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    lbl_dict = {}
    if dict_:
        lbl_dict = load_dict(pickle_path=f"{path}/{name}.dict")
    return model, lbl_dict


def detect_video(model, video_path, save_path):
    # Kmeans_model, Kmeans_cluster_names = load_kmeans_model(
    #     path=KMEANS_MODEL_FOLDER,
    #     dict_=True,
    #     name=KMEANS_MODEL_NAME,
    # )

    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)

    # video_path = '/home/deny/Рабочий стол/CarpetTracker/videos/Test_0.mp4'
    # save_path = 'tracked_video.mp4'
    tracker = Tracker()

    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # model = YOLO('runs/detect/train21/weights/best.pt')
    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    for i in range(int(f)):
        print(f"Processed {i + 1} / {f} frames")
        ret, frame = vc.read()
        res = model(frame)
        tracker.process(res)

        if save_path:
            if tracker.id_coords[-1]:
                tracker_id = tracker.id_coords[-1]
                coords = tracker.coordinates[-1]
                labels = [
                    f"# {tracker_id[i]} {model.model.names[coords[i][-1]]} {coords[i][-2]:0.2f}"
                    for i in range(len(tracker_id))
                ]
                if len(labels) > 1:
                    cl = colors * len(labels)
                else:
                    cl = colors

                fr = tracker.put_box_on_image(
                    save_path=None,
                    results=res,
                    labels=labels,
                    color_list=cl
                )
                fr = np.array(fr)
            else:
                fr = res[0].orig_img[:, :, ::-1].copy()

            headline = f"Обнаружено объектов: {tracker.obj_count}\n"
            fr = add_headline_to_cv_image(
                image=fr,
                headline=headline
            )
            cv_img = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            out.write(cv_img)


def train(weights='yolo8/yolov8n.pt', config='data_custom.yaml', epochs=50, batch_size=4, name=None):
    model = YOLO(weights)
    model.train(data=config, epochs=epochs, batch=batch_size, name=name)


if __name__ == '__main__':
    # TRAIN
    # train(epochs=50, weights='runs/detect/train_camera1/weights/best.pt', config='data_custom_CAM1.yaml')
    # train(epochs=50, weights='runs/detect/train_camera2/weights/best.pt', config='data_custom_CAM2.yaml')
    # train(epochs=50, weights='yolo8/yolov8n.pt', config='data_custom.yaml')
    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom.yaml', batch_size=2, name='train_mix_yolov8x')
    #
    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM1.yaml', batch_size=2, name='camera_1_yolov8x')
    #
    # train(epochs=50, weights='yolo8/yolov8s.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8s')
    # train(epochs=50, weights='yolo8/yolov8m.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8m')
    # train(epochs=50, weights='yolo8/yolov8l.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8l')
    # train(epochs=50, weights='yolo8/yolov8x.pt', config='data_custom_CAM2.yaml', batch_size=2, name='camera_2_yolov8x')

    # PREDICT IMAGE
    # model1 = YOLO('runs/detect/train_camera1/weights/best.pt')
    # img_path='datasets/batch_01_#108664/obj_train_data/images/batch_01_000013.jpg'
    # # # img_path='datasets/От разметчиков/batch_01_#108664/obj_train_data/batch_01/batch_01_001932.jpg'
    # res = model1(img_path)
    # print(res[0].boxes)
    # print(res[0].orig_shape)
    # img = Image.open(img_path)
    # print(img.size[::-1])

    # PREDICT VIDEO
    weights = [
        'runs/detect/train_camera1/weights/best.pt',
        # 'runs/detect/camera_1_yolov8s/weights/best.pt',
        # 'runs/detect/camera_1_yolov8m/weights/best.pt'
    ]
    ws = ['n', 's', 'm']
    for j, w in enumerate(weights):
        model1 = YOLO(w)
        for i, l in enumerate(
                [
                    'videos/test 1_cam 1.mp4',  # 'videos/test 2_cam 1.mp4', 'videos/test 3_cam 1.mp4',
                    # 'videos/test 1_cam 2.mp4', 'videos/test 2_cam 2.mp4', 'videos/test 3_cam 3.mp4'
                ]):
            detect_video(
                model=model1,
                video_path=l,
                save_path=f'temp/tracked_{i}_yolov8{ws[j]}.mp4'
            )
    # model2 = YOLO('runs/detect/train_camera2/weights/best.pt')
    # detect_video(
    #     model=model2,
    #     video_path='videos/Е,16.28,кам 2.mp4',
    #     save_path='temp/tracked_CAM2.mp4'
    # )
    # for i, l in enumerate(['videos/Е,16.28,кам 2.mp4', 'videos/КЛМ,18-00 (2).mp4', 'videos/ЮЗВ,18,00,КАМ2.mp4']):
    #     detect_video(
    #         model=model2,
    #         video_path=l,
    #         save_path=f'temp/tracked_CAM2 {i}.mp4'
    #     )

    pass
