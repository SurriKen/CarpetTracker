import os
import shutil
import wget
from ultralytics import YOLO

yolov7_types = {
    "yolov7": {"Test Size": 640, "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"},
    "yolov7x": {"Test Size": 640, "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt"},
    "yolov7-w6": {"Test Size": 1280,
                  "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt"},
    "yolov7-e6": {"Test Size": 1280,
                  "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt"},
    "yolov7-d6": {"Test Size": 1280,
                  "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt"},
    "yolov7-e6e": {"Test Size": 1280,
                   "link": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt"},
}

def set_yolo_v7(v7_mode="yolov7"):
    if not os.path.isdir("yolov7"):
        os.system("git clone https://github.com/WongKinYiu/yolov7.git")

    if not os.path.isdir(f"{v7_mode}/{v7_mode}.pt"):
        url = yolov7_types[v7_mode]["link"]
        wget.download(url, f"{v7_mode}/{v7_mode}.pt")


# def set_yolo_v8(v8_mode="yolov8n"):
#     if not os.path.isdir("yolov8"):
#         os.system("git clone https://github.com/WongKinYiu/yolov7.git")
#
#     if not os.path.isdir(f"{v8_mode}/{v8_mode}.pt"):
#         url = yolov7_types[v8_mode]["link"]
#         wget.download(url, f"{v8_mode}/{v8_mode}.pt")


def predict_yolo_v7(v7_mode, weights, conf, source, save_path, name):
    try:
        shutil.rmtree(f"{save_path}/{name}")
    except:
        pass
    os.system("cd yolov7")
    os.system(
        f"python3 yolo7/detect.py --weights {weights} --conf {conf} "
        f"--img-size {yolov7_types[v7_mode]['Test Size']} --source {source} --save-txt "
        f"--project {save_path} --name {name}"
    )


def train_yolo_v7(dataset_path, v7_mode, batch, epochs, name, save_path, weights='yolo7/yolo7.pt'):
    try:
        os.remove(f"{dataset_path}/train/labels.cache")
        os.remove(f"{dataset_path}/val/labels.cache")
    except:
        pass

    os.system(
        f"python3 yolo7/train.py --workers 1 --device 0 --batch-size {batch} --epochs {epochs} --weights {weights} "
        f"--data {dataset_path}/data_custom.yaml --img-size {yolov7_types[v7_mode]['Test Size']} "
        f"{yolov7_types[v7_mode]['Test Size']} --cfg {dataset_path}/cfg_custom.yaml "
        f"--hyp {dataset_path}/hyp.scratch.custom.yaml --project {save_path} --name {name}"
    )


# # predict_yolo_v7("yolov7", 0.5, "videos/Air_1.mp4", 'predict', 'predict_yolo_v7_air')
train_yolo_v7(
    dataset_path='datasets/mix_yolov7',
    v7_mode='yolov7',
    batch=2,
    epochs=1,
    name='mix_yolov7',
    save_path='train',
    weights="train/mix_yolov710/weights/last.pt"
)
# for i in range(5):
#     predict_yolo_v7(
#         v7_mode="yolov7",
#         weights='train/mix_yolov710/weights/best.pt',
#         conf=0.7,
#         source=f"datasets/Train_{i}_0s-300s/video/Train_{i}.mp4",
#         save_path='predict',
#         name=f'predict_train_{i}_yolov7'
#     )
# for i in ['Test_2', 'Test_3']:
#     predict_yolo_v7(
#         v7_mode="yolov7",
#         weights='train/mix_yolov710/weights/best.pt',
#         conf=0.7,
#         source=f"videos/{i}.mp4",
#         save_path='predict',
#         name=f'predict_{i}'
#     )
# predict_yolo_v7(
#     v7_mode="yolov7",
#     weights='train/mix_yolov710/weights/best.pt',
#     conf=0.7,
#     source="datasets/Train_0_0s-300s/video/Train_0.mp4",
#     save_path='predict',
#     name='predict_train_0_300s_yolov7'
# )
