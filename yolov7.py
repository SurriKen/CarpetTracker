import os
import wget

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

    if not os.path.isdir(f"yolov7/{v7_mode}.pt"):
        url = yolov7_types[v7_mode]["link"]
        wget.download(url, f"yolov7/{v7_mode}.pt")


def predict_yolo_v7(v7_mode, conf, source, save_path, name):
    # os.system("cd yolov7")
    os.system(
        f"python yolov7/detect.py --weights yolov7/{v7_mode}.pt --conf {conf} "
        f"--img-size {yolov7_types[v7_mode]['Test Size']} --source {source} --save-txt "
        f"--project {save_path} --name {name}"
    )


def train_yolo_v7(dataset_path, v7_mode, batch, epochs, name, save_path):
    try:
        os.remove(f"{dataset_path}/train/labels.cache")
        os.remove(f"{dataset_path}/val/labels.cache")
    except:
        pass

    os.system(
        f"python yolov7/train.py --workers 1 --device 0 --batch-size {batch} --epochs {epochs} "
        f"--data {dataset_path}/data_custom.yaml --img-size {yolov7_types[v7_mode]['Test Size']} "
        f"{yolov7_types[v7_mode]['Test Size']} --cfg {dataset_path}/cfg_custom.yaml --weights yolov7/{v7_mode}.pt "
        f"--hyp {dataset_path}/hyp.scratch.custom.yaml --project {save_path} --name {name}"
    )


# predict_yolo_v7("yolov7", 0.5, "videos/Air_1.mp4", 'predict', 'predict_yolo_v7_air')
# train_yolo_v7(
#     dataset_path='datasets/train2_yolov7',
#     v7_mode='yolov7',
#     batch=4,
#     epochs=50,
#     name='train2_yolov7',
#     save_path='runs'
# )
import torch
print(torch.__version__)
print(torch.cuda.is_available())
torch.zeros(1).cuda()