import os

if not os.path.isdir("yolov7"):
    os.system("git clone https://github.com/WongKinYiu/yolov7.git")

import wget

if not os.path.isdir("yolov7/yolov7.pt"):
    url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    filename = wget.download(url, "yolov7/yolov7.pt")
    print(filename)

