import os
import shutil
import time
import cv2
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from PIL import Image

from utils import get_colors


# Основная функция
def put_box_on_image(frames_path: str, save_path: str, box_path: str = None):
    image_list, box_list, lbl_list, color_list = [], [], [], []
    with os.scandir(frames_path) as folder:
        for f in folder:
            image_list.append(f.name)
    image_list = sorted(image_list)
    if box_path:
        with os.scandir(box_path) as folder:
            for f in folder:
                box_list.append(f.name)
        for box in box_list:
            # box_info = PrepareDataset.read_xml(f"{box_path}/{f'{box}'}")
            with open(f"{box_path}/{f'{box}'}", 'r') as handle:
                box_info = handle.readlines()

            coords = []
            if box_info:
                box_info = box_info.split('\n')
                for b in box_info:
                    bb = b.split(' ')
                    coord = [float(bb[1]), float(bb[2]), float(bb[3]), float(bb[4])]  # x y w h
                    coords.append(coord)
                lbl_list.append(box_info["coords"][-1])
        lbl_list = sorted(lbl_list)
        color_list = get_colors(lbl_list)

    st = time.time()
    count = 0
    print(f"Start processing...\n")
    out = cv2.VideoWriter(
        f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
    # print(f'{save_path}/{video_name}.mp4')
    for img in image_list:
        if (count + 1) % int(len(image_list) * 0.1) == 0:
            print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were processed...")
        name = img.split(".")[0]
        if box_path:
            # print(img)
            if box_type == 'xml':
                if resize:
                    box_info = PrepareDataset.read_xml(
                        xml_path=f"{box_path}/{f'{name}.xml'}",
                        shrink=True,
                        new_width=parameters['size'][0],
                        new_height=parameters['size'][1]
                    )
                else:
                    box_info = PrepareDataset.read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
                boxes, labels = [], []
                # print(box_info)
                for b in box_info["coords"]:
                    boxes.append(b[:-1])
                    labels.append(b[-1])
                bbox = torch.tensor(boxes, dtype=torch.int)
                image = read_image(f"{frames_path}/{img}")
                image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
                image = torchvision.transforms.ToPILImage()(image_true)


if __name__ == '__main__':
    labels = ['carpet']
    color_list = get_colors(labels)
    dataset = 'datasets/DataSetMat_Yolo/train_63sec'
    img_names = []
    save_path = 'datasets/DataSetMat_Yolo/train_63sec/img+box'
    with os.scandir(f"{dataset}/images") as folder:
        for f in folder:
            img_names.append(f.name[:-4])
    for name in img_names:
        img_path = f'{dataset}/images/{name}.jpg'
        box_path = f'{dataset}/labels/{name}.txt'
        img = Image.open(img_path)
        with open(box_path, 'r') as handle:
            box_info = handle.readlines()[0]
            box_info = box_info.split('\n')
        # print(box_info, img.size)
        coord = []
        for box in box_info:
            box = box.split(" ")
            # print(box, len(box))
            coord.append([
                int((float(box[1]) - float(box[3]) / 2) * img.size[0]),
                int((float(box[2]) - float(box[4]) / 2) * img.size[1]),
                int((float(box[1]) + float(box[3]) / 2) * img.size[0]),
                int((float(box[2]) + float(box[4]) / 2) * img.size[1]),
            ])
        print(coord)
        bbox = torch.tensor(coord, dtype=torch.int)
        image = read_image(img_path)
        image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
        image = torchvision.transforms.ToPILImage()(image_true)
        image.save(f'{save_path}/{name}.png')

    pass
