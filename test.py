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
def put_box_on_image(dataset: str, save_path: str, box_path: str = None):
    labels = ['carpet']
    color_list = get_colors(labels)
    # dataset = 'datasets/DataSetMat_Yolo/train_63sec'
    img_names = []
    # save_path = 'datasets/DataSetMat_Yolo/train_63sec/img+box'
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


if __name__ == '__main__':
    # labels = ['carpet']
    # color_list = get_colors(labels)
    # dataset = 'datasets/DataSetMat_Yolo/train_63sec'
    # img_names = []
    # save_path = 'datasets/DataSetMat_Yolo/train_63sec/img+box'
    # with os.scandir(f"{dataset}/images") as folder:
    #     for f in folder:
    #         img_names.append(f.name[:-4])
    # for name in img_names:
    #     img_path = f'{dataset}/images/{name}.jpg'
    #     box_path = f'{dataset}/labels/{name}.txt'
    #     img = Image.open(img_path)
    #     with open(box_path, 'r') as handle:
    #         box_info = handle.readlines()[0]
    #         box_info = box_info.split('\n')
    #     # print(box_info, img.size)
    #     coord = []
    #     for box in box_info:
    #         box = box.split(" ")
    #         # print(box, len(box))
    #         coord.append([
    #             int((float(box[1]) - float(box[3]) / 2) * img.size[0]),
    #             int((float(box[2]) - float(box[4]) / 2) * img.size[1]),
    #             int((float(box[1]) + float(box[3]) / 2) * img.size[0]),
    #             int((float(box[2]) + float(box[4]) / 2) * img.size[1]),
    #         ])
    #     print(coord)
    #     bbox = torch.tensor(coord, dtype=torch.int)
    #     image = read_image(img_path)
    #     image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
    #     image = torchvision.transforms.ToPILImage()(image_true)
    #     image.save(f'{save_path}/{name}.png')

    pass
