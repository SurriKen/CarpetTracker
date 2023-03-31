import cv2
import numpy as np
import torch
import torchvision
import ultralytics
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from tracker import Tracker
from utils import get_colors
from utils import add_headline_to_cv_image


def put_box_on_image(save_path, results, labels, color_list):
    image = results[0].orig_img[:, :, ::-1].copy()
    image = np.transpose(image, (2, 0, 1))
    w, h = image.shape[:2]
    image = torch.from_numpy(image)
    coord = []
    for box in results[0].boxes:
        box = box.boxes.tolist()[0]
        coord.append([
            int(box[0]),
            int(box[1]),
            int(box[2]),
            int(box[3]),
        ])
    bbox = torch.tensor(coord, dtype=torch.int)
    if bbox.tolist():
        image_true = draw_bounding_boxes(
            image, bbox, width=3, labels=labels, colors=color_list, fill=True, font='arial.ttf', font_size=int(h * 0.02))
        image = torchvision.transforms.ToPILImage()(image_true)
    else:
        image = torchvision.transforms.ToPILImage()(image)
    if save_path:
        image.save(f'{save_path}')
    return image


# video_path = '/home/deny/Рабочий стол/CarpetTracker/videos/Test_0.mp4'
# save_path = 'temp/tracked_video.mp4'
# tracker = Tracker()
# count = 0
#
# vc = cv2.VideoCapture()
# vc.open(video_path)
# f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = vc.get(cv2.CAP_PROP_FPS)
# w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# model = YOLO('runs/detect/train21/weights/best.pt')
#
# color_list = get_colors(["carpet"])
#
# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
#
# for i in range(int(f)):
#     print(i)
#     ret, frame = vc.read()
#     res = model(frame)
#     tracker.process(res)
#     count += 1
#     # clear_output()
#
#     if tracker.id_coords[-1]:
#         tracker_id = tracker.id_coords[-1]
#         coords = tracker.coordinates[-1]
#         labels = [
#             f"# {tracker_id[i]} {model.model.names[coords[i][-1]]} {coords[i][-2]:0.2f}"
#             for i in range(len(tracker_id))
#         ]
#         if len(labels) > 1:
#             cl = color_list * len(labels)
#         else:
#             cl = color_list
#
#         fr = put_box_on_image(
#             save_path=None,
#             results=res,
#             labels=labels,
#             color_list=cl
#         )
#         fr = np.array(fr)
#         # print(fr.shape)
#         # print(np.array(frame).max())
#     else:
#         fr = res[0].orig_img[:, :, ::-1].copy()
#         # fr = np.transpose(fr, (2, 0, 1))
#         # print(fr.shape)
#     # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#     # vid_writer.write(fr)
#     headline = f"Обнаружено объектов: {tracker.obj_count}\n"
#     # for s in CARPET_SIZE_LIST:
#     #     if not cluster_pred.get(s):
#     #         x = 0
#     #     else:
#     #         x = cluster_pred.get(s)
#     #     headline = f"{headline}\nSize {s}: {x}"
#     im0 = add_headline_to_cv_image(
#         image=fr,
#         headline=headline
#     )
#     cv_img = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
#     out.write(cv_img)
#     # clear_output()
#
#     # cv2.imshow('yolov8', cv_img)
#
#     if count > 300:
#         break

print(355 in range(100, 500))
