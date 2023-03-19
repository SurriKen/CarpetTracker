import os
import pickle
import time

import cv2
import numpy as np
import torch
import torchvision
import wget
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO

from parameters import KMEANS_MODEL_FOLDER, KMEANS_MODEL_NAME
from utils import get_colors, load_dict

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


def put_box_on_image(save_path, results):
    # labels = ['carpet']
    labels = []
    # dataset = 'datasets/DataSetMat_Yolo/train_63sec'
    # img_path = results[0].path
    img = Image.fromarray(results[0].orig_img[:, :, ::-1])
    img.save(f'orig.jpg')
    image = results[0].orig_img[:, :, ::-1].copy()
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    # img = Image.open(img_path)
    coord = []
    for box in results[0].boxes:
        box = box.boxes.tolist()[0]
        # print(box, len(box))
        coord.append([
            int(box[0]),
            int(box[1]),
            int(box[2]),
            int(box[3]),
        ])
        labels.append(results[0].names[int(box[-1])])
    color_list = get_colors(labels)
    bbox = torch.tensor(coord, dtype=torch.int)
    # image = read_image(img_path)
    # print(image.numpy().shape)
    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
    image = torchvision.transforms.ToPILImage()(image_true)
    if save_path:
        image.save(f'{save_path}')
    return image


def detect_video(model, video_path, save_path):
    Kmeans_model, Kmeans_cluster_names = load_kmeans_model(
        path=KMEANS_MODEL_FOLDER,
        dict_=True,
        name=KMEANS_MODEL_NAME,
    )

    # Get names and colors
    names = ['carpet']
    colors = get_colors(names)

    vc = cv2.VideoCapture()
    vc.open(video_path)
    f = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(int(f)):
        ret, frame = vc.read()
        size = (frame.shape[1], frame.shape[0])
        break

    video_capture = cv2.VideoCapture()
    video_capture.open(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    t0 = time.time()
    coords, total_count, cluster_list = [], [], []
    cur_obj, obj_count, ttc, step = 0, 0, 0, 0
    cluster_pred = {}
    for i in range(int(frames)):
        if (i + 1) % 200 == 0:
            print(f"{i + 1} frames are ready")
        ret, frame = video_capture.read()


    # for path, img, im0s, vid_cap in dataset:
    #
    #     t11 = time.time()
    #     if pred[0].size(dim=0):
    #         total_count.append(step)
    #         # ttc = len(total_count)
    #         lines = []
    #         pred0 = [torch.clone(pred[0])]
    #         for i, det0 in enumerate(pred0):
    #             if webcam:  # batch_size >= 1
    #                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
    #             else:
    #                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    #             gn0 = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    #             det0[:, :4] = scale_coords(img.shape[2:], det0[:, :4], im0.shape).round()
    #             for *xyxy0, conf0, cls0 in reversed(det0):
    #                 xywh0 = (xyxy2xywh(torch.tensor(xyxy0).view(1, 4)) / gn0).view(-1).tolist()  # normalized xywh
    #                 lines.append([cls0.item(), *xywh0])
    #         coords.append(lines)
    #     else:
    #         coords.append([])
    #
    #     t111 = time.time()
    #     if ttc != len(total_count):
    #         obj_count, clust_coords = get_object_count(total_count, coords)
    #         # t112 = time.time()
    #         # print(f"get_object_count {(1E3 * (t112 - t111)):.1f}ms, obj_count = {obj_count}, length = {len(total_count)}")
    #         vecs = get_obj_box_squares(clust_coords)
    #         if vecs.any():
    #             _, lbl_pred = kmeans_predict(
    #                 model=Kmeans_model,
    #                 lbl_dict=Kmeans_cluster_names,
    #                 array=vecs[-1]
    #             )
    #             if len(vecs) > cur_obj:
    #                 cur_obj = len(vecs)
    #                 cluster_list.append(lbl_pred)
    #             else:
    #                 cluster_list[-1] = lbl_pred
    #         cluster_pred = dict(collections.Counter(cluster_list))
    #
    #         ttc = len(total_count)
    #     t12 = time.time()
    #     # Process detections
    #     for i, det in enumerate(pred):  # detections per image
    #         if webcam:  # batch_size >= 1
    #             p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
    #         else:
    #             p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    #
    #         p = Path(p)  # to Path
    #         save_path = str(save_dir / p.name)  # img.jpg
    #         txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             # print(step, img.shape[2:], det, im0.shape)
    #             # torch.Size([384, 640]) tensor([[296.25000, 137.75000, 332.75000, 200.00000,   0.91895,   0.00000]], device='cuda:0') (1080, 1920, 3)
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #
    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 if save_txt:  # Write to file
    #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
    #                     with open(txt_path + '.txt', 'a') as f:
    #                         f.write(('%g ' * len(line)).rstrip() % line + '\n')
    #
    #                 # if save_img or view_img:  # Add bbox to image
    #                 #     label = f'{names[int(cls)]} {conf:.2f}'
    #                 #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    #
    #         # Print time (inference + NMS)
    #         print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, '
    #               f'({(1E3 * (t12 - t11)):.1f}ms) clusterization, ({(1E3 * (time.time() - t01)):.1f}ms) total')
    #
    #         # Stream results
    #         # if dataset.mode != "image":
    #         #     headline = f"Обнаружено объектов: {obj_count}\n"
    #         #     for s in CARPET_SIZE_LIST:
    #         #         if not cluster_pred.get(s):
    #         #             x = 0
    #         #         else:
    #         #             x = cluster_pred.get(s)
    #         #         headline = f"{headline}\nSize {s}: {x}"
    #         #     im0 = add_headline_to_image(
    #         #         image=im0,
    #         #         headline=headline
    #         #     )
    #             # cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    #         if view_img:
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond
    #
    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'image':
    #                 cv2.imwrite(save_path, im0)
    #                 # print(f" The image with the result is saved in: {save_path}")
    #             else:  # 'video' or 'stream'
    #                 if vid_path != save_path:  # new video
    #                     vid_path = save_path
    #                     if isinstance(vid_writer, cv2.VideoWriter):
    #                         vid_writer.release()  # release previous video writer
    #                     if vid_cap:  # video
    #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer.write(im0)
    #     step += 1
    # # print('total_count', len(total_count), total_count)
    # # print('coords', len(coords), coords)
    # # print('obj_count', obj_count)
    # # print('clust_coords', len(clust_coords), clust_coords)
    # print('cluster_pred', cluster_pred)
    #
    # # if save_txt or save_img:
    # #     print(f" The output with the result is saved in: {save_path}")
    # #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    # #     print(f"Results saved to {save_dir}{s}")
    #
    # print(f'Done. ({time.time() - t0:.3f}s)')


img_path = 'datasets/DataSetMat_Yolo/train_23sec/images/Mat_20_sec_1920_mp4-11_jpg.rf.953c8cdfcd29d3261eb51dbd6697cd5a.jpg'
# img_path = '1.mp4'
# model = YOLO('runs/detect/train19/weights/best.pt')
model = YOLO('yolo8/yolov8n.pt')
# print(model.__dict__)
model.train(data='data_custom.yaml', epochs=200)
# results = model(img_path)
# model.predict('videos/Test_0.mp4')
# for box in results[0].boxes:
#     print(box.boxes.tolist())
# print(results[0].orig_img.shape)
# img = Image.open(img_path)
# img = cv2.imread(img_path)
# cv2.imshow('image', img)
# convert_tensor = transforms.ToTensor()
# img = convert_tensor(img)
# r2 = model(img)
# print(r2)
# put_box_on_image(save_path='1.jpg', results=r2)

# os.system('yolo task=detect mode=predict model=yolo8/yolov8n.pt source=1.mp4')

# video_path = '/home/deny/Рабочий стол/CarpetTracker/videos/Test_0.mp4'
# vc = cv2.VideoCapture()
# vc.open(video_path)
# f = vc.get(cv2.CAP_PROP_FRAME_COUNT)
#
# t0 = time.time()
# for i in range(int(f)):
#     ret, frame = vc.read()
#     # size = (frame.shape[1], frame.shape[0])
#     # break
# print(time.time()-t0)
