# import numpy as np
# import skvideo.io
# import torch
# import os
# import torch.nn.functional as F
#
# from dataset_processing import DatasetProcessing
# from parameters import ROOT_DIR
# import json
#
# from tests.test import ResNet3D, VideoClassifier
#
# with open("/media/deny/Новый том/AI/CarpetTracker/kinetics_classnames.json", "r") as f:
#     kinetics_classnames = json.load(f)
#
# print("kinetics_classnames", len(kinetics_classnames), kinetics_classnames)
#
# # Device on which to run the model
# device = "cuda:0"
# # device = "cpu"
#
# # Pick a pretrained model
# model_name = "slow_r50"
import os

import cv2
import numpy as np
import skvideo.io
import torch
from PIL import Image

from dataset_processing import DatasetProcessing
from parameters import ROOT_DIR
from tests.classif_models import VideoClassifier

# # Local path to the parent folder of hubconf.py in the pytorchvideo codebase
# path = '/media/deny/Новый том/AI/CarpetTracker/pytorchvideo'
# # /media/deny/Новый том/AI/CarpetTracker/tests/pytorchvideo/hubconf.py
# # model = torch.hub.load(path, source="local", model=model_name, pretrained=False)
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
# model_url = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth"
#
# checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location=device)
# state_dict = checkpoint["model_state"]
# # print('state_dict', state_dict)
#
# # Apply the state dict to the model
# model.load_state_dict(state_dict)
#
# model = model.eval()
# model = model.to(device)

# kinetics_id_to_classname = {v: k for k, v in kinetics_classnames.items()}
# print(kinetics_id_to_classname)
#
# video_path = "/media/deny/Новый том/AI/CarpetTracker/datasets/class_videos/85x150/4.mp4"
#
#
# def resnet3d_dataset(link: str) -> torch.FloatTensor:
#     side_size = 256
#     mean = [0.45, 0.45, 0.45]
#     std = [0.225, 0.225, 0.225]
#     crop_size = 256
#     num_frames = 8
#
#     video = skvideo.io.vread(os.path.join(ROOT_DIR, link))
#     x_train = video / 255
#     x_train = torch.from_numpy(x_train)
#     x_train = x_train.permute(3, 0, 1, 2)
#     x_train = F.interpolate(x_train, size=(side_size, side_size))
#     mean = torch.as_tensor(mean, dtype=x_train.dtype, device=x_train.device)
#     std = torch.as_tensor(std, dtype=x_train.dtype, device=x_train.device)
#     x_train.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
#     return x_train.type(torch.FloatTensor)
#

# videos = 'datasets/class_videos'
# dataset = DatasetProcessing.create_video_class_dataset_generator(
#     folder_path=videos, split=0.8
# )
# classes = {i: v for i, v in enumerate(dataset.classes)}
#
# x_train = resnet3d_dataset(video_path)
# print(type(x_train), x_train.size())
#
# model = ResNet3D(device=device)
#
# inputs = x_train.to(device)
# preds_pre_act = model(inputs[None, ...])
# post_act = torch.nn.Softmax(dim=1)
# preds = post_act(preds_pre_act).cpu().detach().numpy()
# print(type(preds), preds.shape, preds)
# x = preds.tolist()[0]
# x = [(v, i, classes[i]) for i, v in enumerate(x)]
# x = sorted(x, reverse=True)
# preds = np.argmax(preds, axis=-1)
#
# print(type(preds), preds, classes[int(preds)])
# for i in range(5):
#     print(x[i])

# with torch.no_grad():
#     for sample in dataset:
#         video_path = sample.filepath
#
#         # Initialize an EncodedVideo helper class
#         video = EncodedVideo.from_path(video_path)
#
#         # Select the duration of the clip to load by specifying the start and end duration
#         # The start_sec should correspond to where the action occurs in the video
#         start_sec = 0
#         clip_duration = int(video.duration)
#         end_sec = start_sec + clip_duration
#
#         # Load the desired clip
#         video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
#
#         # Apply a transform to normalize the video input
#         video_data = transform(video_data)
#
#         # Move the inputs to the desired device
#         inputs = video_data["video"]
#         inputs = inputs.to(device)
#
#         # Pass the input clip through the model
#         preds_pre_act = model(inputs[None, ...])
#
#         # Get the predicted classes
#         post_act = torch.nn.Softmax(dim=1)
#         preds = post_act(preds_pre_act)
#
#         # Generate FiftyOne labels from predictions
#         prediction_top_1, predictions_top_5 = parse_predictions(preds, kinetics_id_to_classname, k=5)
#
#         # Add FiftyOne label fields to Sample
#         sample["predictions"] = prediction_top_1
#         sample["predictions_top_5"] = predictions_top_5
#         sample.save()

vid_ex = 'datasets/class_videos/60x90/7.mp4'

def get_x_batch(video_path: list) -> torch.Tensor:
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    x_train = []
    for vid in video_path:
        array = DatasetProcessing.video_to_array(video_path=vid)
        array = VideoClassifier.crop_video_array(array, 10)
        array = resize_video_array(array, size=(256, 256))
        x_train.append(array / 255)
    x_train = torch.from_numpy(np.array(x_train, dtype='float32'))
    x_train = x_train.permute(0, 4, 1, 2, 3)
    x_train = x_train.to('cuda:0')
    # x_train = F.interpolate(x_train, size=(x_train.size()[2], self.frame_size[0], self.frame_size[1]))
    mean = torch.as_tensor(mean, dtype=x_train.dtype, device=torch.device('cuda:0'))
    std = torch.as_tensor(std, dtype=x_train.dtype, device=torch.device('cuda:0'))
    x_train.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    # if 'cuda' in self.device:
    #     return x_train.to(self.torch_device, dtype=torch.float)
    # else:
    return x_train.type(torch.FloatTensor)

def resize_video_array(array, size=(256,256)):
    new_array = []
    for a in array:
        im = Image.fromarray(a)
        im = im.resize(size)
        new_array.append(np.array(im))
    return np.array(new_array)

# ar = get_x_batch([vid_ex])
# print(ar.size)
vid = skvideo.io.vread(fname=os.path.join(ROOT_DIR, vid_ex))
print(vid.shape)
vid = resize_video_array(vid, (256, 256))
print(vid.shape)

def array_to_video(arrays: np.ndarray, save_dir: str = '', size: tuple = (256, 256)):
    out = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
    for a in arrays:
        im = Image.fromarray(a)
        im = im.resize(size)
        cv_img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        out.write(cv_img)
    out.release()

# array_to_video(arrays=vid, save_dir=f'{ROOT_DIR}/tests/111.mp4')

x = get_x_batch(video_path=[vid_ex])
print(x.size())