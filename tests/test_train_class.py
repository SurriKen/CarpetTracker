import os.path
import torch.nn as nn
import torch.nn.functional as F
import time

from nn_classificator import VideoClassifier
from parameters import ROOT_DIR
from utils import load_data


class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5, input_size=(12, 256, 256, 3), concat_axis=2,
                 frame_size=(128, 128), start_channels: int = 32, kernel_size: int = 3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.frame_size = frame_size
        self.concat_axis = concat_axis
        self.conv3d_1 = nn.Conv3d(
            in_channels=input_size[-1], out_channels=start_channels, kernel_size=kernel_size, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=start_channels, out_channels=start_channels * 2,
                                  kernel_size=kernel_size, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=start_channels * 2, out_channels=start_channels * 4, kernel_size=kernel_size,
                                  padding='same', device=device)
        self.conv3d_4 = nn.Conv3d(in_channels=start_channels * 4, out_channels=start_channels * 8, kernel_size=kernel_size,
                                  padding='same', device=device)
        self.dense_3d = nn.Linear(
            in_features=start_channels * 8 * int(input_size[1] / 16) * int(input_size[2] / 16) * input_size[0],
            out_features=256, device=device)
        self.dense_3d_3 = nn.Linear(in_features=256, out_features=num_classes, device=device)
        self.post = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        # print(x.size())
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_3(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_4(x))), (1, 2, 2))
        # print(x.size())
        x = x.reshape(x.size(0), -1)
        x = F.normalize(self.dense_3d(x))
        # print(x.size())
        x = self.post(F.normalize(self.dense_3d_3(x)))
        return x


device = 'cuda:0'
frame_size = (64, 64)
num_frames = 16
concat_axis = 2
start_channels = 32
kernel_size = 3
name = f'model5_{num_frames}f_{frame_size}_ca{concat_axis}'
dataset_path = os.path.join(ROOT_DIR, 'tests/train_class_boxes_model5_Pex.dict')

# device = 'cpu'
dataset = load_data(dataset_path)
dataset = VideoClassifier.create_box_video_dataset(
    dataset=dataset,
    split=0.9,
    frame_size=frame_size,
)

inp = [1, num_frames, *dataset.x_val[0][0][0].shape]
inp[concat_axis] = inp[concat_axis] * 2
print('input size', inp)
model = Net(device=device, num_classes=len(dataset.classes), input_size=tuple(inp[1:]), concat_axis=concat_axis,
            frame_size=frame_size, kernel_size=kernel_size, start_channels=start_channels)
vc = VideoClassifier(num_classes=len(dataset.classes), weights='', input_size=tuple(inp[1:]), name=name, device=device)
vc.model = model

print("Training is started")
vc.train(
    dataset=dataset,
    epochs=50,
    batch_size=4,
    lr=0.00005,
    num_frames=num_frames,
    concat_axis=concat_axis,
)
