import os.path
import torch.nn as nn
import torch.nn.functional as F
from nn_classificator import VideoClassifier
from parameters import ROOT_DIR
from utils import load_data


class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5, input_size=(12, 256, 256, 3), concat_axis=2,
                 frame_size=(128, 128), start_channels: int = 32, kernel_size: int = 3,
                 dense_out: int = 256):
        super(Net, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.frame_size = frame_size
        self.concat_axis = concat_axis
        self.dropout = dropout
        self.conv3d_1 = nn.Conv3d(
            in_channels=input_size[-1], out_channels=start_channels, kernel_size=kernel_size, padding='same',
            device=device)
        self.conv3d_2 = nn.Conv3d(
            in_channels=start_channels, out_channels=start_channels * 2,
            kernel_size=kernel_size, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(
            in_channels=start_channels * 2, out_channels=start_channels * 4,
            kernel_size=kernel_size, padding='same', device=device)
        self.conv3d_4 = nn.Conv3d(
            in_channels=start_channels * 4, out_channels=start_channels * 8,
            kernel_size=kernel_size, padding='same', device=device)
        # self.conv3d_5 = nn.Conv3d(
        #     in_channels=start_channels * 8, out_channels=start_channels * 16,
        #     kernel_size=kernel_size, padding='same', device=device)
        self.dense_3d = nn.Linear(
            in_features=
            start_channels * 8 * int(input_size[1] / (2 ** 4)) * int(input_size[2] / (2 ** 4)) * input_size[0],
            out_features=dense_out, device=device)
        self.dense_3d_3 = nn.Linear(in_features=dense_out, out_features=num_classes, device=device)
        self.post = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_3(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_4(x))), (1, 2, 2))
        # x = F.max_pool3d(F.relu(F.normalize(self.conv3d_5(x))), (1, 2, 2))
        # print(x.size())
        x = x.reshape(x.size(0), -1)
        x = F.normalize(self.dense_3d(F.dropout(x, self.dropout)))
        # print(x.size())
        x = self.post(F.normalize(self.dense_3d_3(x)))
        return x


device = 'cuda:0'
frame_size = (128, 128)
num_frames = 16
concat_axis = 2
start_channels = 32
kernel_size = 3
dense_out = 256
dropout = 0.0
name = f'model6_{num_frames}f_{frame_size}_ca{concat_axis}'
# dataset_path = os.path.join(ROOT_DIR, 'temp/train_class_boxes_model5_Pex.dict')
#
# # device = 'cpu'
# dataset = load_data(dataset_path)
# for cl in dataset.keys():
#     print(cl, len(dataset[cl].keys()), dataset[cl][list(dataset[cl].keys())[0]])

dataset_path = os.path.join(ROOT_DIR, 'temp/crop_frames_28.dict')

# device = 'cpu'
# ds = load_data(dataset_path)
# for cl in dataset.keys():
#     dataset[cl].pop('camera_1')
#     dataset[cl].pop('camera_2')
#     print(cl, len(dataset[cl].keys()), dataset[cl][list(dataset[cl].keys())[0]])

dataset = VideoClassifier.create_box_video_dataset(
    dataset={},
    dataset_path=dataset_path,
    split=0.85,
    test_split=0.05,
    frame_size=frame_size,
)
for k, v in dataset.params.items():
    print(k, v)

inp = [1, num_frames, *dataset.x_val[0][0][0].shape]
inp[concat_axis] = inp[concat_axis] * 2
print('input size', inp)
model = Net(
    device=device, num_classes=len(dataset.classes), input_size=tuple(inp[1:]), concat_axis=concat_axis,
    frame_size=frame_size, kernel_size=kernel_size, start_channels=start_channels, dense_out=dense_out,
)
vc = VideoClassifier(num_classes=len(dataset.classes), weights='', input_size=tuple(inp[1:]), name=name, device=device)
vc.model = model

print("Training is started")
vc.train(
    dataset=dataset,
    epochs=10,
    batch_size=4,
    lr=0.00005,
    num_frames=num_frames,
    concat_axis=concat_axis,
    save_dataset=True
)
