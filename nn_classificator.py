# https://newtechaudit.ru/kak-predskazyvat-budushhee-s-pomoshhyu-keras/
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_processing import DatasetProcessing
import time
from utils import logger

# st = time.time()
vid_ex = 'datasets/class_videos/60x90/7.mp4'

arr = DatasetProcessing.video_to_array(vid_ex)
arr = np.expand_dims(arr, 0) / 255
# arr = arr / 255
lbl = DatasetProcessing.ohe_from_list([1], 5)


# print(arr.shape)
# logger.info(f"-- Video processing time = {round(time.time() - st, 1)} sec")

#
# print(arr.shape, lbl)
# arr.shape[2:]


class Net(nn.Module):

    def __init__(self, device='cpu'):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=5, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=5, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=16, out_channels=5, kernel_size=5, padding='same', device=device)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # n = x.cpu().size()[2]
        # x = F.interpolate(x, size=(n, 256, 256))
        x = F.max_pool3d(F.relu(self.conv3d_1(x)), 2)
        # print(x.size())
        x = F.max_pool3d(F.relu(self.conv3d_2(x)), 2)
        # print(x.size())
        x = F.max_pool3d(F.relu(self.conv3d_3(x)), 2)
        # print(x.size())
        x, _ = torch.max(x, 2)
        # print(x.size())
        x, _ = torch.max(x, 2)
        # print(x.size())
        x, _ = torch.max(x, 2)
        # print(x.size())
        x = F.softmax(x, 1)
        return x


st = time.time()
net = Net(device='cuda:0')
net.zero_grad()
# print(net)
params = list(net.parameters())
# print(len(params))
# print(params[0].size())

cuda0 = torch.device('cuda:0')
arr = torch.from_numpy(arr)
arr = arr.permute(0, 4, 1, 2, 3)
arr = F.interpolate(arr, size=(arr.size()[2], 256, 256))
print(f"-- Input size: {arr.size()}\n")
arr = arr.to(cuda0, dtype=torch.float)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
output = net(arr)
print(f"-- Output tensor: {output}")
logger.info(f"-- Pretrain process time = {round(time.time() - st, 2)} sec\n")

for i in range(2):
    st = time.time()
    optimizer.zero_grad()
    output = net(arr)
    target = torch.tensor([1., 0., 0., 0., 0.], dtype=torch.float, device=cuda0)
    target = target.view(1, -1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    # print(f" --- weights before {list(net.parameters())[0][0][0][0][0]}")
    loss.backward()
    # print(f" --- weights after {list(net.parameters())[0][0][0][0][0]}")
    optimizer.step()
    # print(f" --- weights after optim {list(net.parameters())[0][0][0][0][0]}")
    print(f"-- loss={loss}")
    # print(f"-- Output size: {output.cpu().size()}")
    output = net(arr)
    print(f"-- Output tensor: {output}")
    logger.info(f"-- Predict time = {round(time.time() - st, 2)} sec\n")

model_scripted = torch.jit.script(net)  # Export to TorchScript
model_scripted.save('model_scripted.pt')  # Save

model = torch.jit.load('model_scripted.pt')
out2 = model(arr)
print(f"-- Output tensor for loaded model: {out2}")
