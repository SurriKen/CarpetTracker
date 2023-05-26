import os.path
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_processing import DatasetProcessing, VideoClass
import time
from parameters import ROOT_DIR
from utils import logger, time_converter


class Net(nn.Module):

    def __init__(self, device='cpu'):
        super(Net, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=5, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=5, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=16, out_channels=5, kernel_size=5, padding='same', device=device)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv3d_1(x)), 2)
        x = F.max_pool3d(F.relu(self.conv3d_2(x)), 2)
        x = F.max_pool3d(F.relu(self.conv3d_3(x)), 2)
        x, _ = torch.max(x, 2)
        x, _ = torch.max(x, 2)
        x, _ = torch.max(x, 2)
        x = F.softmax(x, 1)
        return x


class VideoClassifier:

    def __init__(self, name: str = 'model', weights: str = None, device: str = 'cuda:0', frame_size: tuple = (255, 255)):
        self.device = device
        self.frame_size = frame_size
        self.torch_device = torch.device(device)
        self.weights = weights
        self.load_model(weights)
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass
        self.name = name

    def load_model(self, weights: str = '') -> None:
        if weights and weights.split('.')[-1] == 'pt':
            self.model = torch.jit.load(weights)
        else:
            self.model = Net(device=self.device)
            self.model.zero_grad()

    def save_model(self, mode: str = 'last'):
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', self.name, f"{mode}.pt"))  # Save

    def get_x_batch(self, video_path: str, frame_size: tuple[int, int] = (255, 255)) -> torch.Tensor:
        x_train = DatasetProcessing.video_to_array(video_path)
        x_train = np.expand_dims(x_train, 0) / 255
        x_train = torch.from_numpy(x_train)
        x_train = x_train.permute(0, 4, 1, 2, 3)
        x_train = F.interpolate(x_train, size=(x_train.size()[2], frame_size[0], frame_size[1]))
        if 'cuda' in self.device:
            return x_train.to(self.torch_device, dtype=torch.float)
        else:
            return x_train

    def numpy_to_torch(self, array: np.ndarray, frame_size: tuple[int, int] = (255, 255)) -> torch.Tensor:
        if len(array.shape) == 4:
            array = np.expand_dims(array, 0)

        if array.max() > 1.:
            array = array / 255

        array = torch.from_numpy(array)
        array = array.permute(0, 4, 1, 2, 3)
        array = F.interpolate(array, size=(array.size()[2], frame_size[0], frame_size[1]))
        if 'cuda' in self.device:
            return array.to(self.torch_device, dtype=torch.float)
        else:
            return array

    def get_y_batch(self, label: int, num_labels: int) -> torch.Tensor:
        lbl = DatasetProcessing.ohe_from_list([label], num_labels)
        if 'cuda' in self.device:
            lbl = torch.tensor(lbl, dtype=torch.float, device=self.torch_device)
        else:
            lbl = torch.tensor(lbl, dtype=torch.float)
        return lbl.view(1, -1)

    def train(self, dataset: VideoClass, epochs: int, weights: str = '',
              lr: float = 0.001) -> None:

        if weights:
            self.load_model(weights)

        stop = False
        i = 1
        name = f"{self.name}{i}"
        while not stop:
            if self.name in os.listdir(os.path.join(ROOT_DIR, 'video_class_train')):
                name = f"{self.name}{i}"
                i += 1
            else:
                os.mkdir(os.path.join(ROOT_DIR, 'video_class_train', name))
                stop = True

        st = time.time()
        num_classes = len(dataset.classes)
        num_train_batches = len(dataset.x_train)
        train_seq = list(np.arange(num_train_batches))
        num_val_batches = len(dataset.x_val)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = 10000.

        logger_batch_markers = []
        for i in range(10):
            logger_batch_markers.append(int(num_train_batches * (i + 1) / 10))

        for epoch in range(epochs):
            st_ep = time.time()
            shuffle(train_seq)
            train_loss = 0.
            for batch in range(num_train_batches):
                st_b = time.time()
                x_train = self.get_x_batch(video_path=dataset.x_train[train_seq[batch]], frame_size=self.frame_size)
                y_train = self.get_y_batch(label=dataset.y_train[train_seq[batch]], num_labels=num_classes)
                optimizer.zero_grad()
                output = self.model(x_train)
                loss = criterion(output, y_train)
                train_loss += loss
                loss.backward()
                optimizer.step()
                if batch + 1 in logger_batch_markers:
                    logger.info(f"  -- Epoch {epoch+1}, batch {batch+1} / {num_train_batches}, "
                                f"train_loss= {train_loss / batch + 1}"
                                f"aver. batch time = {round((time.time() - st_b) / (batch+1), 2)} sec\n")
            # logger.info(f"\nEpoch {epoch + 1}, train_loss= {train_loss / num_train_batches}"
            #             f"epoch time = {round((time.time() - st_ep) / num_train_batches, 2)} sec\n")

            val_loss = 0

            for val_batch in range(num_val_batches):
                x_val = self.get_x_batch(video_path=dataset.x_val[train_seq[val_batch]], frame_size=self.frame_size)
                y_val = self.get_y_batch(label=dataset.y_val[train_seq[val_batch]], num_labels=num_classes)
                output = self.model(x_val)
                loss = criterion(output, y_val)
                val_loss += loss

            self.save_model(mode='last')
            if val_loss / num_val_batches <= best_loss:
                self.save_model('best')

            logger.info(f"\nEpoch {epoch + 1}, train_loss= {train_loss / num_train_batches},"
                        f"val_loss = {val_loss / num_val_batches},"
                        f"epoch time = {round((time.time() - st_ep), 2)} sec\n")

        logger.info(f"Training is finished, "
                    f"train time = {time_converter(int(time.time() - st))}\n")

    def predict(self, array, weights: str = '', classes: list = None) -> list:
        if classes is None:
            classes = []
        if weights or weights != self.weights:
            self.load_model(weights)
        array = self.numpy_to_torch(array, self.frame_size)
        output = self.model(array)
        output = output.cpu().detach().numpy() if self.device != 'cpu' else output.detach().numpy()
        if classes:
            return [classes[i] for i in list(np.argmax(output, axis=-1))]
        return list(np.argmax(output, axis=-1))


if __name__ == "__main__":
    vid_ex = 'datasets/class_videos/60x90/7.mp4'

    arr = DatasetProcessing.video_to_array(vid_ex)
    arr = np.expand_dims(arr, 0) / 255
    # arr = arr / 255
    lbl = DatasetProcessing.ohe_from_list([1], 5)

    st = time.time()
    # device = 'cuda:0'
    device = 'cpu'
    net = Net(device=device)
    net.zero_grad()
    # print(net)
    params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())

    cuda0 = torch.device(device)
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
        target = torch.tensor(lbl, dtype=torch.float, device=cuda0)
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
    lbl = np.argmax(out2.detach().numpy(), axis=-1) if device=='cpu' else np.argmax(out2.cpu().detach().numpy(), axis=-1)
    print(f"-- Output tensor for loaded model: {out2}, label={lbl[0]}", type(lbl))
