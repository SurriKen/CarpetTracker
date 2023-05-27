import inspect
import os.path
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from dataset_processing import DatasetProcessing, VideoClass
import time
from parameters import ROOT_DIR
from utils import logger, time_converter, plot_and_save_gragh, save_dict_to_table_txt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

logger.info("\n    --- Running test.py ---    \n")


# https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5):
        super(Net, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=5, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=5, padding='same', device=device)
        self.dense = nn.Linear(128*32*32, num_classes, device=device)

    def forward(self, x):
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), 2)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), 2)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_3(x))), 2)
        x = x.view(x.size(0), x.size(2), -1)
        x = torch.mean(x, dim=1)
        x = torch.sigmoid(self.dense(x))
        return x


class VideoClassifier:

    def __init__(self, num_classes: int = 5, name: str = 'model', weights: str = None, device: str = 'cuda:0',
                 frame_size: tuple = (256, 256)):
        self.num_classes = num_classes
        self.device = device
        self.frame_size = frame_size
        self.torch_device = torch.device(device)
        self.weights = weights
        self.load_model(weights)
        self.history = {}
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass
        self.name = name

    def load_model(self, weights: str = '') -> None:
        if weights and weights.split('.')[-1] == 'pt':
            self.model = torch.jit.load(weights)
        else:
            self.model = Net(device=self.device, num_classes=self.num_classes)
            self.model.zero_grad()

    def save_model(self, name, mode: str = 'last'):
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', name, f"{mode}.pt"))  # Save

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

    @staticmethod
    def get_confusion_matrix(y_true, y_pred, classes, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.savefig(save_path)
        plt.close()
        cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        sp = f"{save_path.split('.')[0]}_norm.{save_path.split('.')[-1]}"
        plt.savefig(sp)
        plt.close()

    def fill_history(self, epoch: int = 0, train_loss: float = 0, val_loss: float = 0, status: str = 'fill') -> None:
        if status == 'create':
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        elif status == 'fill':
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

    def train(self, dataset: VideoClass, epochs: int, weights: str = '',
              lr: float = 0.001) -> None:
        try:
            if weights:
                self.load_model(weights)

            stop = False
            i = 1
            name = f"{self.name}{i}"
            while not stop:
                if name in os.listdir(os.path.join(ROOT_DIR, 'video_class_train')):
                    i += 1
                    name = f"{self.name}{i}"
                else:
                    os.mkdir(os.path.join(ROOT_DIR, 'video_class_train', name))
                    stop = True

            st = time.time()
            logger.info("Training is started\n")

            num_classes = len(dataset.classes)
            num_train_batches = len(dataset.x_train)
            train_seq = list(np.arange(num_train_batches))
            num_val_batches = len(dataset.x_val)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.MSELoss()
            best_loss = 10000.

            logger_batch_markers = []
            for i in range(10):
                logger_batch_markers.append(int(num_train_batches * (i + 1) / 10))

            logger.info(f"training parameters:\n"
                        f"- name: {self.name},\n"
                        f"- weights: {weights}\n"
                        f"- save path: {os.path.join(ROOT_DIR, 'video_class_train', name)}\n"
                        f"- optimizer: {optimizer.__dict__.get('_zero_grad_profile_name')}\n"
                        f"- optimizr params: {optimizer.state_dict().get('param_groups')[0]}\n"
                        f"\n- Model structure:\n"
                        f"{inspect.getsource(self.model.__init__)}\n"
                        f"{inspect.getsource(self.model.forward)}\n")

            train_loss_hist, val_loss_hist = [], []
            self.fill_history(status='create')
            for epoch in range(epochs):
                st_ep = time.time()
                shuffle(train_seq)
                train_loss = 0.
                y_true, y_pred = [], []
                for batch in range(num_train_batches):
                    x_train = self.get_x_batch(video_path=dataset.x_train[train_seq[batch]], frame_size=self.frame_size)
                    y_train = self.get_y_batch(label=dataset.y_train[train_seq[batch]], num_labels=num_classes)
                    y_true.append(dataset.classes[dataset.y_train[train_seq[batch]]])
                    optimizer.zero_grad()
                    output = self.model(x_train)
                    y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                    loss = criterion(output, y_train)
                    train_loss += loss.cpu().detach().numpy()
                    loss.backward()
                    optimizer.step()
                    if batch + 1 in logger_batch_markers:
                        logger.info(f"  -- Epoch {epoch + 1}, batch {batch + 1} / {num_train_batches}, "
                                    f"train_loss= {round(train_loss / (batch + 1), 4)}, "
                                    f"average batch time = {round((time.time() - st_ep) * 1000 / (batch + 1), 1)} ms, "
                                    f"time passed = {time_converter(int(time.time() - st))}")

                save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Ep{epoch + 1}_Train_Confusion Matrix.jpg')
                self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm)
                train_loss_hist.append(round(train_loss / num_train_batches, 4))

                val_loss = 0
                y_true, y_pred = [], []
                for val_batch in range(num_val_batches):
                    x_val = self.get_x_batch(video_path=dataset.x_val[val_batch], frame_size=self.frame_size)
                    y_val = self.get_y_batch(label=dataset.y_val[val_batch], num_labels=num_classes)
                    y_true.append(dataset.classes[dataset.y_val[val_batch]])
                    output = self.model(x_val)
                    y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                    loss = criterion(output, y_val)
                    val_loss += loss.cpu().detach().numpy()

                save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Ep{epoch + 1}_Val_Confusion Matrix.jpg')
                self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm)
                val_loss_hist.append(round(val_loss / num_val_batches, 4))

                plot_and_save_gragh(train_loss_hist, 'epochs', 'Loss', 'Train loss',
                                    os.path.join(ROOT_DIR, 'video_class_train', name))
                plot_and_save_gragh(val_loss_hist, 'epochs', 'Loss', 'Val loss',
                                    os.path.join(ROOT_DIR, 'video_class_train', name))
                self.fill_history(
                    epoch=epoch,
                    train_loss=round(train_loss / num_train_batches, 4),
                    val_loss=round(val_loss / num_val_batches, 4),
                )
                save_dict_to_table_txt(self.history, os.path.join(ROOT_DIR, 'video_class_train', name, 'train_history.txt'))

                self.save_model(name=name, mode='last')
                if val_loss / num_val_batches <= best_loss:
                    self.save_model(name=name, mode='best')
                    best_loss = val_loss / num_val_batches
                    logger.info('\nBest weights were saved')

                logger.info(f"\nEpoch {epoch + 1}, train_loss= {round(train_loss / num_train_batches, 4)}, "
                            f"val_loss = {round(val_loss / num_val_batches, 4)}, "
                            f"epoch time = {time_converter(int(time.time() - st_ep))}\n")

            logger.info(f"Training is finished, "
                        f"train time = {time_converter(int(time.time() - st))}\n")
        except Exception as e:
            logger.error(f"Error training: \n{e}")

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
    device = 'cuda:0'
    # device = 'cpu'
    net = Net(device=device, num_classes=5)
    net.zero_grad()
    # print(net)
    params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())

    cuda0 = torch.device(device)
    arr = torch.from_numpy(arr)
    arr = arr.permute(0, 4, 1, 2, 3)
    arr = F.interpolate(arr, size=(arr.size()[2], 256, 256))
    logger.info(f"-- Input size: {arr.size()}\n")
    arr = arr.to(cuda0, dtype=torch.float)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    output = net(arr)
    logger.info(f"-- Output tensor: {output}")
    logger.info(f"-- Pretrain process time = {round(time.time() - st, 2)} sec\n")

    for i in range(5):
        st = time.time()
        optimizer.zero_grad()
        output = net(arr)
        target = torch.tensor(lbl, dtype=torch.float, device=cuda0)
        target = target.view(1, -1)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        # print(f" --- weights before {list(net.parameters())[0][0][0][0][0]}")
        loss.backward()
        # print(f" --- weights after {list(net.parameters())[0][0][0][0][0]}")
        optimizer.step()
        # print(f" --- weights after optim {list(net.parameters())[0][0][0][0][0]}")
        logger.info(f"-- loss (epoch {i + 1})={round(float(loss.cpu().detach().numpy()), 4)}")
        # print(f"-- Output size: {output.cpu().size()}")
        output = net(arr)
        logger.info(f"-- Output tensor (epoch {i + 1}): {output}")
        logger.info(f"-- Predict time (epoch {i + 1}) = {round(time.time() - st, 2)} sec\n")

    # model_scripted = torch.jit.script(net)  # Export to TorchScript
    # model_scripted.save('model_scripted.pt')  # Save
    #
    # model = torch.jit.load('model_scripted.pt')
    # out2 = model(arr)
    # lbl = np.argmax(out2.detach().numpy(), axis=-1) if device=='cpu' else np.argmax(out2.cpu().detach().numpy(), axis=-1)
    # print(f"-- Output tensor for loaded model: {out2}, label={lbl[0]}", type(lbl))
