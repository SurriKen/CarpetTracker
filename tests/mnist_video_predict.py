from dataclasses import dataclass
from collections import Counter
import numpy as np
import time
import inspect
import random
from utils import logger, time_converter, plot_and_save_gragh, save_dict_to_table_txt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import os
from parameters import ROOT_DIR

x_train = np.load(os.path.join(ROOT_DIR, 'tests/x_train.npy'))
y_train = np.load(os.path.join(ROOT_DIR, 'tests/y_train.npy'))
print(x_train.shape, y_train.shape)
split = 0.8


@dataclass
class Dataset:

    def __init__(self):
        self.x_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.x_val: np.ndarray = np.array([])
        self.y_val: np.ndarray = np.array([])
        self.classes = []
        self.train_stat = []
        self.val_stat = []
        self.params = {}


dataset = Dataset()
dataset.classes = ['upgrade', 'downgrade', 'random', 'valley', 'mountain', 'odd', 'non-odd']
dataset.params['split'] = split

idxs = {i: [] for i in range(10)}
for i, id in enumerate(y_train.tolist()):
    idxs[id].append(i)
count = {k: len(v) for k, v in idxs.items()}

data = []
for i in range(5000):
    array = []
    lbl = None
    if i in range(0, 5000, len(dataset.classes)):
        lbl = 0
        for key in sorted(list(idxs.keys())):
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            idxs[key].pop(0)
    elif i in range(1, 5000, len(dataset.classes)):
        lbl = 1
        for key in sorted(list(idxs.keys()), reverse=True):
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            idxs[key].pop(0)
    elif i in range(2, 5000, len(dataset.classes)):
        lbl = 2
        keys = list(idxs.keys())
        random.shuffle(keys)
        for key in keys:
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            idxs[key].pop(0)
    elif i in range(3, 5000, len(dataset.classes)):
        lbl = 3
        for key in [9, 7, 5, 3, 1, 0, 2, 4, 6, 8]:
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            idxs[key].pop(0)
    elif i in range(4, 5000, len(dataset.classes)):
        lbl = 4
        for key in [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]:
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            idxs[key].pop(0)
    elif i in range(5, 5000, len(dataset.classes)):
        lbl = 5
        for key in [1, 3, 5, 7, 9]:
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            array.append(ar)
            idxs[key].pop(0)
        random.shuffle(array)
    elif i in range(6, 5000, len(dataset.classes)):
        lbl = 6
        for key in [0, 2, 4, 6, 8]:
            ar = x_train[idxs[key][0]]
            ar = np.expand_dims(ar, axis=-1)
            ar = np.concatenate([ar, ar, ar], axis=-1)
            array.append(ar)
            array.append(ar)
            idxs[key].pop(0)
        random.shuffle(array)
    data.append((np.array(array), lbl))

random.shuffle(data)
tr = data[:int(len(data) * split)]
val = data[int(len(data) * split):]
dataset.x_train, dataset.y_train = list(zip(*tr))
dataset.x_val, dataset.y_val = list(zip(*val))

ytr = dict(Counter(dataset.y_train))
stat_ytr = {}
for k, v in ytr.items():
    stat_ytr[dataset.classes[k]] = v
dataset.train_stat = stat_ytr

yv = dict(Counter(dataset.y_val))
stat_yv = {}
for k, v in yv.items():
    stat_yv[dataset.classes[k]] = v
dataset.val_stat = stat_yv

dataset.x_train = np.array(dataset.x_train)
dataset.x_val = np.array(dataset.x_val)
dataset.y_train = np.array(dataset.y_train)
dataset.y_val = np.array(dataset.y_val)
print('dataset.train_stat', dataset.train_stat)
print('dataset.val_stat', dataset.val_stat)
print('dataset.x_train', dataset.x_train.shape)
print('dataset.y_train', dataset.y_train.shape)

device = 'cuda:0'

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, device='cuda:0', num_classes: int = 5):
        super(Net, self).__init__()
        # print('num_classes', num_classes)
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=5, padding='same', device=device)
        # print(3)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, padding='same', device=device)
        # print(4)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=device)
        self.dense = nn.Linear(128 * 7 * 7, num_classes, device=device)
        # print('device', device)
        self.post = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(device=device)

    def forward(self, x):
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), 2)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), 2)
        x = F.relu(F.normalize(self.conv3d_3(x)))
        # print(x.size())
        # x = x.reshape(x.size(0), -1)
        x = x.reshape(x.size(0), x.size(2), -1)
        x = torch.mean(x, dim=1)
        # print(x.size())
        x = self.post(self.dense(x))
        return x


def get_x_batch(device, array: np.ndarray) -> torch.Tensor:
    # array = np.expand_dims(array, axis=0) / 255
    # print('get_x_batch', array.shape)
    array = array / 255
    array = torch.from_numpy(array)
    array = array.permute(0, 4, 1, 2, 3)
    if 'cuda' in device:
        return array.to(torch.device(device), dtype=torch.float)
    else:
        return array


from dataset_processing import DatasetProcessing


def get_y_batch(device, label: list, num_labels: int) -> torch.Tensor:
    res = []
    for lbl in label:
        lbl = DatasetProcessing.ohe_from_list([lbl], num_labels)
        res.extend(lbl)

    if 'cuda' in device:
        res = torch.tensor(res, dtype=torch.float, device=torch.device(device))
    else:
        res = torch.tensor(res, dtype=torch.float)
    return res


def fill_history(history=None, epoch: int = 0, train_loss: float = 0, val_loss: float = 0,
                 train_accuracy: float = 0, val_accuracy: float = 0, status: str = 'fill') -> dict:
    if history is None:
        history = {}
    if status == 'create':
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    elif status == 'fill':
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
    return history


def get_confusion_matrix(y_true, y_pred, classes, save_path: str = ''):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if save_path:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.savefig(save_path)
        plt.close()
    return cm


def save_model(model: Net, name, mode: str = 'last'):
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', name, f"{mode}.pt"))  # Save


def accuracy(cm_data: np.ndarray) -> float:
    # x = np.array([[0, 473], [0, 527]])
    acc = 0
    for i in range(len(cm_data)):
        # print(cm_data[i][i] / np.sum(cm_data[i]))
        acc += cm_data[i][i] / np.sum(cm_data[i])
    acc /= len(cm_data)
    return round(acc, 4)


def train(model: Net, dataset: Dataset, epochs: int, batch_size: int = 1, lr: float = 0.005) -> None:
    stop = False
    i = 1
    name = f"mnist3D_test_{i}"
    try:
        os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
    except:
        pass

    while not stop:
        if name in os.listdir(os.path.join(ROOT_DIR, 'video_class_train')):
            i += 1
            name = f"mnist3D_test_{i}"
        else:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train', name))
            stop = True

    st = time.time()
    logger.info("Training is started\n")
    # dataset.classes = ['upgrade', 'downgrade']
    num_classes = len(dataset.classes)
    num_train_batches = int(len(dataset.x_train) / batch_size)
    train_seq = list(np.arange(len(dataset.x_train)))
    num_val_batches = len(dataset.x_val)
    print("num_train_batches", num_train_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer.zero_grad()
    # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    best_loss, best_acc = 10000., 0.

    logger_batch_markers = []
    for i in range(10):
        logger_batch_markers.append(int(num_train_batches * (i + 1) / 10))

    logger.info(f"training parameters:\n"
                f"- name: {name},\n"
                f"- save path: {os.path.join(ROOT_DIR, 'video_class_train', name)}\n"
                f"- optimizer: {optimizer.__dict__.get('_zero_grad_profile_name')}\n"
                f"- optimizr params: {optimizer.state_dict().get('param_groups')[0]}\n"
                f"\n- Model structure:\n"
                f"{inspect.getsource(model.__init__)}\n"
                f"{inspect.getsource(model.forward)}\n")

    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []
    history = fill_history(status='create')
    for epoch in range(epochs):
        st_ep = time.time()
        random.shuffle(train_seq)
        # print("train_seq", dataset.x_train.shape, train_seq)
        train_loss, train_acc = 0., 0.
        y_true, y_pred = [], []
        for batch in range(num_train_batches):
            x_batch = dataset.x_train[train_seq[batch * batch_size:(batch + 1) * batch_size]]
            # print("x_batch", f"{batch}/{num_train_batches}", x_batch.shape, train_seq[batch * batch_size:(batch + 1) * batch_size], batch * batch_size, (batch + 1) * batch_size)
            x_train = get_x_batch(device=device, array=np.array(x_batch))
            y_batch = dataset.y_train[train_seq[batch * batch_size:(batch + 1) * batch_size]]
            y_train = get_y_batch(device=device, label=y_batch, num_labels=num_classes)
            # print(x_train.size, y_train.shape)
            y_true.extend([dataset.classes[i] for i in y_batch])
            output = model(x_train)
            y_pred.extend([dataset.classes[i] for i in np.argmax(output.cpu().detach().numpy(), axis=-1)])
            loss = criterion(output, y_train)
            train_loss += loss.cpu().detach().numpy()

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if batch + 1 in logger_batch_markers:
                cm = get_confusion_matrix(y_true, y_pred, dataset.classes)
                train_acc = accuracy(cm)
                logger.info(
                    f"  -- Epoch {epoch + 1}, batch {batch + 1} / {num_train_batches}, "
                    f"train_loss= {round(train_loss / (batch + 1), 4)}, "
                    f"train_accuracy= {train_acc}, "
                    f"average batch time = {round((time.time() - st_ep) * 1000 / (batch + 1), 1)} ms, "
                    f"time passed = {time_converter(int(time.time() - st))}"
                )

        save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Ep{epoch + 1}_Train_Confusion Matrix.jpg')
        cm = get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm)
        train_acc = accuracy(cm)
        train_loss_hist.append(round(train_loss / num_train_batches, 4))
        train_acc_hist.append(train_acc)

        val_loss = 0
        y_true, y_pred = [], []
        # print("num_val_batches", num_val_batches, dataset.x_val.shape)
        with torch.no_grad():
            for val_batch in range(num_val_batches):
                x_val = get_x_batch(device=device, array=dataset.x_val[val_batch: val_batch + 1])
                # print("val_batch", f"{val_batch}/{num_val_batches}", x_batch.shape,
                #       train_seq[batch * batch_size:(batch + 1) * batch_size], batch * batch_size,
                #       (batch + 1) * batch_size)
                # print("val_batch", dataset.y_val[val_batch: val_batch+1].tolist())
                y_val = get_y_batch(device=device, label=dataset.y_val[val_batch: val_batch + 1].tolist(),
                                    num_labels=num_classes)
                y_true.append(dataset.classes[dataset.y_val[val_batch]])
                output = model(x_val)
                y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                # y_pred.extend([dataset.classes[i] for i in np.argmax(output.cpu().detach().numpy(), axis=-1)])
                loss = criterion(output, y_val)
                val_loss += loss.cpu().detach().numpy()

        save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, f'Ep{epoch + 1}_Val_Confusion Matrix.jpg')
        cm = get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm)
        val_acc = accuracy(cm)
        val_loss_hist.append(round(val_loss / num_val_batches, 4))
        val_acc_hist.append(val_acc)

        plot_and_save_gragh(train_loss_hist, 'epochs', 'Loss', 'Train loss',
                            os.path.join(ROOT_DIR, 'video_class_train', name))
        plot_and_save_gragh(val_loss_hist, 'epochs', 'Loss', 'Val loss',
                            os.path.join(ROOT_DIR, 'video_class_train', name))
        plot_and_save_gragh(train_acc_hist, 'epochs', 'Accuracy', 'Train Accuracy',
                            os.path.join(ROOT_DIR, 'video_class_train', name))
        plot_and_save_gragh(val_acc_hist, 'epochs', 'Accuracy', 'Val Accuracy',
                            os.path.join(ROOT_DIR, 'video_class_train', name))
        history = fill_history(
            history=history,
            epoch=epoch,
            train_loss=round(train_loss / num_train_batches, 4),
            val_loss=round(val_loss / num_val_batches, 4),
            train_accuracy=train_acc,
            val_accuracy=val_acc,
        )
        save_dict_to_table_txt(
            history, os.path.join(ROOT_DIR, 'video_class_train', name, 'train_history.txt'))

        save_model(model=model, name=name, mode='last')
        if val_acc >= best_acc:
            save_model(model=model, name=name, mode='best')
            best_acc = val_acc
            logger.info('\nBest weights were saved')

        logger.info(f"\nEpoch {epoch + 1}, train_loss= {round(train_loss / num_train_batches, 4)}, "
                    f"val_loss = {round(val_loss / num_val_batches, 4)}, "
                    f"train_accuracy= {train_acc}, val_accuracy = {val_acc}, "
                    f"epoch time = {time_converter(int(time.time() - st_ep))}\n")

    logger.info(f"Training is finished, "
                f"train time = {time_converter(int(time.time() - st))}\n")


# print(2)
model = Net(device=device, num_classes=len(dataset.classes))
print("Training is started")
train(
    model=model,
    dataset=dataset,
    epochs=5,
    batch_size=16,
    lr=0.001
)
