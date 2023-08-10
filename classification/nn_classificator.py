import inspect
import os.path
from collections import Counter
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_process.dataset_processing import DatasetProcessing, VideoClass
import time
from utils import *
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class Net(nn.Module):
    def __init__(self, device='cpu', num_classes: int = 5, input_size=(12, 256, 256, 3),
                 frame_size=(128, 128), concat_axis: int = 2):
        super(Net, self).__init__()
        self.input_size = input_size
        self.frame_size = frame_size
        self.concat_axis = concat_axis
        self.conv3d_1 = nn.Conv3d(
            in_channels=input_size[-1], out_channels=32, kernel_size=3, padding='same', device=device)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding='same', device=device)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=device)
        self.conv3d_4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding='same', device=device)
        self.dense_3d = nn.Linear(in_features=256 * int(input_size[1] / 16) * int(input_size[2] / 16) * input_size[0],
                                  out_features=256, device=device)
        self.dense_3d_3 = nn.Linear(in_features=256, out_features=num_classes, device=device)
        self.post = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_1(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_2(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_3(x))), (1, 2, 2))
        x = F.max_pool3d(F.relu(F.normalize(self.conv3d_4(x))), (1, 2, 2))
        x = x.reshape(x.size(0), -1)
        x = F.normalize(self.dense_3d(x))
        x = self.post(F.normalize(self.dense_3d_3(x)))
        return x


class VideoClassifier:

    def __init__(
            self, num_classes: int = 5, name: str = 'model', weights: str = None,
            device: str = 'cuda:0', input_size: tuple = (2, 256, 128, 1), frame_size: tuple[int, int] = (128, 128),
            concat_axis: int = 2):
        self.num_classes = num_classes
        self.device = device
        self.input_size = input_size
        self.frame_size = frame_size
        self.concat_axis = concat_axis
        self.torch_device = torch.device(device)
        self.weights = weights
        self.model = self.load_model(weights)
        self.history = {}
        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass
        self.name = name

    def load_model(self, weights: str = '') -> nn.Module:
        if weights and weights.split('.')[-1] == 'pt':
            self.model = torch.jit.load(weights)
            self.input_size = self.model.input_size
            try:
                self.model.cuda(torch.device('cuda:0'))
            except:
                pass
        else:
            self.model = Net(device=self.device, num_classes=self.num_classes, input_size=self.input_size,
                             frame_size=self.frame_size, concat_axis=self.concat_axis)
        return self.model

    def save_model(self, name, mode: str = 'last') -> None:
        model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        model_scripted.save(os.path.join(ROOT_DIR, 'video_class_train', name, f"{mode}.pt"))  # Save

    @staticmethod
    def save_dataset(dataset: VideoClass, save_folder: str):
        save_data(data=dataset.dataset, folder_path=save_folder, filename='dataset')
        save_data(data=dataset.params, folder_path=save_folder, filename='dataset_params')

    def get_x_batch(self, x_train: list, num_frames: int = None, concat_axis: int = None) -> torch.Tensor:
        if num_frames and 3 < num_frames:
            x1, x2 = [], []
            for batch in x_train:
                b1, b2 = batch[0], batch[1]
                # seq_1, seq_2 = [b1[0]], [b2[0]]
                sequence = list(range(len(b1)))
                idx = VideoClassifier.resize_list(sequence, num_frames)
                # seq_1.extend(b1[idx])
                # seq_1.append(b1[-1])
                # seq_2.extend(b2[idx])
                # seq_2.append(b2[-1])
                b1, b2 = b1[idx], b2[idx]
                x1.append(b1)
                x2.append(b2)
            if concat_axis in [1, 2, 3, -1]:
                # if concat_axis in [1, 2]:
                x_train = np.concatenate([x1, x2], axis=concat_axis)
                # target_shape = [len(x_train), x_train[0][0].shape]
                # target_shape[concat_axis] = target_shape[concat_axis] * 2
                # target_shape[-1] = num_frames
                # x_tr = np.zeros(target_shape)
                # for i in range(len(x1)):
                #     for j in range(len(x1[0])):
                #         x_tr[i, :, :, j:j + 1] = np.concatenate([x1[i], x2[i]], axis=concat_axis - 1)
                #     # x_train = np.concatenate([x, x2[0]], axis=concat_axis - 1)
                # x_train = np.array(x_train)
            else:
                x_train = np.concatenate([x1, x2], axis=1)
                print("Concat_axis is our of range. Choose from None, 0, 1, 2 or -1. "
                      "Used default value concat_axis=None")
        x_train = torch.from_numpy(np.array(x_train))
        if 'cuda' in self.device:
            return x_train.to(self.torch_device, dtype=torch.float)
        else:
            return x_train.type(torch.FloatTensor)

    def numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        array = torch.from_numpy(array)
        if 'cuda' in self.device:
            return array.to(self.torch_device, dtype=torch.float)
        else:
            return array

    def get_y_batch(self, label: list[int, ...], num_labels: int) -> torch.Tensor:
        lbl = DatasetProcessing.ohe_from_list(label, num_labels)
        if 'cuda' in self.device:
            return torch.tensor(lbl, dtype=torch.float, device=self.torch_device)
        else:
            return torch.tensor(lbl, dtype=torch.float)

    @staticmethod
    def get_confusion_matrix(y_true: list, y_pred: list, classes: list, save_path: str = '',
                             get_percent: bool = False) -> np.ndarray:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        if save_path:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot()
            plt.savefig(save_path)
            plt.close()

        if get_percent:
            cm_percent = np.zeros_like(cm).astype('float')
            for i in range(len(cm)):
                total = np.sum(cm[i])
                for j in range(len(cm[i])):
                    cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
            if save_path:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=classes)
                disp.plot()
                plt.savefig(f"{save_path[:-4]}_%.jpg")
                plt.close()
        return cm

    def fill_history(self, epoch: int = 0, train_loss: float = 0, val_loss: float = 0,
                     train_accuracy: float = 0, val_accuracy: float = 0, status: str = 'fill') -> None:
        if status == 'create':
            self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        elif status == 'fill':
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)

    @staticmethod
    def accuracy(cm_data: np.ndarray) -> float:
        acc = 0
        for i in range(len(cm_data)):
            acc += cm_data[i][i] / np.sum(cm_data[i])
        acc /= len(cm_data)
        return round(acc, 4)

    @staticmethod
    def resize_list(sequence: list, length: int) -> list:
        if len(sequence) >= length:
            idx = list(range(len(sequence)))
            x2 = sorted(np.random.choice(idx, size=length, replace=False).tolist())
            y = [sequence[i] for i in x2]
        else:
            idx = list(range(len(sequence)))
            add = length - len(idx)
            idx.extend(np.random.choice(idx[1:-1], add))
            idx = sorted(idx)
            y = [sequence[i] for i in idx]
        return y

    # @staticmethod
    def track_to_array(self, tracks: list[list, list], frame_size: tuple = (128, 128), num_frames: int = 16,
                       concat_axis: int = 2) -> np.ndarray:
        """
        tracks: a list of lists [camera_1, camera_2] where camera_X = [[frame_id_1, frame_id_2, ...],[box_1, box_2, ...]]
        """
        tr1 = tracks[0]
        tr2 = tracks[1]
        if not tr1:
            tr1 = [[], []]
            min_fr = min(tr2[0])
            max_fr = max(tr2[0])
        elif not tr2:
            tr2 = [[], []]
            min_fr = min(tr1[0])
            max_fr = max(tr1[0])
        else:
            min_fr = min([min(tr1[0]), min(tr2[0])])
            max_fr = max([max(tr1[0]), max(tr2[0])])

        sequence = list(range(min_fr, max_fr + 1))
        seq_frame_1, seq_frame_2 = [], []
        for fr in sequence:
            if fr in tr1[0]:
                seq_frame_1.append(tr1[1][tr1[0].index(fr)])
            else:
                seq_frame_1.append([])

            if fr in tr2[0]:
                seq_frame_2.append(tr2[1][tr2[0].index(fr)])
            else:
                seq_frame_2.append([])

        track = {'cl': {'type': {"1": seq_frame_1, "2": seq_frame_2}}}
        dataset = self.create_box_video_dataset(dataset=track, split=1., test_split=0, frame_size=frame_size)
        batch = self.get_x_batch(x_train=dataset.x_train[0: 1], num_frames=num_frames, concat_axis=concat_axis)
        return batch.cpu().numpy()

    @staticmethod
    def create_box_video_dataset(
            dataset: dict, split: float, dataset_path: str = '', test_split: float = 0.05,
            frame_size: tuple = (128, 128)
    ) -> VideoClass:
        if dataset_path:
            dataset = load_data(dataset_path)
        vc = VideoClass()
        vc.dataset = dataset
        vc.params['split'] = split
        vc.params['test_split'] = test_split
        vc.params['box_path'] = dataset_path
        vc.classes = sorted(list(dataset.keys()))

        data = []
        for class_ in dataset.keys():
            cl_id = vc.classes.index(class_)
            for vid in dataset[class_].keys():
                seq_frame_1, seq_frame_2 = [], []
                cameras = list(dataset[class_][vid].keys())
                if dataset[class_][vid] != {camera: [] for camera in cameras} and len(
                        dataset[class_][vid][cameras[0]]) > 2:
                    sequence = list(range(len(dataset[class_][vid][cameras[0]]))) if len(
                        dataset[class_][vid][cameras[0]]) \
                        else list(range(len(dataset[class_][vid][cameras[1]])))
                    for fr in range(len(sequence)):
                        fr1 = np.zeros((frame_size[0], frame_size[1], 1))
                        fr2 = np.zeros((frame_size[0], frame_size[1], 1))

                        if dataset[class_][vid][cameras[0]][fr]:
                            box1 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[0]][fr])]
                            fr1[box1[1]:box1[3], box1[0]:box1[2], :] = 1.
                        # fr1 = np.expand_dims(fr1, axis=-1)
                        seq_frame_1.append(fr1)

                        if dataset[class_][vid][cameras[1]][fr]:
                            box2 = [int(bb * frame_size[i % 2]) for i, bb in
                                    enumerate(dataset[class_][vid][cameras[1]][fr])]
                            fr2[box2[1]:box2[3], box2[0]:box2[2], :] = 1.
                        # fr2 = np.expand_dims(fr2, axis=-1)
                        seq_frame_2.append(fr2)

                    seq_frame_1 = np.array(seq_frame_1)
                    seq_frame_2 = np.array(seq_frame_2)
                    batch = [[seq_frame_1, seq_frame_2], cl_id, (class_, vid)]
                    data.append(batch)

        random.shuffle(data)
        x, y, ref = list(zip(*data))
        y = np.array(y)

        vc.x_train = x[:int(vc.params['split'] * len(x))]
        vc.y_train = y[:int(vc.params['split'] * len(x))]
        vc.params['train_ref'] = ref[:int(vc.params['split'] * len(x))]
        vc.params['train_stat'] = dict(Counter(vc.y_train))

        vc.x_val = x[int(vc.params['split'] * len(x)):int((1 - test_split) * len(x))]
        vc.y_val = y[int(vc.params['split'] * len(x)):int((1 - test_split) * len(x))]
        vc.params['val_ref'] = ref[int(vc.params['split'] * len(x)):int((1 - test_split) * len(x))]
        vc.params['val_stat'] = dict(Counter(vc.y_val))

        vc.x_test = x[int(int((1 - test_split) * len(x))):]
        vc.y_test = y[int(int((1 - test_split) * len(x))):]
        vc.params['test_ref'] = ref[int(int((1 - test_split) * len(x))):]
        vc.params['test_stat'] = dict(Counter(vc.y_test))
        return vc

    def train(self, dataset: VideoClass, epochs: int, batch_size: int = 1, weights: str = '',
              lr: float = 0.005, num_frames: int = 6, concat_axis: int = 2, save_dataset: bool = False,
              load_dataset_path: str = '') -> None:
        if weights:
            self.load_model(weights)
        stop = False
        i = 1
        name = f"{self.name}_{i}"

        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass

        try:
            os.mkdir(os.path.join(ROOT_DIR, 'video_class_train'))
        except:
            pass

        while not stop:
            if name in os.listdir(os.path.join(ROOT_DIR, 'video_class_train')):
                i += 1
                name = f"{self.name}_{i}"
            else:
                os.mkdir(os.path.join(ROOT_DIR, 'video_class_train', name))
                stop = True
        print(os.path.join(ROOT_DIR, 'video_class_train', name))
        # if load_dataset_path:
        #     dataset = self.load_dataset(load_dataset_path)

        if save_dataset:
            self.save_dataset(dataset, os.path.join(ROOT_DIR, 'video_class_train', name))
        st = time.time()
        print("Training is started\n")
        num_classes = len(dataset.classes)
        num_train_batches = int(len(dataset.x_train) / batch_size)
        train_seq = list(np.arange(len(dataset.x_train)))
        num_val_batches = len(dataset.x_val)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss, best_acc = 10000., 0.

        logger_batch_markers = []
        for i in range(10):
            logger_batch_markers.append(int(num_train_batches * (i + 1) / 10))

        lr_steps = []
        for i in range(2):
            lr_steps.append(int(epochs * (i + 1) / 2))

        txt = f"training parameters:\n" \
              f"- name: {name},\n" \
              f"- save path: {os.path.join(ROOT_DIR, '../video_class_train', name)}\n" \
              f"- optimizer: {optimizer.__dict__.get('_zero_grad_profile_name')}\n" \
              f"- optimizr params: {optimizer.state_dict().get('param_groups')[0]}\n" \
              f"\n- Model structure:\n" \
              f"{inspect.getsource(self.model.__init__) if not weights and not self.weights else ''}\n" \
              f"{inspect.getsource(self.model.forward) if not weights and not self.weights else ''}\n"
        save_txt(txt, os.path.join(ROOT_DIR, 'video_class_train', name, f"model_info.txt"))
        print(txt)

        train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []
        self.fill_history(status='create')
        for epoch in range(epochs):
            st_ep = time.time()
            random.shuffle(train_seq)
            train_loss, train_acc = 0., 0.
            y_true, y_pred = [], []
            for batch in range(num_train_batches):
                x_batch = [dataset.x_train[i] for i in train_seq[batch * batch_size:(batch + 1) * batch_size]]
                x_train = self.get_x_batch(x_train=x_batch, num_frames=num_frames, concat_axis=concat_axis)
                y_batch = dataset.y_train[train_seq[batch * batch_size:(batch + 1) * batch_size]]
                y_train = self.get_y_batch(label=y_batch, num_labels=num_classes)
                y_true.extend([dataset.classes[i] for i in y_batch])
                output = self.model(x_train)
                y_pred.extend([dataset.classes[i] for i in np.argmax(output.cpu().detach().numpy(), axis=-1)])
                loss = criterion(output, y_train)
                train_loss += loss.cpu().detach().numpy()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if batch + 1 in logger_batch_markers:
                    save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, 'Train_Confusion Matrix.jpg')
                    cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
                    train_acc = self.accuracy(cm)
                    print(
                        f"  -- Epoch {epoch + 1}, batch {batch + 1} / {num_train_batches}, "
                        f"train_loss= {round(train_loss / (batch + 1), 4)}, "
                        f"train_accuracy= {train_acc}, "
                        f"average batch time = {round((time.time() - st_ep) * 1000 / (batch + 1), 1)} ms, "
                        f"time passed = {time_converter(int(time.time() - st))}"
                    )

                # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2

            if epoch + 1 in lr_steps and epoch + 1 != epochs:
                print(
                    f"  -- Epoch {epoch + 1}, lr was reduced from  {optimizer.param_groups[0]['lr']} "
                    f"to {optimizer.param_groups[0]['lr'] / 2}"
                )
            save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, 'Train_Confusion Matrix.jpg')
            cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
            train_acc = self.accuracy(cm)
            train_loss_hist.append(round(train_loss / num_train_batches, 4))
            train_acc_hist.append(train_acc)

            val_loss = 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for val_batch in range(num_val_batches):
                    x_val = self.get_x_batch(x_train=dataset.x_val[val_batch: val_batch + 1],
                                             num_frames=num_frames, concat_axis=concat_axis)
                    y_val = self.get_y_batch(label=dataset.y_val[val_batch: val_batch + 1],
                                             num_labels=num_classes)
                    y_true.append(dataset.classes[dataset.y_val[val_batch]])
                    output = self.model(x_val)
                    y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])
                    loss = criterion(output, y_val)
                    val_loss += loss.cpu().detach().numpy()

            save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, 'Val_Confusion Matrix.jpg')
            cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
            val_acc = self.accuracy(cm)
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
            self.fill_history(
                epoch=epoch,
                train_loss=round(train_loss / num_train_batches, 4),
                val_loss=round(val_loss / num_val_batches, 4),
                train_accuracy=train_acc,
                val_accuracy=val_acc,
            )
            save_dict_to_table_txt(
                self.history, os.path.join(ROOT_DIR, 'video_class_train', name, 'train_history.txt'))

            self.save_model(name=name, mode='last')
            if val_acc >= best_acc:
                self.save_model(name=name, mode='best')
                best_acc = val_acc
                print('\nBest weights were saved')

            print(f"\nEpoch {epoch + 1}, train_loss= {round(train_loss / num_train_batches, 4)}, "
                  f"val_loss = {round(val_loss / num_val_batches, 4)}, "
                  f"train_accuracy= {train_acc}, val_accuracy = {val_acc}, "
                  f"epoch time = {time_converter(int(time.time() - st_ep))}\n")

        y_true, y_pred = [], []
        num_test_batches = len(dataset.x_test)
        with torch.no_grad():
            for test_batch in range(num_test_batches):
                x_test = self.get_x_batch(x_train=dataset.x_test[test_batch: test_batch + 1],
                                          num_frames=num_frames, concat_axis=concat_axis)
                y_true.append(dataset.classes[dataset.y_test[test_batch]])
                output = self.model(x_test)
                y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])

        save_cm = os.path.join(ROOT_DIR, 'video_class_train', name, 'Test_Confusion Matrix.jpg')
        cm = self.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
        test_acc = self.accuracy(cm)
        print(f'Training is finished, test_acc = {test_acc}, '
              f'train time = {time_converter(int(time.time() - st))}\n')

    def predict(self, array, model: nn.Module, classes: list = None) -> list:
        if classes is None:
            classes = []

        array = self.numpy_to_torch(array)
        with torch.no_grad():
            output = model(array)
        output = output.cpu().detach().numpy() if self.device != 'cpu' else output.detach().numpy()
        if classes:
            return [classes[i] for i in list(np.argmax(output, axis=-1))]
        return list(np.argmax(output, axis=-1))

    @staticmethod
    def evaluate_on_test_data(test_dataset: str, weights: str = '') -> np.ndarray:
        weights_folder = weights[:-len(weights.split('/')[-1])]
        save_cm = f"{weights_folder}Test_Confusion Matrix.jpg"
        dataset = load_data(test_dataset)
        classes = sorted(list(dataset.keys()))
        vc = VideoClassifier(num_classes=len(classes), weights=weights)
        dataset = VideoClassifier.create_box_video_dataset(
            dataset=dataset,
            split=1.0,
            frame_size=vc.model.frame_size,
        )
        num_test_batches = len(dataset.x_train)
        num_frames = vc.model.input_size[0] if vc.model.concat_axis != 1 else int(vc.model.input_size[0] / 2)

        y_true, y_pred = [], []
        with torch.no_grad():
            for test_batch in range(num_test_batches):
                x_test = vc.get_x_batch(
                    x_train=dataset.x_train[test_batch: test_batch + 1], num_frames=num_frames,
                    concat_axis=vc.model.concat_axis)
                y_true.append(dataset.classes[dataset.y_train[test_batch]])
                output = vc.model(x_test)
                # print(y_true[-1], output, classes)
                y_pred.append(dataset.classes[np.argmax(output.cpu().detach().numpy(), axis=-1)[0]])

        cm = vc.get_confusion_matrix(y_true, y_pred, dataset.classes, save_cm, get_percent=True)
        return cm


if __name__ == "__main__":
    pass
