import osimport randomimport reimport shutilimport timefrom collections import Counterfrom dataclasses import dataclassimport matplotlib.path as mpltPathimport cv2import numpy as npimport pandas as pdimport skvideo.ioimport torchimport torchvisionfrom PIL import Image, ImageDrawfrom moviepy.video.io.VideoFileClip import VideoFileClipfrom torchvision.utils import draw_bounding_boxesfrom torchvision.io import read_imagefrom parameters import ROOT_DIR, DATASET_DIRfrom utils import save_data, load_data, remove_empty_xml, read_xml, get_colors, logger, save_txt, load_txt, \    get_name_from_link@dataclassclass VideoClass:    def __init__(self):        self.x_train = []        self.y_train = []        self.x_val = []        self.y_val = []        self.x_test = []        self.y_test = []        self.dataset = {}        # self.classes = []        # self.train_stat = []        # self.val_stat = []        self.params = {}class DatasetProcessing:    def __init__(self):        pass    @staticmethod    def cut_video(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000) -> str:        """        Cut video in given time range.        Args:            video_path: path to video file            save_path: path to save folder            from_time: time to start cut in seconds            to_time: time to finish cut in seconds        Returns: path to saved video file        """        try:            os.mkdir(save_path)        except:            pass        video_capture = cv2.VideoCapture()        video_capture.open(video_path)        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))        duration = int(frame_count / fps)        saved_video_path = f"{save_path}/{video_path.split('/')[-1]}"        if to_time and to_time < duration:            clip = VideoFileClip(video_path).subclip(from_time, to_time)            clip.write_videofile(saved_video_path)        elif from_time:            clip = VideoFileClip(video_path).subclip(from_time, duration)            clip.write_videofile(saved_video_path)        else:            shutil.copy2(video_path, saved_video_path)        print("Video was cut and save")        return saved_video_path    @staticmethod    def draw_polygons(polygons: list, image: np.ndarray, outline=(0, 200, 0), width: int = 5) -> np.ndarray:        if type(polygons[0]) == list:            xy = []            for i in polygons:                xy.append(i[0])                xy.append(i[1])        else:            xy = polygons        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)        points = np.array(xy)        points = points.reshape((-1, 1, 2))        image = cv2.polylines(image, [points], True, outline, width)        return np.array(image)    @staticmethod    def point_in_polygon(point: list, polygon: list[list, ...]) -> bool:        path = mpltPath.Path(polygon)        return path.contains_points([point])[0]    @staticmethod    def synchronize_video(video_path: str, save_path: str = 'datasets', from_frame=None, to_frame=None):        video_capture = cv2.VideoCapture()        video_capture.open(video_path)        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))        vc = cv2.VideoCapture()        vc.open(video_path)        frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))        size = None        for i in range(frames):            ret, frame = video_capture.read()            size = (frame.shape[1], frame.shape[0])            break        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)        for i in range(frame_count):            if (i + 1) % 500 == 0:                print(f"{i + 1} frames are ready")            ret, frame = video_capture.read()            if from_frame and i >= from_frame:                out.write(frame)            if to_frame and i >= to_frame:                break        out.release()    @staticmethod    def video_class_dataset(video_links: list, save_folder: str, separate: bool = False):        # save_folder = os.path.join(ROOT_DIR, save_folder)        try:            os.mkdir(save_folder)        except:            shutil.rmtree(save_folder)            os.mkdir(save_folder)        count = 0        for i, vid in enumerate(video_links):            csv = vid[2]  # os.path.join(ROOT_DIR, vid[2])            obj = 0            data = pd.read_csv(csv)            try:                carpet_size = data['Размер']            except:                carpet_size = data['размер']            start_frame = data['Кадр начала']            end_frame = data['Кадр конца']            # print(len(carpet_size), len(start_frame), len(end_frame))            cls = carpet_size.unique()            for cl in cls:                cl = cl.replace('*', 'x')                if not os.path.isdir(os.path.join(save_folder, cl)):                    os.mkdir(os.path.join(save_folder, cl))                    if separate:                        os.mkdir(os.path.join(save_folder, cl, 'camera_1'))                        os.mkdir(os.path.join(save_folder, cl, 'camera_2'))            cam_1, cam_2 = vid[0], vid[1]  # os.path.join(ROOT_DIR, vid[0]), os.path.join(ROOT_DIR, vid[1])            print(f"\n{csv}\n")            vc1 = cv2.VideoCapture()            vc1.open(cam_1)            w1 = int(vc1.get(cv2.CAP_PROP_FRAME_WIDTH))            h1 = int(vc1.get(cv2.CAP_PROP_FRAME_HEIGHT))            frames1 = int(vc1.get(cv2.CAP_PROP_FRAME_COUNT))            vc2 = cv2.VideoCapture()            vc2.open(cam_2)            frames2 = int(vc2.get(cv2.CAP_PROP_FRAME_COUNT))            w2 = int(vc2.get(cv2.CAP_PROP_FRAME_WIDTH))            h2 = int(vc2.get(cv2.CAP_PROP_FRAME_HEIGHT))            w = min([w1, w2])            h = min([h1, h2])            for j in range(min([frames1, frames2])):                _, frame1 = vc1.read()                _, frame2 = vc2.read()                if j == start_frame[obj]:                    cs = carpet_size[obj].replace('*', 'x')                    if separate:                        out1 = cv2.VideoWriter(                            os.path.join(save_folder, cs, 'camera_1', f"{count}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),                            25, (w1, h1)                        )                        out2 = cv2.VideoWriter(                            os.path.join(save_folder, cs, 'camera_2', f"{count}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),                            25, (w2, h2)                        )                    else:                        out1 = cv2.VideoWriter(                            os.path.join(save_folder, cs, f"{count}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 25,                            (w, h * 2)                        )                if start_frame[obj] <= j <= end_frame[obj]:                    if separate:                        out1.write(frame1)                        out2.write(frame2)                    else:                        size1 = (frame1.shape[1], frame1.shape[0])                        if size1 != (w, h):                            frame1 = cv2.resize(frame1, (w, h))                        size2 = (frame2.shape[1], frame2.shape[0])                        if size2 != (w, h):                            frame2 = cv2.resize(frame2, (w, h))                        frame = np.concatenate((frame1, frame2), axis=0)                        out1.write(frame)                if j == end_frame[obj]:                    if separate:                        out1.release()                        logger.info(f"Video was saved to {os.path.join(save_folder, cs, 'camera_1', f'{count}.mp4')}")                        out2.release()                        logger.info(f"Video was saved to {os.path.join(save_folder, cs, 'camera_2', f'{count}.mp4')}\n")                    else:                        out1.release()                        logger.info(f"Video was saved to {os.path.join(save_folder, cs, f'{count}.mp4')}")                    obj += 1                    count += 1                if obj == len(carpet_size):                    break    @staticmethod    def change_fps(video_path: str, save_path: str, set_fps=25):        video_capture = cv2.VideoCapture()        video_capture.open(video_path)        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"        if fps == 25:            print(f"Video {video_path} already has fps = 25")        else:            # print(f"Video {video_path} has fps = {fps}. Setting fps to 25")            clip = VideoFileClip(video_path)            clip.write_videofile(save_path, fps=set_fps)    @staticmethod    def video2frames(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000, size=(),                     save_video=False):        video_name = get_name_from_link(video_path)        print(video_name)        video_capture = cv2.VideoCapture()        video_capture.open(video_path)        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))        print(f"fps = {fps}, frame_count = {frame_count}")        duration = int(frame_count / fps)        if to_time:            if to_time > int(duration):                to_time = int(duration)            to_path = f"{save_path}/{video_name}_{from_time}s-{to_time}s"        elif from_time:            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"        else:            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"        try:            os.mkdir(to_path)        except:            shutil.rmtree(to_path, ignore_errors=True)            os.mkdir(to_path)        os.mkdir(f"{to_path}/frames")        os.mkdir(f"{to_path}/video")        os.mkdir(f"{to_path}/xml_labels")        # if save_video:        # saved_video_path = DatasetProcessing.cut_video(        #     video_path=video_path,        #     save_path=f"{to_path}/video",        #     from_time=from_time,        #     to_time=to_time        # )        video_capture = cv2.VideoCapture()        video_capture.open(video_path)        fps = video_capture.get(cv2.CAP_PROP_FPS)        frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)        start = int(from_time * fps)        finish = int(to_time * fps)        print(f"Getting frames ({int(frames)} in total)...")        # size = ()        for i in range(0, finish):            if (i + 1) % 200 == 0:                print(f"{i + 1} frames are ready")            ret, frame = video_capture.read()            if i >= start:                if not size:                    size = (frame.shape[1], frame.shape[0])                frame = cv2.resize(frame, size)                cv2.imwrite(f"{to_path}/frames/%05d.png" % i, frame)        video_data = {            "fps": int(fps), "frames": int(frames), 'size': size        }        print(f"frames were got: fps - {int(fps)}, total frames - {int(frames)}, frame size - {size}")        save_data(video_data, to_path, 'data')    @staticmethod    def frames2video(frames_path: str, save_path: str, video_name: str,                     params: str, box_path: str = None, resize=False, box_type='xml'):        parameters = load_data(params)        image_list, box_list, lbl_list, color_list = [], [], [], []        with os.scandir(frames_path) as folder:            for f in folder:                image_list.append(f.name)        image_list = sorted(image_list)        if box_path:            remove_empty_xml(box_path)            with os.scandir(box_path) as folder:                for f in folder:                    box_list.append(f.name)            for box in box_list:                box_info = read_xml(f"{box_path}/{f'{box}'}")                if box_info["coords"][-1] not in lbl_list:                    lbl_list.append(box_info["coords"][-1])            lbl_list = sorted(lbl_list)            color_list = get_colors(lbl_list)        st = time.time()        count = 0        print(f"Start processing...\n")        out = cv2.VideoWriter(            f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])        for img in image_list:            if (count + 1) % int(len(image_list) * 0.1) == 0:                print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were processed...")            name = img.split(".")[0]            if box_path:                if box_type == 'xml':                    if resize:                        box_info = read_xml(                            xml_path=f"{box_path}/{f'{name}.xml'}",                            shrink=True,                            new_width=parameters['size'][0],                            new_height=parameters['size'][1]                        )                    else:                        box_info = read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")                    boxes, labels = [], []                    for b in box_info["coords"]:                        boxes.append(b[:-1])                        labels.append(b[-1])                    bbox = torch.tensor(boxes, dtype=torch.int)                    image = read_image(f"{frames_path}/{img}")                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)                    image = torchvision.transforms.ToPILImage()(image_true)                if box_type == 'txt':                    box_info = read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")                    boxes, labels = [], []                    for b in box_info["coords"]:                        boxes.append(b[:-1])                        labels.append(b[-1])                    bbox = torch.tensor(boxes, dtype=torch.int)                    image = read_image(f"{frames_path}/{img}")                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)                    image = torchvision.transforms.ToPILImage()(image_true)                elif box_type == 'terra':                    image = Image.open(f"{frames_path}/{img}")                else:                    image = Image.open(f"{frames_path}/{img}")            else:                if resize:                    image = Image.open(f"{frames_path}/{img}")                    image = image.resize(parameters['size'])                else:                    image = Image.open(f"{frames_path}/{img}")            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)            out.write(cv_img)            count += 1        out.release()        print(f"\nProcessing is finished! Processing time = {round(time.time() - st, 1)}s\n")    @staticmethod    def put_box_on_image(images: str, labels: str, save_path: str):        try:            os.mkdir(save_path)        except:            shutil.rmtree(save_path)            os.mkdir(save_path)        lbl = ['carpet']        color_list = get_colors(lbl)        img_names = []        empty_box, fill_box = 0, 0        with os.scandir(images) as folder:            for f in folder:                img_names.append(f.name[:-4])        for name in img_names:            img_path = f'{images}/{name}.jpg'            box_path = f'{labels}/{name}.txt'            img = Image.open(img_path)            with open(box_path, 'r') as handle:                box_info = handle.readlines()                if box_info:                    fill_box += 1                    box_info = [re.sub(f'\n', ' ', b) for b in box_info]                else:                    empty_box += 1            coord = []            if box_info:                for box in box_info:                    if box:                        box = box.split(" ")                        coord.append([                            int((float(box[1]) - float(box[3]) / 2) * img.size[0]),                            int((float(box[2]) - float(box[4]) / 2) * img.size[1]),                            int((float(box[1]) + float(box[3]) / 2) * img.size[0]),                            int((float(box[2]) + float(box[4]) / 2) * img.size[1]),                        ])                bbox = torch.tensor(coord, dtype=torch.int)                image = read_image(img_path)                lbl2 = lbl * len(coord)                color_list2 = color_list * len(coord)                image_true = draw_bounding_boxes(image, bbox, width=3, labels=lbl2, colors=color_list2, fill=True)                image = torchvision.transforms.ToPILImage()(image_true)                image.save(f'{save_path}/{name}.png')            else:                image = read_image(img_path)                image = torchvision.transforms.ToPILImage()(image)                image.save(f'{save_path}/{name}.png')            print('fill_box=', fill_box, 'empty_box=', empty_box)    @staticmethod    def fill_empty_box(dataset: str):        empty_text = ""        img_names, lbl_names = [], []        with os.scandir(f"{dataset}/images") as folder:            for f in folder:                img_names.append(f.name[:-4])        with os.scandir(f"{dataset}/labels") as folder:            for f in folder:                lbl_names.append(f.name[:-4])        for name in img_names:            if name not in lbl_names:                save_txt(                    txt=empty_text,                    txt_path=f"{dataset}/labels/{name}.txt"                )    @staticmethod    def form_dataset_for_train(data: list, split: float, save_path: str, condition=None):        """        :param data: list of lists of 2 str and 1 float [[image_folder, corresponding_labels_folder, 0.5], ...]        :param split: float between 0 and 1        :param save_path: str        :param condition: dict        """        if condition is None:            condition = {}        try:            os.mkdir(save_path)            os.mkdir(f"{save_path}/train")            os.mkdir(f"{save_path}/train/images")            os.mkdir(f"{save_path}/train/labels")            os.mkdir(f"{save_path}/val")            os.mkdir(f"{save_path}/val/images")            os.mkdir(f"{save_path}/val/labels")        except:            shutil.rmtree(save_path)            os.mkdir(save_path)            os.mkdir(f"{save_path}/train")            os.mkdir(f"{save_path}/train/images")            os.mkdir(f"{save_path}/train/labels")            os.mkdir(f"{save_path}/val")            os.mkdir(f"{save_path}/val/images")            os.mkdir(f"{save_path}/val/labels")        count = 0        for folders in data:            img_list = []            lbl_list = []            print(f"Data: {folders}")            with os.scandir(folders[0]) as fold:                for f in fold:                    if f.name[-3:] in ['png', 'jpg']:                        if condition.get('orig_shape'):                            img = Image.open(f"{folders[0]}/{f.name}")                            if img.size == condition.get('orig_shape'):                                img_list.append(f.name)                        else:                            img_list.append(f.name)            # print("Image", len(img_list), img_list[0])            with_boxes, no_boxes = [], []            with os.scandir(folders[1]) as fold:                for f in fold:                    if f.name[-3:] in ['txt'] and \                            (f"{f.name.split('.')[0]}.png" in img_list or f"{f.name.split('.')[0]}.jpg" in img_list):                        # lbl_list.append(f.name)                        if load_txt(os.path.join(folders[1], f.name)):                            with_boxes.append(f.name)                        else:                            no_boxes.append(f.name)            # print("Labels", len(with_boxes), len(no_boxes), with_boxes[0])            # with_boxes, no_boxes = [], []            # for lbl_file in lbl_list:            #     if f"{lbl_file.split('.')[0]}.png" in img_list or f"{lbl_file.split('.')[0]}.jpg" in img_list:            #         if load_txt(os.path.join(folders[1], lbl_file)):            #             with_boxes.append(lbl_file)            #         else:            #             no_boxes.append(lbl_file)            random.shuffle(no_boxes)            no_boxes = no_boxes[:len(with_boxes)] if len(no_boxes) > len(with_boxes) else no_boxes            with_boxes.extend(no_boxes)            random.shuffle(with_boxes)            with_boxes = with_boxes[:int(len(with_boxes) * folders[2])] if folders[2] < 1 else with_boxes            # take_num = len(with_boxes)            if len(with_boxes) and len(img_list):                print("Labels", len(with_boxes), len(no_boxes), with_boxes[0])                images = []                for label in with_boxes:                    if f"{label.split('.')[0]}.png" in img_list:                        images.append((f"{label.split('.')[0]}.png", label))                    if f"{label.split('.')[0]}.jpg" in img_list:                        images.append((f"{label.split('.')[0]}.jpg", label))                print("Image", len(images), images[0])                # ids = list(range(len(img_list)))                # z = np.random.choice(ids, take_num, replace=False)                # img_list = [img_list[i] for i in z]                logger.info(f'\n- img_list: {len(images)}\n- lbl_list: {len(with_boxes)}\n')                random.shuffle(images)                delimiter = int(len(images) * split)                for i, img in enumerate(images):                    if i <= delimiter:                        # shutil.copy2(f"{folders[0]}/{img[0]}", f"{save_path}/train/images/{img[0].split('.')[0]}.jpg")                        # shutil.copy2(f"{folders[1]}/{img[1]}", f"{save_path}/train/labels/{img[1].split('.')[0]}.txt")                        shutil.copy2(f"{folders[0]}/{img[0]}", f"{save_path}/train/images/{count}.jpg")                        shutil.copy2(f"{folders[1]}/{img[1]}", f"{save_path}/train/labels/{count}.txt")                        # if f"{img[:-3]}txt" in lbl_list:                        #     shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/train/labels/{count}.txt")                        # else:                        #     save_txt(txt='', txt_path=f"{save_path}/train/labels/{count}.txt")                    else:                        # shutil.copy2(f"{folders[0]}/{img[0]}", f"{save_path}/val/images/{img[0].split('.')[0]}.jpg")                        # shutil.copy2(f"{folders[1]}/{img[1]}", f"{save_path}/val/labels/{img[1].split('.')[0]}.txt")                        shutil.copy2(f"{folders[0]}/{img[0]}", f"{save_path}/val/images/{count}.jpg")                        shutil.copy2(f"{folders[1]}/{img[1]}", f"{save_path}/val/labels/{count}.txt")                        # if f"{img[:-3]}txt" in lbl_list:                        #     shutil.copy2(f"{folders[1]}/{img[:-3]}txt", f"{save_path}/val/labels/{count}.txt")                        # else:                        #     save_txt(txt='', txt_path=f"{save_path}/val/labels/{count}.txt")                    if (count + 1) % 200 == 0:                        logger.info(f"-- prepared {i + 1} / {len(images)} images")                    count += 1    @staticmethod    def video_to_array(video_path: str) -> np.ndarray:        """        Transform video to numpy array        """        return skvideo.io.vread(os.path.join(ROOT_DIR, video_path))    @staticmethod    def ohe_from_list(data: list[int], num: int) -> np.ndarray:        """Transform list of labels to one hot encoding array"""        targets = np.array([data]).reshape(-1)        return np.eye(num)[targets]    @staticmethod    def create_video_class_dataset_generator(folder_path: str, split: float) -> VideoClass:        vc = VideoClass()        vc.params['split'] = split        vc.params['folder_path'] = folder_path        classes = os.listdir(os.path.join(ROOT_DIR, folder_path))        classes = sorted(classes)        vc.classes = classes        data, lbl, stat_lbl = [], [], []        for cl in classes:            content = os.listdir(os.path.join(ROOT_DIR, folder_path, cl))            content = sorted(content)            lbl.extend([classes.index(cl)] * len(content))            for file in content:                data.append(os.path.join(folder_path, cl, file))            logger.info(f"-- Class {cl}, processed {len(content)} videos")        zip_data = list(zip(data, lbl))        random.shuffle(zip_data)        train, val = zip_data[:int(split * len(lbl))], zip_data[int(split * len(lbl)):]        vc.x_train, vc.y_train = list(zip(*train))        vc.x_val, vc.y_val = list(zip(*val))        ytr = dict(Counter(vc.y_train))        stat_ytr = {}        for k, v in ytr.items():            stat_ytr[classes[k]] = v        vc.train_stat = stat_ytr        yv = dict(Counter(vc.y_val))        stat_yv = {}        for k, v in yv.items():            stat_yv[classes[k]] = v        vc.val_stat = stat_yv        return vc    @staticmethod    def generate_video_class_batch(generator_dict: dict, iteration: int, mode: str = 'train'                                   ) -> tuple[np.ndarray, np.ndarray]:        num_classes = len(generator_dict.get('stat').get('classes'))        x_array = DatasetProcessing.video_to_array(generator_dict.get(f'x_{mode}')[iteration])        y_array = DatasetProcessing.ohe_from_list([generator_dict.get(f'y_{mode}')[iteration]], num=num_classes)        return np.expand_dims(np.array(x_array), axis=0), np.array(y_array)if __name__ == '__main__':    # [os.path.join(DATASET_DIR, 'datasets/От разметчиков/60x90/60x90'),    #  os.path.join(DATASET_DIR, 'datasets/От разметчиков/60x90/60x90_boxes'), 1],    # [os.path.join(DATASET_DIR, 'datasets/От разметчиков/85x150/85x150'),    #  os.path.join(DATASET_DIR, 'datasets/От разметчиков/85x150/85x150_boxes'), 1],    # [os.path.join(DATASET_DIR, 'datasets/От разметчиков/115x200/115x200'),    #  os.path.join(DATASET_DIR, 'datasets/От разметчиков/115x200/115x200_boxes'), 1],    # [os.path.join(DATASET_DIR, 'datasets/От разметчиков/115x400/115x400'),    #  os.path.join(DATASET_DIR, 'datasets/От разметчиков/115x400/115x400_boxes'), 1],    # [os.path.join(DATASET_DIR, 'datasets/От разметчиков/150x300/150x300'),    #  os.path.join(DATASET_DIR, 'datasets/От разметчиков/150x300/150x300_boxes'), 1],    img_path = os.path.join(DATASET_DIR, 'datasets/От разметчиков/150x300/150x300')    lbl_path = os.path.join(DATASET_DIR, 'datasets/От разметчиков/150x300/150x300_boxes')    img_content = os.listdir(img_path)    lbl_content = os.listdir(lbl_path)    txt = ""    count = 0    for link in img_content:        name = get_name_from_link(link)        if f"{name}.txt" not in lbl_content:            count += 1            save_txt(txt=txt, txt_path=os.path.join(DATASET_DIR, lbl_path, f"{name}.txt"))    print(f"Empty count = {count}")    pass