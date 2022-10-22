import os
import shutil
import time

import cv2
import torch
import torchvision
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from prepare_dataset import PrepareDataset
from utils import save_dict, load_dict, get_colors


# from imageai.Detection import VideoObjectDetection


class VideoProcessing:

    def __init__(self):
        pass

    @staticmethod
    def init_new_folder(folder_path):
        try:
            os.mkdir(f"{folder_path}/init_video")
        except:
            pass
        try:
            os.mkdir(f"{folder_path}/init_frames")
        except:
            pass
        try:
            os.mkdir(f"{folder_path}/xml_labels")
        except:
            pass
        try:
            os.mkdir(f"{folder_path}/yolo_model")
        except:
            pass
        try:
            os.mkdir(f"{folder_path}/class_model")
        except:
            pass

    @staticmethod
    def cut_video(video_path: str, save_path: str = 'init_frames', max_time=None):
        video_name = video_path.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        if max_time:
            if max_time > int(duration):
                max_time = int(duration)
        try:
            os.mkdir(f"{save_path}")
        except:
            pass

        if max_time:
            p1 = f"{save_path}/{video_name}_{max_time}s"
        else:
            p1 = f"{save_path}/{video_name}_full"
        try:
            os.mkdir(p1)
        except:
            shutil.rmtree(p1, ignore_errors=True)
            os.mkdir(p1)

        VideoProcessing.init_new_folder(p1)
        saved_video_path = f"{p1}/init_video/{video_path.split('/')[-1]}"
        if max_time and max_time > duration:
            clip = VideoFileClip(video_path).subclip(0, max_time)
            clip.write_videofile(saved_video_path)
        else:
            shutil.copy2(video_path, saved_video_path)
        print("Video was cut and save")

    @staticmethod
    def video2frames(video_path: str, save_path: str = 'init_frames', max_time=1000, predict_mode=False):
        video_name = video_path.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        if max_time:
            if max_time > int(duration):
                max_time = int(duration)
            to_path = f"{save_path}/{video_name}_{max_time}s/init_frames" if not predict_mode else f"{save_path}/init"
        else:
            to_path = f"{save_path}/{video_name}_full/init_frames" if not predict_mode else f"{save_path}/init"
        try:
            os.mkdir(to_path)
        except:
            shutil.rmtree(to_path, ignore_errors=True)
            os.mkdir(to_path)
        if not predict_mode:
            video_capture = cv2.VideoCapture()
            saved_video_path = f"{save_path}/{video_name}_{max_time}s/" \
                               f"init_video/{video_path.split('/')[-1]}"
            video_capture.open(saved_video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"Getting frames ({int(frames)} in total)...")
            size = ()
            for i in range(int(frames)):
                if (i + 1) % 200 == 0:
                    print(f"{i + 1} frames are ready")
                ret, frame = video_capture.read()
                size = (frame.shape[1], frame.shape[0])
                cv2.imwrite(to_path + "/%05d.jpg" % i, frame)
            video_data = {
                "fps": int(fps), "frames": int(frames), 'size': size
            }
            print(f"frames were got: fps - {int(fps)}, total frames - {int(frames)}, frame size - {size}")
            save_dict(video_data, f"{save_path}/{video_name}_{max_time}s", 'data')
        else:
            print(f"Getting frames ({int(frame_count)} in total)...")
            size = ()
            for i in range(int(frame_count)):
                if (i + 1) % 200 == 0:
                    print(f"{i + 1} frames are ready")
                ret, frame = video_capture.read()
                size = (frame.shape[1], frame.shape[0])
                cv2.imwrite(to_path + "/%05d.jpg" % i, frame)
            video_data = {
                "fps": int(fps), "frames": int(frame_count), 'size': size
            }
            print(f"frames were got: fps - {int(fps)}, total frames - {int(frame_count)}, frame size - {size}")
            save_dict(video_data, save_path, 'data')

    @staticmethod
    def frames2video(frames_path: str, save_path: str, video_name: str,
                     params: str, box_path: str = None, resize=False, box_type='xml'):
        parameters = load_dict(params)
        image_list, box_list, lbl_list, color_list = [], [], [], []
        with os.scandir(frames_path) as folder:
            for f in folder:
                image_list.append(f.name)
        image_list = sorted(image_list)
        if box_path:
            PrepareDataset.remove_empty_xml(box_path)
            with os.scandir(box_path) as folder:
                for f in folder:
                    box_list.append(f.name)
            for box in box_list:
                box_info = PrepareDataset.read_xml(f"{box_path}/{f'{box}'}")
                if box_info["coords"][-1] not in lbl_list:
                    lbl_list.append(box_info["coords"][-1])
            lbl_list = sorted(lbl_list)
            color_list = get_colors(lbl_list)

        tmp_folder = f"{save_path}/tmp"
        try:
            os.mkdir(tmp_folder)
        except:
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.mkdir(tmp_folder)

        st = time.time()
        count = 0
        print(f"Start preprocessing...\n")
        for img in image_list:
            if (count + 1) % int(len(image_list) * 0.1) == 0:
                print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were preprocessed...")
            name = img.split(".")[0]
            if box_path and f"{name}.xml" in box_list:
                if box_type == 'xml':
                    if resize:
                        box_info = PrepareDataset.read_xml(
                            xml_path=f"{box_path}/{f'{name}.xml'}",
                            shrink=True,
                            new_width=parameters['size'][0],
                            new_height=parameters['size'][1]
                        )
                    else:
                        box_info = PrepareDataset.read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
                    boxes, labels = [], []
                    for b in box_info["coords"]:
                        boxes.append(b[:-1])
                        labels.append(b[-1])
                    bbox = torch.tensor(boxes, dtype=torch.int)
                    # print(box_info)
                    image = read_image(f"{frames_path}/{img}")
                    image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
                    image_true = torchvision.transforms.ToPILImage()(image_true)
                    image_true.save(f"{tmp_folder}/{img}")
                elif box_type == 'terra':
                    pass
                else:
                    pass
            else:
                if resize:
                    image = Image.open(f"{frames_path}/{img}")
                    new_image = image.resize(parameters['size'])
                    new_image.save(f"{tmp_folder}/{img}")
                else:
                    shutil.copy2(f"{frames_path}/{img}", f"{tmp_folder}/{img}")
            count += 1
        print(f"\nPreprocessing is finished! Preprocessing time = {round(time.time() - st, 1)}s\n")

        out = cv2.VideoWriter(
            f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
        count = 0
        st = time.time()
        print("Writing video is started...\n")
        for filename in image_list:
            if (count + 1) % int(len(image_list) * 0.1) == 0:
                print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were writed in video...")
            img = cv2.imread(f"{tmp_folder}/{filename}")
            out.write(img)
            count += 1
        out.release()
        print(f"Video is ready! Path to video: {f'{save_path}/{video_name}.mp4'}. "
              f"Writing time={round(time.time() - st, 1)}s")
        shutil.rmtree(tmp_folder, ignore_errors=True)

    # @staticmethod
    # def video_detection(model_path, video_path, save_path, fps):
    #     https://wellsr.com/python/object-detection-from-videos-with-yolo/
    #     vid_obj_detect = VideoObjectDetection()
    #     vid_obj_detect.setModelTypeAsYOLOv3()
    #     vid_obj_detect.setModelPath(model_path) #.h5
    #     vid_obj_detect.loadModel()
    #     detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    #         input_file_path=video_path,
    #         output_file_path=save_path,
    #         frames_per_second=fps,
    #         log_progress=True,
    #         return_detected_frame=True
    #     )
    #


if __name__ == '__main__':
    # video_path_ = 'init_video/Train_2.mp4'
    # save_path_ = 'init_frames'
    # max_tim = 300
    # dict_path = 'init_frames/Train_2_300s/data.dict'
    video_path_ = 'init_video/Air_1.mp4'
    save_path_ = 'init_frames'
    max_tim = 300
    dict_path = 'init_frames/Air_1_300s/data.dict'
    # VideoProcessing.cut_video(
    #     video_path=video_path_,
    #     save_path=save_path_,
    #     max_time=max_tim
    # )
    # VideoProcessing.video2frames(
    #     video_path=video_path_,
    #     save_path=save_path_,
    #     max_time=max_tim
    # )
    # VideoProcessing.init_new_folder('init_frames/Air_1_24s')
    # VideoProcessing.frames2video(
    #     frames_path='init_frames/Air_1_24s/init_frames',
    #     save_path='init_frames/Air_1_24s',
    #     video_name='Air_1_24s',
    #     params='init_frames/Air_1_24s/data.dict',
    #     box_path='init_frames/Air_1_24s/xml_labels',
    #     resize=False
    # )
