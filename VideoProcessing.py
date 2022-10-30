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

from parameters import PREDICT_PATH
from prepare_dataset import PrepareDataset
from utils import save_dict, load_dict, get_colors


# from imageai.Detection import VideoObjectDetection


class VideoProcessing:

    def __init__(self):
        pass

    @staticmethod
    def cut_video(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000):
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        saved_video_path = f"{save_path}/{video_path.split('/')[-1]}"
        if to_time and to_time < duration:
            clip = VideoFileClip(video_path).subclip(from_time, to_time)
            clip.write_videofile(saved_video_path)
        elif from_time:
            clip = VideoFileClip(video_path).subclip(from_time, duration)
            clip.write_videofile(saved_video_path)
        else:
            shutil.copy2(video_path, saved_video_path)
        print("Video was cut and save")
        return saved_video_path

    @staticmethod
    def video2frames(video_path: str, save_path: str = 'datasets', from_time=0, to_time=1000):
        video_name = video_path.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        # print(video_path, fps)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        if to_time:
            if to_time > int(duration):
                to_time = int(duration)
            to_path = f"{save_path}/{video_name}_{from_time}s-{to_time}s"
        elif from_time:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        else:
            to_path = f"{save_path}/{video_name}_{from_time}s-{duration}s"
        try:
            os.mkdir(to_path)
        except:
            shutil.rmtree(to_path, ignore_errors=True)
            os.mkdir(to_path)
        os.mkdir(f"{to_path}/frames")
        os.mkdir(f"{to_path}/video")
        os.mkdir(f"{to_path}/xml_labels")
        saved_video_path = VideoProcessing.cut_video(
            video_path=video_path,
            save_path=f"{to_path}/video",
            from_time=from_time,
            to_time=to_time
        )
        video_capture = cv2.VideoCapture()
        # saved_video_path = f"{save_path}/{video_name}_{max_time}s/videos/{video_path.split('/')[-1]}"
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
            cv2.imwrite(f"{to_path}/frames/%05d.png" % i, frame)
        video_data = {
            "fps": int(fps), "frames": int(frames), 'size': size
        }
        print(f"frames were got: fps - {int(fps)}, total frames - {int(frames)}, frame size - {size}")
        save_dict(video_data, to_path, 'data')

    # @staticmethod
    # def frames2video(frames_path: str, save_path: str, video_name: str,
    #                  params: str, box_path: str = None, resize=False, box_type='xml'):
    #     parameters = load_dict(params)
    #     image_list, box_list, lbl_list, color_list = [], [], [], []
    #     with os.scandir(frames_path) as folder:
    #         for f in folder:
    #             image_list.append(f.name)
    #     image_list = sorted(image_list)
    #     if box_path:
    #         PrepareDataset.remove_empty_xml(box_path)
    #         with os.scandir(box_path) as folder:
    #             for f in folder:
    #                 box_list.append(f.name)
    #         for box in box_list:
    #             box_info = PrepareDataset.read_xml(f"{box_path}/{f'{box}'}")
    #             if box_info["coords"][-1] not in lbl_list:
    #                 lbl_list.append(box_info["coords"][-1])
    #         lbl_list = sorted(lbl_list)
    #         color_list = get_colors(lbl_list)
    #
    #     tmp_folder = f"{save_path}/tmp"
    #     try:
    #         os.mkdir(tmp_folder)
    #     except:
    #         shutil.rmtree(tmp_folder, ignore_errors=True)
    #         os.mkdir(tmp_folder)
    #
    #     st = time.time()
    #     count = 0
    #     print(f"Start preprocessing...\n")
    #     for img in image_list:
    #         if (count + 1) % int(len(image_list) * 0.1) == 0:
    #             print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were preprocessed...")
    #         name = img.split(".")[0]
    #         if box_path and f"{name}.xml" in box_list:
    #             if box_type == 'xml':
    #                 if resize:
    #                     box_info = PrepareDataset.read_xml(
    #                         xml_path=f"{box_path}/{f'{name}.xml'}",
    #                         shrink=True,
    #                         new_width=parameters['size'][0],
    #                         new_height=parameters['size'][1]
    #                     )
    #                 else:
    #                     box_info = PrepareDataset.read_xml(xml_path=f"{box_path}/{f'{name}.xml'}")
    #                 boxes, labels = [], []
    #                 for b in box_info["coords"]:
    #                     boxes.append(b[:-1])
    #                     labels.append(b[-1])
    #                 bbox = torch.tensor(boxes, dtype=torch.int)
    #                 image = read_image(f"{frames_path}/{img}")
    #                 image_true = draw_bounding_boxes(image, bbox, width=3, labels=labels, colors=color_list, fill=True)
    #                 image_true = torchvision.transforms.ToPILImage()(image_true)
    #                 image_true.save(f"{tmp_folder}/{img}")
    #             elif box_type == 'terra':
    #                 pass
    #             else:
    #                 pass
    #         else:
    #             if resize:
    #                 image = Image.open(f"{frames_path}/{img}")
    #                 new_image = image.resize(parameters['size'])
    #                 new_image.save(f"{tmp_folder}/{img}")
    #             else:
    #                 shutil.copy2(f"{frames_path}/{img}", f"{tmp_folder}/{img}")
    #         count += 1
    #     print(f"\nPreprocessing is finished! Preprocessing time = {round(time.time() - st, 1)}s\n")
    #
    #     out = cv2.VideoWriter(
    #         f'{save_path}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), parameters['fps'], parameters['size'])
    #     count = 0
    #     st = time.time()
    #     print("Writing video is started...\n")
    #     for filename in image_list:
    #         if (count + 1) % int(len(image_list) * 0.1) == 0:
    #             print(f"{round((count + 1) * 100 / len(image_list), 0)}% images were writed in video...")
    #         img = cv2.imread(f"{tmp_folder}/{filename}")
    #         out.write(img)
    #         count += 1
    #     out.release()
    #     print(f"Video is ready! Path to video: {f'{save_path}/{video_name}.mp4'}. "
    #           f"Writing time={round(time.time() - st, 1)}s")
    #     shutil.rmtree(tmp_folder, ignore_errors=True)

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
    for i in range(5):
        video_path_ = f'videos/Train_{i}.mp4'
        save_path_ = 'datasets'
        max_time = 300
        VideoProcessing.video2frames(
            video_path=video_path_,
            save_path=save_path_,
            from_time=0,
            to_time=max_time
        )
    # VideoProcessing.frames2video(
    #     frames_path='init_frames/Air_1_24s/init_frames',
    #     save_path='init_frames/Air_1_24s',
    #     video_name='Air_1_24s',
    #     params='init_frames/Air_1_24s/data.dict',
    #     box_path='init_frames/Air_1_24s/xml_labels',
    #     resize=False
    # )
