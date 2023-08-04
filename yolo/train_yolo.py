import os
from parameters import ROOT_DIR
from yolo.yolo8 import train


# train(epochs=50, weights=os.path.join(ROOT_DIR, 'runs/camera_1/weights/last.pt'),
#       config=os.path.join(ROOT_DIR, 'yolo/data_custom_CAM1.yaml'),
#       name='camera_1_mix5+_8n_400ep')
train(epochs=50, weights=os.path.join(ROOT_DIR, 'runs/camera_2/weights/last.pt'),
      config=os.path.join(ROOT_DIR, 'yolo/data_custom_CAM2.yaml'),
      name='camera_2_mix5+_8n_400ep')



