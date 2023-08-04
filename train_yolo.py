import os
from parameters import ROOT_DIR, DATASET_DIR
from yolo8 import train

# train(epochs=100, weights='runs/detect/univ_200ep3/weights/last.pt',
#       config=os.path.join(ROOT_DIR, 'data_custom_univ.yaml'), name='univ_200ep3')
# train(epochs=100, weights='yolo_weights/yolov8n.pt',
#       config=os.path.join(ROOT_DIR, 'data_custom_diff_CAM1.yaml'), name='camera_1_red_100ep')
# train(epochs=100, weights='yolo_weights/yolov8n.pt',
#       config=os.path.join(ROOT_DIR, 'data_custom_diff_CAM2.yaml'), name='camera_2_red_100ep')
train(epochs=100, weights=os.path.join(ROOT_DIR, 'runs/detect/camera_1_mix4+_8n_250ep/weights/last.pt'),
      config=os.path.join(ROOT_DIR, 'data_custom_CAM1.yaml'), name='camera_1_mix4+_8n_350ep')
train(epochs=100, weights=os.path.join(ROOT_DIR, 'runs/detect/camera_2_mix4+_8n_250ep/weights/last.pt'),
      config=os.path.join(ROOT_DIR, 'data_custom_CAM2.yaml'), name='camera_2_mix4+_8n_350ep')

