# 115x200 {'total': 783, 'filled': 780, 'empty': 3}
# 115x400 {'total': 78, 'filled': 73, 'empty': 5}
# 150x300 {'total': 408, 'filled': 407, 'empty': 1}
# 60x90 {'total': 257, 'filled': 249, 'empty': 8}
# 85x150 {'total': 717, 'filled': 716, 'empty': 1}

POLY_CAM1_IN = [[110, 0], [410, 650], [725, 415], [545, 0]]
POLY_CAM1_OUT = [[0, 0], [0, 120], [270, 750], [540, 735], [775, 525], [855, 315], [760, 0]]
POLY_CAM2_IN = [[140, 0], [140, 150], [260, 185], [260, 0]]
POLY_CAM2_OUT = [[80, 0], [80, 140], [120, 190], [260, 235], [335, 200], [335, 0]]

sh1 = (1920, 1080)
sh2 = (640, 360)

POLY_CAM1_IN = [[b[0] / sh1[0], b[1] / sh1[1]] for b in POLY_CAM1_IN]
POLY_CAM1_OUT = [[b[0] / sh1[0], b[1] / sh1[1]] for b in POLY_CAM1_OUT]
print(f"POLY_CAM1_IN = {POLY_CAM1_IN}")
print(f"POLY_CAM1_OUT = {POLY_CAM1_OUT}")

POLY_CAM2_IN = [[b[0] / sh2[0], b[1] / sh2[1]] for b in POLY_CAM2_IN]
POLY_CAM2_OUT = [[b[0] / sh2[0], b[1] / sh2[1]] for b in POLY_CAM2_OUT]
print(f"POLY_CAM2_IN = {POLY_CAM2_IN}")
print(f"POLY_CAM2_OUT = {POLY_CAM2_OUT}")
