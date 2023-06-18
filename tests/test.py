from tests.test2 import PolyTracker, POLY_CAM1_OUT

x = [[543, 465, 706, 575], [549, 464, 707, 599], [558, 478, 722, 603], [539, 463, 729, 619], [539, 463, 729, 619], [294, 513, 734, 762], [235, 540, 749, 872], [253, 506, 758, 876]]

for b in x:
    point = PolyTracker.get_center(b)
    d = PolyTracker.point_in_polygon(point, POLY_CAM1_OUT)
    print(point, d)

# img_shape = (640, 360)
# diagonal = ((img_shape[0]) ** 2 + (img_shape[1]) ** 2) ** 0.5
# dist_limit = 0.15 * diagonal
# print(dist_limit)