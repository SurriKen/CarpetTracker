from parameters import SPEED_LIMIT_PERCENT, MIN_EMPTY_SEQUENCE, DEAD_LIMIT_PERCENT
from tests.test_tracker import PolyTracker
from utils import get_distance

x, y = [133, 122, 150, 157], [76, 153, 123, 216]

c1 = ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)
c2 = ((y[0] + y[2]) / 2, (y[1] + y[3]) / 2)
# c1 = (x[0], x[1])
# c2 = (y[0], y[1])
d = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
print(d)
speed_limit = get_distance([640, 0], [0, 360]) * SPEED_LIMIT_PERCENT
print('speed_limit', speed_limit)

#
# xxx = [7953, 7954, 7955, 7956]
# xxx2 = [7936, 7937, 7938, 7939, 7940, 7941, 7942, 7943, 7944, 7945, 7946, 7947, 7948, 7949]
# print(list(set(xxx) & set(xxx2)))
# print(list(set(xxx + xxx2)))
#
# d = [True, False] * 5
# print(sum(d))