import numpy as np

xxx = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
       2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
cnt = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6]


def object_counter(class_counter, emp: bool, obj: bool, cur_obj: int, obj_range, total_obj):
    if len(class_counter) < obj_range:
        emp = True
        obj = False
        total_obj = 0
        cur_obj = 0
    else:
        found_obj = class_counter[-1]
        objects = 0
        for j in range(obj_range):
            if class_counter[-(j + 1)] > 0:
                objects += 1
        if emp and objects < obj_range:
            emp = True
            obj = False
        elif emp and objects == obj_range:
            emp = False
            obj = True
            if found_obj < cur_obj:
                pass
            elif found_obj == cur_obj:
                total_obj += cur_obj
            elif found_obj > cur_obj and np.sum(class_counter[-obj_range:]) == found_obj * obj_range:
                total_obj += found_obj - cur_obj
                cur_obj = found_obj
        elif obj and objects > 0:
            emp = False
            obj = True
            if found_obj > cur_obj and np.sum(class_counter[-obj_range:]) == found_obj * obj_range:
                total_obj += found_obj - cur_obj
                cur_obj = found_obj
        elif obj and objects == 0:
            emp = True
            obj = False
            cur_obj = 0
        else:
            emp = False
            obj = True
    return total_obj, emp, obj, cur_obj


emp = True
obj = False
cur_obj = 0
obj_range = 4
total_obj = 0
result = []
for i in range(len(xxx)):
    total_obj, emp, obj, cur_obj = object_counter(
        class_counter=xxx[:(i+1)],
        emp=emp,
        obj=obj,
        cur_obj=cur_obj,
        obj_range=obj_range,
        total_obj=total_obj,
    )
    result.append(total_obj)
print(cnt)
print(result)