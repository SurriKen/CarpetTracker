xxx = [[782, 414, 919, 618, 0.7248424887657166, 0],
       [676, 403, 917, 614, 0.309719979763031, 0],
       [626, 392, 915, 580, 0.25444692373275757, 0]]

def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    inter_w = max(0.0, inter_box[2] - inter_box[0] + 1)
    inter_h = max(0.0, inter_box[3] - inter_box[1] + 1)

    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_area = inter_w * inter_h
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)

def drop_intersect():
    pass

print(box_iou(xxx[2], xxx[0]))