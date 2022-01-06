from shapely.geometry import Polygon
import os
import json
from matplotlib import pyplot as plt
import math
import matplotlib.patches as mpatches

def image_true_pos(gt_data, dt_data, iou_thresh):
    '''
    All instances where there is a good dt for a given gt
    '''

    true_p = 0 # count all true positives

    for gt in gt_data:
        # get ground truth class id (int) and polygon
        gt_class = gt[2]
        poly_gt = gt[1]

        # set found to false - if we find one overlapping dt of the right class,
        # we will not accept more, so this will be set to True an all other overlaps
        # will not be also counted as true positives for the same gt
        found = False

        for dt in dt_data:
            # get detection class id and polygon
            dt_class = dt[2]
            poly_dt = dt[1]

            # If classes match and iou 
            if gt_class == dt_class and not found:
                inter = poly_gt.intersection(poly_dt)
                iou = inter.area/(poly_gt.area + poly_dt.area - inter.area)
                if iou >= iou_thresh:
                    true_p += 1
                    found = True
                    break

    return true_p

def get_poly(bbox):
    [x1, y1, w,h] = bbox
    x2 = x1 + w
    y2 = y1 + h

    poly = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

    return poly

def coco_dt_eval_format(coco_dt_path):
    '''
    Create data in the format:
    im_id, polygon_box, class, confidence
    '''
    with open(coco_dt_path, 'r') as f:
        dts = json.load(f)

    eval_dts = []

    for dt in dts:
        new_dt = [dt['image_id'], get_poly(dt['bbox']), dt['category_id'], dt['score']]
        eval_dts.append(new_dt)
    
    return eval_dts

def coco_gt_eval_format(coco_gt_path):
    with open(coco_gt_path, 'r') as f:
        gts = json.load(f)
    eval_gts = []
    im_ids = []

    for gt in gts['annotations']:
        new_gt = [gt['image_id'], get_poly(gt['bbox']), gt['category_id'], 1.0]
        eval_gts.append(new_gt)
        im_ids.append(gt['image_id'])

    im_ids = list(set(im_ids))
    return eval_gts, im_ids

def get_im_data(eval_data, im_id, conf_thresh):
    im_data = []
    for d in eval_data:
        if d[0] == im_id:
            if d[3] >= conf_thresh:
                im_data.append(d)
    return im_data

def show_pr(dt, gt, iou_thresh):
    # format detections, groud truth
    eval_dts = coco_dt_eval_format(dt)
    eval_gts, im_ids = coco_gt_eval_format(gt)
    print('Data formatted')

    confs = list(range(0,50))
    confs = [float(c)*0.02 for c in confs]
    total_ims = len(im_ids)

    p_curve = []
    r_curve = []

    for conf_thresh in confs:
        all_precision = []
        all_recall = []

        for im_id in im_ids: 
            im_gt = get_im_data(eval_gts, im_id, 0.0)
            im_dt = get_im_data(eval_dts, im_id, conf_thresh)

            tp = image_true_pos(im_gt, im_dt, iou_thresh)

            if len(im_dt) > 0:
                im_p = tp/(len(im_dt))
            else:
                im_p = 1
            if len(im_gt) > 0:
                im_r = tp/(len(im_gt))
            else:
                im_r = 0

            all_precision.append(im_p)
            all_recall.append(im_r)

        full_precision = sum(all_precision)/len(all_precision)
        full_recall = sum(all_recall)/len(all_recall)
        # print(f'Precision: {full_precision}, Recall: {full_recall}')

        p_curve.append(full_precision)
        r_curve.append(full_recall)
    plt.figure()
    plt.plot(confs, p_curve, color = 'b')
    plt.plot(confs, r_curve, color = 'r')
    r_patch = mpatches.Patch(color='red', label='Recall')
    b_patch = mpatches.Patch(color='blue', label = 'Precision')
    plt.legend(handles=[r_patch, b_patch])
    plt.xlabel('Confidence threshold')
    plt.ylabel('Preformance (%)')
    plt.show()

    dist_pr = 1.0
    chosen_p = 0
    chosen_r = 0
    index = 0
    for i,p in enumerate(p_curve):
        check_dist = abs(p - r_curve[i])
        if check_dist < dist_pr:
            dist_pr = check_dist
            chosen_p = p
            chosen_r = r_curve[i]
            index = confs[i]

    print(f'Chosen precision: {chosen_p}, Chosen recall: {chosen_r}, Confidence: {index}')

    return p_curve, r_curve

def final_pr(dt, gt, iou_thresh, conf_thresh):
    '''
    Return final precision and recall, using the confidence threshold you have set based on your validation set
    '''
    # format detections, groud truth
    eval_dts = coco_dt_eval_format(dt)
    eval_gts, im_ids = coco_gt_eval_format(gt)
    print('Data formatted')

    total_ims = len(im_ids)
    
    all_precision = []
    all_recall = []

    for im_id in im_ids: 
        im_gt = get_im_data(eval_gts, im_id, 0.0)
        im_dt = get_im_data(eval_dts, im_id, conf_thresh)

        tp = image_true_pos(im_gt, im_dt, iou_thresh)

        if len(im_dt) > 0:
            im_p = tp/(len(im_dt))
        else:
            im_p = 1
        if len(im_gt) > 0:
            im_r = tp/(len(im_gt))
        else:
            im_r = 0

        all_precision.append(im_p)
        all_recall.append(im_r)

    full_precision = sum(all_precision)/len(all_precision)
    full_recall = sum(all_recall)/len(all_recall)


    print(f'Precision: {full_precision}, Recall: {full_recall}')

    return   


