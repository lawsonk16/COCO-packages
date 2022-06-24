from __future__ import division
import math
import time
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from terminaltables import AsciiTable

import sys
paths = ['/content/drive/MyDrive/Colab Notebooks/scripts']

for p in paths:
    sys.path.append(p)

from coco_utils.coco_help import get_im_ids


def get_class_names(gt_path):
      with open(gt_path, 'r') as f:
          gt = json.load(f)
      clss = gt['categories']

      class_names = []

      for c in clss:
          class_names.append(c['name'])
      return class_names


def yolo_coco_dts(gt_fp, dt_fp, conf_thresh = 0.0):

    # get all image ids in the ground truth
    im_ids = get_im_ids(gt_fp)
    im_ids.sort()

    with open(dt_fp, 'r') as f:
        dts = json.load(f)

    all_dts = []

    for im_id in range(0,im_ids[-1]+1):
        im_dts = get_dts_id(im_id, dts)
        formatted_dts = []
        for dt in im_dts:

            # bounding box
            bbox = dt['bbox']
            bbox = [float(i) for i in bbox]
            [x1, y1, w, h] = bbox
            x2 = x1 + w
            y2 = y1 + h
            score = dt['score']
            cat = float(dt['category_id']) - 1.0
            dt_formatted = [x1, y1, x2, y2, score, cat]
            if score >= conf_thresh:
                formatted_dts.append(dt_formatted)
        if len(formatted_dts) > 0:
            formatted_dts = torch.Tensor(formatted_dts)
            all_dts.append(formatted_dts)
        else:
            all_dts.append(None)

    return all_dts

def yolo_coco_gt(gt_fp):
    all_gt = []

    im_ids = get_im_ids(gt_fp)
    im_ids.sort()

    with open(gt_fp, 'r') as f:
        gts = json.load(f)
    
    for im_id in tqdm(im_ids):
        im_gts = get_gts_id(im_id, gts)
        for gt in im_gts:
            cat = gt['category_id'] - 1.0
            bbox = gt['bbox']
            bbox = [float(a) for a in bbox]
            [x1, y1, w, h] = bbox
            x2 = x1 + w
            y2 = y1 + h
            im_id = float(im_id)
            f_gt = [im_id, cat, x1, y1, x2, y2]
            all_gt.append(f_gt)

    return torch.Tensor(all_gt)

def evaluate_coco(outputs, targets, class_names, iou_thresh = 0.5):

    sample_metrics = get_batch_statistics(outputs, targets, iou_threshold=iou_thresh)

    labels = targets[:, 1].tolist()

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")
    print(f"---- Precision {precision.mean()}")
    print(f"---- Recall {recall.mean()}")

    return {'mAP': AP.mean(), 'precision': precision.mean(), 'recall': recall.mean()}

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue
        
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
        
        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def nms(prediction, conf_thres=0.3, nms_thres=0.8):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, class_score, class_pred)
    """

    
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        
        if image_pred is not None:
            
            image_pred = image_pred[image_pred[:, 4] >= conf_thres]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Class score
            score = image_pred[:, 4]
            # Sort by it
            image_pred = image_pred[(-score).argsort()]
            # Key info
            class_confs, _ = image_pred[:, 4:5].max(1, keepdim=True)
            class_preds, _ = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :4], class_confs.float(), class_preds.float()), 1)

            # Perform non-maximum suppression
            keep_boxes = []
            while detections.size(0):
                large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
                label_match = detections[0, -1] == detections[:, -1]
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                invalid = large_overlap & label_match
                weights = detections[invalid, 4:5]
                # Merge overlapping bboxes by order of confidence
                detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
                keep_boxes += [detections[0]]
                detections = detections[~invalid]
            if keep_boxes:
                output[image_i] = torch.stack(keep_boxes)

    return output

def get_dts_id(im_id, dts):
    im_dts = []
    for d in dts:
        if d['image_id'] == im_id:
            im_dts.append(d)
    return im_dts


def get_gts_id(im_id, gts):
    im_gts = []
    for d in gts['annotations']:
        if d['image_id'] == im_id:
            im_gts.append(d)
    return im_gts

def merge_dts(detections_1, detections_2):
    merged_dts = []
    for i, dts_1 in enumerate(tqdm(detections_1)):
        dts_2 = detections_2[i]
        merged = []

        # process detections from the first set
        if dts_1 is not None:
            dts_1 = dts_1.tolist()
            merged.extend(dts_1)

        # process detections from the second set
        if dts_2 is not None:
            dts_2 = dts_2.tolist()
            merged.extend(dts_2)
        
        # handle the 'None' case
        if len(merged) < 1:
            merged_dts.append(None)

        # add the detections to the larger list
        else:
            merged_dts.append(torch.Tensor(merged))

    return merged_dts