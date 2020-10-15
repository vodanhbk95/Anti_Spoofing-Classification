import cv2
import os
import time
import sys

import numpy as np

#Get the threshold under several fpr
def get_thresholdtable_from_fpr(scores,labels, fpr_list):
    """Calculate the threshold score list from the FPR list
    Args:
      score_target: list of (score,label)
    Returns:
      threshold_list: list, the element is threshold score calculated by the
      corresponding fpr
    """
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            live_scores.append(float(score))
    live_scores.sort(reverse=True)
    live_nums = len(live_scores)
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        threshold_list.append(live_scores[i_sample - 1])
    return threshold_list

#Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    """Calculate the recall score list from the threshold score list.
    Args:
      score_target: list of (score,label)
      threshold_list: list, the threshold list
    Returns:
      recall_list: list, the element is recall score calculated by the
                   correspond threshold
    """
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            hack_scores.append(float(score))
    hack_scores.sort(reverse=True)
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] <= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list