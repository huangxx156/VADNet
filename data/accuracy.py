import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.metrics import roc_auc_score

def psnr(mse):
    psnr = 10 * math.log10(1 / mse)
    psnr = torch.tensor(psnr).cuda()

    return psnr

def acc(preds,gts,dists):

    loss_func_mse = nn.MSELoss(reduction='none')
    psnr_list = []
    for idx in range(len(preds)):
        pred,gt,dist = preds[idx],gts[idx],dists[idx]
        mse_imgs = torch.mean(loss_func_mse((pred +1)/2, (gt +1)/2))
        psnr_list.append(psnr(mse_imgs))

    return psnr_list

def anomaly_score(psnr, max_psnr, min_psnr):

    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list

def score_sum(psnr_set, dist_set, alpha):
    for idx in range(len(psnr_set)):
        psnr_set[idx] = psnr_set[idx][0].cpu().numpy()
        dist_set[idx] = dist_set[idx][0].cpu().numpy()

    list1 = anomaly_score_list(psnr_set)
    list2 = anomaly_score_list_inv(dist_set)
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list1,list2,list_result

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc