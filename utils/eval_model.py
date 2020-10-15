#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.eval_test import get_thresholdtable_from_fpr, get_tpr_from_threshold
from utils.utils import LossRecord

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(Config, model, data_loader, val_version, epoch_num, log_file):

    model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    scores = []
    test_labels = []
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            # outputs = model(inputs)
            # import ipdb; ipdb.set_trace()
            preds = model(inputs)
            preds = F.softmax(preds[0], dim=1)
            # import ipdb; ipdb.set_trace()
            preds = preds.cpu().data.numpy()
            
            preds = preds[:,1].squeeze()
            
            labels = list(labels.squeeze())
            preds = list(preds)

            scores = scores + preds
            test_labels = test_labels + labels

            # loss = 0

            # ce_loss = get_ce_loss(preds[0], labels).item()
            # loss += ce_loss

            # val_loss_recorder.update(loss)
            # val_celoss_recorder.update(ce_loss)

            # if Config.use_dcl and Config.cls_2xmul:
            #     outputs_pred = preds[0] + preds[1][:,0:num_cls] + preds[1][:,num_cls:2*num_cls]
            # else:
            #     outputs_pred = preds[0]
            # # top3_val, top3_pos = torch.topk(outputs_pred, 3)

            # print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

        #     batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
        #     val_corrects1 += batch_corrects1
        #     batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
        #     val_corrects2 += (batch_corrects2 + batch_corrects1)
        #     batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
        #     val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        # val_acc1 = val_corrects1 / item_count
        # val_acc2 = val_corrects2 / item_count
        # val_acc3 = val_corrects3 / item_count

        # calculate tpr
        fpr_list = [0.01, 0.005, 0.001]
        threshold_list = get_thresholdtable_from_fpr(scores,test_labels, fpr_list)
        tpr_list = get_tpr_from_threshold(scores,test_labels, threshold_list)

        # show results
        print('=========================================================================')
        print('TPR@FPR=10E-3: {}\n'.format(tpr_list[0]))
        print('TPR@FPR=5E-3: {}\n'.format(tpr_list[1]))
        print('TPR@FPR=10E-4: {}\n'.format(tpr_list[2]))
        print('=========================================================================')

        log_file.write(val_version  + '\t' +str(tpr_list[0])+'\t' + str(tpr_list[1]) + '\t' + str(tpr_list[2]) + '\n')

        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        # print('% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, since), flush=True)
        print('TPR@FPR=10E-3: %.4f || TPR@FPR=5E-3: %.4f || TPR@FPR=10E-4: %.4f ||time: %d' % (tpr_list[0], tpr_list[1], tpr_list[2], since), flush=True)
        print('--' * 30, flush=True)

    return tpr_list[0], tpr_list[1], tpr_list[2]

