from __future__ import division
import torch
import torch.nn as nn
from models.pred import Encoder, Decoder
from models.prototype import Prototype
from torch.autograd import Variable

class VADNet(nn.Module):
    def __init__(self):
        super(VADNet,self).__init__()

        self.Encoder = Encoder()
        self.Decoder = Decoder(256)
        self.Prototype = Prototype()
    def forward(self,datas,num_pred,mode,feat_proto=None,feat_proto_avg = None,feat_proto_p = None,Q_require_back = False):
        """
        :param datas:      list (1,3,4)\list(1,1,4)
        :param masks:      list (1,3,4)
        :return:
        """
        if mode == 'S_n':
            ## support set ##
            feat_proto_all = []

            data_s = datas[0][0]
            data_s = self.prep_data(data_s)
            imgs_s = data_s[:num_pred]

            # encoder+convsltm ##
            feat_mid4_s,feat_mid3_s,feat_mid2_s,feat_mid1_s= self.Encoder(imgs_s)

            feat_proto_single_s, feat_mid3_s,feat_mid2_s,feat_mid1_s = self.Prototype([feat_mid4_s,
                                                                           feat_mid3_s,feat_mid2_s,feat_mid1_s])
            feat_proto_all.append(feat_proto_single_s)

            return feat_proto_all,feat_proto_single_s

        elif mode == 'S_p':
            assert feat_proto is not None
            ## query set ##
            feat_proto_p = []

            data_s = datas[0][0]

            data_s = self.prep_data(data_s)

            imgs_s = data_s[:num_pred]

            feat_mid4_s,feat_mid3_s,feat_mid2_s,feat_mid1_s = self.Encoder(imgs_s)
            feat_proto_single_s, feat_mid3_s,feat_mid2_s,feat_mid1_s = self.Prototype([feat_mid4_s,
                                                                           feat_mid3_s,feat_mid2_s,feat_mid1_s])

            feat_proto_p.append(feat_proto_single_s)

            return feat_proto_p

        elif mode == 'q':
            assert feat_proto_avg is not None
            if len(datas)==1:
                datas = datas[0][0]
            ## query set ##
            dists = []
            gt_preds = []
            gt = []

            for idx in range(len(datas)-(num_pred)):
                data_q = self.prep_data(datas[idx:num_pred+idx+1])

                imgs_q = data_q[0:num_pred]
                gt_q = data_q[num_pred]

                feat_mid4_q, feat_mid3_q, feat_mid2_q ,feat_mid1_q = self.Encoder(imgs_q)
                feat_proto_single_q, feat_mid3_q,feat_mid2_q,feat_mid1_q = self.Prototype([feat_mid4_q,
                                                                               feat_mid3_q,feat_mid2_q,feat_mid1_q])

                B,C,H,W = feat_proto_single_q.size()

                feat_proto_ = feat_proto[0]
                feat_proto_sub = (feat_proto_).view(B,C,H*W)
                S_avg = torch.mean(feat_proto_sub,dim= -1,keepdim=True)
                S_std = torch.std(feat_proto_sub,-1,keepdim=True)

                gt_pred_q = torch.zeros(gt_q.size()).cuda()
                for _ in range(3):

                    norm_or =  torch.randn(B,C,H*W).cuda()

                    norm_new = norm_or.mul(S_std)+S_avg
                    norm_new = norm_new.view(B,C,H,W)

                    ## predict ##
                    gt_pred_q = gt_pred_q + self.Decoder(norm_new, feat_mid3_q,feat_mid2_q,feat_mid1_q)

                gt_pred_q = gt_pred_q/3
                gt_preds.append(gt_pred_q)
                gt.append(gt_q)

                loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
                dist = loss_fn(feat_proto_,feat_proto_single_q)
                dists.append(dist)

            return gt_preds,gt,dists,

        else:
            print('Mode type error')
            import sys
            sys.exit(1)

    def prep_data(self,img):
        """
        :param img:  # [img0,img1,img2]
        :param gen_labels:   true
        :return:
        """
        for x in range(len(img)):
            img[x] = Variable(img[x].cuda(), requires_grad=True)
        return img
