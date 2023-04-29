import numpy as np
import os,sys
from torchvision import transforms

import torch
torch.backends.cudnn.benckmark = True
from torch.utils import data
from data.dataloader import setup_dataset_test
from models.VADNet import VADNet
from torch.optim.lr_scheduler import StepLR
from data.accuracy import acc,score_sum,AUC

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import dataloader
from multiprocessing.reduction import ForkingPickler
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)
for t in torch._storage_classes:

    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

def init():
    import argparse
    print('Parsing the arguments...')
    args = argparse.ArgumentParser()

    # exp name #
    args.add_argument('--exp_name', type=str, required=True,
                      help='the experiment name')
    # pre-trained #
    args.add_argument('--pretrained', action='store_true', default=False,
                      help='True if with pre_trained model')
    args.add_argument('--pretrained_path', type=str, default='.',
                      help='the path to pre_trained model')

    # logging #
    args.add_argument('--log_path', type=str, default='logging', help='the path to logs')
    args.add_argument('--log_interv', type=int, default=1000, help='Interval betweem log files')

    # model saved #
    args.add_argument('--model_saved_path', type=str, default='./saved_models', help='the path to save models')
    args.add_argument('--model_saved_interv', type=int, default=1000, help='Interval between saved models')

    # test #
    args.add_argument('--set', type=str, default='ped2', help='name of training or test set')
    args.add_argument('--nepoch', type=int, default=1, help='num epochs to run')
    args.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    args.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')

    # Datasets #
    args.add_argument('--dataset_path', type=str, default='.', help='the path to the datasets')

    # few shot #
    args.add_argument('--k_shot',type=int,default=1,help='num of images in one support set')
    args.add_argument('--num_pred',type=int,default=3,help='num of images for predict')

    arg = args.parse_args()
    return arg

def load_pretrained_model(pretrained_path,model,optimizer = None,scheduler=None):
    if pretrained_path[-3:] == 'pth':
        pre_model_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        pre_model_dict_feat = {k.replace('module', 'Encoder'): v for k, v in pre_model_dict['state_dict'].items() if k.replace('module', 'Encoder') in model_dict}
        model.load_state_dict(pre_model_dict_feat,strict=False)

    elif pretrained_path[-3:] == 'tar':
        pretrained_dict = torch.load(pretrained_path)
        if isinstance(pretrained_dict,dict):
            pre_model_dict = pretrained_dict['model_dict']
        else:
            pre_model_dict = pretrained_dict

        model_dict = model.state_dict()

        pre_model_dict_feat = {k:v for k,v in pre_model_dict.items() if k in model_dict}

        # update model #
        model.load_state_dict(pre_model_dict_feat)

        if optimizer is not None:
            optimizer.load_state_dict(pretrained_dict['optimizer'])
            print('Also loaded the optimizer status')
        if scheduler is not None:
            scheduler.load_state_dict(pretrained_dict['scheduler'])
            print('Also loaded the scheduler status')
    else:
        print('pretained_model error!')

def test(dataset_path, k_shot, num_pred,t_idx):
    test_log_path = os.path.join(args.model_saved_path, exp_name, 'test_log_'+str(t_idx))

    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path)

    Testset_sets,img_note_gt,k_shot_list = setup_dataset_test(dataset_path, 'test', k_shot, num_pred)
    Num_scene = len(Testset_sets)
    AUC_set,scene_set = [],[]
    auc_avg = 0

    ## test logging ##
    test_record_file = test_log_path + '/record.txt'
    f3 = open(test_record_file, 'w')
    for idx in range(len(k_shot_list[0])-1):
        f3.write(k_shot_list[0][idx][0])
        f3.write('\n***********************************')
    f3.close()

    for idx in range(Num_scene):

        Testloader0 = Testset_sets[idx]
        Num_video = len(Testloader0)
        anomaly_score_total_list = []
        img_note_gt_all = []

    ## testing ##
        for x in range(Num_video):

            Testsets = Testloader0[x]
            test_loader = data.DataLoader(Testsets, batch_size=1, pin_memory=True, num_workers=0)
            psnr_set, dist_set = [], []

            img_log_path = os.path.join(test_log_path, 'test_' + str(x))
            if not os.path.exists(img_log_path):
                os.makedirs(img_log_path)

            for seq, data_dict in enumerate(test_loader):
                datas = data_dict['datas']
                datas_p = data_dict['datas_p']
                scene_name = data_dict['scene_name']

                if seq == 0:
                    f3 = open(test_record_file, 'r+')
                    f3.read()
                    f3.write('\n')
                    f3.write(scene_name[0])
                    f3.close()

                datas_n = [datas[0][:k_shot]]
                datas_q = datas_p
                if seq == 0:

                    feat_proto_all, feat_proto_avg= model(datas_n, num_pred, mode='S_n')
                gt_preds,gt, dist = model(datas_q, num_pred, mode='q', feat_proto=feat_proto_all,
                                                    feat_proto_avg=feat_proto_avg)

                psnr_list = acc(gt_preds, gt, dist)
                psnr_set.append(psnr_list)
                dist_set.append(dist)

                ## image load ##
                if (seq + 1) % 10 == 0:

                    pred = gt_preds[0]
                    gt = datas_q[0][0][-1]
                    for t in range(pred.size(0)):
                        save_img = img_log_path + '/' + str(seq + 1) + '_' + str(t) + '.jpg'
                        save_img_ori = img_log_path + '/' + str(seq + 1) + '_' + str(t) + '_ori.jpg'

                        pred0 = pred[t].detach().cpu()
                        pred0 = transforms.ToPILImage()(pred0.float())
                        pred0.save(save_img)

                        gt0 = gt[t].detach().cpu()
                        gt0 = transforms.ToPILImage()(gt0.float())
                        gt0.save(save_img_ori)

            anomaly_score_psnr_, anomaly_score_dist_,anomaly_score_sum = score_sum(psnr_set, dist_set,0.6)
            anomaly_score_total_list += anomaly_score_sum
            img_note_gt_all += list(img_note_gt[idx][x])
            ## recode score ##
            save_png = img_log_path + '/avg_score'+'.png'
            save_png_psnr = img_log_path + '/psnr_score' + '.png'
            save_png_dist = img_log_path + '/dist_score' + '.png'
            x0 = np.arange(len(anomaly_score_psnr_))
            anomaly_score_psnr = np.asarray(anomaly_score_psnr_)
            anomaly_score_dist = np.asarray(anomaly_score_dist_)
            anomaly_score_sum0 = np.asarray(anomaly_score_sum)
            note_gt = img_note_gt[idx][x]


            plt.figure()
            plt.plot(x0,note_gt, marker='.', color='r', label='gt_note',lw=2)
            plt.plot(x0, 1-anomaly_score_psnr, marker='.', color='b', label='psnr',lw=1)
            plt.legend()
            plt.xlabel('t_frame')
            plt.ylabel('anomaly_scores')
            plt.savefig(save_png_psnr)

            plt.figure()
            plt.plot(x0, note_gt, marker='.', color='r', label='gt_note', lw=2)
            plt.plot(x0, 1-anomaly_score_dist, marker='.', color='g', label='dist', lw=1)
            plt.legend()
            plt.xlabel('t_frame')
            plt.ylabel('anomaly_scores')
            plt.savefig(save_png_dist)

            plt.figure()
            plt.plot(x0, note_gt, marker='.', color='r', label='gt_note', lw=3)
            plt.plot(x0, 1 - anomaly_score_psnr, marker='.', color='b', label='psnr', lw=1)
            plt.plot(x0, 1 - anomaly_score_dist, marker='.', color='g', label='dist', lw=1)
            plt.plot(x0, 1 - anomaly_score_sum0, marker='.', color='y', label='avg', lw=1)
            plt.legend()
            plt.xlabel('t_frame')
            plt.ylabel('anomaly_scores')
            plt.savefig(save_png)

            plt.cla()
            plt.close('all')

        anomaly_score_total_list = np.asarray(anomaly_score_total_list)
        img_note_gt_all = np.asarray(img_note_gt_all)

        accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - img_note_gt_all, 0))
        AUC_set.append(accuracy)
        scene_set.append(scene_name)
        auc_avg += accuracy
    auc_avg = auc_avg/Num_scene

    print('auc_avg:',auc_avg)

    print('AUC: ', [accuracy * 100 for accuracy in AUC_set], '%')
    print('*' * 100)
    return auc_avg,AUC_set,scene_set




if __name__=='__main__':
    args = init()

    # data init #
    exp_name = args.exp_name
    nepoch = args.nepoch

    pretrained = args.pretrained
    pretrained_path = args.pretrained_path
    dataset_path = args.dataset_path
    SET = args.set
    num_pred = args.num_pred

    k_shot = args.k_shot

    logging_path = os.path.join(args.log_path, exp_name)
    save_model_path = os.path.join(args.model_saved_path, exp_name)
    save_img_path = os.path.join(args.model_saved_path,exp_name,'generate_imgs')
    save_log_path = os.path.join(args.model_saved_path,exp_name,'log.txt')

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    f1 = open(save_log_path, 'w')
    f1.write('test...')
    f1.close()

    model = VADNet()
    gpu_index = args.gpu
    model.cuda()

    ## setup optimizer ##
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = StepLR(optimizer, step_size=3000, gamma=0.8)

    # update model and optimizer #
    if pretrained and pretrained_path is not '.':
        print('loading pretrained modle at %s' % (pretrained_path))
        load_pretrained_model(pretrained_path, model, optimizer,scheduler)
    else:
        pretrained_path = 'best_proto.tar'
        load_pretrained_model(pretrained_path, model,optimizer,scheduler)

    ## testing ##
    for epoch in range(nepoch):
        with torch.no_grad():
            model.eval()
            auc_temp,auc,scene_names = test(dataset_path, k_shot, num_pred,epoch + 1)
