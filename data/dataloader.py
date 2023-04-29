import numpy as np
import torch
import os
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import Dataset

mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class Dataset_loader_test(Dataset):
    def __init__(self,k_shot_list, scene_name,num_pred, data_transform=None):
        self.k_shot_list = k_shot_list
        self.data_transform = data_transform
        self.scene_name = scene_name    # test:[1,12,120]
        self.num_pred = num_pred

        self.image_trajs = generate_img_test(scene_name)
        self.num_image_set = len(self.image_trajs)-num_pred

    def __len__(self):
        length = self.num_image_set

        return length

    def __getitem__(self, item):
        datas = []
        for frames in self.k_shot_list[0]:
            task_list = []
            for idx in range(len(frames)):
                img_path = frames[idx]
                img = Image.open(img_path).convert('RGB')
                if (self.data_transform is not None) & (idx !=len(frames)-1):
                    img = self.data_transform(img)
                    img = (img - mean) / std
                else:
                    img = self.data_transform(img)
                task_list.append(img)
            datas.append(task_list)

        self.img_set = self.image_trajs[item:item+self.num_pred+1]
        datas_p = []
        for frames_p in [self.img_set]:
            task_list_p = []

            for idx in range(len(frames_p)):
                img_path_p = frames_p[idx]
                img_p = Image.open(img_path_p).convert('RGB')
                if (self.data_transform is not None) & (idx !=len(frames)-1):
                    img_p = self.data_transform(img_p)
                    img_p = (img_p - mean) /std
                else:
                    img_p = self.data_transform(img_p)

                task_list_p.append(img_p)
            datas_p.append(task_list_p)

        data = {'datas': [datas], 'datas_p': [datas_p],'scene_name': self.scene_name,'item':item}
        return data

def generate_img_test(video_path):

    img_list = []
    imgs_ = os.listdir(video_path)
    for fileName in imgs_:
        if os.path.splitext(fileName)[-1] == '.jpg':
            filename = os.path.splitext(fileName)[0]
            img_list.append(filename)

    img_list.sort(key=int)
    in_list = []
    for filename in img_list:
        file = os.path.join(video_path,filename+'.jpg')
        in_list.append(file)
    return in_list

def generate_note_test(notes_paths,num_pred):

    notes_list = []
    for i in range(len(notes_paths)):
        notes_path = notes_paths[i]
        notes_ = np.load(notes_path)
        notes_list.append(notes_[num_pred:])

    return notes_list

def generate_img_list(video_paths,k_shot,num_pred):
    """
    :param video_path:
    :param k_shot:    k
    :return:
    """
    import glob

    img_list = []
    for i in range(k_shot+1):
        video_path = video_paths[i]
        imgs_list = os.listdir(video_path)

        img_list_ = random.sample(imgs_list, 1)

        img_idx = int(os.path.splitext(img_list_[0])[0])
        if img_idx>(len(imgs_list) - num_pred - 1):
            img_idx = img_idx-num_pred-1

        img_single = []
        for t in range(num_pred+1):
            img00_idx = img_idx+t
            img00_path = os.path.join(video_path,str(img00_idx).zfill(4)+'.jpg')
            img_single.append(img00_path)

        img_list.append(img_single)

    return [img_list]

def setup_dataset_test(file_path,SET,k_shot,num_pred):
    """
    :param file_path:
    :param SET:
    :param k_shot:
    :param num_pred:
    :return:
    """
    file_path_n = os.path.join(file_path, SET, 'norm')
    file_path_p = os.path.join(file_path, SET, 'abnorm')
    file_path_note = os.path.join(file_path, SET, 'notes')

    dir_list = os.listdir(file_path_n)
    datasets_set = []
    img_note_gt_set = []
    for i in range(len(dir_list)):
        dirs0 = dir_list[i]
        files = os.path.join(file_path_n, dirs0)
        dirs = os.listdir(files)


        files_p = os.path.join(file_path_p, dirs0)
        dirs_p = os.listdir(files_p)
        files_note = os.path.join(file_path_note, dirs0)

        data_transform = transforms.Compose([transforms.Resize((256, 384)), transforms.ToTensor()])
        video_name = random.sample(dirs, k_shot + 1)
        video_path = [os.path.join(file_path_n, dirs0, name) for name in video_name]
        k_shot_list = generate_img_list(video_path, k_shot, num_pred)

        video_path_P = [os.path.join(files_p, name) for name in dirs_p]
        video_path_note = [os.path.join(files_note, name) + '.npy' for name in dirs_p]
        img_note_gt = generate_note_test(video_path_note,num_pred)

        datasets = []

        for t in range(len(video_path_P)):
            scene_name = video_path_P[t]
            dataset = Dataset_loader_test(k_shot_list, scene_name, num_pred,data_transform)
            datasets.append(dataset)
        datasets_set.append(datasets)
        img_note_gt_set.append(img_note_gt)

    return datasets_set,img_note_gt_set,k_shot_list