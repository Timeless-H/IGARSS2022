from datetime import datetime
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
import math
import numpy as np
import h5py
import pandas as pd
import plyfile
from Config import Hparams as Config


def recognize_all_data(rtdir, eval=False, split=None):
    if not eval:
        tr_files = getDataFiles(os.path.join(rtdir, 'train_data_files.txt'), rtdir)
        data_batch_list = []
        label_batch_list = []
        for h5_filename in tr_files:
            temp = h5_filename.split('/', 2)[-1]
            data_batch, label_batch = loadDataFile(os.path.join(rtdir, temp))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        train_data = np.concatenate(data_batch_list, axis=0)
        train_label = np.concatenate(label_batch_list, axis=0)
        print('train_data:', train_data.shape, 'train_label:', train_label.shape)
    else:
        train_data, train_label = [], []

    if split == 'test' or split == 'val':
        print('split under process: {}'.format(split))
        test_files = getDataFiles(f"{rtdir}/{split}_data_files.txt", split=split)
        data_batch_list = []
        label_batch_list = []
        for h5_filename in test_files:
            temp = h5_filename.split('/', 1)[-1]
            data_batch, label_batch = loadDataFile(os.path.join(rtdir, temp))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        test_data = np.concatenate(data_batch_list, axis=0)
        test_label = np.concatenate(label_batch_list, axis=0)
        print('test_data:', test_data.shape, 'test_label:', test_label.shape, np.unique(test_label))
    else:
        test_data, test_label = [], []
    return train_data, train_label, test_data, test_label


def getDataFiles(list_filename, split=None):
    if split == 'test' or split == 'val':
        level1_filelist = [line.rstrip() for line in open(list_filename)]
        level1_filelist = [file for file in level1_filelist if file.split('_', 3)[1] == 'zero']
    else:
        level0 = [line.rstrip() for line in open(list_filename)]
        rtdir = list_filename.rsplit('/', 1)[0]
        level1_filelist = []
        for path in level0:
            thepath = os.path.join(rtdir, (path.lstrip('.')).lstrip('/'))
            level1_temp = [line.rstrip() for line in open(thepath)]
            # if [True for level1_file in level1_filelist if level1_temp == level1_file]:
            #     continue
            level1_filelist.extend(level1_temp)
        # level1_filelist = [file for file in level1_filelist if file.split('_', 3)[1] == 'zero']
    return level1_filelist


def loadDataFile(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label_seg'][:]
    return (data, label)


class TorontoDataloader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 30 every 20000 steps"""
    lr = Config.learning_rate * (0.45 ** (step // 20000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_iou(pred,target,iou_tabel=None):
    ious = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            iou = intersection/union
            ious.append(iou)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
    return np.mean(ious), iou_tabel


def test_seg(model, loader, catdict, num_classes):
    import torch
    from collections import defaultdict
    from tqdm import tqdm
    from torch.autograd import Variable
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    model.eval()
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    with torch.no_grad():
        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            points, target = Variable(points.float()), Variable(target.long())
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred = model(points[:,:3,:],points[:,3:,:])
            mean_iou, iou_tabel = compute_iou(pred,target,iou_tabel)
            pred = pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
            metrics['iou'].append(mean_iou)
        iou_tabel[:,2] = iou_tabel[:,0] /(iou_tabel[:,1]+0.01)
        hist_acc += metrics['accuracy']
        metrics['accuracy'] = np.mean(metrics['accuracy'])
        iou_tabel = pd.DataFrame(iou_tabel, columns=['iou','count','mean_iou'])
        iou_tabel['Category_IOU'] = [cat_value for cat_value in catdict.values()]
        cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou