from torch.utils.data import DataLoader
from torch.autograd import Variable
from pathlib import Path
from collections import defaultdict
from toronto_utils import *
from model_fcns import GACNet, GACNet_PNA, Res_GACNet
from Config import Hparams as cfg
from Config import start_logger
import logging as logger
import torch
import tqdm
from pytictoc import TicToc


def logMemoryUsage(additional_string="[*]"):
    # device = 'cuda:{}'.format(gpu)
    if torch.cuda.is_available():
        logger.info(additional_string + "Memory {:.0f}MiB max, {:.0f}MiB current".format(
            torch.cuda.max_memory_allocated()/1024/1024, torch.cuda.memory_allocated()/1024/1024
        ))

tr_data, tr_label, val_data, val_label = recognize_all_data(cfg.root, eval=False, split=None)  # training phase so val.

tr_dataset = TorontoDataloader(tr_data, tr_label)
tr_dataloader = DataLoader(tr_dataset, batch_size=16, shuffle=True,)
del tr_data, tr_label, tr_dataset
# val_dataset = TorontoDataloader(val_data, val_label)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)  # shuffle or not?
# del val_data, val_label, val_dataset

'''LOG'''
start_logger(cfg)

# print(cfg.class2label)
seg_label_to_cat = {}
for i,cat in enumerate(cfg.class2label.keys()):
    seg_label_to_cat[i] = cat  # keys:value  label:category  num:str
# print(seg_label_to_cat)

num_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=0)
ratio_samples = num_samples/num_samples.sum()
weights = 1 / (ratio_samples + 0.02)
print('Weights: ', weights)

model_name = cfg.model_name

'''TRAIN'''
if model_name == 'backbone':
    model = GACNet(cfg.num_classes, cfg.alpha)
elif model_name == 'backbone_with_pna':
    model = GACNet_PNA(attention='Mix')  # 'Mix' 'GAC'
elif model_name == 'backbone_with_res':
    model = Res_GACNet(mode='gc')  # 'gc' 'gc_no_s'
elif model_name == 'backbone_with_pna_combine':
    pass                                            # fix

paramCount = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Parameters: {:,}".format(paramCount).replace(",", "'"))
model.cuda()
blue = lambda x: '\033[94m' + x + '\033[0m'

history = defaultdict(lambda: list())
best_acc = 0
best_meaniou = 0
step = 0

#  train net,
init_epoch = 0  # todo: optimize coding
epoch = 40  # todo: adjust
# todo: optimize the ff with argparse
optimizer = 'SGD'
decay_rate = 1e-4

if optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
elif optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=decay_rate)

t = TicToc()
avg_time_per_iter = []

for epoch in range(init_epoch, epoch):
    for i, data in tqdm.tqdm(enumerate(tr_dataloader, 0), total=len(tr_dataloader), smoothing=0.9):
        t.tic() #Start timer
        points, target = data
        # print( 'first', points.shape, target.shape)
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        model = model.train()
        pred = model(points[:, :3, :], points[:, 3:, :])
        pred = pred.contiguous().view(-1, cfg.num_classes)
        target = target.view(-1, 1)[:, 0]
        # print(pred.shape, target.shape)
        loss = torch.nn.functional.nll_loss(pred, target)
        history['loss'].append(loss.cpu().data.numpy())

        if i < 5 and epoch < 5:
            logMemoryUsage()

        loss.backward()
        optimizer.step()
        step += 1
        adjust_learning_rate(optimizer, step)

        iter_time = t.tocvalue() #Time elapsed since t.tic()
        avg_time_per_iter.append(iter_time)

    # if epoch % 10 == 0:
    #     train_metrics, train_hist_acc, cat_mean_iou = test_seg(model, dataloader,seg_label_to_cat)
    #     print('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
    #         epoch, blue('train'), history['loss'][-1], train_metrics['accuracy'],np.mean(cat_mean_iou)))
    #     logger.info('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
    #         epoch, 'train', history['loss'][-1], train_metrics['accuracy'],np.mean(cat_mean_iou)))
    #
    logger.info("sec/iter: {} seconds".format(np.array(avg_time_per_iter).mean()))
    if epoch == 6:
        break

    # test_metrics, test_hist_acc, cat_mean_iou = test_seg(model, val_dataloader, seg_label_to_cat, Config.num_classes)
    # mean_iou = np.mean(cat_mean_iou)

    # print('Epoch %d  %s accuracy: %f  meanIOU: %f' % (epoch, blue('test'), test_metrics['accuracy'],mean_iou))
    # logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (epoch, 'test', test_metrics['accuracy'],mean_iou))
    # if test_metrics['accuracy'] > best_acc:
    #     best_acc = test_metrics['accuracy']
    #     torch.save(model.state_dict(), '%s/GAC_Modified_%.3d_%.4f.pth' % (cfg.checkpoints_dir, epoch, best_acc))
    #     logger.info(cat_mean_iou)
    #     logger.info('Save model..')
    #     print('Save model..')
    #     print(cat_mean_iou)
    # if mean_iou > best_meaniou:
    #     best_meaniou = mean_iou
    # print('Best accuracy is: %.5f'%best_acc)
    # logger.info('Best accuracy is: %.5f'%best_acc)
    # print('Best meanIOU is: %.5f'%best_meaniou)
    # logger.info('Best meanIOU is: %.5f'%best_meaniou)