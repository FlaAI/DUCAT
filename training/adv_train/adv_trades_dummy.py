import time

import torch.optim
import torch.nn.functional as F

from adv_lib.trades import generate_trades
from training import _kl_div
from utils.utils import AverageMeter

import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary=None, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['adv'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        DUMMY_NUM_CLASS = 20
        if P.dataset == 'cifar100':
            DUMMY_NUM_CLASS = 200
        if P.dataset == 'tinyimagenet':
            DUMMY_NUM_CLASS = 400

        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        images_adv = generate_trades(model, images, distance=P.distance,
                                     eps_iter=P.alpha, eps=P.epsilon, nb_iter=P.n_iters,
                                     clip_min=0, clip_max=1)

        # --- reconstruct training data and labels with dummy classes --- #

        if epoch >= P.dummy_start and epoch <= P.dummy_end:
            labels_oh = F.one_hot(labels, num_classes=DUMMY_NUM_CLASS)
            labels_oh_dummy = F.one_hot(labels + int(DUMMY_NUM_CLASS / 2), num_classes=DUMMY_NUM_CLASS)

            benign_labels_noh = P.w_benign * labels_oh_dummy + (1 - P.w_benign) * labels_oh
            adv_labels_noh = P.w_adv * labels_oh_dummy + (1 - P.w_adv) * labels_oh

            data_dummy = torch.cat((images, images_adv))
            labels_dummy = torch.cat((benign_labels_noh, adv_labels_noh))

        else:
            data_dummy = images_adv
            labels_dummy = labels

        outputs_dummy = model(data_dummy)
        loss_dummy = criterion(outputs_dummy, labels_dummy)

        # outputs = model(images)
        outputs_adv = model(images_adv)

        # loss_ce = F.cross_entropy(outputs, labels)
        loss_adv = P.beta * _kl_div(outputs_adv, model(images))

        # loss = loss_ce + loss_adv
        loss = loss_dummy + loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_dummy.item(), batch_size)
        losses['adv'].update(loss_adv.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossAdv %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['adv'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossAdv %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['adv'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_adversary', losses['adv'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
