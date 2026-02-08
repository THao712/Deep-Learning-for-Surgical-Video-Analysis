import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.nn import DataParallel
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from models.mix_transformer_evp import mit_b3_evp, mit_b4_evp, mit_b2_evp,mit_b5_evp
from models.data_process import CholecSegmapDataset, M2caiSegmapDataset,RandomCrop, RandomHorizontalFlip,\
                                RandomRotation, ColorJitter, SeqSampler, get_useful_start_idx
import sys

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=88, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=88, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=50, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--horizon', default=5, type=float, help='horizon time (mins)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=1e-5, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov
horizon = args.horizon

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

# print('number of gpu   : {:6d}'.format(num_gpu))
# print('sequence length : {:6d}'.format(sequence_length))
# print('train batch size: {:6d}'.format(train_batch_size))
# print('valid batch size: {:6d}'.format(val_batch_size))
# print('optimizer choice: {:6d}'.format(optimizer_choice))
# print('multiple optim  : {:6d}'.format(multi_optim))
# print('num of epochs   : {:6d}'.format(epochs))
# print('num of workers  : {:6d}'.format(workers))
# print('test crop type  : {:6d}'.format(crop_type))
# print('whether to flip : {:6d}'.format(use_flip))
# print('learning rate   : {:.4f}'.format(learning_rate))
# print('momentum for sgd: {:.4f}'.format(momentum))
# print('weight decay    : {:.4f}'.format(weight_decay))
# print('dampening       : {:.4f}'.format(dampening))
# print('use nesterov    : {:6d}'.format(use_nesterov))
# print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
# print('step for sgd    : {:6d}'.format(sgd_step))
# print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def get_all_data(data_path, seg_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    with open(seg_path, 'rb') as f:
        train_test_segmap_paths = pickle.load(f)

    # 检查错误
    # print('train_test_paths_labels length:', len(train_test_paths_labels))
    # print('train_test_paths_labels:', train_test_paths_labels)

    # code_80 数据
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]

    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]

    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    # segmap 数据
    train_paths_80_seg = train_test_segmap_paths[0]
    val_paths_80_seg = train_test_segmap_paths[1]

    train_num_each_80_seg = train_test_segmap_paths[2]
    val_num_each_80_seg = train_test_segmap_paths[3]

    test_paths_80_seg = train_test_segmap_paths[4]
    test_num_each_80_seg = train_test_segmap_paths[5]

    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))
    print('test_paths_80  : {:6d}'.format(len(test_paths_80)))
    print('test_labels_80 : {:6d}'.format(len(test_labels_80)))
    print('train_paths_80_seg  : {:6d}'.format(len(train_paths_80_seg)))
    print('valid_paths_80_seg  : {:6d}'.format(len(val_paths_80_seg)))
    print('test_paths_80_seg  : {:6d}'.format(len(test_paths_80_seg)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.float32)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.float32)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.float32)

    if train_labels_80.ndim == 1:
        train_labels_80 = train_labels_80[:, np.newaxis]
    if val_labels_80.ndim == 1:
        val_labels_80 = val_labels_80[:, np.newaxis]
    if test_labels_80.ndim == 1:
        test_labels_80 = test_labels_80[:, np.newaxis]

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])

    train_dataset_80 = CholecSegmapDataset(train_paths_80, train_paths_80_seg, train_labels_80, train_transforms)
    val_dataset_80 = CholecSegmapDataset(val_paths_80, val_paths_80_seg, val_labels_80, test_transforms)
    test_dataset_80 = CholecSegmapDataset(test_paths_80, test_paths_80_seg, test_labels_80, test_transforms)
    # train_dataset_80 = M2caiSegmapDataset(train_paths_80, train_paths_80_seg, train_labels_80, train_transforms)
    # val_dataset_80 = M2caiSegmapDataset(val_paths_80, val_paths_80_seg, val_labels_80, test_transforms)
    # test_dataset_80 = M2caiSegmapDataset(test_paths_80, test_paths_80_seg, test_labels_80, test_transforms)

    return train_dataset_80, train_num_each_80, val_dataset_80, val_num_each_80, test_dataset_80, test_num_each_80


sig_f = nn.Sigmoid()


def finetune_model(train_dataset, train_num_each, test_dataset, test_num_each):
    # TensorBoard
    writer = SummaryWriter('bimask_ss_pos/cholec80/stage2_40_40/embedding1/runs/')

    train_dataset_80, train_num_each_80, test_dataset, test_num_each = train_dataset, train_num_each, test_dataset, test_num_each

    train_useful_start_idx_80 = get_useful_start_idx(sequence_length, train_num_each_80)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_test_we_use = len(test_useful_start_idx)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    test_we_use_start_idx = test_useful_start_idx


    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_test_all = len(test_idx)

    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all test use: {:6d}'.format(num_test_all))

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(test_dataset, test_idx),
        num_workers=workers,
        pin_memory=False
    )

    model = mit_b3_evp()

    # 加载预训练好的segformer
    # pretrained_dict = torch.load('./segformer_ckp/mit_b3.pth')
    # new_state_dict = {}
    # for k, v in pretrained_dict.items():
    #     if 'head' not in k and 'prompt' not in k:
    #         new_state_dict[k] = v

    # 从上次第18个最佳epoch开始训练
    embed_dict = torch.load(
        'bimask_ss_pos/cholec80/stage1_32_8_40/embedding1/evpfc_ce_epoch_39_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_88_train_9945_val_8257_test_8002.pth')

    model.load_state_dict(embed_dict, strict=False)


    # # 冻结主干，只训练head和prompt部分
    for name, param in model.named_parameters():
        if "head" not in name and "prompt" not in name:
            param.requires_grad = False

    if use_gpu:
        # model = DataParallel(model)
        model.to(device)


    # criterion_phase = nn.CrossEntropyLoss(size_average=False, weight=torch.from_numpy(weights_train).float().to(device))
    criterion_phase = nn.CrossEntropyLoss(size_average=False)
    criterion_reg = nn.SmoothL1Loss(size_average=False)

    optimizer = None
    exp_lr_scheduler = None
    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.prompt_generator.parameters(), 'lr': learning_rate},
                {'params': model.head.parameters(), 'lr': learning_rate},
                # {'params': model.parameters(), 'lr': learning_rate}
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.prompt_generator.parameters(), 'lr': learning_rate},
                {'params': model.head.parameters(), 'lr': learning_rate},
                # {'params': model.parameters(), 'lr': learning_rate}
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc_phase = 0.0163#同步
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    # 混合精度训练
    scaler = GradScaler()

    for epoch in range(epochs):
        print("-----epoch: %d-----" % epoch)
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []
        for i in range(num_train_we_use_80):
            for j in range(sequence_length):
                train_idx_80.append(train_we_use_start_idx_80[i] + j)

        train_loader_80 = DataLoader(
            train_dataset_80,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset_80, train_idx_80),
            num_workers=workers,
            pin_memory=False
        )

        # Sets the module in training mode.
        model.train()

        train_loss_phase = 0.0
        train_loss_phase_ant = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        running_loss_phase_ant = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader_80):
            # dataload_time = time.time()
            # data_time = dataload_time - train_start_time
            # print()
            # print('data_time: {:.4f}'.format(data_time))
            optimizer.zero_grad()
            if use_gpu:
                inputs, segmaps, labels_phase, labels_phase_ant = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)  # inputs: Tensor: (100, 3, 224, 224); labels_phase: Tensor: (100, ); labels_phase_ant: Tensor:(100，7)
            else:
                inputs, segmaps, labels_phase, labels_phase_ant = data[0], data[1], data[2], data[3]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
            labels_phase_ant = labels_phase_ant[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            segmaps = segmaps.view(-1, sequence_length, 3, 224, 224)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs_phase, outputs_phase_ant = model.forward(inputs, segmaps)  # outputs_phase: Tensor: (B, 7); outputs_phase_ant: Tensor: (B, 7)

                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]
                outputs_phase_ant = outputs_phase_ant[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)
                loss_phase_ant = criterion_reg(outputs_phase_ant, labels_phase_ant)

            # print(labels_phase_ant.shape)
            # print(labels_phase_ant)
            # print(outputs_phase_ant.shape)
            # print(outputs_phase_ant)

                loss = loss_phase + loss_phase_ant
                # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()
            running_loss_phase_ant += loss_phase_ant.data.item()
            train_loss_phase_ant += loss_phase_ant.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            if i % 15 == 14:

                # ...log the running loss
                batch_iters = epoch * num_train_all / sequence_length + i * train_batch_size / sequence_length

                writer.add_scalar('training/loss phase',
                                  running_loss_phase / (train_batch_size * 500 / sequence_length),
                                  batch_iters)
                writer.add_scalar('training/loss phase ant',
                                  running_loss_phase_ant / (train_batch_size * 500 / sequence_length),
                                  batch_iters)
                # ...log the training acc
                writer.add_scalar('training/acc phase',
                                  float(minibatch_correct_phase) / (float(train_batch_size) * 500 / sequence_length),
                                  batch_iters)


                running_loss_phase = 0.0
                running_loss_phase_ant = 0.0
                minibatch_correct_phase = 0.0

            if (i + 1) * train_batch_size >= num_train_all:
                running_loss_phase = 0.0
                running_loss_phase_ant = 0.0
                minibatch_correct_phase = 0.0

            # train_batch_time = time.time() - dataload_time
            # print()
            # print("batch: {:6d} train_batch_time: {:.4f}".format(i, train_batch_time), end='\n')
            # print()

            batch_progress += 1

            if batch_progress * train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            elif batch_progress % 100 == 0:
                percent = round(batch_progress * train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress * train_batch_size, num_train_all), end='\r')

            # DEBUG
            # break

        # todo
        train_elapsed_time = time.time() - train_start_time
        print("train_elapsed_time: {:.4f}".format(train_elapsed_time))
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * sequence_length
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length
        train_average_loss_phase_ant = train_loss_phase_ant / num_train_all * sequence_length

        model.eval()
        test_loss_phase = 0.0
        test_corrects_phase = 0
        test_start_time = time.time()
        test_progress = 0
        test_all_preds_phase = []
        test_all_labels_phase = []
        predict_phase_ant_all = []
        gt_phase_ant_all = []

        in_MAE = []
        pMAE = []
        eMAE = []

        with torch.no_grad():
            for data in test_loader:
                if use_gpu:
                    inputs, segmaps, labels_phase, labels_phase_ant = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)  # inputs: Tensor: (100, 3, 224, 224); labels_phase: Tensor: (100, ); labels_phase_ant: Tensor:(100，7)
                else:
                    inputs, segmaps, labels_phase, labels_phase_ant = data[0], data[1], data[2], data[3]

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
                labels_phase_ant = labels_phase_ant[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                segmaps = segmaps.view(-1, sequence_length, 3, 224, 224)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs_phase, outputs_phase_ant = model.forward(inputs, segmaps)

                    outputs_phase = outputs_phase[sequence_length - 1::sequence_length]
                    outputs_phase_ant = outputs_phase_ant[sequence_length - 1::sequence_length]
                    _, preds_phase = torch.max(outputs_phase.data, 1)
                    loss_phase = criterion_phase(outputs_phase, labels_phase)

                    test_loss_phase += loss_phase.data.item()

                    test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for i in range(len(preds_phase)):
                    test_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    test_all_labels_phase.append(int(labels_phase.data.cpu()[i]))
                for i in range(len(outputs_phase_ant)):
                    predict_phase_ant_all.append(outputs_phase_ant.data.cpu().numpy()[i])
                for i in range(len(labels_phase_ant)):
                    gt_phase_ant_all.append(labels_phase_ant.data.cpu().numpy()[i])

                test_progress += 1
                if test_progress * val_batch_size >= num_test_all:
                    percent = 100.0
                    print('test progress: %s [%d/%d]' % (str(percent) + '%', num_test_all, num_test_all), end='\n')
                elif test_progress % 100 == 0:
                    percent = round(test_progress * val_batch_size / num_test_all * 100, 2)
                    print('test progress: %s [%d/%d]' % (
                    str(percent) + '%', test_progress * val_batch_size, num_test_all),
                          end='\r')

        test_elapsed_time = time.time() - test_start_time
        test_accuracy_phase = float(test_corrects_phase) / float(num_test_we_use)
        test_average_loss_phase = test_loss_phase / num_test_we_use

        predict_phase_ant_all = np.array(predict_phase_ant_all).transpose(1, 0)
        gt_phase_ant_all = np.array(gt_phase_ant_all).transpose(1, 0)

        for y, t in zip(predict_phase_ant_all, gt_phase_ant_all):

            inside_horizon = (t < 1) & (t > 0)
            anticipating = (y > 1 * .1) & (y < 1 * .9)
            e_anticipating = (t < 1 * .1) & (t > 0)

            in_MAE_ins = np.mean(np.abs(y[inside_horizon] * horizon - t[inside_horizon] * horizon))
            if not np.isnan(in_MAE_ins):
                in_MAE.append(in_MAE_ins)

            pMAE_ins = np.mean(np.abs(y[anticipating] * horizon - t[anticipating] * horizon))
            if not np.isnan(pMAE_ins):
                pMAE.append(pMAE_ins)

            eMAE_ins = np.mean(np.abs(y[e_anticipating] * horizon - t[e_anticipating] * horizon))
            if not np.isnan(eMAE_ins):
                eMAE.append(eMAE_ins)

        in_MAE_m = np.mean(in_MAE)
        pMAE_m = np.mean(pMAE)
        eMAE_m = np.mean(eMAE)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(phase): {:4.4f}'
              ' train accu(phase): {:.4f}'
              ' test in: {:2.0f}m{:2.0f}s'
              ' test loss(phase): {:4.4f}'
              ' test accu(phase): {:.4f}'
              ' test in_MAE(phase): {:.4f}'
              ' test pMAE(phase): {:.4f}'
              ' test eMAE(phase): {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_phase,
                      train_accuracy_phase,
                      test_elapsed_time // 60,
                      test_elapsed_time % 60,
                      test_average_loss_phase,
                      test_accuracy_phase,
                      in_MAE_m,
                      pMAE_m,
                      eMAE_m))

        # --- 【新增】自动停止逻辑 ---
        # 设定您的第一阶段最佳 Loss (Loss1)
        target_train_loss = 0.0163

        # 检查当前 epoch 的训练 loss 是否已经小于这一阶段的目标
        if train_average_loss_phase < target_train_loss:
            print(f"\n[Stop Condition Met] Current train loss ({train_average_loss_phase:.4f}) is lower than Stage 1 best loss ({target_train_loss}). Stopping training.")

            # 保存当前达标的模型权重
            save_test_phase = int("{:4.0f}".format(test_accuracy_phase * 10000))
            save_train_phase = int("{:4.0f}".format(train_accuracy_phase * 10000))

            final_name = "evpfc_ce_STOPPED_EARLY" \
                        + "_epoch_" + str(epoch) \
                        + "_loss_" + str(int(train_average_loss_phase * 10000)) \
                        + "_train_" + str(save_train_phase) \
                        + "_test_" + str(save_test_phase)

            save_path = "bimask_ss_pos/cholec80/stage2_40_40/embedding1/" + final_name + ".pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

            break # 跳出 epoch 循环，结束训练
        # ---------------------------

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(train_average_loss_phase)

        # if train_average_loss_phase < best_train_acc_phase:
        if train_accuracy_phase > correspond_train_acc_phase:
            correspond_train_acc_phase = train_accuracy_phase
            correspond_test_acc_phase = test_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        save_test_phase = int("{:4.0f}".format(correspond_test_acc_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))

        base_name1 = "evpfc_ce" \
                    + "_epoch_" + str(epoch) \
                    + "_length_" + str(sequence_length) \
                    + "_opt_" + str(optimizer_choice) \
                    + "_mulopt_" + str(multi_optim) \
                    + "_flip_" + str(use_flip) \
                    + "_crop_" + str(crop_type) \
                    + "_batch_" + str(train_batch_size) \
                    + "_train_" + str(save_train_phase) \
                    + "_test_" + str(save_test_phase)
        # torch.save(best_model_wts, "m2cai_best_model/mit_b3/stage2_20_7_14/embedding2/" + base_name1 + ".pth")
        print("best_epoch", str(best_epoch))
        # torch.save(model.state_dict(), "cholec_binary/stage1_32_8_40/embedding/" + str(epoch   ) + ".pth")
        torch.save(best_model_wts, "bimask_ss_pos/cholec80/stage2_40_40/embedding1/" + base_name1 + ".pth")
    writer.close()



def main():#同步

    train_dataset_80, train_num_each_80, \
        _, _, \
        test_dataset_80, test_num_each_80 = get_all_data('pathfiles/cholec80/train_val_paths_labels_40_40.pkl',
                                                         'pathfiles/cholec80/bimasks_ss_pos_train_val_test_40_40.pkl')

    finetune_model(
        train_dataset_80,
        train_num_each_80,
        test_dataset_80,
        test_num_each_80)


if __name__ == "__main__":

    main()

