import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from models.mix_transformer_evp import mit_b2_evp, mit_b5_evp, mit_b3_evp, mit_b4_evp, mit_b1_evp
from models.data_process import CholecSegmapDataset,M2caiSegmapDataset, RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, CholecDataset,\
                                SeqSampler, get_useful_start_idx, get_useful_start_idx_LFB
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
#from NLBlock import NLBlockimport os, subprocess
import os, subprocess
#os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=200, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=200, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-7, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=1e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--LFB_l', default=40, type=int, help='long term feature bank length')
parser.add_argument('--load_LFB', default=False, type=bool, help='whether load exist long term feature bank')

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

LFB_length = args.LFB_l
load_exist_LFB = args.load_LFB

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

print('number of gpu   : {:6d}'.format(num_gpu))
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


def pil_loader(path):
    # 加载图像文件，输入图像文件的路径 path，并返回一个PIL图像对象，将其转换为RGB格式
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = resnet
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        #self.lstm = nn.Linear(2048, 7)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        #self.dropout = nn.Dropout(p=0.2)
        #self.relu = nn.ReLU()

        #init.xavier_normal_(self.lstm.weight)
        #init.xavier_normal_(self.lstm.all_weights[0][1])
        #init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.resnet.forward(x)
        # x = self.share.forward(x)  # x=Tensor(batchsize, 2048, 1, 1)
        x = x.view(-1, 2048)  # x=Tensor(batchsize, 2048)
        #self.lstm.flatten_parameters()
        #y = self.relu(self.lstm(x))
        #y = y.contiguous().view(-1, 256)
        #y = self.dropout(y)
        #x = self.fc(x)
        return x


def get_long_feature(start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []
    for j in range(len(start_index_list)):
        long_feature_each = []

        # 上一个存在feature的index
        last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]

        for k in range(LFB_length):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])

        long_feature.append(long_feature_each)
    return long_feature


def get_all_data(data_path, seg_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    with open(seg_path, 'rb') as f:
        train_test_segmap_paths = pickle.load(f)

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

    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    # 维度修正：强制将 1D 数组 (N,) 转换为 2D 数组 (N, 1)
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
    train_dataset_80_LFB = CholecSegmapDataset(train_paths_80, train_paths_80_seg, train_labels_80, test_transforms)
    val_dataset_80 = CholecSegmapDataset(val_paths_80, val_paths_80_seg, val_labels_80, test_transforms)
    test_dataset_80 = CholecSegmapDataset(test_paths_80, test_paths_80_seg, test_labels_80, test_transforms)
    # train_dataset_80 = M2caiSegmapDataset(train_paths_80, train_paths_80_seg, train_labels_80, train_transforms)
    # train_dataset_80_LFB = M2caiSegmapDataset(train_paths_80, train_paths_80_seg, train_labels_80, test_transforms)
    # val_dataset_80 = M2caiSegmapDataset(val_paths_80, val_paths_80_seg, val_labels_80, test_transforms)
    # test_dataset_80 = M2caiSegmapDataset(test_paths_80, test_paths_80_seg, test_labels_80, test_transforms)

    return (train_dataset_80, train_dataset_80_LFB), train_num_each_80, val_dataset_80, val_num_each_80, test_dataset_80, test_num_each_80


sig_f = nn.Sigmoid()

# Long Term Feature bank

g_LFB_train = np.zeros(shape=(0, 2048))
g_LFB_val = np.zeros(shape=(0, 2048))
g_LFB_test = np.zeros(shape=(0, 2048))


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    # TensorBoard
    # writer = SummaryWriter('runs/non-local/pretrained_lr5e-7_L40_2fc_copy/')

    (train_num_each_80), \
    (val_dataset, test_dataset), \
    (val_num_each, test_num_each) = train_num_each, val_dataset, val_num_each

    (train_dataset_80, train_dataset_80_LFB) = train_dataset

    train_useful_start_idx_80 = get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    train_useful_start_idx_80_LFB = get_useful_start_idx_LFB(sequence_length, train_num_each_80)
    val_useful_start_idx_LFB = get_useful_start_idx_LFB(sequence_length, val_num_each)
    test_useful_start_idx_LFB = get_useful_start_idx_LFB(sequence_length, test_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)
    num_test_we_use = len(test_useful_start_idx)

    num_train_we_use_80_LFB = len(train_useful_start_idx_80_LFB)
    num_val_we_use_LFB = len(val_useful_start_idx_LFB)
    num_test_we_use_LFB = len(test_useful_start_idx_LFB)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx
    test_we_use_start_idx = test_useful_start_idx

    train_we_use_start_idx_80_LFB = train_useful_start_idx_80_LFB
    val_we_use_start_idx_LFB = val_useful_start_idx_LFB
    test_we_use_start_idx_LFB = test_useful_start_idx_LFB

    #    np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    train_idx_LFB = []
    for i in range(num_train_we_use_80_LFB):
        for j in range(sequence_length):
            train_idx_LFB.append(train_we_use_start_idx_80_LFB[i] + j)

    val_idx_LFB = []
    for i in range(num_val_we_use_LFB):
        for j in range(sequence_length):
            val_idx_LFB.append(val_we_use_start_idx_LFB[i] + j)

    test_idx_LFB = []
    for i in range(num_test_we_use_LFB):
        for j in range(sequence_length):
            test_idx_LFB.append(test_we_use_start_idx_LFB[i] + j)

    # 使用字典推导式替代 zip 解包，防止验证集为空时报错
    dict_train_start_idx_LFB = {value: index for index, value in enumerate(train_we_use_start_idx_80_LFB)}
    dict_val_start_idx_LFB = {value: index for index, value in enumerate(val_we_use_start_idx_LFB)}

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    num_test_all = len(test_idx)

    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))
    print('num of all test use: {:6d}'.format(num_test_all))
    print('num of all train LFB use: {:6d}'.format(len(train_idx_LFB)))
    print('num of all valid LFB use: {:6d}'.format(len(val_idx_LFB)))


    global g_LFB_train
    global g_LFB_val
    global g_LFB_test
    print("loading features!>.........")

    if not load_exist_LFB:

        train_feature_loader = DataLoader(
            train_dataset_80_LFB,
            batch_size=val_batch_size,
            sampler=SeqSampler(train_dataset_80_LFB, train_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )
        val_feature_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=SeqSampler(val_dataset, val_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )

        test_feature_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            sampler=SeqSampler(test_dataset, test_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )

        model_LFB = mit_b3_evp()
        model_LFB.load_state_dict(torch.load(
            "bimask_ss_pos/cholec80/stage2_40_40/embedding1/evpfc_ce_STOPPED_EARLY_epoch_15_loss_162_train_9945_test_8068.pth"))


#可能需要改成：实验权重/bimask_ss_pos/cholec80/stage2_40_40/embedding1/evpfc_ce_epoch_9_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_88_train_9919_test_8116.pth"))



        def get_parameter_number(net):
            total_num = sum(p.numel() for p in net.parameters())
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            # print('Total: {}, Trainable: {}'.format(total_num, trainable_num))
            return trainable_num

        total_papa_num = 0
        total_papa_num += get_parameter_number(model_LFB)

        if use_gpu:
            model_LFB = DataParallel(model_LFB)
            model_LFB.to(device)

        for params in model_LFB.parameters():
            params.requires_grad = False

        model_LFB.eval()

        with torch.no_grad():

            for data in train_feature_loader:
                if use_gpu:
                    inputs, segmaps, labels_phase = data[0].to(device), data[1].to(device), data[2].to(device)  # inputs: Tensor: (100, 3, 224, 224); labels_phase: Tensor: (100, ); labels_phase_ant: Tensor:(100，7)
                else:
                    inputs, segmaps, labels_phase = data[0], data[1], data[2]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)  # inputs: Tensor(400, 1, 3, 224, 224)
                segmaps = segmaps.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs, segmaps).data.cpu().numpy()
                # outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()  # output feature = lt， ndarray=(batchsize,2048)

                g_LFB_train = np.concatenate((g_LFB_train, outputs_feature), axis=0)


                print("train feature length:", len(g_LFB_train))

            for data in val_feature_loader:
                if use_gpu:
                    inputs, segmaps, labels_phase = data[0].to(device), data[1].to(device), data[2].to(device)  # inputs: Tensor: (100, 3, 224, 224); labels_phase: Tensor: (100, ); labels_phase_ant: Tensor:(100，7)
                else:
                    inputs, segmaps, labels_phase = data[0], data[1], data[2]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)  # inputs: Tensor(400, 1, 3, 224, 224)
                segmaps = segmaps.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs, segmaps).data.cpu().numpy()

                g_LFB_val = np.concatenate((g_LFB_val, outputs_feature), axis=0)


                print("val feature length:", len(g_LFB_val))

            for data in test_feature_loader:
                if use_gpu:
                    inputs, segmaps, labels_phase = data[0].to(device), data[1].to(device), data[2].to(device)  # inputs: Tensor: (100, 3, 224, 224); labels_phase: Tensor: (100, ); labels_phase_ant: Tensor:(100，7)
                else:
                    inputs, segmaps, labels_phase = data[0], data[1], data[2]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)  # inputs: Tensor(400, 1, 3, 224, 224)
                segmaps = segmaps.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs, segmaps).data.cpu().numpy()

                g_LFB_test = np.concatenate((g_LFB_test, outputs_feature), axis=0)

                print("test feature length:", len(g_LFB_test))

        print("finish!")
        g_LFB_train = np.array(g_LFB_train)
        g_LFB_val = np.array(g_LFB_val)
        g_LFB_test = np.array(g_LFB_test)
        #'''

        # 自动创建保存目录
        save_dir = "bimask_ss_pos/cholec80/stage2_40_40/LFB1"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")

        with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_train.pkl", 'wb') as f:
            pickle.dump(g_LFB_train, f)

        with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_val.pkl", 'wb') as f:
            pickle.dump(g_LFB_val, f)
        #''
        with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_test.pkl", 'wb') as f:
            pickle.dump(g_LFB_test, f)

def main():
    train_dataset_80, train_num_each_80, \
        val_dataset_80, val_num_each_80, \
        test_dataset_80, test_num_each_80 = get_all_data('pathfiles/cholec80/train_val_paths_labels_40_40.pkl',
                                                         'pathfiles/cholec80/bimasks_ss_pos_train_val_test_40_40.pkl')
    train_model(
        (train_dataset_80),
        (train_num_each_80),
        (val_dataset_80, test_dataset_80),
        (val_num_each_80, test_num_each_80))


if __name__ == "__main__":
    main()

# print('Done')