import os.path
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
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import sys
import jpeg4py as jpeg
import cv2 # 新增：用于处理光流的缩放
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda:0")
sequence_length = 30

#同步
def pil_loader(path, mode='RGB'):
    try:
        # pkl_path = path.replace('jpg', 'pkl').replace('cutMargin', 'cutMargin/pklimgs')
        # if os.path.exists(pkl_path):
        #     with open(pkl_path, 'rb') as f:
        #         return pickle.load(f)
        # else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                # img_rgb = img.convert('RGB')
                img_converted = img.convert(mode)
                return img_converted

    except Exception as e:
        print(f"Error loading image at path {path}: {e}")
        raise



class RandomCrop(object):
    # 随机裁剪图像，它可以指定裁剪的大小和填充值
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):
        # Handle Tensor (for Optical Flow)
        if isinstance(img, torch.Tensor):
            if self.padding > 0:
                # Pad (left, right, top, bottom)
                img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), value=0)

            h, w = img.shape[-2], img.shape[-1]
            th, tw = self.size
            if w == tw and h == th:
                return img

            # Sync logic: Re-use seed logic
            random.seed(self.count // sequence_length)
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            self.count += 1

            return img[..., y1:y1 + th, x1:x1 + tw]

        # Existing PIL logic
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    # 随机水平翻转图像
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1

        if prob < 0.5:
            # Handle Tensor (for Optical Flow)
            if isinstance(img, torch.Tensor):
                # Flip width dimension (last dimension)
                img = img.flip(-1)
                # If flow (2 channels), negate u component (channel 0)
                if img.shape[0] == 2:
                    img[0] = -img[0]
                return img

            # Existing PIL logic
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    # 随机旋转图像
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)

        # Handle Tensor (for Optical Flow)
        if isinstance(img, torch.Tensor):
            # Rotate image grid
            img = TF.rotate(img, angle)
            # If flow, rotate vectors
            if img.shape[0] == 2:
                # Convert angle to radians (negative because Y-axis is down in images usually,
                # or match TF direction. TF.rotate +angle is CCW)
                # If we rotate image CCW, the vector (u,v) should also rotate CCW.
                # Standard rotation matrix for CCW theta:
                # x' = x cos - y sin
                # y' = x sin + y cos
                rad = math.radians(angle)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)

                u, v = img[0].clone(), img[1].clone()
                img[0] = u * cos_a - v * sin_a
                img[1] = u * sin_a + v * cos_a
            return img

        # Existing PIL logic
        return TF.rotate(img, angle)


class ColorJitter(object):
    # 对图像进行颜色增强操作，包括亮度、对比度、饱和度和色相的调整
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_


class SeqSampler(Sampler):
    # 序列采样sampler
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class CholecDataset(Dataset):
    # 创建自定义的数据集，用于训练和验证深度学习模型
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 8: 15]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)


class CholecSegmapDataset(Dataset):
    def __init__(self, file_paths, seg_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.seg_paths = seg_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 8: 15]
        self.transform = transform
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        seg_names = self.seg_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]
        imgs = self.loader(img_names, mode='RGB')  # 保持图像为RGB模式
        segmaps = self.loader(seg_names, mode='RGB')  # 强制将segmaps加载为灰度图像,'L'
        # imgs = jpeg.JPEG(imgs_names).decode()
        # segmaps = jpeg.JPEG(segmaps_names).decode()

        if self.transform is not None:
            imgs = self.transform(imgs)
            segmaps = self.transform(segmaps)
        # segmaps = self.resize(segmaps)
        # segmaps = self.to_tensor(segmaps)
        # imgs_transform_path = os.path.join(os.path.dirname(img_names), os.path.basename(img_names)).replace('cutMargin','trans_imgs')
        # segmaps_transform_path = os.path.join(os.path.dirname(seg_names), os.path.basename(seg_names)).replace('SegMap', 'trans_segmaps')
        #
        # # with open(imgs_transform_path, 'rb') as f:
        # #     f.seek(0)
        # #     imgs = pickle.load(f)
        # #     f.close()
        # #
        # # with open(segmaps_transform_path, 'rb') as f:
        # #     f.seek(0)
        # #     segmaps = pickle.load(f)
        # #     f.close()
        #
        # imgs = jpeg.JPEG(imgs_transform_path).decode().transpose(2, 0, 1)
        # segmaps = jpeg.JPEG(segmaps_transform_path).decode().transpose(2, 0, 1)
        # imgs = transforms.ToTensor()(imgs)
        # segmaps = transforms.ToTensor()(segmaps)


        return imgs, segmaps, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)

class M2caiSegmapDataset(Dataset):
    def __init__(self, file_paths, seg_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.seg_paths = seg_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 1: 9]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        seg_names = self.seg_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]
        imgs = self.loader(img_names, mode='RGB')  # 保持图像为RGB模式
        segmaps = self.loader(seg_names, mode='RGB')  # 强制将segmaps加载为灰度图像,'L'
        # imgs = jpeg.JPEG(imgs_names).decode()
        # segmaps = jpeg.JPEG(segmaps_names).decode()

        if self.transform is not None:
            imgs = self.transform(imgs)
            segmaps = self.transform(segmaps)


        return imgs, segmaps, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_useful_start_idx_LFB(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


class CholecSegmapDataset1(Dataset):
    def __init__(self, file_paths, seg_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.seg_paths = seg_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 8: 15]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        seg_names = self.seg_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]

        imgs = self.loader(img_names)
        segmaps = self.loader(seg_names)

        imgs_transform_path = os.path.join(os.path.dirname(img_names), os.path.basename(img_names)).replace('cutMargin', 'trans_imgs_pkl').replace("\\", "/").replace("jpg", "pkl")
        segmaps_transform_path = os.path.join(os.path.dirname(seg_names), os.path.basename(seg_names)).replace('SegMap', 'trans_segmaps_pkl').replace("\\", "/").replace("jpg", "pkl")


        os.makedirs(os.path.dirname(imgs_transform_path), exist_ok=True)
        os.makedirs(os.path.dirname(segmaps_transform_path), exist_ok=True)
        if os.path.exists(imgs_transform_path) and os.path.getsize(imgs_transform_path) > 0:
            with open(imgs_transform_path, 'rb') as f:
                f.seek(0)
                imgs = pickle.load(f)
                f.close()
        else:
            os.makedirs(os.path.dirname(imgs_transform_path), exist_ok=True)
            if self.transform is not None:
                imgs = self.transform(imgs)
                with open(imgs_transform_path, 'wb') as f:
                    pickle.dump(imgs, f)
                    f.close()
        #     segmaps = self.transform(segmaps)
        #
        #     imgs_trans = imgs.permute(1,2,0).numpy()
        #     imgs_trans = Image.fromarray((imgs_trans*255).astype('uint8'))
        #     segmaps_trans = segmaps.permute(1,2,0).numpy()
        #     segmaps_trans = Image.fromarray((segmaps_trans*255).astype('uint8'))
        #
        #     # 保存替换后的图像
        #     imgs_trans.save(imgs_transform_path)
        #     segmaps_trans.save(segmaps_transform_path)
        #


        if os.path.exists(segmaps_transform_path) and os.path.getsize(segmaps_transform_path) > 0:
            with open(segmaps_transform_path, 'rb') as f:
                f.seek(0)
                segmaps = pickle.load(f)
                f.close()
        else:
            os.makedirs(os.path.dirname(segmaps_transform_path), exist_ok=True)
            if self.transform is not None:
                segmaps = self.transform(segmaps)

            with open(segmaps_transform_path, 'wb') as f:
                pickle.dump(segmaps, f)
                f.close()

        return imgs, segmaps, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)


class CholecFlowDataset(Dataset):
    # 新增类：在 CholectSegmapDataset 基础上加载光流 (.npy)
    # 返回：imgs, segmaps, flow, labels_phase, labels_phase_ant
    def __init__(self, file_paths, seg_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.seg_paths = seg_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 8: 15]
        self.transform = transform
        self.loader = loader
        # [修改] 目标中间尺寸调整为 (250, 250)，与 transforms.Resize((250, 250)) 保持一致
        self.target_size = (250, 250)

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        seg_names = self.seg_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]

        # 1. 加载图像和分割图
        imgs = self.loader(img_names, mode='RGB')
        segmaps = self.loader(seg_names, mode='RGB') # 保持 RGB 模式

        # 2. 加载光流
        # 假设路径结构：.../cutMargin/video_id/frame_id.jpg
        # 光流路径：.../raft_flow_npy/video_id/frame_id.npy
        flow_path = img_names.replace('cutMargin', 'raft_flow_npy').replace('.jpg', '.npy')

        if os.path.exists(flow_path):
            flow = np.load(flow_path) # (H, W, 2) dtype=float32
        else:
            # 容错处理：如果找不到光流文件（虽然应该都有），生成全0光流
            w, h = imgs.size
            flow = np.zeros((h, w, 2), dtype=np.float32)

        # 3. 处理光流：缩放尺寸 + 缩放数值
        # 获取原始尺寸 (H, W)
        h_origin, w_origin = flow.shape[:2]

        # 使用 cv2.resize 调整尺寸 (注意 cv2 是 (width, height))
        flow_resized = cv2.resize(flow, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 计算缩放比例
        scale_x = self.target_size[0] / w_origin
        scale_y = self.target_size[1] / h_origin

        # 调整数值：位移量也需要按比例缩放
        flow_resized[:, :, 0] *= scale_x
        flow_resized[:, :, 1] *= scale_y

        # 转为 Tensor: (H, W, 2) -> (2, H, W)
        flow_tensor = torch.from_numpy(flow_resized).permute(2, 0, 1).float()

        # 4. 应用 Transform
        # 同步增强：必须对 flow 应用与 imgs/segmaps 相同的几何变换
        if self.transform is not None:
            # 这里的 transform 通常是一个 Compose
            # 我们需要遍历它，对 flow 只应用几何变换（RandomCrop, Flip, Rotation）
            # 注意：ColorJitter 等不应应用于 flow

            # Application to Images
            imgs = self.transform(imgs)

            # Application to Segmaps (mask)
            # 注意：如果 transform 包含 ColorJitter，这其实也会改变 mask 的值，
            # 但原始 CholecSegmapDataset 就是这样写的，为了保持一致性我们先维持原样。
            segmaps = self.transform(segmaps)

            # Application to Flow using filtered transforms
            # 我们必须手动调用 transform 中的组件，或者假设 transform 是 Compose
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    apply_to_flow = False
                    # [修改] 增加对 torchvision 原生 Crop 的支持
                    if isinstance(t, (RandomCrop, RandomHorizontalFlip, RandomRotation,
                                      transforms.CenterCrop, transforms.RandomCrop, transforms.RandomHorizontalFlip)):
                        apply_to_flow = True
                    elif isinstance(t, transforms.Resize):
                        # 我们已经在上面 resize 过了，且手动处理了数值缩放。
                        # 如果 pipeline 里的 resize 尺寸与 self.target_size 一致，则相当于 Identity，可以应用。
                        # 鉴于我们 hardcode 了 250，这里允许 resize。
                        apply_to_flow = True

                    if apply_to_flow:
                        flow_tensor = t(flow_tensor)
            else:
                # 如果不是 Compose，可能就是单个 transform，简单判断
                if isinstance(self.transform, (RandomCrop, RandomHorizontalFlip, RandomRotation,
                                               transforms.CenterCrop, transforms.RandomCrop, transforms.RandomHorizontalFlip)):
                    flow_tensor = self.transform(flow_tensor)

        return imgs, segmaps, flow_tensor, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)


img_array = (np.random.rand(250, 250, 3) * 255).astype(np.uint8)

# 将 NumPy 数组转换为 PIL Image 格式
img_pil = Image.fromarray(img_array)

class CholecNoiseDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:, 0]
        self.file_labels_phase_ant = file_labels[:, 8: 15]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        labels_phase_ant = self.file_labels_phase_ant[index]
        imgs = self.loader(img_names)
        rnmaps = (np.random.rand(250, 250, 3) * 255).astype(np.uint8)
        rnmaps = Image.fromarray(rnmaps)

        if self.transform is not None:
            imgs = self.transform(imgs)
            segmaps = self.transform(rnmaps)

        return imgs, rnmaps, labels_phase.astype(np.int64), labels_phase_ant.astype(np.float64)

    def __len__(self):
        return len(self.file_paths)