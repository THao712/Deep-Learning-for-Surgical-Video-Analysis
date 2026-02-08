from visualizer import get_local
get_local.activate()
import cv2
import torchvision.models as models
from models.mix_transformer_evp import mit_b3_evp
import torch
import math
import os
import matplotlib
from PIL import Image, ImageOps
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn.functional as F
from models.data_process import CholecSegmapDataset, M2caiSegmapDataset,RandomCrop, RandomHorizontalFlip,\
                                RandomRotation, ColorJitter, SeqSampler, get_useful_start_idx

import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


img_path = 'data/cholec80/cutMargin/1/3650.jpg'
seg_path = 'data/cholec80/BiMasks_ss/1/3650.jpg'
device = torch.device("cuda:0")

with open(img_path, 'rb') as f:
    with Image.open(f) as img:
        img_converted = img.convert('RGB')

with open(seg_path, 'rb') as f:
    with Image.open(f) as img:
        seg_converted = img.convert('RGB')

train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

inputs = train_transforms(img_converted)
segmaps = train_transforms(seg_converted)
class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
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
        self.fc_ant = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        #self.dropout = nn.Dropout(p=0.2)
        #self.relu = nn.ReLU()

        #init.xavier_normal_(self.lstm.weight)
        #init.xavier_normal_(self.lstm.all_weights[0][1])
        #init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        #self.lstm.flatten_parameters()
        #y = self.relu(self.lstm(x))
        #y = y.contiguous().view(-1, 256)
        #y = self.dropout(y)
        y = self.fc(x)
        y_ant = self.fc_ant(x)
        return y, y_ant

# model = mit_b3_evp()
# embed_dict = torch.load(
#         'ss_bimask/stage2_40_40/embedding1/evpfc_ce_epoch_3_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_88_train_9838_test_8131.pth')
model = resnet_lstm()
embed_dict = torch.load('transsvnet/resnetfc_ce_epoch_18_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_100_train_9969_val_8395_test_8124.pth')

model.load_state_dict(embed_dict, strict=False)
model.to(device)

# # 冻结主干，只训练head和prompt部分
# for name, param in model.named_parameters():
#     if "head" not in name and "prompt" not in name:
#         param.requires_grad = False

inputs = inputs.view(-1, 1, 3, 224, 224)
segmaps = segmaps.view(-1, 1, 3, 224, 224)
with torch.no_grad():
    inputs = inputs.to(device)
    segmaps = segmaps.to(device)
    # outputs_phase, outputs_phase_ant = model.forward(inputs, segmaps)
    outputs_phase, outputs_phase_ant = model.forward(inputs)

cache = get_local.cache
attention_maps = cache['Attention.forward']
print(len(attention_maps))  # 28
# for i in range(len(attention_maps)):
#     print('attention_maps[{}].shape: {}'.format(i, attention_maps[i].shape))
# print(attention_maps[0].shape)  # torch.Size(1, 1,3136,49)
""""
attention_maps[0].shape: (1, 1, 3136, 49)
attention_maps[1].shape: (1, 1, 3136, 49)
attention_maps[2].shape: (1, 1, 3136, 49)
attention_maps[3].shape: (1, 2, 784, 49)
attention_maps[4].shape: (1, 2, 784, 49)
attention_maps[5].shape: (1, 2, 784, 49)
attention_maps[6].shape: (1, 2, 784, 49)
attention_maps[7].shape: (1, 5, 196, 49)
attention_maps[8].shape: (1, 5, 196, 49)
attention_maps[9].shape: (1, 5, 196, 49)
attention_maps[10].shape: (1, 5, 196, 49)
attention_maps[11].shape: (1, 5, 196, 49)
attention_maps[12].shape: (1, 5, 196, 49)
attention_maps[13].shape: (1, 5, 196, 49)
attention_maps[14].shape: (1, 5, 196, 49)
attention_maps[15].shape: (1, 5, 196, 49)
attention_maps[16].shape: (1, 5, 196, 49)
attention_maps[17].shape: (1, 5, 196, 49)
attention_maps[18].shape: (1, 5, 196, 49)
attention_maps[19].shape: (1, 5, 196, 49)
attention_maps[20].shape: (1, 5, 196, 49)
attention_maps[21].shape: (1, 5, 196, 49)
attention_maps[22].shape: (1, 5, 196, 49)
attention_maps[23].shape: (1, 5, 196, 49)
attention_maps[24].shape: (1, 5, 196, 49)
attention_maps[25].shape: (1, 8, 49, 49)
attention_maps[26].shape: (1, 8, 49, 49)
attention_maps[27].shape: (1, 8, 49, 49)
"""
# attn = attention_maps[27]
# # visualize_grid_to_grid_with_cls(attention_maps[4][0,4,:,:], 105,img_converted)
# H = int(math.sqrt(attn.shape[2]))
# C = int(attn.shape[3])
# num_layers = len(attention_maps)
# depths=[3, 4, 18, 3]
# num_heads=[1, 2, 5, 8]
# attn = torch.tensor(attn).permute(0,1,3,2).reshape(1,1,343,343) # (1,49,56,56)
#
# attn.detach().numpy()
# att1 = F.interpolate(attn,size=[224,224],mode='bilinear')
# plt.imshow(img_converted)  # 设置plt可视化图层为原图
# plt.imshow(att1.squeeze().cpu().numpy(), alpha=0.4, cmap='rainbow')  # 这行将attention图叠加显示，透明度0.4
# plt.show()
# plt.axis('off')  # 关闭坐标轴

# visualize_grid_to_grid(attention_maps[27][0, 0, :, :], 0, img_converted)

patch_sizes = {
    3136: 4,  # 56x56 patches
    784: 8,   # 28x28 patches
    196: 16,  # 14x14 patches
    49: 32    # 7x7 patches
}
def visualize_attention_maps(img, attention_maps):
    # img = T.ToPILImage()(img)  # Convert the image tensor to PIL Image
    img = np.array(img)  # Convert the PIL Image to numpy array

    fig, axs = plt.subplots(1, len(attention_maps), figsize=(20, 10))

    for i, att_map in enumerate(attention_maps):
        att_map = att_map.squeeze(0)
        num_heads, hw, p = att_map.shape

        patch_size = patch_sizes[hw]  # Determine patch size based on the number of patches
        att_map = att_map.mean(0).reshape(int(np.sqrt(hw)), int(np.sqrt(hw)),p)

        att_map = att_map.mean(2)
        att_map = att_map / att_map.max()  # Normalize the attention map
        att_map = np.kron(att_map, np.ones((patch_size, patch_size)))  # Upsample to match image size

        axs[i].imshow(img)
        axs[i].imshow(att_map, cmap='rainbow', alpha=0.5)  # Overlay attention map on the image
        axs[i].set_title(f'Attention Map {i + 1}')
        axs[i].axis('off')

    plt.show()


def visualize_attention_maps1(img, attention_maps):
    img = np.array(img)  # Convert the image tensor to numpy array

    fig, axs = plt.subplots(1, len(attention_maps), figsize=(20, 10))

    for i, att_map in enumerate(attention_maps):
        att_map = att_map.squeeze(0)
        num_heads, hw, p = att_map.shape

        patch_size = patch_sizes[hw]  # Determine patch size based on the number of patches

        combined_att_map = np.zeros((int(np.sqrt(hw)) * patch_size, int(np.sqrt(hw)) * patch_size))

        for head in range(num_heads):
            head_map = att_map[head].reshape(int(np.sqrt(hw)), int(np.sqrt(hw)), p)

            for j in range(p):
                map_upsampled = np.kron(head_map[:, :, j], np.ones((patch_size, patch_size)))
                map_upsampled = map_upsampled / map_upsampled.max()  # Normalize the attention map

                combined_att_map += map_upsampled

        combined_att_map = combined_att_map / num_heads  # Normalize combined attention map

        axs[i].imshow(img)
        axs[i].imshow(combined_att_map, cmap='rainbow', alpha=0.5)  # Overlay attention map on the image
        axs[i].set_title(f'Attention Map {i + 1}')
        axs[i].axis('off')

    plt.show()


def visualize_and_save_attention_maps(img, attention_maps, save_dir='transsvnet/attention_maps'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.array(img)  # Convert the image tensor to numpy array

    for i, att_map in enumerate(attention_maps):
        att_map = att_map.squeeze(0)
        num_heads, hw, p = att_map.shape

        patch_size = patch_sizes[hw]  # Determine patch size based on the number of patches

        for head in range(num_heads):
            for feature_dim in range(p):
                head_map = att_map[head].reshape(int(np.sqrt(hw)), int(np.sqrt(hw)), p)
                map_upsampled = np.kron(head_map[:, :, feature_dim], np.ones((patch_size, patch_size)))
                map_upsampled = map_upsampled / map_upsampled.max()  # Normalize the attention map

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img)
                ax.imshow(map_upsampled, cmap='rainbow', alpha=0.5)  # Overlay attention map on the image
                ax.set_title(f'Attention Map {i + 1} - Head {head + 1} - Feature {feature_dim + 1}')
                ax.axis('off')

                save_path = os.path.join(save_dir, f'att_map_{i + 1}_head_{head + 1}_feature_{feature_dim + 1}.png')
                plt.savefig(save_path)
                plt.close(fig)


def visualize_and_save_attention_maps1(img, attention_maps, save_dir='ss_bimask/stage2_40_40/output/attention_maps1'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.array(img)  # Convert the image tensor to numpy array

    for i, att_map in enumerate(attention_maps):
        att_map = att_map.squeeze(0)
        num_heads, hw, p = att_map.shape

        patch_size = patch_sizes[hw]  # Determine patch size based on the number of patches

        combined_att_map = np.zeros((int(np.sqrt(hw)) * patch_size, int(np.sqrt(hw)) * patch_size))

        for head in range(num_heads):
            head_map = att_map[head].reshape(int(np.sqrt(hw)), int(np.sqrt(hw)), p)

            for j in range(p):
                map_upsampled = np.kron(head_map[:, :, j], np.ones((patch_size, patch_size)))
                map_upsampled = map_upsampled / map_upsampled.max()  # Normalize the attention map

                combined_att_map += map_upsampled

        combined_att_map = combined_att_map / num_heads  # Normalize combined attention map

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.imshow(combined_att_map, cmap='rainbow', alpha=0.5)  # Overlay attention map on the image
        ax.set_title(f'Attention Map {i + 1}')
        ax.axis('off')

        save_path = os.path.join(save_dir, f'att_map_{i + 1}.png')
        plt.savefig(save_path)
        plt.close(fig)

visualize_and_save_attention_maps(img_converted, attention_maps)