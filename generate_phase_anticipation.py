# Modified from https://gitlab.com/nct_tso_public/ins_ant/-/blob/master/train_test_scripts/dataloader.py
import os
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt


# generates the ground truth signal over time for a single phase and a single operation
def generate_anticipation_gt_onephase(phase_code,horizon):
    # initialize ground truth signal
    print(phase_code)
    anticipation = torch.zeros_like(phase_code).type(torch.FloatTensor)  # 创建一个与phase_code相同形状的全零tensor，然后将它的数据类型设置为浮点数FloatTensor
    # default ground truth value is <horizon> minutes
    # (i.e. phase will not appear within next <horizon> minutes)
    anticipation_count = horizon
    # iterate through phase-presence signal backwards
    for i in torch.arange(len(phase_code)-1,-1,-1):  # 反向迭代，start是len-1，表示最后一个元素，end是-1，表示索引取到-1的上一位0，step是-1，表示按1递减
        # if phase is present, then set anticipation value to 0 minutes
        if phase_code[i]:
            anticipation_count = 0
        # else increase anticipation value with each (reverse) time step but clip at <horizon> minutes
        # video is sampled at 1fps, so 1 step = 1/60 minutes
        else:
            anticipation_count = min(horizon, anticipation_count + 1/1500)
        anticipation[i] = anticipation_count
    # normalize ground truth signal to values between 0 and 1
    anticipation = anticipation / horizon
    return anticipation


# generates the ground truth signal over time for a single operation
def generate_anticipation_gt(phases, horizon):
    return torch.stack([generate_anticipation_gt_onephase(phase_code,horizon) for phase_code in phases]).permute(1,0)


def plot_phase_anticipation(save_path, phase_gt, phase_pred=None):
    plt.clf()
    plt.figure(figsize=(30, 2*phase_gt.shape[-1]))

    for i in range(phase_gt.shape[-1]):

        ax1=plt.subplot(phase_gt.shape[-1],1,i+1)
        ax1.plot([x for x in range(len(phase_gt[:,i]))], phase_gt[:,i],color="red",linewidth=1)
        if phase_pred is not None:
            ax1.plot([x for x in range(len(phase_pred[:,i]))], phase_pred[:,i],color="blue",linewidth=1)

        plt.ylabel(str(i))
        plt.yticks([0,0.5,1], ['0','0.5','>1'])

    plt.xlabel("frame")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.savefig(save_path, dpi=120,bbox_inches='tight')


if __name__ == "__main__":

    CHOLEC80 = True
    M2CAI16 = False#同步

    if CHOLEC80:

        annotation_path = './data/cholec80/phase_annotations/'
        save_path = './data/cholec80/phase_anticipation_annotations/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        horizon = 5  # max estimate time (mins)
        phase_dict = {}
        phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                        'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']

        for i in range(len(phase_dict_key)):
            phase_dict[phase_dict_key[i]] = i

        annotation_paths = os.listdir(annotation_path)

        for file_name in annotation_paths:

            with open(annotation_path+file_name, "r") as f:
                phases = []
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)  # 跳过第一行
                for i,row in enumerate(reader):
                    phases.append([1 if int(phase_dict[row[1]])==x else 0 for x in range(7)])  # 把phase转换成one-hot编码，7个数字表示7个阶段，1是当前阶段，0是其他阶段

                phases = torch.LongTensor(phases)  # LongTensor是长整型,用于存储整数，一般为int64
                print(phases.shape)  # torch.Size([43326, 7])，第一个视频的
                # print(phases)
                phases = phases.permute(1,0)  # 交换第一个和第二个维度，改变形状成torch.Size([7, 43326])
                print(phases.shape)  # torch.Size([7, 43326])
                # print(phases)

            target_reg = generate_anticipation_gt(phases, horizon=horizon)  # target_reg.shape:torch.Size([43326, 7])


            np.savetxt(save_path+file_name,target_reg)

            print(target_reg.shape)
            print(target_reg)

            plot_phase_anticipation("data/cholec80/anticipation_output/" + file_name.split(".")[0]+".png",target_reg*horizon)

    if M2CAI16:

        annotation_path = './data/m2cai16/phase_annotations/'
        save_path = './data/m2cai16/phase_anticipation_annotations/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        horizon = 5  # max estimate time (mins)

        phase_dict = {}

        phase_dict_key = ['TrocarPlacement', 'Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                          'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']

        for i in range(len(phase_dict_key)):
            phase_dict[phase_dict_key[i]] = i

        annotation_paths = os.listdir(annotation_path)

        for file_name in annotation_paths:

            with open(annotation_path + file_name, "r") as f:

                phases = []

                reader = csv.reader(f, delimiter='\t')

                next(reader, None)  # 跳过第一行

                for i, row in enumerate(reader):
                    phases.append([1 if int(phase_dict[row[1]]) == x else 0 for x in range(8)])  # 把phase转换成one-hot编码，8个数字表示8个阶段，1是当前阶段，0是其他阶段

                phases = torch.LongTensor(phases)  # LongTensor是长整型,用于存储整数，一般为int64

                print(phases.shape)  # ([76425, 8])，第一个视频的

                # print(phases)

                phases = phases.permute(1, 0)  # 交换第一个和第二个维度，改变形状成torch.Size([7, 43326])

                print(phases.shape)

                # print(phases)

            target_reg = generate_anticipation_gt(phases, horizon=horizon)  # target_reg.shape:torch.Size([43326, 7])
            np.savetxt(save_path+file_name,target_reg)

            print(target_reg.shape)
            print(target_reg)

            plot_phase_anticipation("data/m2cai16/anticipation_output/" + file_name.split(".")[0] + ".png", target_reg * horizon)
