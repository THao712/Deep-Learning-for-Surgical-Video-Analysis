import os
import numpy as np
import pickle

# cholec80 version

root_dir = './data/cholec80/'

segmap_dir = os.path.join(root_dir, 'ss_Bimasks_pos_ep10').replace("\\", "/")
phase_dir = os.path.join(root_dir, 'phase_annotations').replace("\\", "/")

# 获取文件夹名和路径
def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists).replace("\\", "/")
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(x))
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths


# 获取文件夹下所有文件的路径
def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists).replace("\\", "/")
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


segmap_dir_names, segmap_dir_paths = get_dirs(segmap_dir)  # 获取segmap图像文件夹名和路径
phase_file_names, phase_file_paths = get_files(phase_dir)

all_info_all = []

for j in range(len(phase_file_names)):
    downsample_rate = 25
    phase_file = open(phase_file_paths[j])
    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths[j]))[0][5:7])
    video_num_dir = int(os.path.basename(segmap_dir_paths[j]))

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            segmap_file_each_path = os.path.join(segmap_dir_paths[j], phase_split[0] + '.jpg').replace("\\", "/")
            info_each.append(segmap_file_each_path)
            info_all.append(info_each)

    all_info_all.append(info_all)

if not os.path.exists('pathfiles/cholec80'):
    os.makedirs('pathfiles/cholec80')

with open('pathfiles/cholec80/bimasks_ss_pos_cholec80.pkl', 'wb') as f:  # 保存为pkl文件
    pickle.dump(all_info_all, f)

with open('pathfiles/cholec80/bimasks_ss_pos_cholec80.pkl', 'rb') as f:
    all_info_80 = pickle.load(f)  # 读取pkl文件

train_segmap_paths_80 = []
val_segmap_paths_80 = []
test_segmap_paths_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

for i in range(40):
    train_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        train_segmap_paths_80.append(all_info_80[i][j][0])
    print(f"video: {i}, train_num_each_80: {train_num_each_80}")

for i in range(40, 48):
    val_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        val_segmap_paths_80.append(all_info_80[i][j][0])
    print(f"video: {i}, val_num_each_80: {val_num_each_80}")

for i in range(40, 80):
    test_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        test_segmap_paths_80.append(all_info_80[i][j][0])
    print(f"video: {i}, test_num_each_80: {test_num_each_80}")

train_val_test_segmap_paths = []
train_val_test_segmap_paths.append(train_segmap_paths_80)  # 0
train_val_test_segmap_paths.append(val_segmap_paths_80)  # 1

train_val_test_segmap_paths.append(train_num_each_80)  # 2
train_val_test_segmap_paths.append(val_num_each_80)  # 3

train_val_test_segmap_paths.append(test_segmap_paths_80)  # 4
train_val_test_segmap_paths.append(test_num_each_80)  # 5

with open('pathfiles/cholec80/bimasks_ss_pos_train_val_test_40_40.pkl', 'wb') as f:
    pickle.dump(train_val_test_segmap_paths, f)
