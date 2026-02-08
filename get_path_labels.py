import os
import numpy as np
import pickle

CHOLEC80 = True
M2CAI16 = False

if CHOLEC80:
    root_dir2 = './data/cholec80/'

    # img_dir2 = os.path.join(root_dir2, 'cutMargin')
    img_dir2 = os.path.join(root_dir2, 'cutMargin')
    phase_dir2 = os.path.join(root_dir2, 'phase_annotations')
    tool_dir2 = os.path.join(root_dir2, 'tool_annotations')
    phase_ant_dir2 = os.path.join(root_dir2, 'phase_anticipation_annotations')

if M2CAI16:
    root_dir2 = './data/m2cai16/'
    img_dir2 = os.path.join(root_dir2, 'cutMargin')
    phase_dir2 = os.path.join(root_dir2, 'phase_annotations')
    phase_ant_dir2 = os.path.join(root_dir2, 'phase_anticipation_annotations')

# train_video_num = 8
# val_video_num = 4
# test_video_num = 2

# print(root_dir)
# print(img_dir)
# print(phase_dir)
# print(tool_dir)
print(root_dir2)
print(img_dir2)
print(phase_dir2)
print(phase_ant_dir2)


# cholec80==================
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):  # listdir列出目录下的文件名
        path = os.path.join(root_dir, lists).replace("\\","/")
        if os.path.isdir(path):  # 检查是否是个路径
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(x))  # 按阿拉伯数字重新排序
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths


def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists).replace("\\","/")
        if not os.path.isdir(path):  # 不是目录，是文件
            file_paths.append(path)
            file_names.append(os.path.basename(path))  # 提取文件名（路径的最后一个部分）
    file_names.sort()  # 重新排序
    file_paths.sort()
    return file_names, file_paths


phase_dict = {}
# cholec80==================
if CHOLEC80:
    img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)  # img_dir_names2:['1','2','3',...,'80'], img_dir_paths2是每个文件夹的路径
    tool_file_names2, tool_file_paths2 = get_files2(tool_dir2)  # tool_file_names2: ['video01-tool.txt',...], tool_file_paths2是每个txt的路径
    phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)  # phase_file_names2: ['video01-phase.txt',...], phase_file_paths2是每个txt的路径
    phase_ant_file_names2, phase_ant_file_paths2 = get_files2(phase_ant_dir2)  # 同上，预测任务的文件名和路径

    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                      'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']



# m2cai16==================
if M2CAI16:
    img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)  # img_dir_names2:['1','2','3',...,'80'], img_dir_paths2是每个文件夹的路径
    phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)  # phase_file_names2: ['video01-phase.txt',...], phase_file_paths2是每个txt的路径
    phase_ant_file_names2, phase_ant_file_paths2 = get_files2(phase_ant_dir2)  # 同上，预测任务的文件名和路径

    phase_dict_key = ['TrocarPlacement', 'Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                      'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']




for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
# cholec80==================

all_info_all2 = []

for j in range(len(phase_file_names2)):
    downsample_rate = 25 
    phase_file = open(phase_file_paths2[j])
    if CHOLEC80:
        tool_file = open(tool_file_paths2[j])
    phase_ant_file = open(phase_ant_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])  # 提取视频编号
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))  # 提取视频编号

    # print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()  # 第一次：phase_split:['Frame', 'Phase']，第二次：['0', 'Preparation']
        if first_line:
            first_line = False  # 存储完标题行就跳过第一行
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg').replace("\\","/")
            info_each.append(img_file_each_path)  # info_each是每张图像的路径和phase的数字label
            info_each.append(phase_dict[phase_split[1]])
            info_all.append(info_each)  # info_all是这个视频里所有图像路径和phase的数字label
    # TODO
    if CHOLEC80:
        count_tool = 0
        first_line = True
        for tool_line in tool_file:
            tool_split = tool_line.split()
            if first_line:
                first_line = False
                continue
            if int(tool_split[0]) % downsample_rate == 0:
                info_all[count_tool].append(int(tool_split[0 + 1]))
                info_all[count_tool].append(int(tool_split[1 + 1]))
                info_all[count_tool].append(int(tool_split[2 + 1]))
                info_all[count_tool].append(int(tool_split[3 + 1]))
                info_all[count_tool].append(int(tool_split[4 + 1]))
                info_all[count_tool].append(int(tool_split[5 + 1]))
                info_all[count_tool].append(int(tool_split[6 + 1]))
                count_tool += 1
            if count_tool == len(info_all) - 1:
                info_all[count_tool].append(int(tool_split[0 + 1]))
                info_all[count_tool].append(int(tool_split[1 + 1]))
                info_all[count_tool].append(int(tool_split[2 + 1]))
                info_all[count_tool].append(int(tool_split[3 + 1]))
                info_all[count_tool].append(int(tool_split[4 + 1]))
                info_all[count_tool].append(int(tool_split[5 + 1]))
                info_all[count_tool].append(int(tool_split[6 + 1]))

    count_phase_ant = 0
    count = 0

    for phase_ant_line in phase_ant_file:
        phase_ant_split = phase_ant_line.split()
        if count % downsample_rate == 0:
            info_all[count_phase_ant].append(float(phase_ant_split[0]))
            info_all[count_phase_ant].append(float(phase_ant_split[1]))
            info_all[count_phase_ant].append(float(phase_ant_split[2]))
            info_all[count_phase_ant].append(float(phase_ant_split[3]))
            info_all[count_phase_ant].append(float(phase_ant_split[4]))
            info_all[count_phase_ant].append(float(phase_ant_split[5]))
            info_all[count_phase_ant].append(float(phase_ant_split[6]))
            if M2CAI16:
                info_all[count_phase_ant].append(float(phase_ant_split[7]))
            count_phase_ant += 1
        count += 1

    all_info_all2.append(info_all)
# cholec80==================

# '''
# with open('./Miccai19.pkl', 'wb') as f:
#     pickle.dump(all_info_all, f)
#
# with open('./Miccai19.pkl', 'rb') as f:
#     all_info_19 = pickle.load(f)
# '''
if CHOLEC80:
    with open('./cholec80.pkl', 'wb') as f:  # 保存为pkl文件
        pickle.dump(all_info_all2, f)

    with open('./cholec80.pkl', 'rb') as f:
        all_info_80 = pickle.load(f)  # 读取pkl文件

    # cholec80==================
    train_file_paths_80 = []
    test_file_paths_80 = []
    val_file_paths_80 = []
    val_labels_80 = []
    train_labels_80 = []
    test_labels_80 = []

    train_num_each_80 = []
    val_num_each_80 = []
    test_num_each_80 = []

    stat = np.zeros(7).astype(int) # 创建7个0值的数组，并把所有元素转换为int
    for i in range(40):
        train_num_each_80.append(len(all_info_80[i]))
        for j in range(len(all_info_80[i])):
            train_file_paths_80.append(all_info_80[i][j][0])
            train_labels_80.append(all_info_80[i][j][1:])
            stat[all_info_80[i][j][1]] += 1  # 统计每个phase的数量

    print(len(train_file_paths_80))
    print(len(train_labels_80))
    print(stat)

    # 验证集: 40-47
    for i in range(40, 48):
        val_num_each_80.append(len(all_info_80[i]))
        for j in range(len(all_info_80[i])):
            val_file_paths_80.append(all_info_80[i][j][0])
            val_labels_80.append(all_info_80[i][j][1:])

    # 测试集: 40-79
    for i in range(40, 80):
        test_num_each_80.append(len(all_info_80[i]))
        for j in range(len(all_info_80[i])):
            test_file_paths_80.append(all_info_80[i][j][0])
            test_labels_80.append(all_info_80[i][j][1:])

    print(len(val_file_paths_80))
    print(len(val_labels_80))

    # cholec80=================


    train_val_test_paths_labels = []
    #train_val_test_paths_labels.append(train_file_paths_19)
    train_val_test_paths_labels.append(train_file_paths_80)  # 0
    train_val_test_paths_labels.append(val_file_paths_80)  # 1

    #train_val_test_paths_labels.append(train_labels_19)
    train_val_test_paths_labels.append(train_labels_80)  # 2
    train_val_test_paths_labels.append(val_labels_80)  # 3

    #train_val_test_paths_labels.append(train_num_each_19)
    train_val_test_paths_labels.append(train_num_each_80)  # 4
    train_val_test_paths_labels.append(val_num_each_80)  # 5


    train_val_test_paths_labels.append(test_file_paths_80)  # 6
    train_val_test_paths_labels.append(test_labels_80)  # 7
    train_val_test_paths_labels.append(test_num_each_80)   # 8


    with open('pathfiles/cholec80/train_val_paths_labels_40_40.pkl', 'wb') as f:
        pickle.dump(train_val_test_paths_labels, f)


if M2CAI16:
    with open('pathfiles/m2cai16/m2cai16_41.pkl', 'wb') as f:  # 保存为pkl文件
        pickle.dump(all_info_all2, f)

    with open('pathfiles/m2cai16/m2cai16_41.pkl', 'rb') as f:
        all_info_16 = pickle.load(f)  # 读取pkl文件

    # m2cai16==================
    train_file_paths_16 = []
    test_file_paths_16 = []
    val_file_paths_16 = []
    val_labels_16 = []
    train_labels_16 = []
    test_labels_16 = []

    train_num_each_16 = []
    val_num_each_16 = []
    test_num_each_16 = []

    stat = np.zeros(8).astype(int)  # 创建7个0值的数组，并把所有元素转换为int
    for i in range(41):
        train_num_each_16.append(len(all_info_16[i]))
        for j in range(len(all_info_16[i])):
            train_file_paths_16.append(all_info_16[i][j][0])
            train_labels_16.append(all_info_16[i][j][1:])
            stat[all_info_16[i][j][1]] += 1  # 统计每个phase的数量

    print(len(train_file_paths_16))
    print(len(train_labels_16))
    print(stat)


    # print(np.max(np.array(train_num_each_80)[:, 0]))  # todo 维度不对啊 ？？？
    # print(np.min(np.array(train_labels_80)[:, 0]))  # todo 维度不对啊 ？？？

    # for i in range(27, 34):
    #     val_num_each_16.append(len(all_info_16[i]))
    #     for j in range(len(all_info_16[i])):
    #         val_file_paths_16.append(all_info_16[i][j][0])
    #         val_labels_16.append(all_info_16[i][j][1:])
    #
    # for i in range(27, 41):
    #     test_num_each_16.append(len(all_info_16[i]))
    #     for j in range(len(all_info_16[i])):
    #         test_file_paths_16.append(all_info_16[i][j][0])
    #         test_labels_16.append(all_info_16[i][j][1:])
    #
    # print(len(val_file_paths_16))
    # print(len(val_labels_16))

    # m2cai16==================

    train_val_test_paths_labels = []
    train_val_test_paths_labels.append(train_file_paths_16)  # 0
    train_val_test_paths_labels.append(val_file_paths_16)  # 1

    train_val_test_paths_labels.append(train_labels_16)  # 2
    train_val_test_paths_labels.append(val_labels_16)  # 3

    # train_val_test_paths_labels.append(train_num_each_19)
    train_val_test_paths_labels.append(train_num_each_16)  # 4
    train_val_test_paths_labels.append(val_num_each_16)  # 5

    train_val_test_paths_labels.append(test_file_paths_16)  # 6
    train_val_test_paths_labels.append(test_labels_16)  # 7
    train_val_test_paths_labels.append(test_num_each_16)  # 8

    with open('pathfiles/m2cai16/train_val_paths_labels_27_7_14.pkl', 'wb') as f:
        pickle.dump(train_val_test_paths_labels, f)

# print('Done')
# print()