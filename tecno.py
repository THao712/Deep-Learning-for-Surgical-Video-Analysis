import torch
from torch import optim
from torch import nn
import numpy as np
from models import mstcn
import pickle, time
import random
from sklearn import metrics
import copy
from torch.utils.tensorboard import SummaryWriter


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.float64)  # (86344, 9)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.float64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.float64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx, \
        test_labels_80, test_num_each_80, test_start_vidx


def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


train_labels_80, train_num_each_80, train_start_vidx, \
    val_labels_80, val_num_each_80, val_start_vidx, \
    test_labels_80, test_num_each_80, test_start_vidx = get_data('pathfiles/cholec80/train_val_paths_labels.pkl')

with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_train.pkl", 'rb') as f:
    g_LFB_train = pickle.load(f)
with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_val.pkl", 'rb') as f:
    g_LFB_val = pickle.load(f)
with open("bimask_ss_pos/cholec80/stage2_40_40/LFB1/evp_LFB_test.pkl", 'rb') as f:
    g_LFB_test = pickle.load(f)

print("load completed")

print("g_LFB_train shape:", g_LFB_train.shape)  # g_LFB_train shape: (86344, 2048)
print("g_LFB_val shape:", g_LFB_val.shape)  # g_LFB_val shape: (21445, 2048)
print("g_LFB_test shape:", g_LFB_test.shape)  # g_LFB_test shape: (98234, 2048)

out_features = 7 * 2
num_workers = 12
batch_size = 1
features_per_seconds = 25
features_subsampling = 1
mstcn_causal_conv = True
learning_rate = 1e-3
min_epochs = 4
max_epochs = 100
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim = 2048
mstcn_stages = 2
horizon = 5

seed = 42
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

weights_train = np.asarray([1.6411019141231247,
                            0.19090963801041133,
                            1.0,
                            0.2502662616859295,
                            1.9176363911137977,
                            0.9840248158200853,
                            2.174635818337618, ])

# weights_train = np.asarray([0.2639588012080849,
#                0.14090451483001626,
#                1.0,
#                0.38530937814605437,
#                0.8951057074266243,
#                0.1306822581894215,
#                0.4749477270967242,
#                0.3697049485015101])

# weights_train = np.asarray([1.43017456,
#                             2.6791701,
#                             0.37750716,
#                             0.97975078,
#                             0.4217459,
#                             2.88874074,
#                             0.7949042,
#                             1.0211039, ])
criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
criterion_reg = nn.SmoothL1Loss()
# criterion_phase = nn.CrossEntropyLoss()

model = mstcn.MultiStageModel_S(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features,
                                mstcn_causal_conv)  # mstcn_f_maps=32， mstcn_f_dim=2048
# model_path = 'evp_best_model/mit_b2/stage2_40_8_40/embedding1/'
# model_name = 'TeCNOevp_epoch_14'
# model.load_state_dict(torch.load(model_path + model_name + '.pth'))
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_accuracy_phase = 0.0
correspond_train_acc_phase = 0.0
best_epoch = 0

train_we_use_start_idx_80 = [x for x in range(40)]
val_we_use_start_idx_80 = [x for x in range(8)]
test_we_use_start_idx_80 = [x for x in range(40)]
# random.shuffle(train_we_use_start_idx_80)

# TensorBoard
writer = SummaryWriter('bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/runs/')

for epoch in range(max_epochs):
    torch.cuda.empty_cache()
    train_idx_80 = []
    model.train()
    train_loss_phase = 0.0
    train_loss_phase_ant = 0.0
    train_corrects_phase = 0
    batch_progress = 0.0
    minibatch_correct_phase = 0.0
    train_start_time = time.time()
    for i in train_we_use_start_idx_80:
        optimizer.zero_grad()
        labels_phase = []
        labels_phase_ant = []
        for j in range(train_start_vidx[i], train_start_vidx[i] + train_num_each_80[i]):
            labels_phase.append(train_labels_80[j][0])
            labels_phase_ant.append(train_labels_80[j][8: 15])
            # labels_phase_ant.append(train_labels_80[j][1: 9])
        labels_phase = torch.LongTensor(np.array(labels_phase))
        labels_phase_ant = torch.Tensor(np.array(labels_phase_ant))
        if use_gpu:
            labels_phase = labels_phase.to(device)
            labels_phase_ant = labels_phase_ant.to(device)
        else:
            labels_phase = labels_phase
            labels_phase_ant = labels_phase_ant
        # print(labels_phase.size())
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = (torch.Tensor(np.array(long_feature))).to(device)  # Tensor:(1, 1734, 2048)
        video_fe = long_feature.transpose(2, 1)  # # Tensor:(1, 2048, 1734)
        # print(long_feature.size())

        # inputs = inputs.view(-1, 3, 224, 224)
        # emb = resnet.forward(inputs).detach()
        # diff = (emb-long_feature[:,-1]).data.cpu().numpy()
        # print(np.sum(np.abs(diff)))
        y_all = model.forward(video_fe)  # Tensor:(2, 1, 14, 1734)
        y_classes = y_all[:, :, :7, :]  # Tensor:(2, 1, 7, 1734)
        y_anticipation = y_all[:, :, 7:, :]
        # y_classes = y_all[:, :, :8, :]  # Tensor:(2, 1, 7, 1734)
        # y_anticipation = y_all[:, :, 8:, :]  # Tensor:(2, 1, 7, 1734)

        stages = y_classes.shape[0]  # y_classes.shape[0]=2, y_classes=Tensor:(2, 1, 8, 3057)
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)  # Tensor:(3057,8)
            ce_loss = criterion_phase(p_classes, labels_phase)
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)

        stages = y_anticipation.shape[0]
        ant_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_anticipation = y_anticipation[j].squeeze().transpose(1, 0)
            ce_loss = criterion_reg(p_anticipation, labels_phase_ant)
            ant_loss += ce_loss
        ant_loss = ant_loss / (stages * 1.0)

        _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)  # _是概率，preds_phase是标签0-6

        loss = clc_loss + ant_loss
        # print(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

        train_loss_phase += clc_loss.data.item()
        train_loss_phase_ant += ant_loss.data.item()

        batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
        train_corrects_phase += batch_corrects_phase
        minibatch_correct_phase += batch_corrects_phase

    train_elapsed_time = time.time() - train_start_time
    train_accuracy_phase = float(train_corrects_phase) / len(train_labels_80)
    train_average_loss_phase = train_loss_phase
    train_average_loss_phase_ant = train_loss_phase_ant
    train_average_loss = train_average_loss_phase + train_average_loss_phase_ant

    writer.add_scalar('training/acc phase', train_accuracy_phase, epoch)
    writer.add_scalar('training/loss phase', train_average_loss_phase, epoch)
    writer.add_scalar('training/loss phase ant', train_average_loss_phase_ant, epoch)

    # Sets the module in evaluation mode.
    model.eval()
    val_loss_phase = 0.0
    val_corrects_phase = 0
    val_start_time = time.time()
    val_progress = 0
    val_all_preds_phase = []
    val_all_labels_phase = []
    val_acc_each_video = []
    in_MAE = []
    pMAE = []
    eMAE = []
    predict_phase_ant_all = []
    gt_phase_ant_all = []

    with torch.no_grad():
        for i in val_we_use_start_idx_80:
            labels_phase = []
            labels_phase_ant = []

            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                labels_phase.append(val_labels_80[j][0])
                labels_phase_ant.append(val_labels_80[j][8: 15])
                # labels_phase_ant.append(val_labels_80[j][1: 9])
            labels_phase = torch.LongTensor(np.array(labels_phase))
            labels_phase_ant = torch.Tensor(np.array(labels_phase_ant))
            if use_gpu:
                labels_phase = labels_phase.to(device)
                labels_phase_ant = labels_phase_ant.to(device)
            else:
                labels_phase = labels_phase
                labels_phase_ant = labels_phase_ant

            long_feature = get_long_feature(start_index=val_start_vidx[i],
                                            lfb=g_LFB_val, LFB_length=val_num_each_80[i])

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_all = model.forward(video_fe)
            y_classes = y_all[:, :, :7, :]  # Tensor:(2, 1, 7, 1734)
            y_anticipation = y_all[:, :, 7:, :]
            # y_classes = y_all[:, :, :8, :]
            # y_anticipation = y_all[:, :, 8:, :]

            stages = y_classes.shape[0]
            clc_loss = 0
            for j in range(stages):  ### make the interuption free stronge the more layers.
                p_classes = y_classes[j].squeeze().transpose(1, 0)
                ce_loss = criterion_phase(p_classes, labels_phase)
                clc_loss += ce_loss
            clc_loss = clc_loss / (stages * 1.0)

            stages = y_anticipation.shape[0]
            ant_loss = 0
            for j in range(stages):  ### make the interuption free stronge the more layers.
                p_anticipation = y_anticipation[j].squeeze().transpose(1, 0)
                ce_loss = criterion_reg(p_anticipation, labels_phase_ant)
                ant_loss += ce_loss
            ant_loss = ant_loss / (stages * 1.0)

            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
            p_classes = y_classes[-1].squeeze().transpose(1, 0)
            loss_phase = criterion_phase(p_classes, labels_phase)

            val_loss_phase += loss_phase.data.item()

            val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / val_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            outputs_phase_ant = y_anticipation[-1].squeeze().transpose(1, 0)
            for j in range(len(outputs_phase_ant)):
                predict_phase_ant_all.append(outputs_phase_ant.data.cpu().numpy()[j])
            for j in range(len(labels_phase_ant)):
                gt_phase_ant_all.append(labels_phase_ant.data.cpu().numpy()[j])

    predict_phase_ant_all = np.array(predict_phase_ant_all).transpose(1, 0)
    gt_phase_ant_all = np.array(gt_phase_ant_all).transpose(1, 0)

    # print(predict_phase_ant_all.shape)
    # print(gt_phase_ant_all.shape)
    # exit()

    for y, t in zip(predict_phase_ant_all, gt_phase_ant_all):

        inside_horizon = (t > 0.0) & (t < 1.0)
        anticipating = (y > 1 * .1) & (y < 1 * .9)
        e_anticipating = (t < 1 * .1) & (t > 0.0)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon] * horizon - t[inside_horizon] * horizon))
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating] * horizon - t[anticipating] * horizon))
        if not np.isnan(pMAE_ins):
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating] * horizon - t[e_anticipating] * horizon))
        if not np.isnan(eMAE_ins):
            eMAE.append(eMAE_ins)

    in_MAE_val = np.mean(in_MAE)
    pMAE_val = np.mean(pMAE)
    eMAE_val = np.mean(eMAE)

    val_elapsed_time = time.time() - val_start_time
    val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
    val_acc_video = np.mean(val_acc_each_video)
    val_average_loss_phase = val_loss_phase

    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
    val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

    writer.add_scalar('validation_epoch/acc phase',
                      float(val_accuracy_phase), epoch)
    writer.add_scalar('validation_epoch/loss phase',
                      float(val_average_loss_phase), epoch)
    writer.add_scalar('validation_epoch/in_MAE phase ant',
                      in_MAE_val,
                      epoch)
    writer.add_scalar('validation_epoch/pMAE ant',
                      pMAE_val,
                      epoch)
    writer.add_scalar('validation_epoch/eMAE phase ant',
                      eMAE_val,
                      epoch)

    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    in_MAE = []
    pMAE = []
    eMAE = []
    predict_phase_ant_all = []
    gt_phase_ant_all = []
    test_start_time = time.time()

    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            labels_phase = []
            labels_phase_ant = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][0])
                labels_phase_ant.append(test_labels_80[j][8: 15])
                # labels_phase_ant.append(test_labels_80[j][1: 9])
            labels_phase = torch.LongTensor(np.array(labels_phase))
            labels_phase_ant = torch.Tensor(np.array(labels_phase_ant))
            if use_gpu:
                labels_phase = labels_phase.to(device)
                labels_phase_ant = labels_phase_ant.to(device)
            else:
                labels_phase = labels_phase
                labels_phase_ant = labels_phase_ant

            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_all = model.forward(video_fe)
            y_classes = y_all[:, :, :7, :]  # Tensor:(2, 1, 7, 1734)
            y_anticipation = y_all[:, :, 7:, :]
            # y_classes = y_all[:, :, :8, :]
            # y_anticipation = y_all[:, :, 8:, :]

            stages = y_classes.shape[0]
            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
            p_classes = y_classes[-1].squeeze().transpose(1, 0)

            test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            outputs_phase_ant = y_anticipation[-1].squeeze().transpose(1, 0)
            for j in range(len(outputs_phase_ant)):
                predict_phase_ant_all.append(outputs_phase_ant.data.cpu().numpy()[j])
            for j in range(len(labels_phase_ant)):
                gt_phase_ant_all.append(labels_phase_ant.data.cpu().numpy()[j])

    test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time

    predict_phase_ant_all = np.array(predict_phase_ant_all).transpose(1, 0)
    gt_phase_ant_all = np.array(gt_phase_ant_all).transpose(1, 0)

    for y, t in zip(predict_phase_ant_all, gt_phase_ant_all):

        inside_horizon = (t > 0.0) & (t < 1.0)
        anticipating = (y > 1 * .1) & (y < 1 * .9)
        e_anticipating = (t < 1 * .1) & (t > 0.0)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon] * horizon - t[inside_horizon] * horizon))
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating] * horizon - t[anticipating] * horizon))
        if not np.isnan(pMAE_ins):
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating] * horizon - t[e_anticipating] * horizon))
        if not np.isnan(eMAE_ins):
            eMAE.append(eMAE_ins)

    in_MAE_test = np.mean(in_MAE)
    pMAE_test = np.mean(pMAE)
    eMAE_test = np.mean(eMAE)

    print('epoch: {:4d}'
          ' train in: {:2.0f}m{:2.0f}s'
          ' train loss(phase): {:4.4f}'
          ' train loss(phase ant): {:4.4f}'
          ' train loss(phase all): {:4.4f}'
          ' train accu(phase): {:.4f}'
          ' valid in: {:2.0f}m{:2.0f}s'
          ' valid loss(phase): {:4.4f}'
          ' valid accu(phase): {:.4f}'
          ' valid accu(video): {:.4f}'
          ' valid in_MAE(phase): {:.4f}'
          ' valid pMAE(phase): {:.4f}'
          ' valid eMAE(phase): {:.4f}'
          ' test in: {:2.0f}m{:2.0f}s'
          ' test accu(phase): {:.4f}'
          ' test accu(video): {:.4f}'
          ' test in_MAE(phase): {:.4f}'
          ' test pMAE(phase): {:.4f}'
          ' test eMAE(phase): {:.4f}'
          .format(epoch,
                  train_elapsed_time // 60,
                  train_elapsed_time % 60,
                  train_average_loss_phase,
                  train_average_loss_phase_ant,
                  train_average_loss,
                  train_accuracy_phase,
                  val_elapsed_time // 60,
                  val_elapsed_time % 60,
                  val_average_loss_phase,
                  val_accuracy_phase,
                  val_acc_video,
                  in_MAE_val,
                  pMAE_val,
                  eMAE_val,
                  test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_accuracy_phase,
                  test_acc_video,
                  in_MAE_test,
                  pMAE_test,
                  eMAE_test))

    print("val_precision_each_phase:", val_precision_each_phase)
    print("val_recall_each_phase:", val_recall_each_phase)
    print("val_precision_phase", val_precision_phase)
    print("val_recall_phase", val_recall_phase)
    print("val_jaccard_phase", val_jaccard_phase)

    if val_accuracy_phase > best_val_accuracy_phase:
        best_val_accuracy_phase = val_accuracy_phase
        best_test_accuracy_phase = test_accuracy_phase
        correspond_train_acc_phase = train_accuracy_phase
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch

    save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
    save_test_phase = int("{:4.0f}".format(best_test_accuracy_phase * 10000))
    save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
    base_name = "TeCNOevp" \
                + "_epoch_" + str(best_epoch)

    base_name_each = "TeCNOevp" \
                     + "_epoch_" + str(epoch)
    torch.save(model.state_dict(), "bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/" + base_name_each + ".pth")
    print("best_epoch", str(best_epoch))


