# Deep-Learning-for-Surgical-Video-Analysis
基于深度学习的手术视频分析算法


```markdown
# 🧠 Surgical Phase Anticipation Pipeline

> 基于 SegFormer + MS-TCN + Transformer 的多阶段手术阶段预测模型训练框架

## 📁 项目结构概览


code_80/
├── train_evp.py            # 第一阶段：主干网络训练
├── finetune_evp.py         # 第二阶段：主干网络微调
├── generate_evp_LFB.py     # 特征提取与存储
├── tecno.py                # MS-TCN 训练
├── tecno_trans.py          # Transformer 训练
└── trans_SV_output.py      # 验证/测试集推理预测

data/cholec80/
├── phase_anticipation_annotations/    # 标注文件
└── anticipation_output/               

bimask_ss_pos/cholec80/     # 训练结果存储
├── stage1_32_8_40/         # 第一阶段结果
├── stage2_40_40/           # 第二阶段结果
│   ├── embedding1/         # SegFormer 参数
│   ├── LFB1/               # 提取的空间特征
│   ├── TeCNO1-2/           # MS-TCN 模型参数
│   ├── TeCNOt1-2/          # Transformer 模型参数
│   └── output/             # 预测结果
└── runs/                   # TensorBoard 日志

```
## 🚀 快速开始

### 1. 环境准备
训练前，切换到我们要用的环境，pycharm连接好服务器后在电脑pycharm软件-终端-服务器终端执行下面的命令，第一次使用需要连接服务器，要连的时候和我（佟）说（几乎用不到，一般调试是在本地跑，用服务器是调试没问题的时候）
my_srtp_env是我们autodl服务器上配好的环境
```bash
conda activate my_srtp_env
```

### 2. 主干网络训练（两阶段）
激活上面的环境后执行下面的命令：
#### 第一阶段训练
```bash
cd /root/autodl-tmp/data/code_80
```
```bash
nohup python train_evp.py > train_log.txt 2>&1 &
```
#### 查看训练日志
```bash
tail -f train_log.txt
```
按Ctrl+c可以退出查看模式

终止服务器上的代码运行：服务器终端执行：
```bash
nvidia-smi
```
查看PID，执行kill PID(数字)

#### 第二阶段微调（所有的命令基本都是在code_80目录下执行，即执行cd /root/autodl-tmp/data/code_80切换到code_80目录）
```bash
nohup python finetune_evp.py > finetune_log.txt 2>&1 &
```
注意要修改： target_train_loss的值为上一阶段训练的最佳epoch的loss值
#### 查看训练日志
```bash
tail -f finetune_log.txt
# 按 Ctrl+C 退出查看
```
#### TensorBoard查看
从服务器把训练结果下载到本地（运行runs文件夹下的内容，选中runs文件夹即可，不要管是文件夹下的哪个文件，目前只能我来做，我会在训练完之后上传到我们的github），执行：
```bash
tensorboard --logdir="D:\srtp\dataset\code_80\bimask_ss_pos\cholec80\stage1_32_8_40\embedding1\runs"
```
（查看其他代码的运行日志时更改runs文件夹的路径即可，每个训练阶段都会有个runs文件夹产生，用来检测损失、准确率等指标的变化，具体路径根据自己电脑上的路径去更改，在本地的环境运行，应该都配好了吧（））

一、二阶段训练是为了训练我们的主干网络即SegFormer的参数，作为特征提取器

### 3. 特征提取
使用训练好的 SegFormer 提取特征，注意也要在服务器的code_80目录下执行：

```bash
nohup python generate_evp_LFB.py > generate_LFB_log.txt 2>&1 &
```
```bash
tail -f generate_LFB_log.txt
```
这个主要是用训练好的SegFormer提取特征并存储，用于后续的MS-TCN，TransFormer模型参数训练、以及用训练好的模型对验证集和测试集作预测

### 4. 时序模型训练
这个文件负责训练MS-TCN的参数：随机初始化参数后，使用generate_evp_LFB提取好的特征进行参数训练与优化，得到训练好的MS-TCN模型，.pth就是它的参数文件

#### MS-TCN 训练
```bash
nohup python tecno.py > tecno_log.txt 2>&1 &
```
```bash
tail -f tecno_log.txt
```

#### Transformer 训练

这个文件是训练Transformer的参数，先用训练好的MS-TCN模型，对generate_evp_LFB提取好的空间特征再提取时序特征，然后与空间特征一起输入Transformer进行参数训练与优化，得到训练好的Transformer模型，.pth为其参数文件

```bash
nohup python tecno_trans.py > tecno_trans_log.txt 2>&1 &
```
```bash
tail -f tecno_trans_log.txt
```

### 5. 推理预测
这个文件是用训练好的模型（MS-TCN和Transformer）对验证集和测试集作预测，得到结果，其输入是generate_evp_LFB.py提取的空间特征、两个模型的参数文件，加载训练好的模型进行预测：

```bash
nohup python trans_SV_output.py > trans_SV_output_log.txt 2>&1 &
```
```bash
tail -f trans_SV_output_log.txt
```
下面是模型加载示例(`trans_SV_output.py)，tecno.py和tecno_trans.py都是类似的
```bash
model = mstcn.MultiStageModel_S(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model_path = 'bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/'
model_name = 'TeCNOevp_epoch_15'
model.load_state_dict(torch.load(model_path + model_name + '.pth'))这段代码负责加载MS-TCN的参数，.pth文件为最好的epoch对应的参数文件

model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1_path = 'bimask_ss_pos/cholec80/stage2_40_40/TeCNO_t1-2/'
model1_name = 'TeCNOevp_trans1_3_5_1_length_30_epoch_1_train_9780_val_9210.pth'  
model1.load_state_dict(torch.load(model1_path + model1_name))这段代码负责加载Transformer模型的参数，.pth文件为最好的epoch对应的参数文件
```
加载好之后就可以执行代码作预测了，目前我们就完成了仇学姐ESIP的复现，后续我们加光流的时候要重来一遍流程，每次训练得到一个最好的epoch后，后续代码的执行都要注意修改参数的加载路径


## 🛠️ 训练流程详解

### 📊 流程概览

```mermaid
graph TD
    A[第一阶段训练<br>SegFormer] --> B[第二阶段微调<br>SegFormer]
    B --> C[特征提取<br>generate_evp_LFB.py]
    C --> D[MS-TCN训练<br>tecno.py]
    D --> E[Transformer训练<br>tecno_trans.py]
    E --> F[推理预测<br>trans_SV_output.py]
    F --> G[输出预测结果]
```

### 一、主干网络训练（SegFormer）

**目标**：训练 SegFormer 作为特征提取器

- **第一阶段**：基础训练
- **第二阶段**：微调优化
- **输出位置**：`bimask_ss_pos/cholec80/stage1_32_8_40/embedding1/`、`bimask_ss_pos/cholec80/stage2_40_40/embedding1/`

### 二、特征提取

**文件**：`generate_evp_LFB.py`

**功能**：使用训练好的 SegFormer 提取视频的空间特征并存储，供后续时序模型使用。

### 三、时序模型训练

#### 1. MS-TCN 训练
- **文件**：`tecno.py`
- **输入**：提取的空间特征
- **输出**：MS-TCN 模型参数（`.pth` 文件）
- **复现阶段输出的最好轮次(epoch)，即最佳参数**：`TeCNOevp_epoch_15.pth`

#### 2. Transformer 训练
- **文件**：`tecno_trans.py`
- **输入**：空间特征 + MS-TCN 模型提取的时序特征（加载tecno.py训练出的参数）
- **输出**：Transformer 模型参数（`.pth` 文件）
- **复现阶段输出的最好轮次(epoch)**：`TeCNOevp_trans1_3_5_1_length_30_epoch_1_train_9780_val_9210.pth`

### 四、推理预测

**文件**：`trans_SV_output.py`

**功能**：加载训练好的 MS-TCN 和 Transformer 模型，对验证集和测试集进行预测，生成最终结果，保存在code_80\bimask_ss_pos\cholec80\stage2_40_40\output中。

## 📂 数据与标注说明

### 标注文件格式
`data/cholec80/phase_anticipation_annotations/`

- **格式**：二维矩阵
- **行**：视频的每一帧（按时间顺序）
- **列**：7个手术阶段
- **数值含义**：
  - `0`：当前正在进行该阶段
  - `>0`：归一化后的倒计时数值（0~1），表示距离该阶段发生的时间，数值越大则离这个阶段越远，可以看一下你们文件夹里的txt文件，在pycharm打开更好看，1表示最远距离，即仇学姐规定的5分钟，我们不用改动

### 输出可视化
`data/cholec80/anticipation_output/`

- **内容**：包含7个子图的大图，每个子图对应一个手术阶段
- **X轴**：帧数（时间）
- **Y轴**：距离下一阶段发生的时间（分钟）
- **红色曲线**：Ground Truth 信号（锯齿状波形）

## 🔧 监控与调试

### TensorBoard 可视化
下面是一个例子
```bash
tensorboard --logdir="D:\srtp\dataset\code_80\bimask_ss_pos\cholec80\stage1_32_8_40\embedding1\runs"
```

> **注意**：路径需根据本地实际情况调整，在本地已配置好的环境中运行（大家应该都配好了吧，不能运行的话就检查一下有没有装tensorboard库，没装的话报错的时候也会提示的）

### 终止训练进程
1. 查看 GPU 进程：服务器终端执行下面的命令
```bash
nvidia-smi
```
2. 执行上述命令后会得到一个方框，找PID，根据显示的 PID 终止进程：
```bash
kill <PID>
```

## 📊 模型文件说明

| 目录/文件 | 说明 |
|-----------|------|
| `stage1_32_8_40/` | 第一阶段训练结果 |
| `stage2_40_40/` | 第二阶段训练结果 |
| `embedding1/` | SegFormer 模型参数 |
| `LFB1/` | 提取的空间特征数据 |
| `TeCNO1-2/` | MS-TCN 模型参数 |
| `TeCNOt1-2/` | Transformer 模型参数 |
| `output/` | 验证集和测试集的预测结果 |

## 🔗 各脚本关系详解

### 1. `tecno.py` - 第一阶段训练器
- **角色**：训练基础的 MS-TCN 模型
- **输入**：预提取的视频特征（LFB features）
- **输出**：MS-TCN 模型权重文件
- **关键点**：这是整个流水线的地基，不依赖其他模型文件

### 2. `tecno_trans.py` - 第二阶段训练器
- **角色**：训练 Transformer 模型
- **输入**：视频特征 + MS-TCN 的输出
- **依赖**：需要先训练好 `tecno.py` 生成的 MS-TCN 模型
- **输出**：Transformer 模型权重文件

### 3. `trans_SV_output.py` - 推理生成器
- **角色**：加载完整模型进行预测
- **输入**：视频特征 + MS-TCN 权重 + Transformer 权重
- **依赖**：需要前两个阶段训练好的模型
- **输出**：最终的预测结果文件

### 4. `get_path_labels.py` 
- **角色**：生成这类文件：train_val_paths_labels_40_40.pkl
- **关键点**：如果要获得train_val_paths_labels_32_8_40.pkl，则要修改代码中的划分方式，下面是40_40.pkl的生成代码，即40个训练集，8个验证集，40个测试集，这里验证集虽然和测试集有重合，但是跑代码的时候忽略掉了，所以不用管
 ```python
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

```

### 5. `get_segmap_path.py` 
- **角色**：生成这类文件：bimasks_ss_pos_train_val_test_40_40.pkl
- **关键点**：如果要获得bimasks_ss_pos_train_val_test_32_8_40.pkl，则要修改代码中的划分方式，下面是40_40.pkl的生成代码，即40个训练集，8个验证集，40个测试集，这里验证集虽然和测试集有重合，但是跑代码的时候忽略掉了，所以不用管
 ```python
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
```

### 6. `visualizer.py` 
- **角色**：from .visualizer import get_local,这种头文件引用如果报错的话，就把visualizer.py放到与代码相同的目录下，比如trans_SV_output.py里面有from .visualizer import get_local这个引用，但是.visualizer下面有红色波浪线，而trans_SV_output.py在code_80目录下，那就把visualizer.py也放到code_80下面


## 📝 代码示例

### 模型加载示例(trans_SV_output.py)，tecno.py和tecno_trans.py都是类似的
```python
# 加载 MS-TCN 参数
model = mstcn.MultiStageModel_S(mstcn_stages, mstcn_layers, mstcn_f_maps, 
                                mstcn_f_dim, out_features, mstcn_causal_conv)
model_path = 'bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/'
model_name = 'TeCNOevp_epoch_15'
model.load_state_dict(torch.load(model_path + model_name + '.pth'))

# 加载 Transformer 参数
model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1_path = 'bimask_ss_pos/cholec80/stage2_40_40/TeCNO_t1-2/'
model1_name = 'TeCNOevp_trans1_3_5_1_length_30_epoch_1_train_9780_val_9210.pth'
model1.load_state_dict(torch.load(model1_path + model1_name))
```

## ⚠️ 注意事项

1. **训练顺序**：必须按照 `主干网络 → 特征提取 → MS-TCN → Transformer → 推理` 的顺序执行
2. **环境配置**：确保在正确的 conda 环境中运行所有命令，运行的时候区分一下本地和服务器，一般来说服务器执行的话是在我这里弄，你们有兴趣的话可以学学怎么连接服务器
3. **路径适配**：TensorBoard 路径和模型加载路径需根据实际环境调整
4. **日志监控**：使用 `tail -f` 命令实时监控训练进度
5. **模型选择**：选择验证集性能最好的 epoch 对应的模型文件进行后续训练和推理


### 服务器清理
删除文件后清空转移到服务器回收站内的文件
```bash
du -sh /root/autodl-tmp/.Trash-0
```

```bash
ls -la /root/autodl-tmp/.Trash-0/files
```

```bash
rm -rf /root/autodl-tmp/.Trash-0/files/*
```
