# Deep-Learning-for-Surgical-Video-Analysis
基于深度学习的手术视频分析算法
🧪 Cholec80 手术阶段预测模型训练指南

基于 SegFormer + MS-TCN + Transformer 的端到端手术阶段预测系统 | 完整训练流程 | 可视化结果生成

🌐 项目结构

code_80/
├── bimask_ss_pos/
│   ├── cholec80/
│   │   ├── stage1_32_8_40/       # 第一阶段：SegFormer主干训练
│   │   │   └── embedding1/        # 模型权重与TensorBoard日志
│   │   └── stage2_40_40/          # 第二阶段：时序模型训练
│   │       ├── TeCNO1-2/          # MS-TCN模型参数
│   │       └── TeCNO_t1-2/        # Transformer模型参数
├── data/
│   └── cholec80/
│       ├── phase_anticipation_annotations/  # 标注数据（7阶段倒计时矩阵）
│       └── anticipation_output/             # 预测结果可视化（含GT对比图）
├── train_evp.py                   # 阶段1：SegFormer主干训练
├── finetune_evp.py                # 阶段2：SegFormer微调
├── generate_evp_LFB.py            # 特征提取（生成LFB）
├── tecno.py                       # MS-TCN训练
├── tecno_trans.py                 # Transformer训练
├── trans_SV_output.py             # 预测与结果生成
└── requirements.txt               # 环境依赖

🛠️ 环境配置
bash
激活训练环境（请提前创建）
conda activate my_srtp_env

安装依赖（首次运行需执行）
pip install -r requirements.txt

🔁 完整训练流程（严格按顺序执行）

✅ 阶段1：SegFormer主干网络训练
bash
cd /root/autodl-tmp/data/code_80
nohup python train_evp.py --train 88 --val 88 --work 8 > train_log.txt 2>&1 &
tail -f train_log.txt  # 实时查看日志（Ctrl+C退出）
终止：nvidia-smi → kill [PID]
📌 输出：bimask_ss_pos/cholec80/stage1_32_8_40/embedding1/ 下的模型权重

✅ 阶段2：SegFormer微调
bash
cd /root/autodl-tmp/data/code_80
nohup python finetune_evp.py > finetune_log.txt 2>&1 &
tail -f finetune_log.txt
📌 输出：微调后的SegFormer权重（用于特征提取）

✅ 阶段3：特征提取（生成LFB）
bash
cd /root/autodl-tmp/data/code_80
nohup python generate_evp_LFB.py > generate_LFB_log.txt 2>&1 &
tail -f generate_LFB_log.txt
📌 输出：data/cholec80/ 下的LFB特征文件

✅ 阶段4：MS-TCN训练
bash
cd /root/autodl-tmp/data/code_80
nohup python tecno.py > tecno_log.txt 2>&1 &
tail -f tecno_log.txt
📌 输出：bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/ 下的MS-TCN模型

✅ 阶段5：Transformer训练
bash
cd /root/autodl-tmp/data/code_80
nohup python tecno_trans.py > tecno_trans_log.txt 2>&1 &
tail -f tecno_trans_log.txt
📌 输出：bimask_ss_pos/cholec80/stage2_40_40/TeCNO_t1-2/ 下的Transformer模型

📊 预测与结果生成
bash
cd /root/autodl-tmp/data/code_80
nohup python trans_SV_output.py > trans_SV_output_log.txt 2>&1 &
tail -f trans_SV_output_log.txt
📌 输出：  
- data/cholec80/anticipation_output/：含7子图的预测可视化（X=帧数, Y=倒计时分钟）  
- 红色曲线 = Ground Truth（锯齿状波形）  
- 蓝色曲线 = 模型预测结果

📚 关键文件说明
文件/目录   说明   格式
phase_anticipation_annotations/   标注数据   每行=1帧，7列=7阶段归一化倒计时（0=当前阶段，>0=距开始时间）

anticipation_output/   预测可视化   大图含7子图，红=GT，蓝=预测

TeCNOevp_epoch_*.pth   MS-TCN权重   PyTorch模型文件

TeCNOevp_tran_epoch_.pth   Transformer权重   PyTorch模型文件

🧠 模型加载示例（预测时参考）
python
加载MS-TCN
model = mstcn.MultiStageModel_S(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model.load_state_dict(torch.load('bimask_ss_pos/cholec80/stage2_40_40/TeCNO1-2/TeCNOevp_epoch_15.pth'))

加载Transformer
model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1.load_state_dict(torch.load('bimask_ss_pos/cholec80/stage2_40_40/TeCNO_t1-2/TeCNOevp_trans1_3_5_1_length_30_epoch_1_train_9780_val_9210.pth'))

🔄 训练流程依赖关系
步骤   脚本   输入依赖   输出   作用
1   train_evp.py   原始视频帧   SegFormer权重   主干特征提取器训练

2   finetune_evp.py   阶段1权重   微调SegFormer   优化特征表示

3   generate_evp_LFB.py   阶段2权重   LFB特征文件   生成时序特征库

4   tecno.py   LFB特征   MS-TCN权重   时序建模（多阶段）

5   tecno_trans.py   LFB + MS-TCN权重   Transformer权重   时序优化与平滑

6   trans_SV_output.py   全部权重 + LFB   可视化结果   验证/测试集预测

💡 使用提示
1. 严格顺序执行：前一阶段完成后再启动下一阶段（检查日志确认训练结束）
2. 监控训练：
   bash
   tensorboard --logdir="bimask_ss_pos/cholec80/stage1_32_8_40/embedding1/runs"
   3. 资源管理：
   - 每阶段训练前确认GPU空闲：nvidia-smi
   - 日志文件过大时及时清理：> train_log.txt（清空）
4. 结果验证：  
   检查 anticipation_output/ 中生成的PNG文件，确认红蓝曲线对齐程度

✅ 验证清单
- [ ] 环境已激活 (conda activate my_srtp_env)
- [ ] 阶段1训练完成（检查 embedding1/ 有 .pth 文件）
- [ ] 阶段3生成LFB特征（检查 data/cholec80/ 有特征文件）
- [ ] 阶段6生成可视化结果（检查 anticipation_output/ 有PNG图）
- [ ] TensorBoard可正常查看训练曲线

🌟 提示：完整流程约需24-48小时（依GPU性能）。建议使用nohup后台运行，避免SSH断连中断训练。  
📬 问题反馈：请提供对应阶段的日志片段 + 错误截图  
📜 本指南已通过Cholec80数据集实测验证 | 最后更新：2026-02-08
