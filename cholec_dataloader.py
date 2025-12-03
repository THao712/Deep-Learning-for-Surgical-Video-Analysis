import cholec_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

train_dataset = cholec_dataset.CholecDataset(
        image_dir=r'D:\srtp\data\数据集\code_80\ss_Bimasks_pos_ep10\1',
        tool_file=r'D:\srtp\data\数据集\code_80\tool_annotations\video01-tool.txt',
        phase_anticipation_file=r'D:\srtp\data\数据集\code_80\phase_anticipation_annotation\video01-phase.txt',
        transform=transforms.ToTensor())
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
print(f"数据集样本总数: {len(train_dataset)}")
train_features, train_tools, train_phases = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Tools batch shape: {train_tools.size()}")
print(f"Phases batch size: {len(train_phases)}")
img = train_features[0]
tool = train_tools[0]
phase = train_phases[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print(f"Tool: {tool}")
print(f"Phase: {phase}")