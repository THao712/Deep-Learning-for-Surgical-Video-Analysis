from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import re

def numerical_sort(value):
    numbers=re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

class CholecDataset(Dataset):
    def __init__(self, image_dir, tool_file, phase_anticipation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir), key=numerical_sort)
        # 读取 tool_annotations
        self.tool_dict = {}
        with open(tool_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过表头
                parts = line.strip().split('\t')
                frame = int(parts[0])
                tools = [int(x) for x in parts[1:]]
                self.tool_dict[frame] = tools
        # 读取 phase_anticipation_annotation，每行是一个长字符串
        self.phase_list = []
        with open(phase_anticipation_file, 'r') as f:
            for line in f:
                self.phase_list.append(line.strip())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # phase_anticipation_annotation: 直接用 idx 索引
        phase = self.phase_list[idx] if idx < len(self.phase_list) else ""
        # tool_annotations: Frame=idx*25
        tool = self.tool_dict.get(idx*25, [0]*7)
        tool = torch.tensor(tool, dtype=torch.float32)
        return image, tool, phase

# 测试代码
if __name__ == "__main__":
    dataset = CholecDataset(
        image_dir=r'D:\srtp\data\数据集\code_80\BIMask_ss\1',
        tool_file=r'D:\srtp\data\数据集\code_80\tool_annotations\video01-tool.txt',
        phase_anticipation_file=r'D:\srtp\data\数据集\code_80\phase_anticipation_annotation\video01-phase.txt',
        transform=None
    )
    print(f"数据集样本总数: {len(dataset)}")
    for i in range(4):
        image, tool, phase = dataset[i]
        print(f"样本 {i}:")
        print(f"  图片名: {dataset.image_names[i]}")
        print(f"  tool: {tool}")
        print(f"  phase: {phase}")
        print(f"  图片尺寸: {image.size if hasattr(image, 'size') else 'N/A'}")
        # 可视化图片
        plt.figure(figsize=(3,3))
        plt.imshow(image)
        plt.title(f"样本 {i}: {dataset.image_names[i]}")
        plt.axis('off')
        plt.show()


