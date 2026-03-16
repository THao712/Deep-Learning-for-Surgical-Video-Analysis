import torch.nn as nn
import torch
from torchvision import models
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.nn.functional as F

# from mmseg.ops import resize


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    # if warning:
    #     if size is not None and align_corners:
    #         input_h, input_w = tuple(int(x) for x in input.shape[2:])
    #         output_h, output_w = tuple(int(x) for x in size)
    #         if output_h > input_h or output_w > input_w:
    #             if ((output_h > 1 and output_w > 1 and input_h > 1
    #                  and input_w > 1) and (output_h - 1) % (input_h - 1)
    #                     and (output_w - 1) % (input_w - 1)):
    #                 warnings.warn(
    #                     f'When align_corners={align_corners}, '
    #                     'the output would more aligned if '
    #                     f'input size {(input_h, input_w)} is `x+1` and '
    #                     f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, num_classes):
        super(SegFormerHead, self).__init__()
        self.input_transform = 'multiple_select'
        self.embedding_dim = 2048
        self.embedding_dim1 = 2048
        self.in_index = [0, 1, 2, 3]
        self.align_corners = False
        self.dropout = nn.Dropout2d(0.1)

        self.in_channels = in_channels
        # assert len(feature_strides) == len(in_channels)
        # assert min(feature_strides) == feature_strides[0]
        # self.feature_strides = feature_strides
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        # todo
        # decoder_params = kwargs['decoder_para
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        # self.linear_c = nn.Sequential(nn.Conv2d(self.embedding_dim, self.embedding_dim1, 2, stride=2),
        #                              nn.BatchNorm2d(self.embedding_dim1))

        self.linear_fuse = ConvModule(
            in_channels=self.embedding_dim*4,
            out_channels=self.embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        # resnet, tecno需要的部分
        # resnet = models.resnet50()
        # pretrained_path = "pathfiles/resnet50-0676ba61.pth"
        # state_dict = torch.load(pretrained_path)
        # resnet.load_state_dict(state_dict)
        # self.resnet = resnet
        # self.share = torch.nn.Sequential()
        # # self.share.add_module("conv1", resnet.conv1)
        # # self.share.add_module("bn1", resnet.bn1)
        # # self.share.add_module("relu", resnet.relu)
        # # self.share.add_module("maxpool", resnet.maxpool)
        # # self.share.add_module("layer1", resnet.layer1)
        # self.share.add_module("layer2", resnet.layer2)
        # self.share.add_module("layer3", resnet.layer3)
        # self.share.add_module("layer4", resnet.layer4)
        # self.share.add_module("avgpool", resnet.avgpool)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # self.linear_proj = nn.Linear(self.embedding_dim, self.embedding_dim1)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        self.fc_ant = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 7))

        # self.linear_pred = nn.Conv2d(self.embedding_dim, self.num_classes, kernel_size=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs, return_features=False):
        # 特征融合
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        # c1=Tensor(B,32,56,56), c2=Tensor(B,64,28,28), c3=Tensor(B,160,14,14), c4=Tensor(B,256,7,7)
        ############## MLP decoder on C1-C4 ###########

        n, _, h, w = c4.shape  # _:256, h=7, w=7, n=B

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])  # _c4=Tensor(B,2048,7,7)
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)  # _c4=Tensor(B,2048,7,7)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])  # _c3=Tensor(B,2048,14,14)
        _c3 = resize(_c3, size=c4.size()[2:],mode='bilinear',align_corners=False)  # _c3=Tensor(B,2048,7,7)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])  # _c2=Tensor(B,2048,28,28)
        _c2 = resize(_c2, size=c4.size()[2:],mode='bilinear',align_corners=False)  # _c2=Tensor(B,2048,7,7)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])  # _c1=Tensor(B,2048,56,56)
        _c1 = resize(_c1, size=c4.size()[2:],mode='bilinear',align_corners=False)  # _c1=Tensor(B,2048,7,7)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))  # _c=Tensor(B,2048,7,7)

        # _c = _c.reshape(n, self.embedding_dim1, -1)  # _c=Tensor(B,2048,7,7)
        # _c = self.linear_c(_c)
        x = self.dropout(_c)  # x=Tensor(B,2048,7,7)


        # 转换为序列
        # x = self.share.forward(x)
        x = self.avgpool(x)  # x=Tensor(B,2048,1,1)
        x = torch.flatten(x, 1)  # x=Tensor(B,2048)
        x = x.view(-1, 2048)  # x=Tensor(B,2048)
        
        # 如果需要返回特征向量 (Mode 2: generate_evp_LFB.py)
        if return_features:
            return x

        # 生成预测结果 (Mode 1: train_evp.py)
        y = self.fc(x)  # y=Tensor(B,7)
        y_ant = self.fc_ant(x)  # y_ant=Tensor(B,7)

        return y, y_ant

