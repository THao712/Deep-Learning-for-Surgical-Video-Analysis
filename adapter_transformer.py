import sys
import os

# 1. 保留这段代码：为了能引用父目录 code_80 下的 generate_LFB
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from generate_LFB import device

# 2. 修改这里：
# 错误写法: from transformer2_3_1 import Transformer2_3_1
# 正确写法: 加一个点 (.) 表示引用同级目录下的模块
from .transformer2_3_1 import Transformer2_3_1

import math
from timm.layers import DropPath, to_2tuple, trunc_normal_


sequence_length = 30

# todo 要把train embedding.py的预处理加到前面 改变图像尺寸250->224

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 如果输入的 img_size 是一个 int，则将其转换为一个 tuple
        patch_size = to_2tuple(patch_size)  # 如果输入的 patch_size 是一个 int，则将其转换为一个 tuple

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # H=32, W=32
        self.num_patches = self.H * self.W  # 32*32=1024
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, input_type, freq_nums,
                 handcrafted_tune, embedding_tune, adaptor, img_size):
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()

        if self.handcrafted_tune:
            self.handcrafted_generator = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                                           embed_dim=self.embed_dim // self.scale_factor)

        if self.embedding_tune:
            self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim // self.scale_factor)

        if self.adaptor == 'adaptor':
            self.lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim // self.scale_factor),
                nn.GELU(),
            )
            self.shared_mlp = nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)

        elif self.adaptor == 'fully_shared':
            self.fully_shared_mlp = nn.Sequential(
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim // self.scale_factor),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim),
            )

        elif self.adaptor == 'fully_unshared':
            self.fully_unshared_mlp = nn.Sequential(
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim // self.scale_factor),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_handcrafted(self, x):
        if self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)

        B = x.shape[0]
        handcrafted, H, W = self.handcrafted_generator(x)

        return handcrafted

    def init_prompt(self, embedding_feature, handcrafted_feature):
        if self.embedding_tune:
            embedding_generator = self.embedding_generator
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature

        return handcrafted_feature, embedding_feature

    def get_embedding_feature(self, x):
        if self.embedding_tune:
            embedding_generator = self.embedding_generator
            embedding_feature = embedding_generator(x)

            return embedding_feature
        else:
            return None

    def get_handcrafted_feature(self, x):
        if self.handcrafted_tune:
            handcrafted_generator = self.handcrafted_generator
            handcrafted_feature = handcrafted_generator(x)

            return handcrafted_feature
        else:
            return None

    def get_prompt(self, x, prompt):
        feat = 0
        if self.handcrafted_tune:
            feat += prompt[0]
        if self.embedding_tune:
            feat += prompt[1]

        if self.adaptor == 'adaptor':
            lightweight_mlp = self.lightweight_mlp
            shared_mlp = self.shared_mlp

            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)

        elif self.adaptor == 'fully_shared':
            fully_shared_mlp = self.fully_shared_mlp
            feat = fully_shared_mlp(feat)

        elif self.adaptor == 'fully_unshared':
            fully_unshared_mlp = self.fully_unshared_mlp
            feat = fully_unshared_mlp(feat)

        x = x + feat

        return x

class GaussianFilter(nn.Module):
    def __init__(self):
        super(GaussianFilter, self).__init__()
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def conv_gauss(self, img):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, self.kernel, groups=img.shape[1])
        return out


class Adaptor(nn.Module):
    def __init__(self, embed_dims):
        super(Adaptor, self).__init__()
        self.scale_factor = 4
        self.embed_dims = embed_dims
        self.num_classes = 14
        self.len_q = 30
        self.adaptor = 'adaptor'
        self.lightweight_mlp = nn.Sequential(
                                            nn.Linear(self.embed_dims, self.embed_dims // self.scale_factor),
                                            nn.GELU()
        )
        self.shared_mlp = nn.Linear(self.embed_dims // self.scale_factor, self.embed_dims)

    def lt_forward(self, x, prompt):
        feat = torch.zeros(prompt.shape)
        feat = prompt  # Tensor(1, 2048, 1976)
        feat = torch.transpose(feat, 1, 2)  # Tensor(1, 1976, 2048)
        feat = feat.squeeze(0)  # Tensor(1976, 2048)
        x = x.squeeze(0)  # Tensor(1976, 2048)

        if self.adaptor == 'adaptor':
            feat = self.lightweight_mlp(feat)  # Tensor(1976, 512)
            feat = self.shared_mlp(feat)  # Tensor(1976, 2048)

        x = x + feat  # Tensor(1976, 2048)
        x = x.unsqueeze(0)  # Tensor(1, 1976, 2048)

        return x

    def gt_forward(self, x, prompt):
        feat = torch.zeros(prompt.shape)
        feat = prompt  # Tensor(1976,30,14)
        feat = feat.flatten(1)  # Tensor(1976, 420)
        x = x.flatten(1)  # Tensor(1976, 420)

        if self.adaptor == 'adaptor':
            feat = self.lightweight_mlp(feat)  # Tensor(1976, 105)
            feat = self.shared_mlp(feat)  # Tensor(1976, 420)

        x = x + feat  # Tensor(1976, 2048)
        x = x.view(-1, self.len_q, self.num_classes)  # Tensor(1976, 14, 30)

        return x

    def forward(self, x, prompt):
        y = self.lt_forward(x, prompt)
        return y

class Transformer(nn.Module):
    def __init__(self,
                 mstcn_f_maps,
                 mstcn_f_dim,
                 out_features,
                 len_q,
                 img_size=224,
                 in_chans=3,
                 embed_dim=64, # todo
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(Transformer, self).__init__()
        # Trans_SVNet
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7*2=14
        self.len_q = len_q
        # 控制一次送入注意力的样本数，降低峰值显存。
        self.chunk_size = 256

        # self.adaptor1 = Adaptor(mstcn_f_dim)  # lt的adaptor
        # self.adaptor2 = Adaptor(out_features * len_q)  # gt的adaptor

        # 注意力维度与Mamba隐藏维解耦，避免 d_k=d_v=512 时显存爆炸。
        attn_dim = min(64, self.num_f_maps)
        n_heads = 4
        self.transformer = Transformer2_3_1(
            d_model=out_features,
            d_ff=self.num_f_maps,
            d_k=attn_dim,
            d_v=attn_dim,
            n_layers=1,
            n_heads=n_heads,
            len_q=sequence_length
        )

        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)

    def original_forward(self, x, long_feature):
        out_features = x.transpose(1, 2)  # out_features: Tensor(1, 1976, 14)
        inputs = []
        # x是gt,long_feature是lt

        # 如果当前位置 i 小于 self.len_q - 1，那么就需要在前补0，补到 self.len_q - 1 - i 的位置
        # 如果当前位置 i 大于等于 self.len_q - 1，那么它直接从 out_features 中截取一个长度为 self.len_q 的数据片段，包括当前位置
        for i in range(out_features.size(1)):  # out_feature: Tensor(1, 1976, 14), size(1)=1976
            if i < self.len_q - 1:  # len_q = 30
                input = torch.zeros((1, self.len_q - 1 - i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i + 1]], dim=1)
            else:
                input = out_features[:, i - self.len_q + 1:i + 1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)  # inputs是g~t: Tensor(1976, 30, 14)

        # TODO 这里给feas或者long_feature加上prompt
        feas = torch.tanh(self.fc(long_feature).transpose(0,
                                                     1))  # lt->l~t, feas是l~t: Tensor(1976, 1, 14), lfb_seg是加了adaptor的lt, feas:Tensor(1976, 1, 14)
        output = self.transformer(inputs, feas)  # output: Tensor(1976, 1, 14)
        # output = output.transpose(1,2)
        # output = self.fc(output)

        return output


    # def forward(self, x, long_feature, segmap_lt=None, segmap_gt=None):
    #     out_features = x.transpose(1,2)  # out_features: Tensor(1, 1976, 14)
    #     segmap_gt = segmap_gt.transpose(1,2)
    #     inputs = []
    #     inputs_seg = []
    #
    #     # x是long feature
    #
    #     # todo adaptor 这里的x不是输入的x，这里的x是segMap
    #     # B = x.shape[0]
    #     # outs = []
    #     # if self.handcrafted_tune:
    #     #     handcrafted_feature = self.prompt_generator.init_handcrafted(x)
    #     # else:
    #     #     handcrafted_feature = None
    #
    #     # prompt = self.prompt_generator.init_prompt(long_feature, segmap)
    #     # lfb_seg = self.prompt_generator.get_prompt(long_feature, segmap)
    #
    #
    #     lfb_seg = self.adaptor1.lt_forward(long_feature, segmap_lt)  # lfb_seg是加了adaptor的lt, Tensor(1, 1976, 2048)
    #
    #
    #     # 如果当前位置 i 小于 self.len_q - 1，那么就需要在前补0，补到 self.len_q - 1 - i 的位置
    #     # 如果当前位置 i 大于等于 self.len_q - 1，那么它直接从 out_features 中截取一个长度为 self.len_q 的数据片段，包括当前位置
    #     for i in range(out_features.size(1)):  # out_feature: Tensor(1, 1976, 14), size(1)=1976
    #         if i<self.len_q-1:  # len_q = 30
    #             input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
    #             input_seg = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
    #             input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
    #             input_seg = torch.cat([input_seg, segmap_gt[:, 0:i+1]], dim=1)   # Tensor(1, 31,14
    #         else:
    #             input = out_features[:, i-self.len_q+1:i+1]
    #             input_seg = segmap_gt[:, i-self.len_q+1:i+1]
    #         inputs.append(input)
    #         inputs_seg.append(input_seg)
    #     inputs = torch.stack(inputs, dim=0).squeeze(1)  # inputs是g~t: Tensor(1976, 30, 14)
    #     inputs_seg = torch.stack(inputs_seg, dim=0).squeeze(1)  # inputs_seg: Tensor(1976, 30, 14)
    #
    #     inputs_adaptor = self.adaptor2.gt_forward(inputs, inputs_seg)  # inputs_adaptor: Tensor(1976, 14, 30)
    #
    #
    #     # TODO 这里给feas或者long_feature加上prompt
    #     # feas = torch.tanh(self.fc(long_feature).transpose(0,1))
    #     feas_adaptor = torch.tanh(self.fc(lfb_seg).transpose(0,1))  #lt->l~t, feas是l~t: Tensor(1976, 1, 14), lfb_seg是加了adaptor的lt, feas:Tensor(1976, 1, 14)
    #
    #     output = self.transformer(inputs_adaptor, feas_adaptor)  # output: Tensor(1976, 1, 14)
    #     #output = output.transpose(1,2)
    #     #output = self.fc(output)
    #
    #     return output


