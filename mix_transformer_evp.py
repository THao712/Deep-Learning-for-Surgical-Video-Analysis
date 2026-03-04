import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
# from mmseg.models.builder import BACKBONES
# from mmseg.utils import get_root_logger
# from mmcv.runner import load_checkpoint
from .segformer_head import SegFormerHead
import math
from functools import reduce
from operator import mul

from torchvision import transforms

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果没有指定 out_features，则默认等于 in_features
        hidden_features = hidden_features or in_features  # 如果没有指定 hidden_features，则默认等于 in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)  # 对模型的参数进行初始化

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

from visualizer import get_local

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    @get_local('attn')
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    HFC tune
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 如果输入的 img_size 是一个 int，则将其转换为一个 tuple
        patch_size = to_2tuple(patch_size)  # 如果输入的 patch_size 是一个 int，则将其转换为一个 tuple

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
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


class MixVisionTransformerEVP(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=14, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.head = SegFormerHead(embed_dims, num_classes)

        self.apply(self._init_weights)

        # vpt config
        self.scale_factor = 4
        self.prompt_type = 'highpass'
        self.tuning_stage = str(1234)
        self.input_type = 'gaussian'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dims, self.tuning_stage, self.depths,
                                                self.input_type, self.freq_nums,
                                                self.handcrafted_tune, self.embedding_tune, self.adaptor,
                                                img_size)

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

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, y, flow=None):
        # x is image, y is segmap, flow is optical flow
        x = x.view(-1, 3, 224, 224)
        y = y.view(-1, 3, 224, 224)

        if flow is not None:
             # Ensure flow is (B, 2, 224, 224). Just in case.
             # If using DataLoader from CholecFlowDataset, it should be correct.
             flow = flow.view(-1, 2, 224, 224)

        B = x.shape[0]
        outs = []

        # Call init_prompts instead of init_handcrafted
        if self.handcrafted_tune:
            # This returns tuple of fused prompts for 4 stages
            handcrafted_prompts = self.prompt_generator.init_prompts(y, flow)
            # handcrafted1, handcrafted2, handcrafted3, handcrafted4 = handcrafted_prompts
            # Unpacking will happen inside loop or by indexing if we change structure
            # Original code unpacked them:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = handcrafted_prompts
        else:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None

        # stage 1
        x, H, W = self.patch_embed1(x)  # x=Tensor(B,3136,32), H=56, W=56
        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)  # prompt=tuple(2): (FusedPrompt, EmbPrompt)
        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)  # x=Tensor(B,3136,32)
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # x=Tensor(B,32,56,56)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        if '2' in self.tuning_stage:
            prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        if '3' in self.tuning_stage:
            prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt3, 3, i)
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        if '4' in self.tuning_stage:
            prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            if '4' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)  # outs: list(4): Tensor(B,32,56,56), Tensor(B,64,28,28), Tensor(B,160,14,14), Tensor(B,256,7,7)

        return outs

    def forward(self, x, y, flow=None):
        x = self.forward_features(x, y, flow)  # x=list(4): Tensor(B,32,56,56), Tensor(B,64,28,28), Tensor(B,160,14,14), Tensor(B,256,7,7)
        x = self.head(x)

        return x



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)




device = "cuda" if torch.cuda.is_available() else "cpu"

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


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,)
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 2 / 4, -4 / 4, 2 / 4, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1 / 2, -2 / 2, 1 / 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        self.srm_layer.weight.data = torch.Tensor(
            [[filter1, filter1, filter1],
             [filter2, filter2, filter2],
             [filter3, filter3, filter3]]
        )

        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def conv_srm(self, img):
        out = self.srm_layer(img)
        return out


class PromptGenerator(nn.Module):
    # todo 看他的结构
    def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor
        self.img_size = img_size # Store img_size for later use

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()
        if self.input_type == 'srm':
            self.srm_filter = SRMFilter()
        if self.input_type == 'all':
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)
        if self.input_type == 'bimask':
            self.bimask_pos_embed = nn.Parameter(torch.zeros(3, img_size, img_size))

        # ============================================================
        # New Components for Optical Flow and Fusion
        # ============================================================
        self.flow_tune = True # Enable flow tuning by default or based on config

        # 1. Flow Encoders (Similar to handcrafted_generator for Segmaps)
        # Input channels = 2 for Optical Flow
        if self.flow_tune:
            if '1' in self.tuning_stage:
                self.flow_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=2,
                                                        embed_dim=self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.flow_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[0] // self.scale_factor,
                                                       embed_dim=self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.flow_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[1] // self.scale_factor,
                                                       embed_dim=self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.flow_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[2] // self.scale_factor,
                                                       embed_dim=self.embed_dims[3] // self.scale_factor)

        # 2. Unification Adapters (Seg & Flow) & 3. Video Projectors
        # Target Dim for Unification is (embed_dim // scale) / 2

        for i, stage in enumerate(['1', '2', '3', '4']):
            if stage in self.tuning_stage:
                dim_original = self.embed_dims[i]
                dim_prompt = dim_original // self.scale_factor
                dim_unified = dim_prompt // 2

                # [新增] Linear Projection Layer for Segmentation
                # "随后该向量被输入至一个可训练的线性投影层"
                seg_projector = nn.Linear(dim_prompt, dim_prompt)
                setattr(self, f'seg_projector{stage}', seg_projector)

                # [新增] Linear Projection Layer for Optical Flow
                # "然后通过可训练的线性投影层将初级特征向量投影至本研究所需的特征维度"
                flow_projector = nn.Linear(dim_prompt, dim_prompt)
                setattr(self, f'flow_projector{stage}', flow_projector)

                # Adapter for Segmentation: dim_prompt -> dim_unified
                # "适配器架构为两个多层感知机(MLP)，激活函数选择GELU"
                seg_adapter = nn.Sequential(
                    Mlp(in_features=dim_prompt, hidden_features=dim_prompt, out_features=dim_unified, act_layer=nn.GELU),
                    Mlp(in_features=dim_unified, hidden_features=dim_unified, out_features=dim_unified, act_layer=nn.GELU)
                )
                setattr(self, f'seg_unification_adapter{stage}', seg_adapter)

                # Adapter for Flow: dim_prompt -> dim_unified
                flow_adapter = nn.Sequential(
                    Mlp(in_features=dim_prompt, hidden_features=dim_prompt, out_features=dim_unified, act_layer=nn.GELU),
                    Mlp(in_features=dim_unified, hidden_features=dim_unified, out_features=dim_unified, act_layer=nn.GELU)
                )
                setattr(self, f'flow_unification_adapter{stage}', flow_adapter)

                # Video Frame Projector: dim_original -> dim_prompt (to match concatenated dimensions)
                video_projector = nn.Linear(dim_original, dim_prompt)
                setattr(self, f'video_projector{stage}', video_projector)


        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                                        embed_dim=self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[0] // self.scale_factor,
                                                       embed_dim=self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[1] // self.scale_factor,
                                                       embed_dim=self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                       in_chans=self.embed_dims[2] // self.scale_factor,
                                                       embed_dim=self.embed_dims[3] // self.scale_factor)

        if self.embedding_tune:
            if '1' in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)

        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

            if '2' in self.tuning_stage:
                for i in range(self.depths[1]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

            if '3' in self.tuning_stage:
                for i in range(self.depths[2]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

            if '4' in self.tuning_stage:
                for i in range(self.depths[3]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])

        elif self.adaptor == 'fully_shared':
            # 创建四个完全共享的多层感知机模型，共享相同的结构和参数
            self.fully_shared_mlp1 = nn.Sequential(
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                    )
            self.fully_shared_mlp2 = nn.Sequential(
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                    )
            self.fully_shared_mlp3 = nn.Sequential(
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                    )
            self.fully_shared_mlp4 = nn.Sequential(
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                    )

        elif self.adaptor == 'fully_unshared':
            # 根据给定的 self.depths 数组长度分别创建多个单独的、不同深度的多层感知机模型
            for i in range(self.depths[0]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                )
                setattr(self, 'fully_unshared_mlp1_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[1]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                )
                setattr(self, 'fully_unshared_mlp2_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[2]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                )
                setattr(self, 'fully_unshared_mlp3_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[3]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                )
                setattr(self, 'fully_unshared_mlp4_{}'.format(str(i)), fully_unshared_mlp1)

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
        # This method is being replaced by init_prompts to support flow fusion
        # Keeping it for compatibility if needed, but MixVisionTransformerEVP will call init_prompts
        return self.init_prompts(x, None)

    def init_prompts(self, segmap, flow=None):
        # x renamed to segmap for clarity
        x = segmap

        # --- 1. Process Segmentation Map (Existing Logic) ---
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)
        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)
        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)
        elif self.input_type == 'bimask':
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]

        # Segmap Features
        seg_feats = [None] * 4
        if '1' in self.tuning_stage:
            seg_feats[0], H1, W1 = self.handcrafted_generator1(x)
        if '2' in self.tuning_stage:
            # Reshape previous stage output to image-like format for next stage input
            prev = seg_feats[0].reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            seg_feats[1], H2, W2 = self.handcrafted_generator2(prev)
        if '3' in self.tuning_stage:
            prev = seg_feats[1].reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
            seg_feats[2], H3, W3 = self.handcrafted_generator3(prev)
        if '4' in self.tuning_stage:
            prev = seg_feats[2].reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
            seg_feats[3], H4, W4 = self.handcrafted_generator4(prev)

        # --- 2. Process Optical Flow (New Logic) ---
        flow_feats = [None] * 4
        if self.flow_tune and flow is not None:
             # Flow processing mirrors Segmap processing but with flow_generators
            if '1' in self.tuning_stage:
                flow_feats[0], fH1, fW1 = self.flow_generator1(flow)
            if '2' in self.tuning_stage:
                prev = flow_feats[0].reshape(B, fH1, fW1, -1).permute(0, 3, 1, 2).contiguous()
                flow_feats[1], fH2, fW2 = self.flow_generator2(prev)
            if '3' in self.tuning_stage:
                prev = flow_feats[1].reshape(B, fH2, fW2, -1).permute(0, 3, 1, 2).contiguous()
                flow_feats[2], fH3, fW3 = self.flow_generator3(prev)
            if '4' in self.tuning_stage:
                prev = flow_feats[2].reshape(B, fH3, fW3, -1).permute(0, 3, 1, 2).contiguous()
                flow_feats[3], fH4, fW4 = self.flow_generator4(prev)

        # --- 3. Fusion & Unification ---
        fused_prompts = [None] * 4
        for i, stage in enumerate(['1', '2', '3', '4']):
            if stage in self.tuning_stage:
                seg_feat = seg_feats[i]

                if flow_feats[i] is not None:
                    flow_feat = flow_feats[i]

                    # [新增] Apply Linear Projections First
                    # Generator Output (B, N, C) -> Linear -> Projected Feature
                    seg_proj_layer = getattr(self, f'seg_projector{stage}')
                    flow_proj_layer = getattr(self, f'flow_projector{stage}')

                    seg_feat = seg_proj_layer(seg_feat)
                    flow_feat = flow_proj_layer(flow_feat)

                    # Apply Unification Adapters
                    seg_adapter = getattr(self, f'seg_unification_adapter{stage}')
                    flow_adapter = getattr(self, f'flow_unification_adapter{stage}')

                    # Note: Adapters expect (B, N, C). MLP defined earlier handles this.
                    # Mlp.forward(x, H, W) requires H, W.
                    # But our unification adapters are nn.Sequential(Mlp, Mlp).
                    # Wait, Mlp class forward signature is (x, H, W). nn.Sequential won't pass H, W automatically!

                    # We have H, W from generator outputs (e.g. H1, W1).
                    # We need to construct the H, W list.
                    Hs = [H1, H2, H3, H4]
                    Ws = [W1, W2, W3, W4]
                    cur_H, cur_W = Hs[i], Ws[i]

                    # Manual forward for sequential containing Mlp layers that need H, W
                    # Or since I defined them as nn.Sequential(Mlp, Mlp), I can't just call it.
                    # I must unpack or define valid forward.

                    # Let's perform step-by-step
                    # Seg
                    s_out = seg_feat
                    for layer in seg_adapter:
                        s_out = layer(s_out, cur_H, cur_W)

                    # Flow
                    f_out = flow_feat
                    for layer in flow_adapter:
                        f_out = layer(f_out, cur_H, cur_W)

                    # Concatenate
                    # Both are (B, N, C_unified) -> Concat to (B, N, 2*C_unified) = (B, N, C_prompt)
                    fused_prompts[i] = torch.cat([s_out, f_out], dim=-1)
                else:
                    # Fallback if no flow provided (though user request implies robust fusion)
                    # For compatibility, just return seg_feat (but dims might mismatch later)
                    # Ideally we should strictly follow the architecture.
                    # Assuming flow is always present for this mode.
                    # If flow is None, we might need a dummy or skip.
                    fused_prompts[i] = seg_feat

        # Return tuple matching 4 stages
        return tuple(fused_prompts)


    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature

        return handcrafted_feature, embedding_feature

    def get_embedding_feature(self, x, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(x)

            return embedding_feature
        else:
            return None

    def get_handcrafted_feature(self, x, block_num):
        if self.handcrafted_tune:
            handcrafted_generator = getattr(self, 'handcrafted_generator{}'.format(str(block_num)))
            handcrafted_feature = handcrafted_generator(x)

            return handcrafted_feature
        else:
            return None

    def get_prompt(self, x, prompt, block_num, depth_num):
        # Revised get_prompt for Synergistic Fusion

        # 'prompt' argument passed from forward_features is a tuple: (handcrafted_feature, embedding_feature)
        # handcrafted_feature corresponds to our 'fused_prompts[i]' (Seg + Flow)
        # embedding_feature corresponds to the projected video frame embedding (projected via embedding_generator)

        fused_prompt_feat, embedding_feat = prompt

        # 2. Synergistic Addition
        # "逐元素相加，从而得到最终的光流-器械协同提示"
        # We use embedding_feat directly as it is already the projected video frame embedding
        synergistic_prompt = embedding_feat + fused_prompt_feat

        # 3. Input to Prompt Adapter Block (Existing Logic)
        feat = synergistic_prompt

        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))

            feat = lightweight_mlp(feat) # Input dim: C_prompt
            feat = shared_mlp(feat)      # Output dim: C_original (restores dimension)

        elif self.adaptor == 'fully_shared':
            fully_shared_mlp = getattr(self, 'fully_shared_mlp' + str(block_num))
            feat = fully_shared_mlp(feat)

        elif self.adaptor == 'fully_unshared':
            fully_unshared_mlp = getattr(self, 'fully_unshared_mlp' + str(block_num) + '_' + str(depth_num))
            feat = fully_unshared_mlp(feat)

        # 4. Add to original input
        x = x + feat


        return x


# @BACKBONES.register_module()
class mit_b0_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b0_evp, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# @BACKBONES.register_module()
class mit_b1_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b1_evp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# @BACKBONES.register_module()
class mit_b2_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b2_evp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# @BACKBONES.register_module()
class mit_b3_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b3_evp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# @BACKBONES.register_module()
class mit_b4_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b4_evp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# @BACKBONES.register_module()
class mit_b5_evp(MixVisionTransformerEVP):
    def __init__(self, **kwargs):
        super(mit_b5_evp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)