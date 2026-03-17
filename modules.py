import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.proj = nn.Conv2d(in_chans, embed_dim*4, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1]//2))
        self.projection = nn.Conv2d(in_chans, embed_dim*4, kernel_size=1, stride=2, bias=False)
        self.norm = nn.LayerNorm(embed_dim * 4)

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
        x = self.projection(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # x= B, H*W, C
        x = self.norm(x)

        return x, H, W


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

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()
        if self.input_type == 'srm':
            self.srm_filter = SRMFilter()
        if self.input_type == 'fft':
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)

        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                                        embed_dim=self.embed_dims[0])
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=1, stride=2,
                                                       in_chans=self.embed_dims[0]*4,
                                                       embed_dim=self.embed_dims[1])
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=1, stride=2,
                                                       in_chans=self.embed_dims[1]*4,
                                                       embed_dim=self.embed_dims[2])
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=1, stride=2,
                                                       in_chans=self.embed_dims[2]*4,
                                                       embed_dim=self.embed_dims[3])

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


    def init_hfcfeas(self, x):
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)

        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)

        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)

        return x

    def init_handcrafted(self, x):
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)

        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)

        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)

        # return x
        B = x.shape[0]
        # get prompting

        if '1' in self.tuning_stage:
            handcrafted1, H1, W1 = self.handcrafted_generator1(x)  # H1=56, W1=56, handcrafted1==Tensor(100,3136,16)
            # 调整维度后：H=112, W=112, handcrafed1 = Tensor(10,12544,256)
        else:
            handcrafted1 = None

        if '2' in self.tuning_stage:
            handcrafted2, H2, W2 = self.handcrafted_generator2(handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())  # H2=28, W2=28, handcrafted2==Tensor(100,784,32)
            # 调整维度后：H=56, W=56, handcrafed2 = Tensor(10,3136,512)
        else:
            handcrafted2 = None

        if '3' in self.tuning_stage:
            handcrafted3, H3, W3 = self.handcrafted_generator3(handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())  # W3=14, H3=14, handcrafted3==Tensor(100,196,64)
            # 调整维度后：H=28, W=28, handcrafed3 = Tensor(10,784,1024)
        else:
            handcrafted3 = None

        if '4' in self.tuning_stage:
            handcrafted4, H4, W4 = self.handcrafted_generator4(handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())  # H4=7, W4=7, handcrafted4==Tensor(100,49,128)
            # 调整维度后：H=14, W=14, handcrafed4 = Tensor(10,196,2048)
        else:
            handcrafted4 = None

        return handcrafted1, handcrafted2, handcrafted3, handcrafted4

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
        feat = 0
        if self.handcrafted_tune:
            feat += prompt[0]
        if self.embedding_tune:
            feat += prompt[1]

        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))

            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)

        elif self.adaptor == 'fully_shared':
            fully_shared_mlp = getattr(self, 'fully_shared_mlp' + str(block_num))
            feat = fully_shared_mlp(feat)

        elif self.adaptor == 'fully_unshared':
            fully_unshared_mlp = getattr(self, 'fully_unshared_mlp' + str(block_num) + '_' + str(depth_num))
            feat = fully_unshared_mlp(feat)

        x = x + feat

        return x

    def fft(self, x, rate, prompt_type):
        # 快速傅里叶变换 Fast Fourier Transform
        mask = torch.zeros(x.shape).to(device)
        w, h = x.shape[-2:]  # h=224,w=224
        line = int((w * h * rate) ** .5 // 2)  # line=56
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1  # mask=Tensor(100,3,224,224)

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))  # fft=Tensor(100,3,224,224)

        if prompt_type == 'highpass':
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv


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

    def srm_layer(self, img):
        out = self.srm_layer(img) # Attention: method name same as attribute name might cause issue if not careful, but looking at usage in PromptGenerator: x = self.srm_filter.srm_layer(x). Wait, srm_layer is nn.Conv2d.
        # In mix_transformer_evp.py, method is named conv_srm but attribute is srm_layer.
        # In modules.py PromptGenerator: x = self.srm_filter.srm_layer(x)
        # If srm_filter is instance of SRMFilter, accessing .srm_layer gives the attribute (the Conv2d layer).
        # So I don't need a forward method or helper method if the code accesses the attribute directly.
        # But wait, PromptGenerator calls: x = self.srm_filter.srm_layer(x). This is calling the Conv2d layer directly.
        # So I just need to define the class with the attribute.

        # Let's double check modules.py usage:
        # if self.input_type == 'srm':
        #    x = self.srm_filter.srm_layer(x)

        # So the class definition I copied has self.srm_layer attribute.
        return out
