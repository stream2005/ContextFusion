import torch
import math
import itertools
import torch.nn as nn
from timm.layers import trunc_normal_, SqueezeExcite
from timm.models import register_model, build_model_with_cfg
from ska import SKA
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
import torch
import torch.nn as nn
class SGLKP(nn.Module):
    def __init__(self, dim, lks_list, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.multi_scale_convs = nn.ModuleList([
            Conv2d_BN(dim//2, dim//2, ks=lks, pad=(lks-1)//2, groups=dim//2)
            for lks in lks_list
        ])
        self.se_gff = SE_GFF(in_channels_list=[dim//2]*len(lks_list))
        self.cv3 = Conv2d_BN(dim//2, dim//2)
        self.cv4 = nn.Conv2d(dim//2, sks**2 * dim//groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim//groups,
                               num_channels=sks**2*dim//groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv1(x))
        scale_features = [conv(x) for conv in self.multi_scale_convs]
        fused_features = self.se_gff(scale_features)  # 返回同尺寸特征列表
        aggregated = torch.stack(fused_features).sum(dim=0)
        x = self.act(self.cv3(aggregated))
        w = self.norm(self.cv4(x))
        return w.view(w.size(0), self.dim//self.groups, self.sks**2, *w.shape[-2:])

class GFF(nn.Module):
    def __init__(self, in_channels_list):
        super(GFF, self).__init__()
        self.num_levels = len(in_channels_list)
        self.gate_convs = nn.ModuleList()
        for c in in_channels_list:
            self.gate_convs.append(
                nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x_list):
        assert len(x_list) == self.num_levels
        gate_maps = []
        for i, x in enumerate(x_list):
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            gate_maps.append(gate)
        outputs = []
        for l in range(self.num_levels):
            x_l = x_list[l]
            G_l = gate_maps[l]
            term1 = (1 + G_l) * x_l
            term2 = 0
            for i in range(self.num_levels):
                if i != l:
                    G_i = gate_maps[i]
                    G_i_resized = G_i
                    term2 += (1 - G_l) * G_i_resized * x_list[i]
            
            fused = term1 + term2
            outputs.append(fused)
        
        return outputs

class SE_GFF(nn.Module):
    """
    Enhanced Gated Fully Fusion (GFF) Module with Squeeze-and-Excitation.

    Key Improvements:
    1. SE blocks introduce channel-wise attention to complement spatial gating
    2. Assumes uniform feature dimensions across levels (no interpolation)
    3. Hybrid gating: Spatial gates (1x1 conv) × Channel gates (SE)

    Args:
        in_channels_list (list[int]): Input channels for each feature level.
        reduction_ratio (int, optional): SE squeeze ratio. Default: 16.
    """
    def __init__(self, in_channels_list, reduction_ratio=16):
        super(SE_GFF, self).__init__()
        self.num_levels = len(in_channels_list)

        # Spatial gating (1x1 conv + Sigmoid)
        self.spatial_gates = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1) for c in in_channels_list
        ])

        # Channel-wise gating (SE blocks)
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, max(4, c // reduction_ratio), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(4, c // reduction_ratio), c, kernel_size=1),
                nn.Sigmoid()
            ) for c in in_channels_list
        ])

    def forward(self, x_list):
        """
        Forward pass with hybrid gating mechanism.

        Args:
            x_list (list[Tensor]): Multi-level features of identical spatial dimensions.
                Each tensor shape: [N, C, H, W]

        Returns:
            list[Tensor]: Refined features for each level, same shapes as inputs.

        Mathematical Formulation:
            X̃_l = (1 + G_l^{space}⊙G_l^{channel}) ⊙ X_l
                 + (1 - G_l^{space}⊙G_l^{channel}) ⊙ ∑_{i≠l} G_i^{space}⊙G_i^{channel} ⊙ X_i
        where ⊙ denotes element-wise multiplication.
        """
        # Validate uniform spatial dimensions
        assert len({x.shape[-2:] for x in x_list}) == 1, \
               "All input features must have identical H,W dimensions"

        # Compute hybrid gates (spatial × channel)
        hybrid_gates = []
        for i, x in enumerate(x_list):
            spatial_gate = torch.sigmoid(self.spatial_gates[i](x))  # [N,1,H,W]
            channel_gate = self.se_blocks[i](x)  # [N,C,1,1]
            hybrid_gates.append(spatial_gate * channel_gate)  # [N,C,H,W] via broadcast

        # Perform gated fusion
        outputs = []
        for l in range(self.num_levels):
            # Term 1: (1 + G_l) ⊙ X_l
            term1 = (1 + hybrid_gates[l]) * x_list[l]

            # Term 2: ∑_{i≠l} (1 - G_l) ⊙ G_i ⊙ X_i
            term2 = sum(
                (1 - hybrid_gates[l]) * hybrid_gates[i] * x_list[i]
                for i in range(self.num_levels) if i != l
            )

            outputs.append(term1 + term2)

        return outputs

class DFConv2d_BN(torch.nn.Sequential):
    def __init__(self, a, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('t', TranBLC())
        self.add_module('c', DCNv4(a, ks, stride, pad, dilation, groups, output_bias=False))
        self.add_module('rt', TranBCHW())
        self.add_module('bn', torch.nn.BatchNorm2d(a))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
class TranBLC(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class TranBCHW(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, L, C = x.shape
        H=int(math.sqrt(L))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=H)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        #print(q.transpose(-2, -1).shape, k.shape, v.shape)
        #print(self.attention_biases[:, self.attention_bias_idxs].shape)
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            #+
            #(self.attention_biases[:, self.attention_bias_idxs]
            # if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=15, sks=3, groups=8)
        #self.sglkp = LKP(dim, lks_list=[11, 15, 19], sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

class Block(torch.nn.Module):    
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 distillation=False,):
        super().__init__()

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
                           )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                resolution_ = (resolution - 1) // 2 + 1
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_

        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
            
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0.c', 'classifier': ('head.linear', 'head_dist.linear'),
        **kwargs
    }

default_cfgs = dict(
    lsnet_t = _cfg(),
    lsnet_t_distill = _cfg(),
    lsnet_s = _cfg(),
    lsnet_s_distill = _cfg(),
    lsnet_b = _cfg(),
    lsnet_b_distill = _cfg(),
)

def _create_lsnet(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        LSNet,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs,
    )
    return model

@register_model
def lsnet_t(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_t" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation, 
                  img_size=224,
                  patch_size=8,
                  embed_dim=[64, 128, 256, 384],
                  depth=[0, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_s(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_s" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[96, 192, 320, 448],
                  depth=[1, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_b(num_classes=1000, distillation=False, pretrained=False, **kwargs):
    model = _create_lsnet("lsnet_b" + ("_distill" if distillation else ""),
                  pretrained=pretrained,
                  num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[128, 256, 384, 512],
                  depth=[4, 6, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  )
    return model

@register_model
def lsnet_t_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_t(**kwargs)

@register_model
def lsnet_s_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_s(**kwargs)

@register_model
def lsnet_b_distill(**kwargs):
    kwargs["distillation"] = True
    return lsnet_b(**kwargs)
