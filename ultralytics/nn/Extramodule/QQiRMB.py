# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from einops import rearrange
# from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE 
# from timm.models.layers import DropPath

# __all__ = ['QQiRMB']

# inplace = True  # Global variable

# class LayerNorm2d(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
#         super().__init__()
#         self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

#     def forward(self, x):
#         x = rearrange(x, 'b c h w -> b h w c').contiguous()
#         x = self.norm(x)
#         x = rearrange(x, 'b h w c -> b c h w').contiguous()
#         return x

# def get_norm(norm_layer='in_1d'):
#     eps = 1e-6
#     norm_dict = {
#         'none': nn.Identity,
#         'in_1d': partial(nn.InstanceNorm1d, eps=eps),
#         'in_2d': partial(nn.InstanceNorm2d, eps=eps),
#         'in_3d': partial(nn.InstanceNorm3d, eps=eps),
#         'bn_1d': partial(nn.BatchNorm1d, eps=eps),
#         'bn_2d': partial(nn.BatchNorm2d, eps=eps),
#         'bn_3d': partial(nn.BatchNorm3d, eps=eps),
#         'gn': partial(nn.GroupNorm, eps=eps),
#         'ln_1d': partial(nn.LayerNorm, eps=eps),
#         'ln_2d': partial(LayerNorm2d, eps=eps),
#     }
#     return norm_dict[norm_layer]

# def get_act(act_layer='relu'):
#     act_dict = {
#         'none': nn.Identity,
#         'relu': nn.ReLU,
#         'relu6': nn.ReLU6,
#         'silu': nn.SiLU
#     }
#     return act_dict[act_layer]

# class ConvNormAct(nn.Module):
#     def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
#                  skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
#         super(ConvNormAct, self).__init__()
#         self.has_skip = skip and dim_in == dim_out
#         padding = math.ceil((kernel_size - stride) / 2)
#         self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
#         self.norm = get_norm(norm_layer)(dim_out)
#         self.act = get_act(act_layer)(inplace=inplace)
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

#     def forward(self, x):
#         shortcut = x
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         if self.has_skip:
#             x = self.drop_path(x) + shortcut
#         return x

# class QuantumConvLayer(nn.Module):
#     def __init__(self, num_qubits, layers=1):
#         super().__init__()
#         # Define the quantum circuit, using multiple quantum layers for more expressive power
#         self.q_layers = nn.ModuleList([nn.Linear(num_qubits, num_qubits) for _ in range(layers)])  # Placeholder for quantum layers

#     def forward(self, x):
#         # Assuming input size is (batch, channels, height, width), we first flatten channels to qubits
#         B, C, H, W = x.shape
#         x = rearrange(x, 'b c h w -> (b h w) c')  # Prepare for quantum processing

#         # Apply each quantum layer sequentially
#         for layer in self.q_layers:
#             x = layer(x)

#         x = rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W)
#         return x

# class QQiRMB(nn.Module):
#     def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
#                  act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=8, window_size=7,
#                  attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False,
#                  use_quantum=False):
#         super().__init__()
#         dim_out = dim_in
#         self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
#         dim_mid = int(dim_in * exp_ratio)
#         self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
#         self.attn_s = attn_s
#         self.use_quantum = use_quantum

#         if self.attn_s:
#             assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
#             self.dim_head = dim_head
#             self.window_size = window_size
#             self.num_head = dim_in // dim_head
#             self.scale = self.dim_head ** -0.5
#             self.attn_pre = attn_pre
#             self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
#                                   act_layer='none')
#             if use_quantum:
#                 self.v = QuantumConvLayer(num_qubits=dim_mid)
#             else:
#                 self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
#                                      norm_layer='none', act_layer=act_layer, inplace=inplace)
#             self.attn_drop = nn.Dropout(attn_drop)
#         else:
#             if v_proj:
#                 if use_quantum:
#                     self.v = QuantumConvLayer(num_qubits=dim_mid)
#                 else:
#                     self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
#                                          act_layer=act_layer, inplace=inplace)
#             else:
#                 self.v = nn.Identity()
        
#         self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
#                                       groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
#         self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

#         self.proj_drop = nn.Dropout(drop)
#         self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
#         self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

#     def forward(self, x):
#         shortcut = x
#         x = self.norm(x)
#         B, C, H, W = x.shape
#         if self.attn_s:
#             # Padding
#             if self.window_size <= 0:
#                 window_size_W, window_size_H = W, H
#             else:
#                 window_size_W, window_size_H = self.window_size, self.window_size
#             pad_l, pad_t = 0, 0
#             pad_r = (window_size_W - W % window_size_W) % window_size_W
#             pad_b = (window_size_H - H % window_size_H) % window_size_H
#             x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
#             n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
#             x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            
#             # Attention with quantum augmentation
#             b, c, h, w = x.shape
#             qk = self.qk(x)
#             qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
#                            dim_head=self.dim_head).contiguous()
#             q, k = qk[0], qk[1]
#             attn_spa = (q @ k.transpose(-2, -1)) * self.scale
#             attn_spa = attn_spa.softmax(dim=-1)
#             attn_spa = self.attn_drop(attn_spa)
            
#             if self.attn_pre:
#                 x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
#                 x_spa = attn_spa @ x
#                 x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
#                                   w=w).contiguous()
#                 x_spa = self.v(x_spa)
#             else:
#                 v = self.v(x)
#                 v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
#                 x_spa = attn_spa @ v
#                 x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
#                                   w=w).contiguous()
#             # Unpadding
#             x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
#             if pad_r > 0 or pad_b > 0:
#                 x = x[:, :, :H, :W].contiguous()
#         else:
#             x = self.v(x)

#         x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

#         x = self.proj_drop(x)
#         x = self.proj(x)

#         x = (shortcut + self.drop_path(x)) if self.has_skip else x
#         return x

# # Example usage of iRMB with quantum augmentation
# if __name__ == "__main__":
#     image_size = (1, 64, 224, 224)
#     image = torch.rand(*image_size)
    
#     # Instantiate iRMB with Quantum Layer Integration
#     model = QQiRMB(64, use_quantum=True)
#     out = model(image)
#     print(out.size())
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE 
from timm.models.layers import DropPath

__all__ = ['QQiRMB']

inplace = True  # Global variable

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]

def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]

class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class QuantumConvLayer(nn.Module):
    def __init__(self, num_qubits, layers=1):
        super().__init__()
        # Define the quantum circuit, using multiple quantum layers for more expressive power
        self.q_layers = nn.ModuleList([nn.Linear(num_qubits, num_qubits) for _ in range(layers)])  # Placeholder for quantum layers
        self.rotation_gates = nn.ModuleList([nn.Linear(num_qubits, num_qubits) for _ in range(layers)])  # Rotation gates as placeholders

    def forward(self, x):
        # Assuming input size is (batch, channels, height, width), we first flatten channels to qubits
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')  # Prepare for quantum processing

        # Apply each quantum layer sequentially, including rotation gates
        for q_layer, rot_gate in zip(self.q_layers, self.rotation_gates):
            x = q_layer(x)
            x = torch.sin(rot_gate(x))  # Applying rotation gate (using sine as a simple example)

        x = rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W)
        return x

class QQiRMB(nn.Module):
    def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=8, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False,
                 use_quantum=False):
        super().__init__()
        dim_out = dim_in
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.use_quantum = use_quantum

        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            if use_quantum:
                self.v = QuantumConvLayer(num_qubits=dim_mid)
            else:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                     norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                if use_quantum:
                    self.v = QuantumConvLayer(num_qubits=dim_mid)
                else:
                    self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                         act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # Padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            
            # Attention with quantum augmentation
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # Unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

# Example usage of iRMB with quantum augmentation
if __name__ == "__main__":
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)
    
    # Instantiate iRMB with Quantum Layer Integration
    model = QQiRMB(64, use_quantum=True)
    out = model(image)
    print(out.size())
