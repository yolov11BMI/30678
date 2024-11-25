import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.efficientnet_blocks import SqueezeExcite as SE
from timm.models.layers import DropPath
import pennylane as qml

# Modified quantum_convolution function using qubits and rotation gates
def quantum_convolution(input_tensor):
    n_qubits = 4  # Example: using 4 qubits
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(inputs):
        # Embed classical data
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)  # Rotate each qubit based on input values

        # Apply quantum gates (example: rotation and entanglement)
        for i in range(n_qubits):
            qml.RY(0.1, wires=i)
            qml.RZ(0.2, wires=i)

        # Entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # Apply the quantum circuit for each channel of the input tensor
    batch_size, channels, height, width = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)

    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    pixel_value = input_tensor[b, c, h, w].item()
                    quantum_output = quantum_circuit([pixel_value] * n_qubits)
                    output_tensor[b, c, h, w] = torch.tensor(sum(quantum_output))

    return output_tensor

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
        self.is_quantum = kernel_size != 1  # Apply quantum convolution for kernels other than 1x1
        self.kernel_size = kernel_size

        if self.is_quantum:
            self.conv = None  # Placeholder, quantum convolution logic will be applied in forward
        else:
            padding = math.ceil((kernel_size - stride) / 2)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        if self.is_quantum:
            # Apply quantum convolution logic here
            x = quantum_convolution(x)
        else:
            x = self.conv(x)
        
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class QiRMB(nn.Module):
    def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=8, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        dim_out = dim_in
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=True)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=True)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=True)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # Attention operations as per the original model
            pass
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()

        self.attn = QiRMB(c)
        self.ffn = nn.Sequential(ConvNormAct(c, c * 2, 1), ConvNormAct(c * 2, c, 1, act_layer='none'))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_QiRMB(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvNormAct(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
