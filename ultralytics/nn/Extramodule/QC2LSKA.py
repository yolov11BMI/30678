import torch
import torch.nn as nn

__all__ = ['QLSKA', 'C2PSA_QLSKA']

# Embedded QCNNConv2d definition
class QCNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QCNNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # Define rotation angle theta as a learnable parameter for each channel
        self.theta = nn.Parameter(torch.rand(in_channels) * 3.14159)  # Initialize angles randomly
        
        # Define initial qubit states (two states per qubit) as learnable parameters
        self.qubits_real = nn.Parameter(torch.rand(in_channels, 2))
        self.qubits_imag = nn.Parameter(torch.rand(in_channels, 2))

    def apply_hadamard(self):
        """Simulate a Hadamard gate to put qubits in superposition."""
        h_factor = 1 / torch.sqrt(torch.tensor(2.0))
        # Apply Hadamard transform
        self.qubits_real.data = h_factor * (self.qubits_real + self.qubits_imag)
        self.qubits_imag.data = h_factor * (self.qubits_real - self.qubits_imag)

    def rotation_gate(self, x):
        """Apply a simulated rotation on the input based on theta for each channel."""
        cos_theta = torch.cos(self.theta).view(-1, 1, 1)
        sin_theta = torch.sin(self.theta).view(-1, 1, 1)
        return x * cos_theta + torch.flip(x, [-1]) * sin_theta  # Mimic rotation operation
    
    def forward(self, x):
        self.apply_hadamard()  # Apply a Hadamard gate to simulate superposition
        x = self.rotation_gate(x)  # Apply rotation gate on the input
        return self.conv(x)  # Apply convolution after rotation



class QLSKA(nn.Module):
    def __init__(self, dim, k_size=11):
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            self.conv0h = QCNNConv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = QCNNConv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = QCNNConv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = QCNNConv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = QCNNConv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = QCNNConv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = QCNNConv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = QCNNConv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = QCNNConv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = QCNNConv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = QCNNConv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = QCNNConv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = QCNNConv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = QCNNConv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = QCNNConv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = QCNNConv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)

        # Ensure conv1 is defined regardless of k_size
        self.conv1 = QCNNConv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


# Define autopad
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = QCNNConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = QLSKA(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA_QLSKA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# Sample run for testing
if __name__ == "__main__":
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    model = C2PSA_QLSKA(64, 64)
    out = model(image)
    print(out.size())
