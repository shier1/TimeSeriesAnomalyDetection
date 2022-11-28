import torch
import torch.nn as nn
from itertools import repeat
import collections.abc

def _ntuple(n):
    """
    判断类型
    @arge传递重复矩阵元素的次数
    返回一个元组
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LayerScale(nn.Module):
    """
    对参数inplace和gamma初始化
    返回符合判断条件的矩阵
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))#初始化参数gamma

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    创建一个区域块将上述类整合到一起
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ConvFC(nn.Module):
    """
    构建网络
    模型结构包括卷积层跟全连接层
    """
    def __init__(self, in_features, n_class, mlp_ratio, num_heads, dropout_ratio, drop_path_ratio, attn_drop_ratio) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, in_features), padding=(0, 3), padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0),  #, padding_mode='replicate'
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0), #, padding_mode='replicate'
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.msa1 = Block(dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=dropout_ratio, attn_drop=attn_drop_ratio, drop_path=drop_path_ratio)
        self.msa2 = Block(dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=dropout_ratio, attn_drop=attn_drop_ratio, drop_path=drop_path_ratio)
        # self.msa3 = Block(dim=256, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=dropout_ratio, attn_drop=attn_drop_ratio, drop_path=drop_path_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(64)
        )
        self.classifier = nn.Linear(in_features=64, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def _forward_features(self, x:torch.Tensor):
        x.unsqueeze_(dim=1)
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(2)
        x = x + self.msa1(x)
        x = x + self.msa2(x)
        #x = x + self.msa3(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

    def forward(self, x:torch.Tensor):
        features = self._forward_features(x)
        logit = self.classifier(features)
        # sigmoid_out = self.sigmoid(logit)
        return logit


class ConvAttention(nn.Module):
    def __init__(self, ):
        self


def get_model():
    """
    自主调参，可对卷积层与全连接层进行参数调节
    """
    return ConvFC(in_features=9, 
                  n_class=2, 
                  mlp_ratio=8,
                  num_heads=4,
                  dropout_ratio=.2, 
                  drop_path_ratio=.2, 
                  attn_drop_ratio=.0)
