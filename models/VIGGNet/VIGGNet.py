import torch.nn as nn
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath
from gcn_lib import Grapher, act_layer


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.ChannelAttention = ChannelAttention(F_int)
        self.SpatialAttention = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        x = x * self.ChannelAttention(psi)  # 通道注意力
        x = x * self.SpatialAttention(psi)

        return x


class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class Att_up(nn.Module):
    def __init__(self, channels1, channels2, use_att=False, useskip=True):
        super().__init__()
        self.use_att = use_att
        self.useskip = useskip
        if use_att == True:
            self.attention1 = Attention_block(channels2, channels2, channels2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels1, channels2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels2),
            nn.ReLU()
        )
        if useskip == True:
            self.conv = DoubleConv(channels2 * 2, channels2)
        else:
            self.conv = DoubleConv(channels2, channels2)

    def forward(self, x1, x2):
        x = self.up(x1)
        if self.use_att == True:
            x2 = self.attention1(x, x2)
        if self.useskip == True:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VIGGNet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=False):
        super(VIGGNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))

        self.stem = Stem(out_dim=512, act='gelu')
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, 14, 14))
        self.attention = Attention_block(512, 512, 512)
        self.up1 = Att_up(512, 256,use_att=True)
        self.up2 = Att_up(256, 128,use_att=True)
        self.up3 = Att_up(128, 64,use_att=True)
        self.up4 = Att_up(64, 32,use_att=True)
        self.outc = (OutConv(32, num_classes))
        self.conv = DoubleConv(1024, 512)
        dpr = [x.item() for x in torch.linspace(0, 0, 6)]
        self.backbone = self.basic_backbone(512,6, dpr, 6, 1, 1, 14 * 14)

    def basic_backbone(self, in_channels, num_blocks, dpr, num_knn, reduce_ratios, max_dilation, HW):
        self.layers = nn.ModuleList([])
        act = 'gelu'
        norm = 'batch'
        bias = True
        epsilon = 0.2
        stochastic = False
        conv = 'mr'
        for j in range(num_blocks):
            self.layers += [
                Seq(Grapher(in_channels, num_knn, max_dilation, conv, act, norm,
                            bias, stochastic, epsilon, reduce_ratios, n=HW, drop_path=dpr[j],
                            relative_pos=True),
                    FFN(in_channels, in_channels * 4, act=act, drop_path=dpr[j])
                    )]
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x_gnn = self.stem(x) + self.pos_embed
        x_gnn = self.backbone(x_gnn)
        x5_gnn = self.attention(x_gnn, x5)
        x5 = torch.cat([x5, x5_gnn], dim=1)
        x5 = self.conv(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits