
import torch.nn as nn
import torch
class AttPoolBlock(nn.Module):
    def __init__(self, expand_dim):
        super(AttPoolBlock, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(1, expand_dim, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(expand_dim, 1, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, l, d = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return torch.sum(x * y.expand_as(x), dim=1)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    output = torch.randn(size=(16, 200, 256)).to(device)

    model = AttPoolBlock(256).to(device)

    out = model(output)

    print(out.shape)
