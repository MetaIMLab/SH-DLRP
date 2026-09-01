import torch.nn as nn
import torch
class ModalAtt(nn.Module):
    def __init__(self, embed_dim):
        super(ModalAtt, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(4, embed_dim, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(embed_dim, 4, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):


        y1 = self.gap(x1)
        y2 = self.gap(x2)
        y3 = self.gap(x3)
        y4 = self.gap(x4)
        # y5 = self.gap(x5)

        ws = torch.cat((y1, y2, y3, y4), dim=-1)


        ws = self.fc(ws)
        w1, w2, w3, w4= torch.unsqueeze(ws[:, 0], dim=1), torch.unsqueeze(ws[:, 1], dim=1), torch.unsqueeze(ws[:, 2], dim=1), torch.unsqueeze(ws[:, 3], dim=1)

        x1, x2, x3, x4= x1* w1.expand_as(x1), x2* w2.expand_as(x2), x3* w3.expand_as(x3), x4* w4.expand_as(x4)

        return x1, x2, x3, x4


class ModalAttention(nn.Module):
    def __init__(self, embed_dim, modal_num):
        super(ModalAttention,  self).__init__()
        # 全局平均池化(Fsq操作)
        self.projs = nn.ModuleList()
        self.embed_dim = embed_dim
        self.modal_num = modal_num


        for i in range(modal_num):
            self.projs.append(nn.Linear(embed_dim, 1))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, xs):
        b_size = xs[0].shape[0]

        x_weight = []
        for proj, x in zip(self.projs, xs):
            s = proj(x)
            x_weight.append(s)

        # s1 = self.weights[0](x1)
        # s2 = self.weights[1](x2)
        # s3 = self.weights[2](x3)
        # s4 = self.weights[3](x4)
        # s5 = self.weights[4](x5)
        # s6 = self.weights[5](x6)
        # s7 = self.weights[6](x7)

        weight = self.softmax(torch.cat(x_weight, dim=-1)).unsqueeze(dim=-1).repeat(1, 1, self.embed_dim)

        value = torch.stack(xs, dim=1)

        weight_value = value * weight

        out_value = weight_value.view((b_size, -1))

        return out_value


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x1 = torch.randn(size=(8, 256)).to(device)
    x2 = torch.randn(size=(8, 256)).to(device)
    x3 = torch.randn(size=(8, 256)).to(device)
    x4 = torch.randn(size=(8, 256)).to(device)
    x5 = torch.randn(size=(8, 256)).to(device)
    x6 = torch.randn(size=(8, 256)).to(device)
    x7 = torch.randn(size=(8, 256)).to(device)


    # model = ModalAtt(256).to(device)

    model = ModalAttention(embed_dim=256, modal_num=7).to(device)

    out = model((x1, x2, x3, x4, x5, x6, x7))

    print(out.shape)
