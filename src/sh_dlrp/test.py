import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

class attnet(nn.Module):
    def __init__(self):
        super(attnet, self).__init__()

        attention_net = Attn_Net_Gated(512, D=256, dropout=0., n_classes=1)
        fc = [nn.Linear(1000, 512), nn.ReLU(), nn.Dropout(0.)]
        fc.append(attention_net)

        self.attn = nn.Sequential(*fc)


    def forward(self, x):
        return self.attn(x)



if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = attnet()
    x = torch.randn(size=(3, 42482, 1000))

    A = model(x)

    A = A.permute(0, 2, 1)
    A = F.softmax(A, dim=-1)

    print("A shape:", A.shape)

    topk = torch.topk(A, 100)
    print("topk shape:", topk.values.shape)

    print("x shape:", x.shape)
    # print(topk.values.permute(0, 2, 1).shape)

    for batch in range(x.shape[0]):
        batch_top_p_ids = torch.topk(A[batch], 100)[1][-1]
        # batch_top_p_ids = topk.values[batch][0]
        print(batch_top_p_ids.shape)
        print(x[batch].shape)
        batch_top_p = torch.index_select(x[batch], dim=0, index=batch_top_p_ids)

        if batch == 0:
            top_p = torch.unsqueeze(batch_top_p, dim=0)
        else:
            top_p = torch.cat((top_p, torch.unsqueeze(batch_top_p, dim=0)), dim=0)

    # top_p = torch.index_select(x, dim=1, index=temp.values.permute(0, 2, 1))

    print(A .shape)

    print(top_p.shape)
