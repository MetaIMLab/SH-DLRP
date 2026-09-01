''' Model '''
import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, feature_dim):
        super(CNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):

        x = self.feature(x)



        x = x.view(x.size()[0], -1)



        return self.classifier(x)



class MultiModalCNNTest(nn.Module):

    def __init__(self, recon, feature_dim):
        super(MultiModalCNNTest, self).__init__()

        self.CNNs = nn.ModuleList()

        for i in range(3):
            self.CNNs.append(CNN(feature_dim))


        self.fc_text = nn.Sequential(
            nn.Linear(19, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.fc_radio_1 = nn.Sequential(
            nn.Linear(1935, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.fc_radio_2 = nn.Sequential(
            nn.Linear(1935, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.fc_radio_3 = nn.Sequential(
            nn.Linear(1935, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.ctx = nn.Parameter(torch.zeros(4,3,224,224))

    def forward(self, img_swt, img_swv, img_bus, report, swt_fea, swv_fea, bus_fea, train):


        # if train:
        #     img_embed_1 = self.CNNs[0](self.ctx)
        # else:
        img_embed_1 = self.CNNs[0](img_swt)
        img_embed_2 = self.CNNs[1](img_swv)
        img_embed_3 = self.CNNs[2](img_bus)


        text_embed = self.fc_text(report)
        radio_embed_1 = self.fc_radio_1(swt_fea)
        radio_embed_2 = self.fc_radio_2(swv_fea)
        radio_embed_3 = self.fc_radio_3(bus_fea)

        comb = torch.cat((img_embed_1, img_embed_2, img_embed_3, text_embed ), dim=-1)

        # comb = torch.cat((text_embed, radio_embed_1, radio_embed_2, radio_embed_3, img_embed_1, img_embed_2, img_embed_3),dim=-1)



        return self.classifier(comb)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x1 = torch.randn(size=(8, 3, 224, 224))
    x2 = torch.randn(size=(8, 3, 224, 224))
    x3 = torch.randn(size=(8, 3, 224, 224))

    text = torch.randn(size=(8, 19))

    fea_1 = torch.randn(size=(8, 1935))

    fea_2 = torch.randn(size=(8, 1935))
    fea_3 = torch.randn(size=(8, 1935))

    model = MultiModalCNNTest(recon=False, feature_dim=128).to(device)

    print(model(x1, x2, x3, text, fea_1, fea_2, fea_3, train=False).size())
    # print(module(test_x).size())
