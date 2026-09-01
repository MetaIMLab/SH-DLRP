import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import time
import cv2
import yaml
from easydict import EasyDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

import pandas as pd
import os
import os.path as osp
import torch.nn.functional as F
warnings.filterwarnings('ignore')
import torch.nn as nn

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class Dataset(data.Dataset):
    def __init__(
        self,
        swt_root,
        swv_root,
        bus_root,
        swt_fea_root,
        swv_fea_root,
        bus_fea_root,
        histology_root,
        report_root,
        augmentations,
        padding_size,
        samples,
        global_seed=42,
    ):

        self.swt_root = swt_root
        self.swv_root = swv_root
        self.bus_root = bus_root

        self.swt_fea_root = swt_fea_root
        self.swv_fea_root = swv_fea_root
        self.bus_fea_root = bus_fea_root

        self.histology_root = histology_root

        self.report_root = report_root

        self.samples = sorted(samples)

        self.global_seed = global_seed


        # self.samples = os.listdir(histology_root)
        #
        #
        # self.samples = [i for i in self.samples if i.endswith('pt')]

        self.transform = augmentations
        self.padding_size = padding_size

    def __getitem__(self, idx):
        seed = self.global_seed + idx

        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        name = self.samples[idx]

        sample_name = name.replace(".pt", "")

        swt = cv2.imread(self.swt_root + '/' + name.replace('.pt', '.jpg'))
        swv = cv2.imread(self.swv_root + '/' + name.replace('.pt', '.jpg'))
        bus = cv2.imread(self.bus_root + '/' + name.replace('.pt', '.jpg'))
        if swt is None or swv is None or bus is None:
            raise FileNotFoundError("Missing ultrasound image for sample '{}'".format(name))

        swt = cv2.cvtColor(swt, cv2.COLOR_BGR2LAB)
        swv = cv2.cvtColor(swv, cv2.COLOR_BGR2LAB)
        bus = cv2.cvtColor(bus, cv2.COLOR_BGR2LAB)

        trans_swt = self.transform(image=swt)
        trans_swv = self.transform(image=swv)
        trans_bus = self.transform(image=bus)

        swt_fea_0_root, swt_fea_1_root, swt_fea_2_root = osp.join(self.swt_fea_root, "0"), osp.join(self.swt_fea_root, "1"), osp.join(
            self.swt_fea_root, "2")
        swv_fea_0_root, swv_fea_1_root, swv_fea_2_root = osp.join(self.swv_fea_root, "0"), osp.join(self.swv_fea_root, "1"), osp.join(
            self.swv_fea_root, "2")
        bus_fea_0_root, bus_fea_1_root, bus_fea_2_root = osp.join(self.bus_fea_root, "0"), osp.join(self.bus_fea_root, "1"), osp.join(
            self.bus_fea_root, "2")


        label_1 = torch.tensor(pd.read_csv(osp.join(self.report_root, name.replace(".pt", ".csv")), header=None).values.tolist()[0][0]).to(torch.int64)
        label_2 = torch.tensor(
            pd.read_csv(osp.join(self.report_root, name.replace(".pt", ".csv")), header=None).values.tolist()[0][1]).to(
            torch.int64)

        # assert label
        report = torch.tensor(pd.read_csv(osp.join(self.report_root, name.replace(".pt", ".csv")), header=None).values.tolist()[0][2:]).to(torch.float32)
        # report = F.normalize(report.float(), dim=0)

        # swt_fea_0 = pd.read_csv(osp.join(swt_fea_0_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        swt_fea_1 = pd.read_csv(osp.join(swt_fea_1_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        swt_fea_2 = pd.read_csv(osp.join(swt_fea_2_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]

        # swv_fea_0 = pd.read_csv(osp.join(swv_fea_0_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        swv_fea_1 = pd.read_csv(osp.join(swv_fea_1_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        swv_fea_2 = pd.read_csv(osp.join(swv_fea_2_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]

        # bus_fea_0 = pd.read_csv(osp.join(bus_fea_0_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        bus_fea_1 = pd.read_csv(osp.join(bus_fea_1_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        bus_fea_2 = pd.read_csv(osp.join(bus_fea_2_root, name.replace(".pt", ".csv")), header=None).values.tolist()[1][37:]
        #

        bus_fea_0 = pd.read_csv(
            osp.join(bus_fea_0_root, name.replace(".pt", ".csv")),
            header=None,
        ).iloc[1].tolist()[37:]
        swt_fea_0 = pd.read_csv(
            osp.join(swt_fea_0_root, name.replace(".pt", ".csv")),
            header=None,
        ).iloc[1].tolist()[37:]
        swv_fea_0 = pd.read_csv(
            osp.join(swv_fea_0_root, name.replace(".pt", ".csv")),
            header=None,
        ).iloc[1].tolist()[37:]

        #
        swt_fea_0, swt_fea_1, swt_fea_2 = (torch.FloatTensor(list(map(float, swt_fea_0))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, swt_fea_1))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, swt_fea_2))).to(torch.float32))
        swv_fea_0, swv_fea_1, swv_fea_2 = (torch.FloatTensor(list(map(float, swv_fea_0))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, swv_fea_1))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, swv_fea_2))).to(torch.float32))
        bus_fea_0, bus_fea_1, bus_fea_2 = (torch.FloatTensor(list(map(float, bus_fea_0))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, bus_fea_1))).to(torch.float32),
                                           torch.FloatTensor(list(map(float, bus_fea_2))).to(torch.float32))



        # feature_0 = torch.FloatTensor(list(map(float, feature_0))).to(torch.float32)
        # feature_1 = torch.FloatTensor(list(map(float, feature_1))).to(torch.float32)
        # feature_2 = torch.FloatTensor(list(map(float, feature_2))).to(torch.float32)

        # swt_fea = F.normalize(torch.cat((swt_fea_0, swt_fea_1, swt_fea_2), dim=0),dim=0)
        # swv_fea = F.normalize(torch.cat((swv_fea_0, swv_fea_1, swv_fea_2), dim=0),dim=0)
        # bus_fea = F.normalize(torch.cat((bus_fea_0, bus_fea_1, bus_fea_2), dim=0),dim=0)

        swt_fea = F.normalize(swt_fea_0, dim=0)
        swv_fea = F.normalize(swv_fea_0, dim=0)
        bus_fea = F.normalize(bus_fea_0, dim=0)

        histology_feature = torch.load(
            osp.join(self.histology_root, name), map_location=torch.device('cpu')
        )
        if histology_feature.ndim != 2:
            raise ValueError("Expected WSI features with shape [num_tiles, feature_dim], got {}".format(
                tuple(histology_feature.shape)
            ))
        if histology_feature.shape[0] > self.padding_size:
            raise ValueError(
                "WSI feature count {} exceeds padding_size {} for sample '{}'".format(
                    histology_feature.shape[0], self.padding_size, name
                )
            )
        histology_feature = histology_feature.to(torch.float32)

        mask = torch.zeros(self.padding_size, dtype=torch.bool)
        mask[:histology_feature.shape[0]] = True

        #The pad of histological features needs to be placed after the mask
        pad = nn.ZeroPad2d(padding=(0, 0, 0, self.padding_size - histology_feature.shape[0]))
        histology_feature = pad(histology_feature)

        return label_1, trans_swt['image'], trans_swv['image'], trans_bus['image'], swt_fea, swv_fea, bus_fea, histology_feature, mask, report, label_2, sample_name

    def __len__(self):
        return len(self.samples)




# def get_image_num(image_root):
def give_augmentations(config, train):
    if train == True:
        augmentations = A.Compose([
            A.Normalize(),
            A.Resize(config.image_size, config.image_size, interpolation=cv2.INTER_NEAREST),
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            ToTensorV2()
        ])
    else:
        augmentations = A.Compose([
            A.Normalize(),
            A.Resize(config.image_size, config.image_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ])
    return augmentations

def get_multicenter_dataloader(config):
    config = config.dataset.MBC

    padding_size = config.padding_size

    train_radiology_root = config.multi_center.train_radiology_root
    test_radiology_root = config.multi_center.test_radiology_root

    train_histology_root = config.multi_center.train_histology_root
    test_histology_root = config.multi_center.test_histology_root

    train_swt_root = os.path.join(train_radiology_root, 'swt')
    train_swv_root = os.path.join(train_radiology_root, 'swv')
    train_bus_root = os.path.join(train_radiology_root, 'bus')
    test_swt_root = os.path.join(test_radiology_root, 'swt')
    test_swv_root = os.path.join(test_radiology_root, 'swv')
    test_bus_root = os.path.join(test_radiology_root, 'bus')

    train_swt_fea_root = os.path.join(train_radiology_root, 'swt_fea')
    train_swv_fea_root = os.path.join(train_radiology_root, 'swv_fea')
    train_bus_fea_root = os.path.join(train_radiology_root, 'bus_fea')
    test_swt_fea_root = os.path.join(test_radiology_root, 'swt_fea')
    test_swv_fea_root = os.path.join(test_radiology_root, 'swv_fea')
    test_bus_fea_root = os.path.join(test_radiology_root, 'bus_fea')

    train_report_root = os.path.join(train_radiology_root, 'rep')
    test_report_root = os.path.join(test_radiology_root, 'rep')



    train_augmentation = give_augmentations(config, train=True)
    test_augmentation = give_augmentations(config, train=False)

    train_val_sample = sorted(os.listdir(train_histology_root))
    train_val_sample = [i for i in train_val_sample if i.endswith('pt')]

    labels = [
        int(pd.read_csv(os.path.join(train_report_root, sample.replace(".pt", ".csv")), header=None).iloc[0, 0])
        for sample in train_val_sample
    ]
    train_sample, val_sample = train_test_split(
        train_val_sample,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    test_sample = sorted(os.listdir(test_histology_root))
    test_sample = [i for i in test_sample if i.endswith('pt')]

    train_dataset = Dataset(train_swt_root, train_swv_root, train_bus_root, train_swt_fea_root, train_swv_fea_root, train_bus_fea_root, train_histology_root, train_report_root, train_augmentation, padding_size, train_sample, global_seed=42)
    val_dataset = Dataset(train_swt_root, train_swv_root, train_bus_root, train_swt_fea_root, train_swv_fea_root, train_bus_fea_root, train_histology_root, train_report_root, test_augmentation, padding_size, val_sample, global_seed=42)
    test_dataset = Dataset(test_swt_root, test_swv_root, test_bus_root, test_swt_fea_root, test_swv_fea_root, test_bus_fea_root, test_histology_root, test_report_root, test_augmentation, padding_size, test_sample, global_seed=42)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=False)
    val_loader = data.DataLoader(val_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   pin_memory=False)
    test_loader = data.DataLoader(test_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   pin_memory=False)




    return train_loader, val_loader, test_loader


def get_multi_center_kfold_dataloader(config, seed):

    g = torch.Generator()
    g.manual_seed(seed)

    config = config.dataset.MBC

    padding_size = config.padding_size

    train_val_all_sample = sorted(os.listdir(config.kfold.train_val_histology_root))

    train_val_all_sample = [i for i in train_val_all_sample if i.endswith('pt')]

    kfold_report_root = os.path.join(config.kfold.train_val_radiology_root, 'rep')
    train_val_labels = [
        int(pd.read_csv(os.path.join(kfold_report_root, sample.replace(".pt", ".csv")), header=None).iloc[0, 0])
        for sample in train_val_all_sample
    ]
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    test_augmentation = give_augmentations(config, train=False)
    train_loaders, val_loaders, test_loaders = [], [], []
    for train_idx, val_idx in kf.split(train_val_all_sample, train_val_labels):

        train_sample = np.array(train_val_all_sample)[train_idx]
        test_sample = np.array(train_val_all_sample)[val_idx]

        radiology_root = config.kfold.train_val_radiology_root

        swt_root = os.path.join(radiology_root, 'swt')
        swv_root = os.path.join(radiology_root, 'swv')
        bus_root = os.path.join(radiology_root, 'bus')

        swt_fea_root = os.path.join(radiology_root, 'swt_fea')
        swv_fea_root = os.path.join(radiology_root, 'swv_fea')
        bus_fea_root = os.path.join(radiology_root, 'bus_fea')

        report_root = os.path.join(radiology_root, 'rep')

        histology_root = config.kfold.train_val_histology_root
        train_augmentation = give_augmentations(config, train=True)
        train_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, train_augmentation, padding_size, train_sample, global_seed=seed)
        val_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, test_augmentation, padding_size, test_sample, global_seed=seed)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       pin_memory=False,
                                       generator=g
                                       )
        val_loader = data.DataLoader(val_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers,
                                       pin_memory=False,
                                        generator=g
                                     )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)


    test_radiology_root = config.kfold.test_radiology_root
    test_histology_root = config.kfold.test_histology_root

    test_swt_root = os.path.join(test_radiology_root, 'swt')
    test_swv_root = os.path.join(test_radiology_root, 'swv')
    test_bus_root = os.path.join(test_radiology_root, 'bus')

    test_swt_fea_root = os.path.join(test_radiology_root, 'swt_fea')
    test_swv_fea_root = os.path.join(test_radiology_root, 'swv_fea')
    test_bus_fea_root = os.path.join(test_radiology_root, 'bus_fea')

    test_report_root = os.path.join(test_radiology_root, 'rep')

    test_sample = sorted(os.listdir(test_histology_root))
    test_sample = [i for i in test_sample if i.endswith('pt')]

    test_loaders = []
    for _ in train_loaders:
        test_dataset = Dataset(
            test_swt_root, test_swv_root, test_bus_root,
            test_swt_fea_root, test_swv_fea_root, test_bus_fea_root,
            test_histology_root, test_report_root, test_augmentation,
            padding_size, test_sample, global_seed=seed,
        )
        test_loaders.append(data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
            generator=g,
        ))

    return train_loaders, val_loaders, test_loaders



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_mix_kfold_dataloader(config, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    config = config.dataset.MBC

    padding_size = config.padding_size

    all_sample = sorted(os.listdir(config.mix.histology_root))

    all_sample = [i for i in all_sample if i.endswith('pt')]

    mix_report_root = os.path.join(config.mix.radiology_root, 'rep')
    all_labels = [
        int(pd.read_csv(os.path.join(mix_report_root, sample.replace(".pt", ".csv")), header=None).iloc[0, 0])
        for sample in all_sample
    ]
    train_val_sample, test_sample = train_test_split(
        all_sample,
        test_size=0.2,
        random_state=seed,
        stratify=all_labels,
    )

    train_val_labels = [
        int(pd.read_csv(os.path.join(mix_report_root, sample.replace(".pt", ".csv")), header=None).iloc[0, 0])
        for sample in train_val_sample
    ]
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

    radiology_root = config.mix.radiology_root

    swt_root = os.path.join(radiology_root, 'swt')
    swv_root = os.path.join(radiology_root, 'swv')
    bus_root = os.path.join(radiology_root, 'bus')

    swt_fea_root = os.path.join(radiology_root, 'swt_fea')
    swv_fea_root = os.path.join(radiology_root, 'swv_fea')
    bus_fea_root = os.path.join(radiology_root, 'bus_fea')

    report_root = os.path.join(radiology_root, 'rep')

    histology_root = config.mix.histology_root

    test_augmentation = give_augmentations(config, train=False)
    train_loaders, val_loaders, test_loaders = [], [], []
    for train_idx, val_idx in kf.split(train_val_sample, train_val_labels):

        train_sample = np.array(train_val_sample)[train_idx]
        val_sample = np.array(train_val_sample)[val_idx]
        train_augmentation = give_augmentations(config, train=True)
        train_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, train_augmentation, padding_size, train_sample, global_seed=seed)
        val_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, test_augmentation, padding_size, val_sample, global_seed=seed)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       pin_memory=False,
                                       worker_init_fn=seed_worker,
                                       generator=g
                                       )
        val_loader = data.DataLoader(val_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers,
                                       pin_memory=False,
                                       worker_init_fn=seed_worker,
                                       generator=g
                                     )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)


        test_dataset = Dataset(
            swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root,
            bus_fea_root, histology_root, report_root, test_augmentation,
            padding_size, test_sample, global_seed=seed,
        )
        test_loaders.append(data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        ))

    return train_loaders, val_loaders, test_loaders

def get_mix_dataloader(config):
    config = config.dataset.MBC

    radiology_root = config.mix.radiology_root

    padding_size = config.padding_size

    all_sample = sorted(os.listdir(config.mix.histology_root))

    all_sample = [i for i in all_sample if i.endswith('pt')]

    all_labels = [
        int(
            pd.read_csv(
                osp.join(
                    radiology_root,
                    'rep',
                    sample.replace(".pt", ".csv"),
                ),
                header=None,
            ).iloc[0, 0]
        )
        for sample in all_sample
    ]
    train_sample, test_sample = train_test_split(
        all_sample,
        test_size=1 - config.mix.ratio,
        random_state=42,
        stratify=all_labels,
    )



    swt_root = os.path.join(radiology_root, 'swt')
    swv_root = os.path.join(radiology_root, 'swv')
    bus_root = os.path.join(radiology_root, 'bus')

    swt_fea_root = os.path.join(radiology_root, 'swt_fea')
    swv_fea_root = os.path.join(radiology_root, 'swv_fea')
    bus_fea_root = os.path.join(radiology_root, 'bus_fea')

    report_root = os.path.join(radiology_root, 'rep')
    histology_root = config.mix.histology_root
    train_augmentation = give_augmentations(config, train=True)
    test_augmentation = give_augmentations(config, train=False)

    train_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, train_augmentation, padding_size, train_sample, global_seed=42)
    test_dataset = Dataset(swt_root, swv_root, bus_root, swt_fea_root, swv_fea_root, bus_fea_root, histology_root, report_root, test_augmentation, padding_size, test_sample, global_seed=42)

    train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers,
                                       pin_memory=False)
    test_loader = data.DataLoader(test_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers,
                                       pin_memory=False)

    return train_loader, test_loader


def get_single_center_kfold_dataloader(config, seed=42):
    return get_multi_center_kfold_dataloader(config, seed)




if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    config = EasyDict(yaml.load(open('../config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    # train_loader, val_loader = get_multicenter_dataloader(config)
    # train_num = 0
    # for i, image_batch in enumerate(train_loader):
    #     print(len(image_batch[0]))
    #
    #     train_num += len(image_batch[0])
    # print(train_num)
    #
    # test_num = 0
    # for i, image_batch in enumerate(val_loader):
    #     print(len(image_batch[0]))
    #
    #     test_num += len(image_batch[0])
    # print(test_num)

    train_loaders, val_loaders, test_loader = get_single_center_kfold_dataloader(config)
    # train_num = 0
    # for i, image_batch in enumerate(train_loader):
    #     print(len(image_batch[0]))
    #
    #     train_num += len(image_batch[0])
    # print(train_num)

    # test_num = 0
    # for i, image_batch in enumerate(test_loader):
    #     modal_swt, modal_swv, modal_bus = image_batch
    #     if i == 0:
    #         print(modal_swt[0])
    #     print(modal_swt[0].size())
    #     print(modal_swt[1].size())
    #     test_num += 1
    # print(train_num)
    # print(test_num)
    # print(train_num + test_num)


    # train_loader, val_loader = get_unimodal_dataloader(config)
