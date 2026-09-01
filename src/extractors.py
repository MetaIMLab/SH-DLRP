# from src.vim.vim import VisionMamba
from src.vit.vit import VisionTransformer
from src.resnet.resnet import ResNet, BasicBlock
import torch
import os

def give_extractor(config):


    # if config.finetune.extractor_choose == 'vim':
    #     extractor = VisionMamba(**config.extractors.vim)
    #     if config.extractors.vim.use_pretrain:
    #         assert os.path.exists(config.extractors.vim.pretrain_dir), "weights file: '{}' not exist.".format(config.extractors.vim.pretrain_dir)
    #
    #         checkpoint = torch.load(config.extractors.vim.pretrain_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    #         extractor.load_state_dict(checkpoint["model"])

    if config.finetune.extractor_choose == 'vit':
        extractor = VisionTransformer(**config.extractors.vit)
        if config.extractors.vit.use_pretrain:
            assert os.path.exists(config.extractors.vit.pretrain_dir), "weights file: '{}' not exist.".format(
                config.extractors.vit.pretrain_dir)

            has_logits = True

            weights_dict = torch.load(config.extractors.vit.pretrain_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            del_keys = ['head.weight', 'head.bias'] if has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]

            extractor.load_state_dict(weights_dict, strict=False)

    elif config.finetune.extractor_choose == 'resnet':
        extractor = ResNet(block=BasicBlock, **config.extractors.resnet)
        if config.extractors.resnet.use_pretrain:
            extractor.load_state_dict(torch.load(config.extractors.resnet.pretrain_dir, map_location='cpu'))

    else:
        assert 0, "Must choose a extractor!"

    return extractor