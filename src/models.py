

# from src_cls.SwinUMamba.SwinUMamba3 import SwinUMamba3
from src.cnn.cnn import MultiModalCNN
from src.cnn.cnn_test import MultiModalCNNTest
from src.densenet.densenet import MultiModalDenseNet
# from src.vit.vit import MultiModalVisionTransformer
from src.sh_dlrp.sh_dlrp import SHDLRP

from src.extractors import give_extractor

def give_model(config):

    if config.finetune.model_choose == 'cnn':
        model = MultiModalCNN(**config.models.cnn)

    elif config.finetune.model_choose == 'cnn_test':
        model = MultiModalCNNTest(**config.models.cnn)

    elif config.finetune.model_choose == 'densenet':
        print("Model:densenet")
        model = MultiModalDenseNet(**config.models.densenet)

    # elif config.finetune.model_choose == 'vit':
    #     print("Model:vit")
    #     model = MultiModalVisionTransformer(**config.models.vit)

    elif config.finetune.model_choose == 'sh_dlrp':
        print("Model:SH-DLRP")
        extractor = give_extractor(config)
        model = SHDLRP(extractor=extractor, extractor_choose=config.finetune.extractor_choose, **config.models.sh_dlrp)
    else:
        assert 0, "Must choose a model!"



    return model
