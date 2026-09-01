import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
import time
try:
    from objprint import objstr
except ImportError:
    from pprint import pformat

    def objstr(value):
        return pformat(value)

import torch
import yaml
import random
from src.utils import Logger

from easydict import EasyDict

from torch import nn, optim

from src import utils
from src.models import give_model

import pandas as pd

import numpy as np



from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_score = np.asarray(y_score)

    if y_true.size == 0:
        raise ValueError("Cannot calculate metrics for an empty data loader")

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_true, y_score[:, 1]) if np.unique(y_true).size == 2 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = _safe_div(tp, tp + fn)
    spe = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    tpr = sen
    fpr = _safe_div(fp, tn + fp)
    fnr = _safe_div(fn, fn + tp)
    return f1, acc, auc, sen, spe, ppv, npv, tpr, fpr, fnr



def train_epoch(model, train_loader, criterion, optimizer, warmup, lr_scheduler, epoch, device):
    model.train()
    total_loss = 0
    y_pred = []
    y_true = []
    y_score = []
    for i, image_batch in enumerate(train_loader):

        label_1, swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, label_2, sample_name =image_batch[0], image_batch[1],image_batch[2],image_batch[3],image_batch[4],image_batch[5],image_batch[6], image_batch[7], image_batch[8], image_batch[9], image_batch[10], image_batch[11]

        if epoch < warmup:
            iteration = epoch * len(train_loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        label_1 = label_1.to(device)
        label_2 = label_2.to(device)



        swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report = swt.to(device), swv.to(device), bus.to(
            device), swt_fea.to(device), swv_fea.to(device), bus_fea.to(device), wsi.to(device), mask.to(
            device), report.to(device)
        # image_batch[0] = torch.cat((modal_TC[0], modal_VC[0], modal_VG[0]), dim=1)
        input = (swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, sample_name)


        model_output = model(input, is_test=False)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output

        loss = criterion(logits, label_1)

        y_score.append(logits)
        y_true.append(label_1)
        y_pred.append(logits.argmax(dim=-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Training [{}/{}] Loss: {}".format(i + 1, len(train_loader), loss))

        total_loss += loss.item()

    y_score = torch.cat(y_score).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy().reshape(-1)
    y_true = torch.cat(y_true).detach().cpu().numpy()

    f1, acc, auc, sen, spe, ppv, npv, tpr, fpr, fnr = _compute_metrics(y_true, y_pred, y_score)

    epoch_loss = total_loss / len(train_loader)

    return y_score, y_pred, y_true, epoch_loss, f1, acc, auc, sen, spe, ppv, npv, tpr, fpr, fnr


def eval_epoch(model, data_loader, criterion, device, train_acc, train_auc, train_sen, acc_threshold=0.9,
               auc_threshold=0.9, sen_threshold=0.9, adjust_labels=True, is_test=False):

    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    y_score = []
    sample_names = []
    with torch.no_grad():
        for i, image_batch in enumerate(data_loader):
            label_1, swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, label_2, sample_name = image_batch[0], \
            image_batch[1], image_batch[2], image_batch[3], image_batch[4], image_batch[5], image_batch[6], image_batch[
                7], image_batch[8], image_batch[9], image_batch[10], image_batch[11]

            label_1 = label_1.to(device)
            label_2 = label_2.to(device)

            swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report = swt.to(device), swv.to(device), bus.to(
                device), swt_fea.to(device), swv_fea.to(device), bus_fea.to(device), wsi.to(device), mask.to(
                device), report.to(device)

            input = (swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, sample_name)

            model_output = model(input, is_test)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output

            loss = criterion(logits, label_1)

            sample_names.append(sample_name)
            y_score.append(logits)
            y_true.append(label_1)
            y_pred.append(logits.argmax(dim=-1))
            print(f"{'Test' if is_test else 'Validation'} [{i + 1}/{len(data_loader)}] Loss: {loss.item():.4f}")

            total_loss += loss.item()

    # 转换为numpy数组进行计算
    y_score = torch.cat(y_score).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy().reshape(-1)
    y_true = torch.cat(y_true).detach().cpu().numpy()

    f1, acc, auc, sen, spe, ppv, npv, tpr, fpr, fnr = _compute_metrics(y_true, y_pred, y_score)

    epoch_loss = total_loss / len(data_loader)

    return y_score, y_pred, y_true, epoch_loss, f1, acc, auc, sen, spe, ppv, npv, tpr, fpr, fnr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






if __name__ == '__main__':

    seed = 42

    set_seed(seed)


    torch.multiprocessing.set_start_method('spawn', force=True)

    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    if config.trainer.num_epochs <= 0:
        raise ValueError("trainer.num_epochs must be positive")
    utils.same_seeds(seed)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + '-' + config.models.sh_dlrp.modal + '-' + str(datetime.now()).replace(' ', '-')


    device = "cuda" if torch.cuda.is_available() else "cpu"


    Logger(logging_dir)
    # accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    print(objstr(config))

    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr


    print('Load Dataloader...')


    from src.dataloader import (
        get_mix_kfold_dataloader,
        get_multicenter_dataloader,
        get_multi_center_kfold_dataloader,
    )

    split = config.trainer.split.lower()
    if split == "mxkf":
        train_loaders, val_loaders, test_loaders = get_mix_kfold_dataloader(config, seed)
    elif split == "kf":
        train_loaders, val_loaders, test_loaders = get_multi_center_kfold_dataloader(config, seed)
    elif split == "mc":
        train_loader, val_loader, test_loader = get_multicenter_dataloader(config)
        train_loaders, val_loaders, test_loaders = (
            [train_loader],
            [val_loader],
            [test_loader],
        )
    else:
        raise ValueError(
            "Unsupported training split '{}'. Use 'kf', 'mxkf', or 'mc'.".format(split)
        )





    print(len(train_loaders))

    train_score_list, train_acc_list, train_auc_list, train_sen_list, train_spe_list, train_ppv_list, train_npv_list, train_tpr_list, train_fpr_list, train_fnr_list = [], [], [], [], [], [], [], [], [], []
    val_score_list, val_acc_list, val_auc_list, val_sen_list, val_spe_list, val_ppv_list, val_npv_list, val_tpr_list, val_fpr_list, val_fnr_list = [], [], [], [], [], [], [], [], [], []
    best_test_score, best_test_acc, best_test_auc, best_test_sen, best_test_spe, best_test_ppv, best_test_npv, best_test_tpr, best_test_fpr, best_test_fnr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for k, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        test_loader = test_loaders[k]

        best_train_acc = 0
        best_val_acc = 0
        best_selection = -float("inf")
        best_eopch = -1
        starting_epoch = 0

        print('Load Model...')

        lr = 0.001
        warmup = min(10, max(0, config.trainer.num_epochs - 1))



        criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.3, 0.7]).to(device))
        # criterion = nn.CrossEntropyLoss()

        model = give_model(config)


        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, config.trainer.num_epochs - warmup)
        )
        lr_steps = lr / (warmup * len(train_loader)) if warmup else lr


        best_score, best_acc, best_auc, best_sen, best_spe, best_ppv, best_npv, best_tpr, best_fpr, best_fnr = (0,) * 10
        t_score, t_acc, t_auc, t_sen, t_spe, t_ppv, t_npv, t_tpr, t_fpr, t_fnr = (0,) * 10
        save_path = os.path.join(
            os.getcwd(),
            "model_store",
            config.finetune.checkpoint + "-" + config.models.sh_dlrp.modal + "-" + config.trainer.split,
        )
        os.makedirs(save_path, exist_ok=True)
        best_test_name = "best-test-" + str(k) + ".pth"

        for epoch in range(starting_epoch, config.trainer.num_epochs):
            # 训练
            start_time = time.time()

            y_score, y_pred, y_true, train_loss, train_score, train_acc, train_auc, train_sen, train_spe, train_ppv, train_npv, train_tpr, train_fpr, train_fnr = train_epoch(
                model, train_loader,
                criterion, optimizer, warmup,
                warmup_lr_scheduler, epoch, device)


            val_y_score, val_y_pred, val_y_true, val_loss, val_score, val_acc, val_auc, val_sen, val_spe, val_ppv, val_npv, val_tpr, val_fpr, val_fnr = eval_epoch(
                model, val_loader, criterion, device, train_acc, train_auc, train_sen, is_test=False)

            if epoch >= warmup:
                lr_scheduler.step()

            print(
                "K{} Epoch:{}: Train loss: {:.4f} Train score:{:.4f} Train acc:{:.4f} Train auc:{:.4f}  Train sen:{:.4f}  Train spe:{:.4f}  Train ppv:{:.4f}  Train npv:{:.4f}  Train tpr:{:.4f}  Train fpr:{:.4f} Train fnr:{:.4f}".format(
                    str(k), epoch + 1, train_loss, train_score, train_acc, train_auc, train_sen, train_spe, train_ppv,
                    train_npv, train_tpr, train_fpr, train_fnr))
            print(
                "K{} Epoch:{}: Val loss: {:.4f} Val score:{:.4f} Val acc:{:.4f} Val auc:{:.4f} Val sen:{:.4f} Val spe:{:.4f} Val ppv:{:.4f}  Val npv:{:.4f}  Val tpr:{:.4f}  Val fpr:{:.4f}  Val fnr:{:.4f}".format(
                    str(k), epoch + 1, val_loss, val_score, val_acc, val_auc, val_sen, val_spe, val_ppv, val_npv,
                    val_tpr,
                    val_fpr, val_fnr))

            val_selection = val_auc if not np.isnan(val_auc) else val_acc
            if best_eopch == -1 or val_selection > best_selection:

                print("Best Result!")

                t_score, t_acc, t_auc, t_sen, t_spe, t_ppv, t_npv, t_tpr, t_fpr, t_fnr = train_score, train_acc, train_auc, train_sen, train_spe, train_ppv, train_npv, train_tpr, train_fpr, train_fnr
                best_score, best_acc, best_auc, best_sen, best_spe, best_ppv, best_npv, best_tpr, best_fpr, best_fnr = val_score, val_acc, val_auc, val_sen, val_spe, val_ppv, val_npv, val_tpr, val_fpr, val_fnr
                best_selection = val_selection

                y_score, y_pred, y_true = pd.DataFrame(y_score), pd.DataFrame(y_pred), pd.DataFrame(y_true)
                val_y_score, val_y_pred, val_y_true = pd.DataFrame(val_y_score), pd.DataFrame(val_y_pred), pd.DataFrame(val_y_true)

                y_score.to_csv(os.path.join(save_path, "y_score-" + str(k) + ".csv"), header=False, index=False)
                y_pred.to_csv(os.path.join(save_path, "y_pred-" + str(k) + ".csv"), header=False, index=False)
                y_true.to_csv(os.path.join(save_path, "y_true-" + str(k) + ".csv"), header=False, index=False)

                val_y_score.to_csv(os.path.join(save_path, "val_y_score-" + str(k) + ".csv"), header=False, index=False)
                val_y_pred.to_csv(os.path.join(save_path, "val_y_pred-" + str(k) + ".csv"), header=False, index=False)
                val_y_true.to_csv(os.path.join(save_path, "val_y_true-" + str(k) + ".csv"), header=False, index=False)

                torch.save(model.state_dict(), os.path.join(save_path, best_test_name))

                best_eopch = epoch

            end_time = time.time()

            all_time = end_time - start_time

        model.load_state_dict(torch.load(os.path.join(save_path, best_test_name), map_location=device))
        test_y_score, test_y_pred, test_y_true, test_loss, test_score, test_acc, test_auc, test_sen, test_spe, test_ppv, test_npv, test_tpr, test_fpr, test_fnr = eval_epoch(
            model, test_loader, criterion, device, t_acc, t_auc, t_sen, is_test=True)
        print(
            "K{}: Test loss: {:.4f} Test score:{:.4f} Test acc:{:.4f} Test auc:{:.4f} Test sen:{:.4f} Test spe:{:.4f} Test ppv:{:.4f} Test npv:{:.4f} Test tpr:{:.4f} Test fpr:{:.4f} Test fnr:{:.4f}".format(
                str(k), test_loss, test_score, test_acc, test_auc, test_sen, test_spe, test_ppv, test_npv,
                test_tpr, test_fpr, test_fnr))
        pd.DataFrame(test_y_score).to_csv(os.path.join(save_path, "test_y_score-" + str(k) + ".csv"), header=False, index=False)
        pd.DataFrame(test_y_pred).to_csv(os.path.join(save_path, "test_y_pred-" + str(k) + ".csv"), header=False, index=False)
        pd.DataFrame(test_y_true).to_csv(os.path.join(save_path, "test_y_true-" + str(k) + ".csv"), header=False, index=False)

        del model
        del train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_score_list.append(t_score)
        train_acc_list.append(t_acc)
        train_auc_list.append(t_auc)
        train_sen_list.append(t_sen)
        train_spe_list.append(t_spe)
        train_ppv_list.append(t_ppv)
        train_npv_list.append(t_npv)
        train_tpr_list.append(t_tpr)
        train_fpr_list.append(t_fpr)
        train_fnr_list.append(t_fnr)

        val_score_list.append(best_score)
        val_acc_list.append(best_acc)
        val_auc_list.append(best_auc)
        val_sen_list.append(best_sen)
        val_spe_list.append(best_spe)
        val_ppv_list.append(best_ppv)
        val_npv_list.append(best_npv)
        val_tpr_list.append(best_tpr)
        val_fpr_list.append(best_fpr)
        val_fnr_list.append(best_fnr)

    print(train_score_list)
    print(train_acc_list)
    print(train_auc_list)
    print(train_sen_list)
    print(train_spe_list)
    print(train_ppv_list)
    print(train_npv_list)
    print(train_tpr_list)
    print(train_fpr_list)
    print(train_fnr_list)

    print(
        " Train score:{:.4f} ~ {:.4f} Train acc:{:.4f} ~ {:.4f} Train auc:{:.4f} ~ {:.4f}  Train sen:{:.4f} ~ {:.4f}  Train spe:{:.4f} ~ {:.4f}  Train ppv:{:.4f} ~ {:.4f}  Train npv:{:.4f} ~ {:.4f}  Train tpr:{:.4f} ~ {:.4f}  Train fpr:{:.4f} ~ {:.4f}  Train fnr:{:.4f} ~ {:.4f}".format(
            np.mean(train_score_list), np.std(train_score_list),
            np.mean(train_acc_list), np.std(train_acc_list),
            np.mean(train_auc_list), np.std(train_auc_list),
            np.mean(train_sen_list), np.std(train_sen_list),
            np.mean(train_spe_list), np.std(train_spe_list),
            np.mean(train_ppv_list), np.std(train_ppv_list),
            np.mean(train_npv_list), np.std(train_npv_list),
            np.mean(train_tpr_list), np.std(train_tpr_list),
            np.mean(train_fpr_list), np.std(train_fpr_list),
            np.mean(train_fnr_list), np.std(train_fnr_list))
    )

    print(val_score_list)
    print(val_acc_list)
    print(val_auc_list)
    print(val_sen_list)
    print(val_spe_list)
    print(val_ppv_list)
    print(val_npv_list)
    print(val_tpr_list)
    print(val_fpr_list)
    print(val_fnr_list)

    print(
        "Val score:{:.4f} ~ {:.4f} Val acc:{:.4f} ~ {:.4f} Val auc:{:.4f} ~ {:.4f} Val sen:{:.4f} ~ {:.4f} Val spe:{:.4f} ~ {:.4f} Val ppv:{:.4f} ~ {:.4f}  Val npv:{:.4f} ~ {:.4f}  Val tpr:{:.4f} ~ {:.4f}  Val fpr:{:.4f} ~ {:.4f}  Val fnr:{:.4f} ~ {:.4f}".format(
            np.mean(val_score_list), np.std(val_score_list),
            np.mean(val_acc_list), np.std(val_acc_list),
            np.mean(val_auc_list), np.std(val_auc_list),
            np.mean(val_sen_list), np.std(val_sen_list),
            np.mean(val_spe_list), np.std(val_spe_list),
            np.mean(val_ppv_list), np.std(val_ppv_list),
            np.mean(val_npv_list), np.std(val_npv_list),
            np.mean(val_tpr_list), np.std(val_tpr_list),
            np.mean(val_fpr_list), np.std(val_fpr_list),
            np.mean(val_fnr_list), np.std(val_fnr_list))
    )
