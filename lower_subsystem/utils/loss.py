from collections import defaultdict

import torch
from detectron2.data import build_detection_test_loader

from utils.albumentations_mapper import AlbumentationsMapper


def _dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = float(sum(d[key] for d in dict_list)) / len(dict_list)
    return mean_dict


class ValidationLossEval:
    def __init__(self, cfg, model, dataset_name):
        self.model = model
        mapper = AlbumentationsMapper(cfg, is_train=False)
        self.data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def get_loss(self):
        losses = []
        for idx, inputs in enumerate(self.data_loader):
            with torch.no_grad():
                loss_dict = self.model(inputs)
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
            losses.append(loss_dict)
        mean_loss_dict = _dict_mean(losses)
        return mean_loss_dict


class MeanTrainLoss:
    def __init__(self):
        self._mean_dict = defaultdict(float)
        self._count = 0

    def update(self, loss_dict):
        self._count += 1
        for k, v in loss_dict.items():
            self._mean_dict[k] += (v - self._mean_dict[k]) / self._count

    def get_losses(self):
        return self._mean_dict

    def get_total_loss(self):
        return sum(loss for loss in self._mean_dict.values())

    def reset(self):
        self._mean_dict.clear()
        self._count = 0
