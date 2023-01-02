# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:09:17 2020
@author: ZHU Haoren
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetPrice(Dataset):
    def __init__(self, data):
        self.data = data

    #    def set_mode(self, train=True):
    #        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def _collate_fn(batch):
    sorted_seq = sorted(batch, key=lambda line: line[0][0], reverse=False)
    minibatch_size = len(batch)
    # print(minibatch_size) 512
    # print(sorted_seq[0]) (23, 23)
    idxs = []
    input_frames = []
    output_frames = []
    for i in range(minibatch_size):
        idxs.append(sorted_seq[i][0])

        input_frames.append(sorted_seq[i][1])

        output_frames.append(sorted_seq[i][2])

    # Consider torch.int rather than torch.long
    return torch.tensor(idxs, dtype=torch.float), \
           torch.tensor(input_frames, dtype=torch.float), \
           torch.tensor(output_frames, dtype=torch.float)


class DataLoaderPrice(DataLoader):

        def __init__(self, *args, **kwargs):
            super(DataLoaderPrice, self).__init__(*args, **kwargs)  # 对父类进行初始化
            self.collate_fn = _collate_fn
