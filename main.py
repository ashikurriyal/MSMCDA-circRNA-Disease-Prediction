import torch
import random
import numpy as np
from datapro2 import Simdata_pro,loading_data

from train import train_test
# from train_fea import train_test
import warnings

import os
import sys


class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 32
        self.ratio = 0
        self.epoch =2
        self.gcn_layers = 2
        self.view = 2    
        self.fm = 128
        self.nhid=64 
        self.hidden_dim=64
        self.fd = 128
        self.miRNA_number= 585
        self.disease_number=88
        self.fcDropout = 0.5
        self.dropout = 0.5
        self.shared_unit_num=4
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main():
    param = Config()
    simData = Simdata_pro(param)  #构图构边，所有的
    train_data = loading_data(param)  ##加载 数据,选取正负样本
    result= train_test(simData, train_data, param, state='valid')
    


if __name__ == "__main__":
    main()
