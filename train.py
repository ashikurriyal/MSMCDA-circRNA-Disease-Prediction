import time
import torch
import random
from datapro2 import CVEdgeDataset
from model import AMHMDA, EmbeddingM, EmbeddingD,ACMF,MDI##*
import os
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import copy
import warnings
import matplotlib.pyplot as plt###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torch as th
from torch_geometric.data import Data

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.to(device).manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):##*
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric,fpr,tpr = caculate_metrics(y_pre, y_true)
    return metric,fpr,tpr 


def caculate_metrics(pre_score, real_score):
    y_true = real_score###
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u) 

    y_score = [0 if j < 0.5 else 1 for j in y_pre]


    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result,fpr,tpr


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))

def train_test(simData, train_data, param,state):
    epo_metric = []
    valid_metric = []
    train_losses = []
    valid_losses = []
    train_edges = train_data['train_Edges'] ##先保存到dict字典里面，然后再从dict字典里面取出来
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']

    # 保存所有fold的验证指标
    all_valid_metrics = []
    # 假设 valid_metric 里包含 FPR 和 TPR
    all_fpr = []
    all_tpr = []
    kfolds = param.kfold
    # edgeIndex = train_data
    # trainEdges = EdgeDataset(edgeIndex, True)
    # testEdges = EdgeDataset(edgeIndex, False)
    # kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    # setup_seed(42)
    torch.manual_seed(99)

    # trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

    if state=='valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)
            
        
        for i in range(kfolds):
            train_losses = []
            valid_losses = []
            a = i + 1  
            model = AMHMDA(EmbeddingM(param), EmbeddingD(param),ACMF(param), MDI(param))##*
            # model.cuda()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  ###

            print(f'################Fold {i + 1} of {kfolds}################')
            # get train set and valid set 
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            
            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            valid_metric = []

            print("-----training-----")
            for e in range(param.epoch):
                running_loss = 0.0  ###
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()
                for f, item in enumerate(trainLoader):
                    data, label = item    ##data就是训练边的list
                    train_data = data.to(device)  
                    true_label = label.to(device)  #

                    pre_score,lossc,lossd= model(simData, train_data) ##*
                    train_loss = torch.nn.BCELoss()   
                    loss1 = train_loss(pre_score, true_label)
                    loss_cl=lossc+lossd
                    loss=loss1+loss_cl
                    # loss=loss1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()  ###好像没有用到这个东西running_loss
                    print(f"After batch {f + 1}: loss= {loss:.3f};", end='\n')###
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())
                end = time.time() 
                print('Time：%.2f \n' % (end - start))
                train_losses.append(running_loss / len(trainLoader))
                
                ##########开始验证############
                valid_loss=0
                valid_score, valid_label = [], []  ###
                model.eval()
                with torch.no_grad():
                    print("-----validing-----")
                    for f, item in enumerate(validLoader):
                        data, label = item
                        train_data = data.to(device)  ##torch.Size([32, 2])
                        pre_score,_,_= model(simData, train_data)
                        valid_loss += torch.nn.BCELoss()(pre_score, label.to(device)).item()
                        batch_score = pre_score.cpu().detach().numpy()
                    
                        valid_score = np.append(valid_score, batch_score)
                        valid_label = np.append(valid_label, label.numpy())
                    end = time.time()
                    print('Time：%.2f \n' % (end - start))
                    valid_losses.append(valid_loss / len(validLoader))

                    metric,fpr,tpr = get_metrics(valid_score, valid_label)
                    valid_metric.append(metric)
                    save_path = "AMHMDA-fast+contractive+GCN/savemodel/circR2_share_KG_fold_{}.pkl".format(a)
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), save_path)  ##      
            all_valid_metrics.append(np.array(valid_metric))
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve for Fold {a}')
            plt.legend()
            plt.savefig(f'image/loss_curve_circR2_share_KG_fold_{a}.png')

    # if state == 'valid':
            valid_metric = np.array(valid_metric)
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 0], label='AUC')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 1], label='AUPR')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 2], label='Accuracy')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 3], label='F1')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 4], label='Recall')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 5], label='Precision')
            plt.xlabel('Epoch')  
            plt.xlabel('Epoch')   
            plt.ylabel('Metrics')
            plt.title(f'Validation Metrics for Fold {a}')
            plt.legend()
            plt.savefig(f'image/validation_metrics_circR2_share_KG_fold_{a}.png')    
            with open('text/valid_metrics_circrna.txt', 'a') as file:   ##改动
    # 写入标题行 
                file.write('AUC\tAUPR\tAccuracy\tF1\tRecall\tPrecision\n')
                
                # 遍历每一行并写入文件
                for epoch in range(valid_metric.shape[0]):
                    # 将每一行的数据转换为制表符分隔的字符串
                    line = '\t'.join(map(str, valid_metric[epoch])) + '\n'
                    file.write(line)
                print('数据已写入 valid_metrics.txt 文件中')
                
            mean_valid_metrics = np.mean(all_valid_metrics, axis=0)  # 修改5：计算五折均值
                # 保存均值结果到另一个文件
        with open('text/mean_valid_metrics_circrna.txt', 'w') as file:  ##改动
            file.write('这是完整的circ2的数据集的数据')
            file.write('AUC\tAUPR\tAccuracy\tF1\tRecall\tPrecision\n')
            for epoch in range(mean_valid_metrics.shape[0]):
                line = '\t'.join(map(str, mean_valid_metrics[epoch])) + '\n'  # 修改7：确保均值结果写入 7 个指标
                file.write(line)
            print('五折均值已保存到 mean_valid_metrics_circR2v2.0.txt 文件中')
    else:
        test_score, test_label = [], []
        testEdges = CVEdgeDataset(test_edges, test_labels)
        testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
        model = AMHMDA(EmbeddingM(param), EmbeddingD(param),ACMF(param), MDI(param)) 
        model.load_state_dict(torch.load('AMHMDA-fast+contractive+GCN/savemodel/circR2_share_KG_fold_1.pkl'))
        model.to(device)
        model.eval()
        with torch.no_grad():
            start = time.time()
            for i, item in enumerate(testLoader):
                data, label = item
                test_data = data.to(device)
                pre_score,_,_= model(simData, test_data)
                print(test_data)
                
                batch_score = pre_score.cpu().detach().numpy()
                test_score = np.append(test_score, batch_score)
                test_label = np.append(test_label, label.numpy())
            end = time.time()
            print('Time：%.2f \n' % (end - start))      
            metrics = get_metrics(test_score, test_label)
            
    # Not for testing
    # print(np.array(valid_metric))
    # cv_metric = np.mean(valid_metric, axis=0)
    # print_met(cv_metric)

    

    return kfolds
    


