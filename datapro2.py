import numpy as np
import os
import torch
import csv
import scipy.sparse as sp
import torch.utils.data.dataset as Dataset
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
from GCN import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def mp_data(dm):
    """
    构建元路径
    dmd: dm * md
    mdm: md * dm
    """
    D, M = dm.shape
    md = dm.T
    dmd = np.matmul(dm, md)

    mdm = np.matmul(md, dm)

    return dmd, mdm

def loading_data(param):
    ratio = param.ratio

    md_matrix = pd.read_csv(('circR2Disease/CircR2Disease_Association.csv'), encoding='utf-8-sig',header=None)

    rng = np.random.default_rng(seed=99)  
    pos_samples = np.where(md_matrix == 1) 
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # get the edge of negative samples
    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]

    idx_split = int(n_pos_samples * ratio)

    ##seed=42. The data classes in test and train are the same as those in './train_test/'
    test_pos_edges = pos_samples_shuffled[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))


    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
 

    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    edge_idx_dict['true_md'] = md_matrix  ##(585, 88)
    non_zero_indices = np.transpose(np.nonzero(md_matrix))

    edge_idx_dict['train_md'] = non_zero_indices  ##(650, 2)
    # edge_idx_dict['edges']=np.vstack((train_edges, test_edges))
    
    return edge_idx_dict



def read_csv(path):
    with open(path, 'r', newline='',encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    """
    构建一个 edge 的列表，添加边。
    
    参数:
    matrix - 一个 PyTorch 张量
    
    返回:
    一个形状为 (2, num_edges) 的 LongTensor，其中每列表示一条边的两个节点索引。
    """
    # 获取非零元素的索引
    edge_index = matrix.nonzero(as_tuple=False).t()
    return edge_index

def Simdata_pro(param):
    dataset = dict()

    "miRNA functional sim"
    mm_f_matrix = read_csv('circR2Disease/circFuncSimilarity_circR2.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix.to(device), 'edges': mm_f_edge_index.to(device)}
    # dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}
    "disease semantic sim"

    dd_s_matrix = read_csv('circR2Disease/disease_semantic_similarity2.csv') 
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix.to(device), 'edges': dd_s_edge_index.to(device)}
    "miRNA Gaussian sim"
    mm_g_matrix = read_csv('circR2Disease/circR2_rna_GaussianSimilarity.csv')
    mm_g_edge_index = get_edge_index(mm_g_matrix)
    dataset['mm_g'] = {'data_matrix': mm_g_matrix.to(device), 'edges': mm_g_edge_index.to(device)}
    "disease Gaussian sim"
    dd_g_matrix = read_csv('circR2Disease/circR2_dis_GaussianSimilarity.csv')
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix.to(device), 'edges': dd_g_edge_index.to(device)}

    mm_I_matrix = read_csv('circR2Disease/circR2_intercircSimilarity.csv')
    mm_I_edge_index = get_edge_index(mm_I_matrix)
    dataset['mm_I'] = {'data_matrix': mm_I_matrix.to(device),'edges': mm_I_edge_index.to(device)}
    dd_I_matrix = read_csv('circR2Disease/circR2_interdisSimilarity.csv')

    dd_I_edge_index = get_edge_index(dd_I_matrix)
    dataset['dd_I'] = {'data_matrix': dd_I_matrix.to(device),'edges': dd_I_edge_index.to(device)}
    md_a_matrix= read_csv('circ2Disease/Circ2Disease_Association1.csv')
    md_edge_index = get_edge_index(md_a_matrix)
    dataset['md']= {'data_matrix': md_a_matrix.to(device), 'edges': md_edge_index.to(device)}
    
    
    
    # dataset['md']= {'data_matrix': md_a_matrix, 'edges': md_edge_index}
    circRNA_nodes=md_a_matrix.shape[0]
    disease_nodes=md_a_matrix.shape[1]
    # ####开始写可达性矩阵的代码#####

    cdc,dcd=mp_data(md_a_matrix)
    dataset['cdc']={'data_matrix': cdc}
    dataset['dcd']={'data_matrix': dcd}
    cdc_edge_index = get_edge_index(cdc)
    dataset['cdc_I'] = {'data_matrix': cdc.to(device), 'edges': cdc_edge_index.to(device)}
    dmd_edge_index = get_edge_index(dcd)
    dataset['dcd_I'] = {'data_matrix': dcd.to(device), 'edges': dmd_edge_index.to(device)}
    return dataset



class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):

        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label




