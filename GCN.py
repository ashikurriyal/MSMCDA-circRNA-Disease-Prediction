import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import numpy as np
import scipy.sparse as sp
import torch
import torch as th


def add_self_loops(adj):
    # 给邻接矩阵 adj 添加自环
    identity = torch.eye(adj.size(0)).to(adj.device)  # 生成单位矩阵
    adj_with_self_loops = adj + identity  # 添加自环
    return adj_with_self_loops
class seGCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(seGCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=3)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj): ##结点特征矩阵，连接矩阵
        seq_fts = self.fc(seq)
        out = torch.mm(adj, seq_fts)  # 使用带自环的邻接矩阵
        if self.bias is not None:
            out += self.bias
        return self.act(out)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj): 
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj): ###手动标准化邻接矩阵
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+sp.eye(adj.shape[0])
    return adj_normalized.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(   ###稀疏矩阵的行和列垂直堆叠
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)
##上面的值都是转换成稀疏张量用的

