from torch import nn as nn
import torch
from math import sqrt
# import dgl.function as fn
import torch as th
import torch.nn.functional as F
# from dgl import DGLGraph
# from dgl.nn.pytorch import RelGraphConv
import numpy as np



class Cross_stitch(nn.Module):
    def __init__(self):
        super(Cross_stitch,self).__init__()
        # self.out_dim=out_dim
        self.w_aa = nn.Parameter(th.Tensor(1,))
        self.w_aa.data=th.tensor(np.random.random(),requires_grad=True)
        
        self.w_ab=nn.Parameter(th.Tensor(1,))
        self.w_ab.data=th.tensor(np.random.random(),requires_grad=True)
        self.w_ba=nn.Parameter(th.Tensor(1,))
        self.w_ba.data=th.tensor(np.random.random(),requires_grad=True)
        self.w_bb=nn.Parameter(th.Tensor(1,))
        self.w_bb.data=th.tensor(np.random.random(),requires_grad=True)
        # np.random.random()
        
        
        print(self.w_aa)
    def forward(self,drug_cnn,drug_kg,):
        drug_cnn_=self.w_aa*drug_cnn+self.w_ab*drug_kg
        drug_kg_=self.w_ba*drug_cnn+self.w_bb*drug_kg
        
        # print('shared parameters: w_aa:{:.4f}, w_ab:{:.4f}, w_ba:{:.4f}, w_bb:{:.4f}'.format(self.w_aa,self.w_ab,self.w_ba,self.w_bb))
        
        return drug_cnn_,drug_kg_


if __name__=='__main__':
    au=AttentionUnit(input_dim=10)
    drug_cnn=th.Tensor(1,10)
    drug_kg=th.Tensor(1,10)
    au(drug_cnn, drug_kg)
    




class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name

    def forward(self, x):
        return self.bn(x)


class BnodeEmbedding(nn.Module):
    def __init__(self, embedding, dropout, freeze=False):
        super(BnodeEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embedding, dtype=torch.float32).detach(), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):

        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.out1=nn.Linear(16,1)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)#batchsize*featuresize
        if self.outBn: x = self.bns(x) if len(x.shape) == 2 else self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, dropout, layers, resnet, actFunc, outBn=False, outAct=True, outDp=True):
        super(GCN, self).__init__()
        self.gcnlayers = layers
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        Z_zero = x# batchsize*node_num*featuresize 128*66*128
        m_all = Z_zero[:, 0, :].unsqueeze(dim=1)#batchsize*1*featuesize torch.Size([128, 1, 128])
        d_all = Z_zero[:, 1, :].unsqueeze(dim=1) #torch.Size([128, 1, 128])

        for i in range(self.gcnlayers):
            a = self.out(torch.matmul(L, x))
            if self.outBn:
                if len(L.shape) == 3:
                    a = self.bns(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = self.bns(a)  #torch.Size([128, 66, 128])
            if self.outAct: a = self.actFunc(a)
            if self.outDp: a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a  #torch.Size([128, 66, 128])
            m_this = x[:, 0, :].unsqueeze(dim=1) #torch.Size([128, 1, 128])
            d_this = x[:, 1, :].unsqueeze(dim=1)
            m_all = torch.cat((m_all, m_this), 1)
            d_all = torch.cat((d_all, d_this), 1)


        return m_all, d_all



class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers):
        super(LayerAtt, self).__init__()
        self.layers = gcnlayers + 1
        self.inSize = inSize
        self.outSize = outSize
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.actfun2 = nn.ReLU()
        self.attcnn = nn.Conv1d(in_channels=self.layers, out_channels=1, kernel_size=1, stride=1,
                            bias=True)

    def forward(self, x):# batchsize*gcn_layers*featuresize
        Q = self.q(x)   
        K = self.k(x)
        V = self.v(x)
        out = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm  #torch.Size([128, 3, 3])
        alpha = self.actfun1(out)# according to gcn_layers #torch.Size([128, 3, 3])
        z = torch.bmm(alpha, V)  #torch.Size([128, 3, 128])
        # cnnz = self.actfun2(z)  

        # cnnz = self.attcnn(x) #做消融用这个
        ########layatt
        cnnz = self.attcnn(z)  #torch.Size([128, 1, 128])
        ###########
        # cnnz = self.actfun2(cnnz)
        finalz = cnnz.squeeze(dim=1)  #torch.Size([128, 128])

        return finalz
