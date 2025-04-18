import torch
from torch import nn
from otherlayers import *
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch as th
from torch import nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn.functional as F
import scipy.sparse as sp
import dgl
from torch_geometric.nn import GCNConv
from otherlayers import Cross_stitch
import torch.nn.init as init
import math
class AMHMDA(nn.Module):
    def __init__(self, c_emd1, d_emd1,acmf, mdi):
        super(AMHMDA, self).__init__()
        self.Xc1 = c_emd1
        self.Xd1= d_emd1
        self.acmf=acmf
        self.md_supernode = mdi
        ##AMHMDA(EmbeddingM(param), EmbeddingD(param),embeddingC(param),embeddingD(param),ACMF(param), MDI(param))
    def forward(self, sim_data, train_data):##*
        Em1,Em2= self.Xc1(sim_data)  ###c-embedding1
        Ed1,Ed2= self.Xd1(sim_data)  
        lossc,lossd,cm1,cm2,dm1,dm2=self.acmf(sim_data,Em1,Ed1,Em2,Ed2)
        mFea1, dFea1 = pro_data(train_data, cm1, dm1) ##先构成所有的数据，然后从中筛选出训练集啥 的
        mFea2, dFea2 = pro_data(train_data, cm2, dm2)
        pre_asso= self.md_supernode(mFea1,mFea2, dFea1,dFea2)  ##预测部分
        return pre_asso,lossc,lossd



def pro_data(data, em, ed):
    edgeData = data.t().to(device)

    mFeaData = em  #torch.Size([853, 128])
    dFeaData = ed   #torch.Size([591, 128])
    m_index = edgeData[0]   
    d_index = edgeData[1]
    Em = torch.index_select(mFeaData, 0, m_index) #torch.Size([128, 128])
    Ed = torch.index_select(dFeaData, 0, d_index)

    return Em, Ed



class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args

        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_g = GCNConv(self.args.fm, self.args.fm)
        self.fc_list=nn.Linear(args.miRNA_number,self.args.fm,bias=True)
        nn.init.xavier_uniform_(self.fc_list.weight)
        if self.fc_list.bias is not None:
            self.fc_list.bias.data.fill_(0.0)
        self.gcn_I1 = GCNConv(self.args.fm, self.args.fm) 
        self.gcn_I2 = GCNConv(self.args.fm, self.args.fm)
        self.gcn1=GCNConv(args.miRNA_number,self.args.fm) 
        self.num_shared_layer=self.args.shared_unit_num
        self.shared_units=nn.ModuleList()
        shared_unit_num=self.args.shared_unit_num  
        for i in range(shared_unit_num):  ####等于1
            self.shared_units.append(Cross_stitch())
            if self.args.dropout > 0:
                self.feat_drop = nn.Dropout(self.args.dropout)
            else: 
                self.feat_drop = lambda x: x
        self.fc1_x = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view * self.args.gcn_layers)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_x = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)  
        miRNA_number = len(data['mm_f']['data_matrix'])
        x_m = torch.randn(miRNA_number, self.args.fm) 
        mm_f=data['mm_f']['data_matrix'].to(device)
        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.to(device), data['mm_f']['edges'].to(device),data['mm_f']['data_matrix'][
           data['mm_f']['edges'][0].to(device), data['mm_f']['edges'][1]].to(device))) 

        x_m_g1 = torch.relu(self.gcn_x1_g(x_m.to(device), data['mm_g']['edges'].to(device),data['mm_g']['data_matrix'][
         data['mm_g']['edges'][0], data['mm_g']['edges'][1]].to(device)))  ##585.128

        reach_matrix=data['mm_I']['data_matrix']  
        mps1=data['cdc_I']['edges']    
        feat=self.fc_list(reach_matrix) 

        circ_mp1= torch.relu(self.gcn_I1(feat.to(device),mps1.to(device)))  ##585.128
        x_m_g1,circ_mpout1=self.shared_units[0](x_m_g1,circ_mp1) 
        x_m_g1,circ_mpout1=self.shared_units[1](x_m_g1,circ_mpout1) 
        x_m_g1,circ_mpout1=self.shared_units[2](x_m_g1,circ_mpout1) 
        x_m_g1,circ_mpout1=self.shared_units[3](x_m_g1,circ_mpout1)

        circ_mp2=torch.relu(self.gcn_I2(circ_mpout1,mps1.to(device)))+circ_mpout1
        x_m_g2 = torch.relu(self.gcn_x2_g(x_m_g1, data['mm_g']['edges'].to(device),data['mm_g']['data_matrix'][
        data['mm_g']['edges'][0], data['mm_g']['edges'][1]].to(device)))      
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].to(device),data['mm_f']['data_matrix'][
          data['mm_f']['edges'][0].to(device), data['mm_f']['edges'][1]].to(device))) 
       
        
        XM = torch.cat((x_m_f1,x_m_f2,x_m_g1,x_m_g2), 1).t() 
        XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1) #torch.Size([1, 6, 128, 853])


##注意力机制部分
        globalAvgPool_x = nn.AvgPool2d((self.args.fm, miRNA_number), (1, 1))
        x_channel_attention = globalAvgPool_x(XM)  #torch.Size([1, 6, 1, 1])

        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)  #torch.Size([1, 6])
        x_channel_attention = self.fc1_x(x_channel_attention)  #torch.Size([1, 30])
        x_channel_attention = torch.relu(x_channel_attention)
        x_channel_attention = self.fc2_x(x_channel_attention) #torch.Size([1, 6])
        x_channel_attention = self.sigmoidx(x_channel_attention)  #torch.Size([1, 6])
        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1) #torch.Size([1, 6, 1, 1])
        XM_channel_attention = x_channel_attention * XM  #torch.Size([1, 6, 128, 853])
        XM_channel_attention = torch.relu(XM_channel_attention) # torch.Size([1, 6, 128, 853])
        # XM_channel_attention = torch.relu(XM)
        x = self.cnn_x(XM_channel_attention) #torch.Size([1, 1, 128, 853])
        x = x.view(self.args.fm, miRNA_number).t()  #torch.Size([585, 128])

        return x,circ_mp2

##生成特征视图
class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y3_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y3_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_I1 = GCNConv(self.args.fd, self.args.fd)  ###考虑考虑
        self.gcn_I2 = GCNConv(self.args.fd, self.args.fd)
        self.gcn_I3 = GCNConv(self.args.fd, self.args.fd)
        self.gcn1=GCNConv(args.disease_number,self.args.fd)   ###考虑考虑
        self.fc_list=nn.Linear(args.disease_number,self.args.fd,bias=True)
        nn.init.xavier_uniform_(self.fc_list.weight)
        if self.fc_list.bias is not None:
            self.fc_list.bias.data.fill_(0.0)
        self.num_shared_layer=self.args.shared_unit_num
        self.shared_units=nn.ModuleList()
        shared_unit_num=self.args.shared_unit_num   ##构建共享单元
        for i in range(shared_unit_num):  ####等于1
            self.shared_units.append(Cross_stitch())

            if self.args.dropout > 0:
                self.feat_drop = nn.Dropout(self.args.dropout)
            else:
                self.feat_drop = lambda x: x
        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view  * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view  * self.args.gcn_layers,
                               out_features=self.args.view  * self.args.gcn_layers)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_y = nn.Conv2d(in_channels=self.args.view  * self.args.gcn_layers, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)  
        disease_number = len(data['dd_s']['data_matrix'])
        x_d = torch.randn(disease_number, self.args.fd)  


        y_d_s1 = torch.relu(self.gcn_y1_s(x_d.to(device), data['dd_s']['edges'].to(device),data['dd_s']['data_matrix'][
         data['dd_s']['edges'][0], data['dd_s']['edges'][1]].to(device)))


        y_d_g1 = torch.relu(self.gcn_y1_g(x_d.to(device), data['dd_g']['edges'].to(device),data['dd_g']['data_matrix'][
        data['dd_g']['edges'][0], data['dd_g']['edges'][1]].to(device)))
       
        reach_matrix=data['dd_I']['data_matrix']  ##88,88
        
        feat=self.fc_list(reach_matrix)
        circ_mp1= torch.relu(self.gcn_I1(feat.to(device),data['dcd_I']['edges'].to(device)))  ##88.128
    
        y_d_g1,circ_mpout1=self.shared_units[0](y_d_g1 ,circ_mp1) 
        y_d_g1,circ_mpout1=self.shared_units[1](y_d_g1,circ_mpout1)
        y_d_g1,circ_mpout1=self.shared_units[2](y_d_g1,circ_mpout1)  
        y_d_g1,circ_mpout1=self.shared_units[3](y_d_g1,circ_mpout1) 

        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].to(device),data['dd_s']['data_matrix'][
         data['dd_s']['edges'][0], data['dd_s']['edges'][1]].to(device)))       
        circ_mp2=torch.relu(self.gcn_I2(circ_mpout1,data['dcd_I']['edges'].to(device)))+circ_mpout1        
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].to(device),data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].to(device))) 

        YD = torch.cat((y_d_s1,y_d_s2,y_d_g1,y_d_g2), 1).t() 
        YD = YD.view(1, self.args.view  * self.args.gcn_layers, self.args.fd, -1) 

        globalAvgPool_y = nn.AvgPool2d((self.args.fm, disease_number), (1, 1))
        y_channel_attention = globalAvgPool_y(YD)  

        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), -1) ###1.6
        y_channel_attention = self.fc1_y(y_channel_attention)  
        y_channel_attention = torch.relu(y_channel_attention)  ##1.30
        y_channel_attention = self.fc2_y(y_channel_attention)  ##1.6
        y_channel_attention = self.sigmoidy(y_channel_attention)  ##1.6  
        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), y_channel_attention.size(1), 1, 1)  

        YD_channel_attention = y_channel_attention * YD  
        YD_channel_attention = torch.relu(YD_channel_attention) 


        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.fd, disease_number).t()  #torch.Size([591, 128])

        return y,circ_mp2





def loss_contrastive_m(m1,m2):
    m1,m2= (m1/th.norm(m1)),(m2/th.norm(m2))
    pos_m1_m2 = th.sum(m1 * m2, dim=1, keepdim=True)  
    neg_m1 = th.matmul(m1, m1.t())  
    neg_m2 = th.matmul(m2, m2.t())  
    neg_m1 = neg_m1 - th.diag_embed(th.diag(neg_m1))  
    neg_m2 = neg_m2 - th.diag_embed(th.diag(neg_m2))  
    pos_m = th.mean(th.cat([pos_m1_m2],dim=1),dim=1)  
    neg_m = th.mean(th.cat([neg_m1, neg_m2], dim=1), dim=1)  
    loss_m = th.mean(F.softplus(neg_m-pos_m))
    return loss_m



class ACMF( nn.Module): 
    def __init__(self, args):
        super(ACMF, self).__init__()
        self.args=args

     

    def forward(self,sim_data, c_embeding1,d_embeding1,c_embeding2,d_embeding2):  ##sample 是训练数据，miRNA, disease是聚合的相似性信息。
        
        lossc=loss_contrastive_m(c_embeding1,c_embeding2)
        lossd=loss_contrastive_m(d_embeding1,d_embeding2)
       
        return lossc,lossd,c_embeding1,c_embeding2,d_embeding1,d_embeding2

class MDI(nn.Module):
    def __init__(self, args):
        super(MDI, self).__init__()

        self.args=args
        self.fcDropout = args.fcDropout
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()
        self.device=args.device
        self.fcLinear = MLP(self.args.fm, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)
        self.fcLinear1 = MLP(2*self.args.fm, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)
    def forward(self, em1,em2,ed1,ed2):
    
        R=em1
        D=ed1
        node_embed = (R * D).squeeze(dim=1)  ##32.128
        pre_part = self.fcLinear(node_embed)   ##torch.Size([32, 1])
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)
        
        return pre_a
