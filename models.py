# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch import GATConv, GraphConv
from utils import loss_function

class GCN(nn.Module):
    def __init__(self, in_size, out_size):
        super(GCN, self).__init__()
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(in_size, out_size, activation = nn.Tanh()))
        self.gcn_layers.append(GraphConv(out_size, out_size, bias=False))
        
    def forward(self, adj, feat):
        for i in range(len(self.gcn_layers)):
            feat = self.gcn_layers[i](adj, feat)
            
        return feat

class PathAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super(PathAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Tanh(),
            nn.Linear(out_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class PDFCNetlayer(nn.Module):
    def __init__(self, features, out_size, layer_num_heads, dropout):
        super(PDFCNetlayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        
        for key in features.keys():
            self.gat_layers.append(GATConv(in_feats = features[key].shape[1],
                                           out_feats = out_size,
                                           num_heads = layer_num_heads,
                                           feat_drop = dropout,
                                           attn_drop = dropout,
                                           activation = F.relu6))
                
        self.semantic_attention = PathAttention(in_size=out_size * layer_num_heads, \
                                                out_size = out_size)
        
    def forward(self, graph, features):
        semantic_embeddings = []
        i = 0
        for key in features.keys():
            semantic_embeddings.append(self.gat_layers[i](graph[key], features[key]).flatten(1))
            i += 1
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)
    
class PDFC(nn.Module):
    def __init__(self, features, sen_in_size, out_size, num_heads, dropout):
        super(PDFC, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(PDFCNetlayer(features, out_size, num_heads, dropout))
        self.fcn = nn.Linear(out_size * num_heads, out_size)
        
        self.gcn = GCN(sen_in_size, out_size)
      
        
    def forward(self, graph, features, adj_g, sen_feat, adj, k, normalized = False):
        for scnlayer in self.layers:
            F = scnlayer(graph, features)
        
        F = self.fcn(F)
        P = self.gcn(adj_g, sen_feat)
        
        tr, ZF = loss_function(F, P, adj, k, normalized)
        return tr, ZF

        