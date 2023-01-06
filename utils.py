# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
import pandas as pd
from scipy import sparse
import dgl

#load feaure and matrix
def load_data(dirfile, dataname, paths = None, eta= 0.5):
    filenames = os.listdir(dirfile)
    data_dict = {}
    for name in filenames:
        temp = name.split('.')[0]
        
        file = dirfile + name
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_dict[temp] = data
    
    features = {}
    features_file = 'datasets/'+ dataname +'/features/'
    fea_names = os.listdir(features_file)
    features_list = {}
    for name in fea_names:
        temp = name.split('.')[0]
        file  = features_file + name
        with open(file, 'rb') as f:
            feature = pickle.load(f)
            features_list[temp] = feature
    
    metap = data_dict['metap']
    metap2 = data_dict['metap2']
    N, _ = feature.shape
    mp = np.zeros((N, N))
    mp2 = np.zeros((N, N))
    pmp = np.zeros((N, N))
    pmp2 = np.zeros((N, N))
    
    if dataname == 'MOOCCube':
        for key in paths.keys():
            if 'UCK' in key:
                features[key] = torch.Tensor(features_list['UCK'].toarray())
            elif 'UVK' in key:
                features[key] = torch.Tensor(features_list['UVK'].toarray())
            elif 'UK' in key:
                features[key] = torch.Tensor(features_list['UK'].toarray())
        
        for key in metap.keys():
            if 'KK' in key:
                pmp += paths[key] * metap[key].toarray()
                pmp2 += metap2[key].toarray()
            else:
                mp += paths[key] * metap[key].toarray()
                mp2 += metap2[key].toarray()
    
    elif dataname == 'DBLP':
        for key in paths.keys():
            if 'AVP' in key:
                features[key] = torch.Tensor(features_list['AVP'].toarray())
            elif 'AP' in key:
                features[key] = torch.Tensor(features_list['AP'].toarray())
        
        for key in metap.keys():
            if 'PP' in key:
                pmp += paths[key] * metap[key].toarray()
                pmp2 += metap2[key].toarray()
            else:
                mp += paths[key] * metap[key].toarray()
                mp2 += metap2[key].toarray()
    
    elif dataname == 'Movielens':
        for key in paths.keys():
            if 'UM' in key:
                features[key] = torch.Tensor(features_list['UM'].toarray())
        
        for key in metap.keys():
            if 'MM' in key:
                pmp += paths[key] * metap[key].toarray()
                pmp2 += metap2[key].toarray()
            else:
                mp += paths[key] * metap[key].toarray()
                mp2 += metap2[key].toarray()
        
    adj = mp + eta * pmp
    return features, metap, adj, [pmp2, mp2]
    
#generate degree matrix
def process_adj_matrix(adj):
    adj = (adj-adj.min())/(adj.max()-adj.min())
    for i in range(len(adj)):
        adj[i,i] = 1
    
    adj_g = sparse.csr_matrix(adj)
    adj_g = dgl.from_scipy(adj_g)
    
    return adj, adj_g

#meta-paths to dgl
def metap_to_dgl(metapaths):
    graph = {}
    
    for key in metapaths.keys():
        N = metapaths[key].shape[0]
        temp = sparse.lil_matrix(metapaths[key])
        for i in range(N):
            temp[i,i] = 1
        
        graph[key] = dgl.from_scipy(temp)
    return graph

def unnormalized_adj(adj):
    
    degrees = np.sum(adj, 1)
    D = np.mat(np.diag(degrees))
    L = D - adj
    
    return torch.Tensor(L), torch.Tensor(D)

# load sensitive attributes
def sensitive_attribute(sensitiveFile, dataname):
    
    df = pd.read_csv(sensitiveFile)
    if dataname == 'Movielens':
        sensitive = df['age'].values
    else:
        sensitive = df['gender'].values
    sensitive = torch.Tensor(sensitive)
    return sensitive


def load_sensitive(sensitive):
    sens_unique = torch.unique(sensitive)
    
    h = len(sens_unique)
    sensitiveNEW = torch.ones(len(sensitive), dtype=torch.float32)
    sensitiveNEW = sensitiveNEW - sensitive
    
    F = sensitiveNEW.reshape((len(sensitiveNEW), h-1))
    for ell in range(1, len(sens_unique)):
        temp = [i for i in sensitiveNEW if i == ell]
        groupsize = len(temp)
        F = F - groupsize/(len(F))
        
    U, _, _ = torch.linalg.svd(F)
    return U[:, 1:]

#Cholesky decomposition
def orthonorm1(F, adj, eps=1e-7):
    L, D = unnormalized_adj(adj)
    outer_prod = torch.matmul(torch.matmul(F.T, D), F)
    outer_prod = outer_prod + eps * torch.eye(outer_prod.shape[0])

    L = torch.linalg.cholesky(outer_prod)
    L_inv = torch.linalg.inv(L)
    return L_inv

def loss_function(F, P, adj, k, normalized = False):
    try:
        QF = orthonorm1(F, adj)
        ZF = torch.matmul(F, QF)
    except:
        ZF = F
    
    try:
        QP = orthonorm1(P, adj)
        ZP = torch.matmul(P, QP.T)
    except:
        ZP = P
    
    L, _ = unnormalized_adj(adj)
    
    Z = torch.cat((ZF, ZP), 1)
    L0 = torch.matmul(torch.matmul(Z.T, L), Z)
    eigvalue, eigvector = torch.linalg.eigh(L0)
    eigv = eigvector[:, :k]
    
    symm = torch.matmul(torch.matmul(torch.matmul(torch.matmul(eigv.T, Z.T), L), Z), eigv)
    symm = (symm + symm.T)/2
    tr = torch.trace(symm)
    
    return tr, torch.matmul(Z, eigv)


