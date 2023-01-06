# -*- coding: utf-8 -*-


import time
import argparse
import numpy as np

from utils import load_data, sensitive_attribute, process_adj_matrix, \
    metap_to_dgl, load_sensitive
from models import PDFC
import torch
from metrics import get_clustering, balance_constraints, structural_constraints, \
    balance_Euc, balance_standard, structural_standard, structural_Euc
import warnings
warnings.filterwarnings("ignore")


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    loaddata_time = time.time()
    args.out_size = args.k
    
    if args.dataname == 'MOOCCube':
        dirfile = 'datasets/'+args.dataname +'/path_info/' + 'knn_' + str(args.knn) + '/'
        sensitiveFile = 'datasets/'+args.dataname +'/sensitive/Attributes.csv'
        paths = {'UKU': 0.5, 'UVKVU': 0.25, 'UCKCU': 0.25, \
                 'UKKU': 0.5, 'UVKKVU': 0.25, 'UCKKCU': 0.25}
    elif args.dataname == 'DBLP':
        dirfile = 'datasets/'+args.dataname +'/path_info/' + 'knn_' + str(args.knn) + '/'
        sensitiveFile = 'datasets/'+args.dataname +'/sensitive/Attributes.csv'
        paths ={'APA': 0.6, 'AVPVA': 0.4, \
                    'APPA': 0.6, 'AVPPVA': 0.4}
    elif args.dataname == 'Movielens':
        dirfile = 'datasets/'+args.dataname +'/path_info/' + 'knn_' + str(args.knn) + '/'
        sensitiveFile = 'datasets/'+args.dataname +'/sensitive/Attributes.csv'
        paths ={'UMU': 1, 'UMMU': 1}
        
    features, metap, adj, metap_structural = load_data(dirfile, args.dataname, paths, eta = args.eta)
    sensitive = sensitive_attribute(sensitiveFile, args.dataname)
    
    sen_dict = balance_standard(sensitive)
    sen_feat = load_sensitive(sensitive)
    sen_balance_stand = np.round(min(list(sen_dict.values()))/max(list(sen_dict.values())), 4)
    print('load data information finished.',
          'knn:', args.knn, 'eta:', args.eta,
          'time:', time.time() - loaddata_time)
    
    standard_structural, _ = structural_standard(metap_structural)
    standard_structural = np.round(standard_structural, 4)
    print('structural constraints:', standard_structural)
    print('balance constraints', sen_balance_stand)
    
    adj, adj_g = process_adj_matrix(adj)
    graph = metap_to_dgl(metap)
    
    start1_time = time.time()
    model = PDFC(features, sen_in_size = sen_feat.shape[1], out_size = args.out_size, 
                  num_heads = args.num_heads, dropout = args.dropout)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_loss = 1000000
    current_ZF = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        loss, ZF= model(graph, features, adj_g, sen_feat, adj, args.k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print('epoch: ',epoch+1, '\t loss: ',np.round(loss.item() ,4), \
                  'time: ', np.round(time.time() - start_time, 4))
            start_time = time.time()
            
        if loss < curr_loss:
            curr_loss = loss
            current_ZF = ZF
    
    labels = get_clustering(current_ZF, args.k)
    end1_time = time.time()
    Ncuts = np.round(curr_loss.item() ,4)
    T = np.round((end1_time - start1_time), 4)
    
    #structural
    structural, structural_ratio = structural_constraints(features, args.k, labels, args.dataname)
    StrEuc = structural_Euc(metap_structural, structural_ratio)
    StrF = structural[0]
    mean_StrF = structural[1]
    
    # balance
    BalEuc = balance_Euc(args.k, labels, sensitive)
    balance = balance_constraints(args.k, labels, sensitiveFile, args.dataname)
    BalF = balance[0]
    mean_BalF = balance[1]
    
    print('eta',args.eta, 
          'StrEuc:', StrEuc, 'StrF:', StrF, 'StrFE:', abs(round(mean_StrF-standard_structural, 4)),
          'BalEuc:', BalEuc, 'BalF:', BalF, 'BalFE:', abs(round(mean_BalF-sen_balance_stand, 4)),
          'ObjE:', Ncuts, 'T:', T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PDFC')
    
    parser.add_argument('--seed', type=int, default=100, help='Random seed.')
    parser.add_argument('--dataname', type=str, 
                        default= 'MOOCCube')
    parser.add_argument('--num_heads', type = int, default = 4, help = 'Attention head')
    parser.add_argument('--hidden_units', type = list, default = [8])
    
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out_size', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.3)
        
    parser.add_argument('--knn', type=int, default= 50, help = 'Number of KNN neighbors')
    parser.add_argument('--eta', type=float, default=0.4, help = 'Control the balance')
    parser.add_argument('--lr', type=float, default = 0.05, help= 'Learning rate')
    parser.add_argument('--k', type=int, default=30, help = 'Clusters')
    
    args = parser.parse_args()
    main(args)
    
