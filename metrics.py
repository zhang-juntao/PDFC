# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from utils import sensitive_attribute
import numpy as np
import pandas as pd

def get_clustering(Z, k):
    Z = Z.detach().numpy()
    kmeans = KMeans(k,init='k-means++', n_init=50).fit(Z)
    labels = kmeans.labels_

    return labels
    
def clusters(k, labels):
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = []
    for i in range(len(labels)):
        if labels[i] in cluster_dict.keys():
            cluster_dict[labels[i]].append(i)
            
    return cluster_dict

#----------------------Balance constrints-----------------------------
def balance_standard(sensitive):
    balance_stand = {}
    unique_sen = np.unique(sensitive)
    
    for i in unique_sen:
        temp = [j for j in sensitive if j== i]
        balance_stand[i] = len(temp) / len(sensitive)
    
    return balance_stand

def clusters_ratio(k, labels, sensitive):
    unique_sen = np.unique(sensitive)
    cluster_dict = clusters(k, labels)
    cluster_ratio = {}
    
    for key in cluster_dict.keys():
        cluster_ratio[key] = {}
        cluster = cluster_dict[key]
        for i in unique_sen:
            temp = [i for j in cluster if sensitive[j] == i]
            cluster_ratio[key][i] = len(temp) / len(cluster)
            
    return cluster_ratio


def balance_Euc(k, labels, sensitive):
    sensitive = sensitive.detach().numpy()
    cluster_ratio = clusters_ratio(k, labels, sensitive)
    balance_stand = balance_standard(sensitive)
    balance_eucD = 0
    
    for key in cluster_ratio.keys():
        euclidean_dis = 0
        for k in cluster_ratio[key].keys():
            euclidean_dis += np.power((balance_stand[k]-cluster_ratio[key][k]), 2)
        balance_eucD += np.sqrt(euclidean_dis)
    
    return np.round(balance_eucD, 4)


def balance_constraints(k, labels, sensitiveFile, dataname):
    cluster_dict = clusters(k, labels)
    balance_dict = {}
    cluster_gender = {}
    
    sensitive = sensitive_attribute(sensitiveFile, dataname)
    sensitive = sensitive.detach().numpy()
    
    
    for key in cluster_dict.keys():
        cluster = cluster_dict[key]
        cluster_gender[key] = [sensitive[i] for i in cluster]
        
        woman_num = sum(cluster_gender[key])
        man_mum = len(cluster_gender[key]) - woman_num
        
        if len(cluster_gender[key]) > 1:
            if woman_num != 0 and man_mum != 0:
                if man_mum >= woman_num:
                    balance_dict[key] = np.round(woman_num/man_mum, 4)
                else:
                    balance_dict[key] = np.round(man_mum/woman_num, 4)
            else:
                balance_dict[key] = 0
        else:
            balance_dict[key] = 0
            
    balance = list(balance_dict.values())
    balance_error = sum(balance)/len(balance)
    return min(balance), np.round(balance_error, 4)

#----------------------Structural constrints-----------------------------
def structural_standard(temp = list):
    pmp = temp[0].sum()
    mp = temp[1].sum()
    
    structural_dict = {}
    structural_dict['pmp'] = pmp / (pmp + mp)
    structural_dict['mp'] = mp / (pmp + mp)
    
    standard_structural_con = pmp / mp
    return standard_structural_con, structural_dict


def get_feature(features, dataname):
    user_feat = {}
    if dataname == 'MOOCCube':
        for key in features.keys():
            if 'UKK' in key:
                user_feat['UK'] = features[key]
            elif 'UCK' in key:
                user_feat['UCK'] = features[key]
            elif 'UVK' in key:
                user_feat['UVK'] = features[key]
    elif dataname == 'DBLP':
        for key in features.keys():
            if 'APP' in key:
                user_feat['AP'] = features[key]
            elif 'AVP' in key:
                user_feat['AVP'] = features[key] 
    elif dataname == 'Movielens':
        for key in features.keys():
            if 'UMM' in key:
                user_feat['UM'] = features[key]
        
    return user_feat

def structural_constraints(features, k, labels, dataname):
    user_feat = get_feature(features, dataname)
    cluster_dict = clusters(k, labels)
    Prere_file = 'datasets/' + dataname + '/prerequisite/Prere.csv'
    prere = pd.read_csv(Prere_file, header=None).values
    
    structural_dict = {}
    structural_ratio = {}
    
    if dataname == 'MOOCCube':
        uk = user_feat['UK']
        uck = user_feat['UCK']
        uvk = user_feat['UVK']
    
        for key in cluster_dict.keys():
            cluster = cluster_dict[key]
            if len(cluster) > 1:
                temp_uk = uk[cluster[0]]
                temp_uck = uck[cluster[0]]
                temp_uvk = uvk[cluster[0]]
                
                for i in range(1, len(cluster)):
                    temp_uk = np.vstack((temp_uk, uk[cluster[i]]))
                    temp_uck = np.vstack((temp_uck, uck[cluster[i]]))
                    temp_uvk = np.vstack((temp_uvk, uvk[cluster[i]]))
                
                sim_uk = np.matmul(temp_uk, temp_uk.T)
                dif_uk = np.matmul(np.matmul(temp_uk, prere), temp_uk.T)
                for i in range(len(dif_uk)):
                    dif_uk[i][i] = 0
                    
                sim_uc = np.matmul(temp_uck, temp_uck.T)
                dif_uc = np.matmul(np.matmul(temp_uck, prere), temp_uck.T)
                for i in range(len(dif_uc)):
                    dif_uc[i][i] = 0
                    
                sim_uv = np.matmul(temp_uvk, temp_uvk.T)
                dif_uv = np.matmul(np.matmul(temp_uvk, prere), temp_uvk.T)
                for i in range(len(dif_uv)):
                    dif_uv[i][i] = 0
                    
                similary_k = sim_uk.sum() + sim_uc.sum() + sim_uv.sum()
                difference_k = dif_uk.sum() + dif_uc.sum() + dif_uv.sum()
                structural_dict[key] = np.round(difference_k/similary_k, 4)
                
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = similary_k / (similary_k + difference_k)
                structural_ratio[key]['pmp'] = difference_k / (similary_k + difference_k)
            else:
                structural_dict[key] = 0
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = 0
                structural_ratio[key]['pmp'] = 0
    
    elif dataname == 'DBLP':
        ap = user_feat['AP']
        avp = user_feat['AVP']
        
        for key in cluster_dict.keys():
            cluster = cluster_dict[key]
            if len(cluster) > 1:
                temp_ap = ap[cluster[0]]
                temp_avp = avp[cluster[0]]
                
                for i in range(1, len(cluster)):
                    temp_ap = np.vstack((temp_ap, ap[cluster[i]]))
                    temp_avp = np.vstack((temp_avp, avp[cluster[i]]))
                
                sim_ap = np.matmul(temp_ap, temp_ap.T)
                dif_ap = np.matmul(np.matmul(temp_ap, prere), temp_ap.T)
                for i in range(len(dif_ap)):
                    dif_ap[i][i] = 0
                    
                sim_avp = np.matmul(temp_avp, temp_avp.T)
                dif_avp = np.matmul(np.matmul(temp_avp, prere), temp_avp.T)
                for i in range(len(dif_avp)):
                    dif_avp[i][i] = 0
                    
                similary_k = sim_ap.sum() + sim_avp.sum()
                difference_k = dif_ap.sum() + dif_avp.sum()
                structural_dict[key] = np.round(difference_k/similary_k, 4)
                
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = similary_k / (similary_k + difference_k)
                structural_ratio[key]['pmp'] = difference_k / (similary_k + difference_k)
            else:
                structural_dict[key] = 0
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = 0
                structural_ratio[key]['pmp'] = 0
                
    elif dataname == 'Movielens':
        um = user_feat['UM']
        for key in cluster_dict.keys():
            cluster = cluster_dict[key]
            if len(cluster) > 1:
                temp_um = um[cluster[0]]
                for i in range(1, len(cluster)):
                    temp_um = np.vstack((temp_um, um[cluster[i]]))
                
                sim_um = np.matmul(temp_um, temp_um.T)
                dif_um = np.matmul(np.matmul(temp_um, prere), temp_um.T)
                for i in range(len(dif_um)):
                    dif_um[i][i] = 0
                    
                similary_k = sim_um.sum()
                difference_k = dif_um.sum()
                structural_dict[key] = np.round(difference_k/similary_k, 4)
                
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = similary_k / (similary_k + difference_k)
                structural_ratio[key]['pmp'] = difference_k / (similary_k + difference_k)
            else:
                structural_dict[key] = 0
                structural_ratio[key] = {}
                structural_ratio[key]['mp'] = 0
                structural_ratio[key]['pmp'] = 0
    
    structural = list(structural_dict.values())
    min_structural = min(structural)
    avg_structural = np.round(sum(structural)/len(structural), 4)
    
    return [min_structural, avg_structural], structural_ratio

def structural_Euc(temp, structural_ratio):
    _, structural_dict = structural_standard(temp)
    structural_eucD = 0
    
    for i in structural_ratio.keys():
        structural_dis = 0
        for key in structural_ratio[i].keys():
            structural_dis += np.power((structural_ratio[i][key] - structural_dict[key]), 2)
        structural_eucD += np.sqrt(structural_dis)
            
    return np.round(structural_eucD, 4)


