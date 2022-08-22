import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import datasets.line_dataset as line_dataset
import datasets.peptides_functional
from config import config

def get_adjacency_list_with_features(data):
    n = int(torch.max(data.edge_index)+1)
    adj_list = [[] for i in range(n)]
    for i in range(data.edge_index.T.shape[0]):
        a,b=data.edge_index.T[i]
        feature=data.edge_attr[i]
        adj_list[a].append((int(b),feature))
    return adj_list

def get_spanning_tree(data):
    adj_list=get_adjacency_list_with_features(data)
    used=set()
    new_edge_index=[[],[]]
    new_edge_attr=[]
    def dfs(v):
        used.add(v)
        for child,feature in adj_list[v]:
            if child in used:
                continue
            new_edge_index[0].append(v)
            new_edge_index[1].append(child)
            new_edge_index[0].append(child)
            new_edge_index[1].append(v)
            new_edge_attr.append(feature)
            new_edge_attr.append(feature)
            dfs(child)
    dfs(0)
    return new_edge_index,new_edge_attr

class SpanningTreeTransform(object):
    def __call__(self, data):
        new_edge_index,new_edge_attr=get_spanning_tree(data)
        print("dei:",data.edge_index)
        print("ei: ",new_edge_index)

        data.edge_index=torch.tensor(new_edge_index,dtype=torch.int64)
        if len(new_edge_attr)==0:
            data.edge_attr=torch.rand((0,config.EDGE_SIZE))
        else: 
            data.edge_attr=torch.vstack(new_edge_attr).float()#None
        print(data.edge_attr)
        n_nodes=(len(new_edge_index[0])//2)+1
        data.x=data.x[0:n_nodes]

        data=line_dataset.transform_to_hld_tree(data)
        return data

def get_molhiv_tree_dataset(split="train"):
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv",pre_transform=SpanningTreeTransform())
    split_idx = dataset.get_idx_split()
    dataset=dataset[split_idx[split]]
    return dataset

def get_peptides_tree_dataset(split="train"):
    dataset = datasets.peptides_functional.PeptidesFunctionalDataset(pre_transform=SpanningTreeTransform())
    split_idx = dataset.get_idx_split()
    dataset=dataset[split_idx[split]]
    return dataset