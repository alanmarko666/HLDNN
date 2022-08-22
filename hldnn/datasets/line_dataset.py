#Abstraktne Dataset data vygeneruje ulozi a nacita
import sys
sys.path.append('.')

import torch
from datasets.dataset import Dataset
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# na zaciatok chceme mat tie data ako keby na jednej usecke
# ale teda asi to chceme mat reprezentovane ako graf.
# to znamena ze pre jeden graf pred transformaciou
# (ktoru neviem kedy chceme robit) chceme mat node features
# (farba tych nodov), y (globalnu output feature)
# hrany (medzi vrcholmi)

## vkazdom pripade strom musi generovat samostatna funkcia
# ta funkcia bude transformacia z toho typu Data -> Data
# s tym ze to tam prida nove vrcholy a nove hrany
# a vlastnosti ze ci je to pravy alebo lavy syn alebo whatever

# 



import torch
import torch_geometric
import torch.nn.functional as F
import random
from config import config

def gen_task1_data(n):
    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    parity = torch.sum(bin_features[beg:end+1])%2
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=parity, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_task2_data(n): #is something inside of an interval
    bin_features = F.one_hot(torch.randint(0,n,(1,)), num_classes=n)#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    parity = torch.sum(bin_features[beg:end+1])%2
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=parity, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_task3_data(n): #count 1s inside of an interval
    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.sum(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_task4_data(n): #shortest path length
    bin_features = torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.sum(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

#find minimum
def gen_task5_data(n): #find minimum
    bin_features = torch.randint(torch.randint(0,20,(1,)).item(),40,(n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.min(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

def generate_tree_task_1_data(n):
    def find_answer_task1(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(1,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task1(data,beg)
    data.y=result
    return  data

def generate_tree_task_2_data(n):#parity, random dots
    def find_answer_task1(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task1(data,beg)%2
    data.y=result
    return  data

def generate_tree_task_3_data(n):#parity, random dots
    def find_answer_task(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task(data,beg)
    data.y=result
    return  data

def get_adjacency_list_with_features(data):
    n = int(torch.max(data.edge_index)+1)
    adj_list = [[] for i in range(n)]
    for i in range(data.edge_index.T.shape[0]):
        a,b=data.edge_index.T[i]
        feature=data.edge_attr[i]
        adj_list[a].append((int(b),feature))
    return adj_list

def generate_edge_test_task(n):
    def find_answer_task(data,start):
        adj_list=get_adjacency_list_with_features(data)
        def dfs(v,parent):
            count=0
            found=data.x[v][2].item() == 1
            for child,feature in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child+feature.item()
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,1, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    edge_attr=torch.randint(0,2,(n-1,)).float()
    edge_attr=torch.hstack([edge_attr,edge_attr])
    edge_attr=torch.unsqueeze(edge_attr,1)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index,edge_attr=edge_attr)
    result=find_answer_task(data,beg)%2
    data.y=result
    return  data
    

def print_lin_task_data(d):
    print("features:", d.x, "\nedge_index:", d.edge_index, "\nanswer:", d.y)
#print_lin_task_data(gen_task1_data(5))




def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def edge_index_to_tensor(edge_index):
    device=getDevice()
    rol, col = zip(*edge_index)
    edge_index = [rol, col] 
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device) 
    return edge_index.to(device).type(torch.long)

def nearest_power_of_2(x):  
    return 0 if x == 0 else 2**((x - 1).bit_length()-1)

#Creates edge index for a rooted binary tree with specified number of leafs
def build_bst_edges(n_nodes, root, depth, x, root_state=2):
    if n_nodes==1:
        return torch.tensor([[],[]]), \
            torch.tensor([depth],dtype=torch.long), \
            1, \
            torch.tensor([root_state],dtype=torch.long), \
            x[0:1]

    n_left=nearest_power_of_2(n_nodes)
    n_right=n_nodes-n_left

    depth_list=[depth]
    node_states=[root_state]
    node_features=torch.zeros((1,x.size(dim=1)))


    edge_index_list=[(root+1,root)]
    edge_index_left,depths_left,count_left,states_left,feat_left=build_bst_edges(n_left,root+1,depth+1,x[0:n_left],0)
    depth_list.extend(depths_left)
    node_states.extend(states_left)

    edge_index_list.append((root+1+count_left,root))
    edge_index_right,depths_right,count_right,states_right,feat_right=build_bst_edges(n_right,root+1+count_left,depth+1,x[n_left:n_nodes],1)
    depth_list.extend(depths_right)
    node_states.extend(states_right)

    edge_index=edge_index_to_tensor(edge_index_list)
    edge_index=torch.cat([edge_index,edge_index_left,edge_index_right],dim=1)

    return edge_index.type(torch.long), \
        torch.tensor(depth_list,dtype=torch.long), \
        count_left+count_right+1, \
        torch.tensor(node_states,dtype=torch.long), \
        torch.cat([node_features,feat_left,feat_right],dim=0)

#Takes in task data and transforms them so that they contain binary tree structure
def transform_to_binary_tree(data):
    n_nodes=data.x.size(dim=0)
    edge_index,depths, n_bst_nodes, node_states,node_features=build_bst_edges(n_nodes,0,0,data.x)

    return torch_geometric.data.Data.from_dict({**(data.to_dict()),
                                    'x':node_features,
                                    'edge_index':edge_index, #nodes pointing to parents
                                    'states':node_states, #0-left son, 1-right son, 2-head
                                    'depths':depths}) # depths of each node


def generate_synthetic_tree_edges(n):
    import networkx as nx
    tree = nx.random_tree(n=n)
    edges = torch.Tensor(list(map(list,tree.edges))).T.long()
    edges = torch.hstack([edges, edges[[1,0]]])
    return edges

def get_adjacency_list(edge_index):
    n = 1 if edge_index.shape[1]==0 else int(torch.max(edge_index)+1)
    adj_list = [[] for i in range(n)]
    for a,b in edge_index.T:
        adj_list[int(a)].append(int(b))
    print(adj_list)
    return adj_list

def find_hld(edge_index):
    n = 1 if edge_index.shape[1]==0 else int(torch.max(edge_index)+1)

    adj_list = get_adjacency_list(edge_index)

    
    def find_heavy_child(adj_list, root=0):
        size = [0 for i in range(n)]
        heavy_child = [-1 for i in range(n)]
        def dfs(x, parent=-1):
            sz = 1
            heaviest_child = (0,-1)
            for child in adj_list[x]:
                if child!=parent:
                    ch_size = dfs(child, x)
                    heaviest_child = max(heaviest_child, (ch_size, child))
                    sz += ch_size
            size[x] = sz
            heavy_child[x] = heaviest_child[1]
            print("find_heavy_child", x, parent, heavy_child[x])
            return sz
        dfs(root)
        return heavy_child
    
    #builds list of heavy paths where each element is tuple of index and list of heavy subpath
    def build_hld(adj_list, heavy_child, root=0):
        #t=0
        def dfs(x, l, parent = -1):
            #nonlocal t
            #t+=1
            #if t>100:return []
            #print("build_hld", x, parent, heavy_child[x])
            sub_paths = []
            for child in adj_list[x]:
                if child!=parent and child!=heavy_child[x]:
                    sub_paths.append(dfs(child, [], x))
            l.append((x,sub_paths))
            if heavy_child[x]!=-1:
                dfs(heavy_child[x], l=l, parent=x)
            return l
        return dfs(root, [])
    
    heavy_child = find_heavy_child(adj_list)
    tree_of_paths = build_hld(adj_list, heavy_child)
    print(heavy_child)
    print(tree_of_paths)
    return tree_of_paths

def build_hld_graph(data):
    n = 1 if data.edge_index.shape[1]==0 else int(torch.max(data.edge_index)+1)
    adj_list = get_adjacency_list(data.edge_index)

    #if edges have attributes
    parent_edge_features = None
    if data.edge_attr is not None:
        num_edge_features = data.edge_attr.shape[1]
        parent_edge_features = torch.zeros((n, num_edge_features))
        parent_dict = {}
        def populate_parent_dict(x, parent = -1):
            parent_dict[x] = parent
            for i in adj_list[x]:
                if i != parent:
                    populate_parent_dict(i, x)
        populate_parent_dict(0)
        parent_tensor = torch.Tensor([parent_dict[i] for i in range(n)]).long()
        mask1 = parent_tensor[data.edge_index[0]]==data.edge_index[1]
        parent_edge_features[data.edge_index[0][mask1]] = data.edge_attr[mask1]
        mask2 = parent_tensor[data.edge_index[1]]==data.edge_index[0]
        parent_edge_features[data.edge_index[1][mask2]] = data.edge_attr[mask2]
        
        #data.x = torch.cat([data.x, parent_edge_features], dim=1)

    avail = n
    edges = []
    depth_list = {}
    node_states = {}
    
    parent_light_edge_features = {}
    parent_edge_features_to_add = []
    #state: 0-left son, 1-right son, 2-head, 3-local head
    def dfs(l, parent =-1, depth = 0, state=2):
        nonlocal avail
        if l == 0: return
        me = 0
        if len(l)>1:
            me = avail
            avail += 1
            parent_edge_features_to_add.append(l[len(l)//2][0])
            dfs(l[:len(l)//2], me, depth+1, state=0)
            dfs(l[len(l)//2:], me, depth+1, state=1)
        else:
            me=l[0][0]
            for subtree in l[0][1]:
                dfs(subtree, me, depth+1, state=3)
        
        depth_list[me]=depth
        node_states[me]=state
        if state == 3:
            parent_light_edge_features[me] = l[0][0]

        if parent!=-1:
            edges.append((me, parent))


    l = find_hld(data.edge_index)
    dfs(l)
    print(avail, depth_list)
    if data.edge_attr is not None:
        print(parent_edge_features_to_add, parent_edge_features.shape)
        parent_edge_features = torch.cat([parent_edge_features, parent_edge_features[parent_edge_features_to_add]])
        indexes_of_parent_features = [(parent_light_edge_features[i] if i in parent_light_edge_features else 0) for i in range(avail)]
        parent_light_edge_features = parent_edge_features[indexes_of_parent_features]
    else:
        parent_edge_features = torch.zeros((avail, 0))
        parent_light_edge_features = torch.zeros((avail, 0))

    return torch.tensor(edges, dtype=torch.long).T, \
        torch.tensor([depth_list[i] for i in range(avail)], dtype=torch.long), \
        torch.tensor([node_states[i] for i in range(avail)], dtype=torch.long), \
        parent_edge_features, \
        parent_light_edge_features
    #return edge_index.type(torch.long), \
    #    torch.tensor(depth_list,dtype=torch.long), \
    #    count_left+count_right+1, \
    #    torch.tensor(node_states,dtype=torch.long), \
    #    torch.cat([node_features,feat_left,feat_right],dim=0)
    
    
def transform_to_hld_tree(data):
    edge_index,depths, node_states, parent_edge_features, parent_light_edge_features = build_hld_graph(data)
    data.edge_index=edge_index
    print(data.x.shape, depths.shape)
    node_features = torch.cat([data.x, torch.zeros((depths.shape[0]-data.x.shape[0], data.x.shape[1]))])

    return torch_geometric.data.Data.from_dict({**(data.to_dict()),
                                    'x':node_features,
                                    'edge_index':edge_index, #nodes pointing to parents
                                    'states':node_states, #0-left son, 1-right son, 2-head, 3-
                                    'depths':depths, # depths of each node
                                    'parent_edge_features':parent_edge_features,
                                    'parent_light_edge_features': parent_light_edge_features}) 

    




def get_tree_dataset(count=10000,min_n=2,max_n=128,generator=generate_tree_task_1_data,name="",force_download=True):
    def gen_f():
        #n=random.choices([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],weights=(30,30,30,5,5,5,3,3,3,1,1,1,1,1,1,1,1,1,1),k=1)[0]
        n=random.randint(min_n,max_n)#random.choices([2,4,8,16,32,64,128],weights=(30,30,30,30,30,30,30),k=1)[0]
        return transform_to_hld_tree(generator(n))
    return Dataset(count, gen_f,force_download=force_download, root=name)

def get_line_dataset(count=10000,min_n=2,max_n=128,generator=gen_task1_data,name="",force_download=True):
    def gen_f():
        #n=random.choices([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],weights=(30,30,30,5,5,5,3,3,3,1,1,1,1,1,1,1,1,1,1),k=1)[0]
        n=random.randint(min_n,max_n)#random.choices([2,4,8,16,32,64,128],weights=(30,30,30,30,30,30,30),k=1)[0]
        return transform_to_binary_tree(generator(n))
    return Dataset(count, gen_f,force_download=force_download, root=name)


#Processor-spaja dva vrcholy vnutri intervalacu

#Root MLP - zoberie hodnoty rootov a spravi novu hodnotu

#Suma - scitame vysledky s root MLP

#Merger - zoberie concatenatovane vysledky zo sumy a pociatocnu hodnotu listu intervalaca a cez MLP mergne do mensej hodnoty

#    return torch_geometric.data.Data.from_dict({**(data.to_dict()),
#                                    'x':node_features,
#                                    'edge_index':edge_index, #nodes pointing to parents
#                                    'states':node_states, #0-left son, 1-right son, 2-main head, 3-other heads
#                                    'light_son_count': pocet_light_synov, #
#
#                                    'depths':depths}) 

