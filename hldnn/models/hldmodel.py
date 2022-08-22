import torch
import torch_geometric
import torch.nn.functional as F
import random
import models.mlps as mlps
import torch.nn as nn
from config import config
import torch_scatter

class EncodeModule(nn.Module):
    def __init__(self):
        super(EncodeModule,self).__init__()
        self.encoder=mlps.WideMLP(config.IN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)

    def forward(self,data):
        data.x=self.encoder(data.x)
        return data


class ProcessModule(nn.Module):
    def __init__(self,):
        super(ProcessModule,self).__init__()
        self.merger=mlps.WideMLP(config.HIDDEN_SIZE*2+config.EDGE_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)
        self.light_edge_processor=mlps.WideMLP(config.HIDDEN_SIZE+config.EDGE_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)
        self.light_edge_merger=mlps.WideMLP(config.HIDDEN_SIZE*2,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)

    def forward(self,data):
        parents = torch.zeros(data.x.shape[0],dtype=torch.long)
        parents[data.edge_index[0]] = data.edge_index[1]

        max_depth=torch.max(data.depths)
        for depth in range(max_depth,0,-1):
            #Interval tree
            mask_depth=data.depths==depth

            left_sons_mask=mask_depth & (data.states==0)
            right_sons_mask=mask_depth & (data.states==1)

            left_parents=parents[left_sons_mask]
            right_parents=parents[right_sons_mask]

            left=torch.zeros_like(data.x)
            right=torch.zeros_like(data.x)

            left = torch_scatter.scatter_add(data.x[left_sons_mask], left_parents, out=left,dim=0)
            right = torch_scatter.scatter_add(data.x[right_sons_mask], right_parents, out=right,dim=0)

            x_parents=torch.cat([left,right, data.parent_edge_features],dim=1)
            x_parents=self.merger(x_parents)

            parents_mask=torch.zeros((data.x.shape[0],1))
            parents_mask=torch_scatter.scatter_add(torch.ones((data.x.shape[0],1))[left_sons_mask], left_parents, out=parents_mask,dim=0).long()!=0


            #Merging two heavy paths
            heads_mask=mask_depth & (data.states==3)
            heads_parents=parents[heads_mask]

            processed_heads=self.light_edge_processor(torch.cat([data.x, data.parent_light_edge_features], dim=1)[heads_mask])

            merged_heads=torch.zeros_like(data.x)
            merged_heads = torch_scatter.scatter_add(processed_heads, heads_parents, out=merged_heads,dim=0)

            designated_leafs_mask=torch.zeros((data.x.shape[0],1))
            designated_leafs_mask=torch_scatter.scatter_add(torch.ones((data.x.shape[0],1))[heads_mask], heads_parents, out=designated_leafs_mask,dim=0).long()!=0

            x_designated_leafs=torch.where(designated_leafs_mask, data.x,torch.zeros_like(data.x))
            x_merged=torch.cat([x_designated_leafs,merged_heads],dim=1)
            x_merged=self.light_edge_merger(x_merged)
            
            data.x = torch.where(parents_mask, x_parents, 
                torch.where(designated_leafs_mask, x_merged, data.x))
            
        return data

class DecodeModule(nn.Module):
    def __init__(self,):
        super(DecodeModule,self).__init__()
        self.decoder=mlps.WideMLP(config.HIDDEN_SIZE,config.OUT_SIZE,config.HIDDEN_SIZE,1,last_activation=nn.Identity)

    def forward(self,x):
        pred=self.decoder(x)
        if config.BINARY_OUTPUT:
            pred = torch.sigmoid(pred)
        return pred


class ReadoutModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,data):
        return data.x[data.depths==0]


class HLDGNNModel(nn.Module):
    def __init__(self,):
        super(HLDGNNModel,self).__init__()
        self.encoding_module=EncodeModule()
        self.process_module=ProcessModule()
        self.decode_module=DecodeModule()
        self.readout_module=ReadoutModule()

    def forward(self,data):
        # Encode input features in leafs
        data=self.encoding_module(data)

        # Process HLD trees from bottom to top
        data=self.process_module(data)

        # Final decoding from heads
        data.y_computed=self.decode_module(
            self.readout_module(data)
        )

        return data