import torch
import torch.nn.functional as F
import torch.nn as nn
class Config:
    def __init__(self,):
        super(Config,self).__init__()
        self.HIDDEN_SIZE=64
        self.IN_SIZE=3
        self.OUT_SIZE=1
        self.EDGE_SIZE = 0
        self.MOLECULES=False
        self.criterion=F.mse_loss
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MoleculesConfig(Config):
    def __init__(self,):
        super(Config,self).__init__()
        self.HIDDEN_SIZE=64
        self.IN_SIZE=9
        self.OUT_SIZE=1
        self.MOLECULES=True #Use OGB Evaluator and loss
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PeptidesConfig(Config):
    def __init__(self,):
        super(Config,self).__init__()
        self.HIDDEN_SIZE=64
        self.EDGE_SIZE=3
        self.IN_SIZE=9
        self.OUT_SIZE=10
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","AVERAGE_PRECISION",]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion=nn.BCEWithLogitsLoss()

class EdgeTestConfig(Config):
    def __init__(self,):
        super(Config,self).__init__()
        self.HIDDEN_SIZE=64
        self.EDGE_SIZE=1
        self.IN_SIZE=3
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","AVERAGE_PRECISION",]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion=nn.BCEWithLogitsLoss()

config = PeptidesConfig()

