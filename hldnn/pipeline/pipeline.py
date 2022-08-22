import torch_geometric
import torch_scatter
from tqdm import tqdm
import torch
from config import config
from ogb.graphproppred import Evaluator
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
import pipeline.metrics as metrics

class Pipeline:
    def __init__(self,network,train_dataset,validation_dataset,test_dataset,epoch_num,batch_size=32,lr=0.001):
        self.device = config.device
        self.epoch_num=epoch_num
        self.batch_size=batch_size
        self.lr=lr
        self.criterion=config.criterion
        self.network=network
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.validation_dataset=validation_dataset

        if config.MOLECULES:
            self.molecules_evaluator = Evaluator(name = 'ogbg-molhiv')
            self.aucm_criterion = AUCMLoss().to(self.device)

    def evaluate_with(self,dataset,name):
        self.network.eval()
        with torch.no_grad():
            testloader = torch_geometric.data.DataLoader(dataset, self.batch_size, shuffle=False,num_workers=config.NUM_WORKERS)

            temp_sum_test_loss = 0

            y_true = []
            y_pred = []

            for test_batch in testloader:
                # Send Batch to Device
                test_batch.to(self.device)
                
                test_batch=self.network(test_batch)

                # Forward and Loss
                temp_sum_test_loss += self.criterion(test_batch.y_computed.reshape(test_batch.y.size()), test_batch.y.float()).detach().item()
               
                pred=test_batch.y_computed.reshape(test_batch.y.size())
                y_true.append(test_batch.y.detach())
                y_pred.append(pred.detach())

            temp_avg_test_loss = temp_sum_test_loss / len(testloader)

            y_true = torch.cat(y_true,dim=0)
            y_pred = torch.cat(y_pred,dim=0)
            y_true=y_true if y_true.dim()==2 else torch.unsqueeze(y_true,dim=1)
            y_pred=y_pred if y_pred.dim()==2 else torch.unsqueeze(y_pred,dim=1)

            for metric in config.METRICS:
                if metric == "AUCM":
                    input_dict = {"y_true": y_true,"y_pred": y_pred}
                    print(name+" rocauc: ", self.molecules_evaluator.eval(input_dict))
                elif metric == "LOSS":
                    print(name+" loss: ",temp_avg_test_loss)
                elif metric == "AVERAGE_PRECISION":
                    print(y_pred[0:3])
                    print(name+" AP: ",metrics.eval_ap(y_true.numpy(),y_pred.numpy()))


    def evaluate(self):
        self.evaluate_with(self.validation_dataset,"Validation")
        self.evaluate_with(self.test_dataset,"Test")

    def train(self):
        print(self.network)

        train_dataset=self.train_dataset

        print("Training started")

        data_loader = torch_geometric.loader.DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
        print("Data loader created")

        self.network = self.network.to(self.device)
        if config.MOLECULES:
            optimizer=PESG(self.network,loss_fn=self.aucm_criterion,lr=0.1,momentum=0.9,margin=1.0,epoch_decay = 0.003, weight_decay = 0.0001)
        else:
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        train_loss = []
        test_loss = []

        for epoch in range(self.epoch_num):
            sum_loss =0
            for step, data in tqdm(enumerate(data_loader)):
                self.network.train()
                optimizer.zero_grad()
                data = data.to(self.device)
                data = self.network(data)

                pred=data.y_computed.reshape(data.y.size())

                if config.MOLECULES:
                    is_labeled = data.y == data.y
                    loss =  self.aucm_criterion(pred.to(torch.float32)[is_labeled].reshape(-1, 1),
                                  data.y.to(torch.float32)[is_labeled].reshape(-1, 1))
                else:
                    loss = self.criterion(pred,data.y.float())

                sum_loss  += loss.detach().item()

                loss.backward()
                optimizer.step()

                #Logging
                train_loss.append(loss.detach().item())
            sum_loss/=len(data_loader)

            # Print Information
            print('-'*20)
            print('Epoch', epoch)
            print("Train metric: ",sum_loss)
            self.evaluate()