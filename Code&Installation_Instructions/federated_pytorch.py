import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy
from sklearn.preprocessing import StandardScaler


def loss_classifier(predictions,labels):
        
        m = nn.LogSoftmax(dim=1)
        # m = nn.Softmax(dim=1)
        loss = nn.NLLLoss()
        pred = m(predictions).view(-1)
        lab = labels.view(-1).type(torch.LongTensor)
        pred = pred
        
        return loss(pred , lab)
    

def loss_dataset(model, dataset, loss_f):
        """Compute the loss of `model` on `dataset`"""
        loss=0
        
        for idx,(features,labels) in enumerate(dataset):
            
            predictions= model(features)
            loss+=loss_f(predictions,labels)
        
        loss/=idx+1
        return loss
    
def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f):
        print("In local training")
        model_0=deepcopy(model)
        
        for e in range(epochs):
            local_loss=train_step(model,model_0,mu,optimizer,train_data,loss_f)
            
        return float(local_loss.detach().numpy())
    
def difference_models_norm_2(model_1, model_2):
        """Return the norm 2 difference between the two model parameters
        """
        
        tensor_1=list(model_1.parameters())
        tensor_2=list(model_2.parameters())
        
        norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) for i in range(len(tensor_1))])
        
        return norm
    
def set_to_zero_model_weights(model):
        """Set all the parameters of a model to 0"""

        for layer_weigths in model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)
            
def average_models(model, clients_models_hist:list , weights:list):
        

        """Creates the new model of a given iteration with the models of the other
        clients"""
        
        new_model=deepcopy(model)
        set_to_zero_model_weights(new_model)

        for k,client_hist in enumerate(clients_models_hist):
            
            for idx, layer_weights in enumerate(new_model.parameters()):

                contribution=client_hist[idx].data*weights[k]
                layer_weights.data.add_(contribution)
                
        return new_model
    

    
    
def train_step(model, model_0, mu:int, optimizer, train_data, loss_f):
        """Train `model` on one epoch of `train_data`"""
        size = len(train_data.dataset)
        for batch, (X, y) in enumerate(train_data):
        # Compute prediction and loss
            pred = model(X)
            loss = loss_f(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return loss
        # total_loss=0
        
        # for idx, (features,labels) in enumerate(train_data):
            
        #     optimizer.zero_grad()
            
        #     predictions= model(features)
            
        #     loss=loss_f(predictions,labels)
        #     # loss+=mu/2*difference_models_norm_2(model,model_0)
        #     total_loss+=loss
            
        #     loss.backward()
        #     optimizer.step()
            
        # return total_loss/(idx+1)
    
def accuracy_dataset(model, dataset):
        """Compute the accuracy of `model` on `dataset`"""
        
        correct=0
        
        for features,labels in iter(dataset):
            
            predictions= model(features)
            
            _,predicted=predictions.max(1,keepdim=True)
            if predicted.numpy()[0][0] == np.argmax(labels.numpy()[0][0]):
                correct+=1
            # correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
            
        accuracy = 100*correct/len(dataset.dataset)
            
        return accuracy

def plot_acc_loss(title:str, loss_hist:list, acc_hist:list):
    plt.figure()
    
    plt.suptitle(title)

    plt.subplot(1,2,1)
    lines=plt.plot(loss_hist)
    plt.title("Loss")
    plt.legend(lines,["C1", "C2", "C3"])

    plt.subplot(1,2,2)
    lines=plt.plot(acc_hist )
    plt.title("Accuracy")
    plt.legend(lines, ["C1", "C2", "C3"])
    plt.savefig('img3.png',dpi=200)
    
def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, mu=0, file_name="test", epochs=5, lr=10**-2, decay=1e-5):
        """ all the clients are considered in this implementation of FedProx
        Parameters:
            - `model`: common structure used by the clients and the server
            - `training_sets`: list of the training sets. At each index is the 
                training set of client "index"
            - `n_iter`: number of iterations the server will run
            - `testing_set`: list of the testing sets. If [], then the testing
                accuracy is not computed
            - `mu`: regularization term for FedProx. mu=0 for FedAvg
            - `epochs`: number of epochs each client is running
            - `lr`: learning rate of the optimizer
            - `decay`: to change the learning rate at each iteration
        
        returns :
            - `model`: the final global model 
        """
            
        loss_f=loss_classifier
        
        #Variables initialization
        K=len(training_sets) #number of clients
        n_samples=sum([len(db.dataset) for db in training_sets])
        weights=([len(db.dataset)/n_samples for db in training_sets])
        print("Clients' weights:",weights)
        
        
        loss_hist=[[float(loss_dataset(model, dl, loss_f).detach()) for dl in training_sets]]
        acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
        server_hist=[[tens_param.detach().numpy() for tens_param in list(model.parameters())]]
        models_hist = []
        
        
        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc = acc_hist[-1][0]
        # server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
        print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
        
        for i in range(n_iter):
            
            clients_params=[]
            clients_models=[]
            clients_losses=[]
            
            for k in range(K):
                
                local_model=deepcopy(model)
                # local_optimizer=optim.SGD(local_model.parameters(),lr=lr,weight_decay=decay)
                local_optimizer=optim.Adam(local_model.parameters())
                
                local_loss=local_learning(local_model,mu,local_optimizer,training_sets[k],epochs,loss_f)
                
                clients_losses.append(local_loss)
                    
                #GET THE PARAMETER TENSORS OF THE MODEL
                list_params=list(local_model.parameters())
                list_params=[tens_param.detach() for tens_param in list_params]
                clients_params.append(list_params)    
                clients_models.append(deepcopy(local_model))
            
            
            #CREATE THE NEW GLOBAL MODEL
            model = average_models(deepcopy(model), clients_params, 
                weights=weights)
            models_hist.append(clients_models)
            
            #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) for dl in training_sets]]
            acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

            server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
            # server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
            server_acc = acc_hist[-1][0]
            a=10
            if not np.isnan(server_loss):
                a=server_loss
            print(f'====> i: {i+1} Loss: {a} Server Test Accuracy: {server_acc}')
            

            server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(model.parameters())])
            
            #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
            lr*=decay
                
        return model, loss_hist, acc_hist
    
    
class SequenceModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=128, n_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            batch_first = True,
            num_layers = n_layers, # Stack LSTMs
            dropout = 0.5  # This model works on a lot of regularisation
        )

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()  # For distrubuted training

        _, (hidden, _) = self.lstm(x)
        # We want the output from the last layer to go into the final
        # regressor linear layer
        out = hidden[-1] 

        return self.classifier(out)
    


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(100*10, 512),
        nn.Linear(512, 128),
        nn.Linear(128, 11))
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
class LSTM_PT(nn.Module):
    def __init__(self, n_features, hidden_dims = [80,80], seq_length = 250, batch_size=64, n_predictions=1, dropout=0.3):
        super(LSTM_PT, self).__init__()
        
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.seq_length = seq_length
        self.num_layers = len(self.hidden_dims)
        self.batch_size = batch_size
        # self.device = device
        
        print(f'number of layers :{self.num_layers}')
        
        self.lstm1 = nn.LSTM(    
            input_size = n_features, 
            hidden_size = hidden_dims[0],
            batch_first = True,
            dropout = dropout,
            num_layers = self.num_layers)
        
        self.linear = nn.Linear(self.hidden_dims[0], n_predictions) 
        
        
        self.hidden = (
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]),
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0])
            )  
            
    def init_hidden_state(self):
        #initialize hidden states (h_n, c_n)
        
        self.hidden = (
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]),
            torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0])
            ) 
        
    
    def forward(self, sequences):
        
        batch_size, seq_len, n_features = sequences.size() #batch_first
                
        # LSTM inputs: (input, (h_0, c_0))
        #input of shape (seq_len, batch, input_size)....   input_size = num_features
        #or (batch, seq_len, input_size) if batch_first = True
        
        lstm1_out , (h1_n, c1_n) = self.lstm1(sequences, (self.hidden[0], self.hidden[1])) #hidden[0] = h_n, hidden[1] = c_n
                        
        #Output: output, (h_n, c_n)
        #output is of shape (batch_size, seq_len, hidden_size) with batch_first = True
                       
        last_time_step = lstm1_out[:,-1,:] #lstm_out[:,-1,:] or h_n[-1,:,:]
        y_pred = self.linear(last_time_step)
        #output is shape (N, *, H_out)....this is (batch_size, out_features)
        
        return y_pred
    
scaler = StandardScaler()            
class CustomDataset(Dataset):
    def __init__(self, x_arr,y_arr):
        self.data = x_arr
        self.labels = y_arr

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32).reshape(1,11)
        
        scaler.fit_transform(arr)
        return arr, label
    
def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_uniform_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        
if __name__ == "__main__":
    X1_train = np.load('x1_train.npy')
    y1_train = np.load('y1_train.npy')
    
    X1_test = np.load('x1_test.npy')
    y1_test = np.load('y1_test.npy')
    
    X2_train = np.load('x2_train.npy')
    y2_train = np.load('y2_train.npy')
    
    X2_test = np.load('x2_test.npy')
    y2_test = np.load('y2_test.npy')
    
    X3_train = np.load('x3_train.npy')
    y3_train = np.load('y3_train.npy')
    
    X3_test = np.load('x3_test.npy')
    y3_test = np.load('y3_test.npy')
    
    X4_train = np.load('x4_train.npy')
    y4_train = np.load('y4_train.npy')
    
    X4_test = np.load('x4_test.npy')
    y4_test = np.load('y4_test.npy')
    
    n_iter = 10
    
    n_timesteps, n_features, n_outputs = X1_train.shape[1], X1_train.shape[2], y1_train.shape[1]
    # model_0 = NeuralNetwork()
    
    model_0 = LSTM_PT(n_features,seq_length=100,batch_size=1,n_predictions=n_outputs)
    model_0.apply(initialize_weights)
    # model_0 = SequenceModel(n_features,n_outputs)
    
    
    d1 = DataLoader(CustomDataset(X1_train, y1_train))
    d2 = DataLoader(CustomDataset(X2_train, y2_train))
    d3 = DataLoader(CustomDataset(X3_train, y3_train))
    d4 = DataLoader(CustomDataset(X4_train, y4_train))
    
    d1_t = DataLoader(CustomDataset(X1_test, y1_test))
    d2_t = DataLoader(CustomDataset(X2_test, y2_test))
    d3_t = DataLoader(CustomDataset(X3_test, y3_test))
    d4_t = DataLoader(CustomDataset(X4_test, y4_test))
    
    model_f, loss_hist_FA_iid, acc_hist_FA_iid = FedProx( model_0,[d1,d3,d4], n_iter,[d1_t], epochs =2, lr =0.001, mu=0)
    
    
    
    

    plot_acc_loss("FedAvg HAR", loss_hist_FA_iid, acc_hist_FA_iid)