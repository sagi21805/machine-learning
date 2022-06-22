import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fastai.vision.all import Module
from IPython.core.debugger import set_trace

class GeneralFit(Module):
    def __init__(self, input_size, output_size, hidden_size=5):
        self.linear_in  = nn.Linear(input_size, hidden_size)
        self.hidden     = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear_in(x))
        x = torch.relu(self.hidden(x))
        x = self.linear_out(x)
        return x
        
    def accuracy_multi(x, y):
        return (x.argmax(-1) == y).float().mean()

    def fit(x_train, y_train, x_test, y_test, model, criterion, optimizer, num_epochs):
        loss_history      = [] # to save the loss at each epoch.
        loss_test_history = [] # to save the test loss at each epoch.
        out_history       = [] # to save the parameters at each epoch
        acc_train_history = []
        acc_test_history  = [] 
        for ii, epoch in enumerate(range(num_epochs)):
            # forward
            model.train()
            out = model(x_train)
            loss = criterion(out, y_train)
            acc_train = GeneralFit.accuracy_multi(out, y_train)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # test
            model.eval()
            out_test  = model(x_test)
            loss_test = criterion(out_test, y_test)
            acc_test  = GeneralFit.accuracy_multi(out_test, y_test)
            
            loss_history.append(loss.item())
            loss_test_history.append(loss_test.item())
            acc_train_history.append(acc_train)
            acc_test_history.append(acc_test)
            if ii == 0:
                out_history = out.detach().cpu().numpy()
            else:
                out_history = np.concatenate((out_history, out.detach().cpu().numpy()), axis=-1)
            
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))
        return loss_history, loss_test_history, out_history, acc_train_history, acc_test_history
    
iris = datasets.load_iris()
X = iris.data   # Get training attributes
y = iris.target # Get labels
input_size = X.shape[-1]
cats = np.sum(np.unique(y)).astype(int)
print('Number of samples:', X.shape[0])
print('Number of attributes:', input_size)
print('Number of categories:', cats)

df = pd.DataFrame({k:X[:,i] for i,k in enumerate(iris['feature_names'])})
fig, ax = plt.subplots(figsize=(12,12), dpi=150)
pd.plotting.scatter_matrix(df, figsize=(12,12), c=y, s=200, alpha=1, ax=ax)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=2020)

model = GeneralFit(int(input_size), int(cats))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model = GeneralFit(int(input_size), int(cats))
# criterion = nn.BCEWithLogitsLoss() #includes the sigmoid function inside it
# optimizer = optim.Adam(model.parameters(), lr=0.001)

x_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

loss_train, loss_test, out, acc_train, acc_test = GeneralFit.fit(x_train, y_train, x_test, y_test, model, criterion, optimizer, num_epochs=1000)

plt.figure(figsize=(6,3), dpi=120)
plt.plot(loss_train, color='red', label='train loss')
plt.plot(loss_test, color='green', label='test loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

