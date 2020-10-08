# -*- coding: utf-8 -*-
'''
Name: Diabetes_prediction.py
Auth: long_ouyang
Time: 2020/10/6 15:15
'''

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#-----------------------------------------------------------------------#
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len
#-----------------------------------------------------------------------#
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid() # sigmoid = 1 / ( 1 + exp(-z) )

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
#-----------------------------------------------------------------------#
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#-----------------------------------------------------------------------#
if __name__ == '__main__':
    dataset = DiabetesDataset('diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()