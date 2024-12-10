import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


ds_transform=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])


ds_train=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

ds_test=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)


for image_batch,label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model=models.MyModel()
    
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test_accuracy:{acc_test*100:.2f}%')

model=models.MyModel()

loss_fn=torch.nn.CrossEntropyLoss()

learning_rate=1e-3
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#models.train(model,dataloader_test,loss_fn,optimizer)

#acc_test=models.test_accuracy(model,dataloader_test)
#print(f'test_accuracy:{acc_test*100:.2f}%')

n_epochs=5

for k in range(n_epochs):
    print(f'epoch{k+1}/{n_epochs}',end=':',flush=True)

    time_start=time.time()    
    loss_train=models.train(model,dataloader_test,loss_fn,optimizer)
    time_end=time.time()
    print(f'train loss:{loss_train:3f}{time_end-time_start:1f}s',end='',end='')
    
    time_start=time.time()   
    loss_test=models.test(model,dataloader_test,loss_fn)
    time_end=time.time()
    print(f'test loss:{loss_test}{time_end-time_start:1f}s',end='')

    time_start=time.time()   
    acc_train=models.test_accuracy(model,dataloader_train)
    time_end=time.time()
    print(f'train accuracy:{acc_train*100:.3f}%{time_end-time_start:1f}s',end='')
    
    time_start=time.time()   
    acc_test=models.test_accuracy(model,dataloader_test)
    time_end=time.time()
    print(f'test accuracy:{acc_test*100:.3f}%{time_end-time_start:1f}s',end='')
