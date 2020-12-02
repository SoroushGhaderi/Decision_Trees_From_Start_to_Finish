import torch
from CustomDataset import custom_dataset
from custom_model import Net

my_model = Net()
iris_dataset = custom_dataset('./Datasets/iris/iris.csv')
# iris_dataset_test = custom_dataset('./Datasets/iris/iris_test.csv')

# for i in range(len(iris_dataset)):
#     sample = iris_dataset[i]
#     print(i, sample[0], sample[1])

from torch.utils.data import random_split, DataLoader
iris_train, iris_valid, iris_test = random_split(iris_dataset, (100, 20, 30))

dataloader = DataLoader(iris_train, batch_size=10, shuffle=True)
# dataloader_test = DataLoader(iris_test, batch_size=100, shuffle=True)

num_epoch = 500
for epoch in range(num_epoch):
    print(epoch)
    for data_batch, label_batch in dataloader:
        print(my_model(data_batch))
