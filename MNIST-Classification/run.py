# Author	: Cristopher McIntyre Garcia 
# Email	 : cmcin019@uottawa.ca
# S-N	   : 300025114

# Imports
import math
import matplotlib.pyplot as plt
import numpy as np
from random import random
from tqdm import tqdm 
from os import system
import os

# Torch imports 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

from torch.autograd import Variable
torch.cuda.is_available()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SoftMax_Regression(nn.Module):
	def __init__(self, drop=False, bn=False):
		super(SoftMax_Regression, self).__init__()
		self.name = f'SoftMax_Regression - dropout={drop} bn={bn}'
		self.linear = nn.Linear(28*28, 10) 
		self.soft_max = nn.Softmax(1)
		self.drop = drop
		self.dropout = nn.Dropout(p=0.2)
		self.bn = bn
		self.batchnorm = nn.BatchNorm1d(28*28)

	def forward(self, image):
		a = image.view(-1, 28*28)
		if self.drop:
			a = self.dropout(a)
		if self.bn:
			a = self.batchnorm(a)
		a = self.linear(a)
		a = self.soft_max(a)
		return a

class MLP(nn.Module):
	def __init__(self, width=28, depth=1, drop=False, bn=False):
		super(MLP, self).__init__()
		s = width
		self.name = f'MLP - width={width} depth={depth} dropout={drop} bn={bn}'
		self.first = nn.Linear(28*28, s*s)
		self.hidden = nn.Linear(s*s, s*s)
		self.final = nn.Linear(s*s, 10)
		self.soft_max = nn.Softmax(1)
		self.depth = depth
		self.drop = drop
		self.dropout = nn.Dropout(p=0.2)
		self.bn = bn
		self.batchnorm = nn.BatchNorm1d(s*s)

	def forward(self, image):
		h = 1
		a = image.view(-1, 28*28)
		a = F.relu(self.first(a))
		for _ in range(self.depth):
			a = F.relu(self.hidden(a))
		if self.drop:
			a = self.dropout(a)
		a = self.final(a)
		a = self.soft_max(a)
		return a

class CNN(nn.Module):
	def __init__(self, k=3, conv_layers=2, drop=False, bn=False):
		super(CNN, self).__init__()
		l = 28 - (k//2)*2*conv_layers
		self.l = l
		self.name = f'CNN - k={k} layers={conv_layers} dropout={drop} bn={bn}'
		# print([(16 // math.pow(2, (conv_layers)), 16 // math.pow(2, (conv_layers-x-2)), k) for x in range(conv_layers-1)])
		self.convs = nn.ModuleList(
			[
				nn.Conv2d(1, int(16 // math.pow(2, (conv_layers-1))), k)
			] +
			[
				nn.Conv2d(int(16 // math.pow(2, int(conv_layers-x-1))), int(16 // math.pow(2, (conv_layers-x-2))), k) for x in range(conv_layers-1)
			]
		)
		self.fc = nn.Linear(16 * l * l, 10)
		self.soft_max = nn.Softmax(1)
		self.drop = drop
		self.dropout = nn.Dropout(p=0.2)
		self.bn = bn
		self.batchnorm = nn.BatchNorm1d(16 * l * l)

	def forward(self, image):
		for c in self.convs:
			image = F.relu(c(image))
		a = image.view(-1, 16 * self.l * self.l)
		if self.drop:
			a = self.dropout(a)
		if self.drop:
			a = self.batchnorm(a)
		a = self.fc(a)
		a = self.soft_max(a)
		return a

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Load training dataset
train_dataset = datasets.MNIST(
	root = 'data',
	train = True,
	transform = transforms.ToTensor(),
	download = True,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Load testing dataset
test_dataset = datasets.MNIST(
	root = 'data',
	train = False,
	transform = transforms.ToTensor(),
	download = False,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


# Intantiate models
sm_r = SoftMax_Regression().to(device=device)

mlp_28_2 = MLP(width=28, depth=2).to(device=device)
cnn_3_2 = CNN(k=3, conv_layers=2).to(device=device)


mlp_28_1 = MLP(width=28, depth=1).to(device=device)
cnn_3_3 = CNN(k=3, conv_layers=3).to(device=device)

mlp_56_1 = MLP(width=56, depth=1).to(device=device)
cnn_5_3 = CNN(k=5, conv_layers=3).to(device=device)

mlp_56_3 = MLP(width=56, depth=3).to(device=device)
cnn_5_1 = CNN(k=5, conv_layers=1).to(device=device)

mlp_28_3 = MLP(width=28, depth=3).to(device=device)
cnn_3_1 = CNN(k=3, conv_layers=1).to(device=device)

mlp_112_2 = MLP(width=112, depth=2).to(device=device)
cnn_7_1 = CNN(k=7, conv_layers=1).to(device=device)

mlp_112_1 = MLP(width=112, depth=1).to(device=device)
cnn_7_3 = CNN(k=7, conv_layers=3).to(device=device)


# Models with dropout
sm_r_dr = SoftMax_Regression(drop=True).to(device=device)

mlp_28_1_dr = MLP(width=28, depth=1, drop=True).to(device=device)
cnn_3_3_dr = CNN(k=3, conv_layers=3, drop=True).to(device=device)

mlp_56_1_dr = MLP(width=56, depth=1, drop=True).to(device=device)
cnn_5_3_dr = CNN(k=5, conv_layers=3, drop=True).to(device=device)

mlp_56_3_dr = MLP(width=56, depth=3, drop=True).to(device=device)
cnn_5_1_dr = CNN(k=5, conv_layers=1, drop=True).to(device=device)

mlp_28_3_dr = MLP(width=28, depth=3, drop=True).to(device=device)
cnn_3_1_dr = CNN(k=3, conv_layers=1, drop=True).to(device=device)

mlp_112_2_dr = MLP(width=112, depth=2, drop=True).to(device=device)
cnn_7_1_dr = CNN(k=7, conv_layers=1, drop=True).to(device=device)

mlp_112_1_dr = MLP(width=112, depth=1, drop=True).to(device=device)
cnn_7_3_dr = CNN(k=7, conv_layers=3, drop=True).to(device=device)


# Models with batch normalization
sm_r_bn = SoftMax_Regression(bn=True).to(device=device)

mlp_28_1_bn = MLP(width=28, depth=1, bn=True).to(device=device)
cnn_3_3_bn = CNN(k=3, conv_layers=3, bn=True).to(device=device)

mlp_56_1_bn = MLP(width=56, depth=1, bn=True).to(device=device)
cnn_5_3_bn = CNN(k=5, conv_layers=3, bn=True).to(device=device)

mlp_56_3_bn = MLP(width=56, depth=3, bn=True).to(device=device)
cnn_5_1_bn = CNN(k=5, conv_layers=1, bn=True).to(device=device)

mlp_28_3_bn = MLP(width=28, depth=3, bn=True).to(device=device)
cnn_3_1_bn = CNN(k=3, conv_layers=1, bn=True).to(device=device)

mlp_112_2_bn = MLP(width=112, depth=2, bn=True).to(device=device)
cnn_7_1_bn = CNN(k=7, conv_layers=1, bn=True).to(device=device)

mlp_112_1_bn = MLP(width=112, depth=1, bn=True).to(device=device)
cnn_7_3_bn = CNN(k=7, conv_layers=3, bn=True).to(device=device)


# Models with dropout & batch normalization
sm_r_dr_bn = SoftMax_Regression(drop=True, bn=True).to(device=device)

mlp_28_1_dr_bn = MLP(width=28, depth=1, drop=True, bn=True).to(device=device)
cnn_3_3_dr_bn = CNN(k=3, conv_layers=3, drop=True, bn=True).to(device=device)

mlp_56_1_dr_bn = MLP(width=56, depth=1, drop=True, bn=True).to(device=device)
cnn_5_3_dr_bn = CNN(k=5, conv_layers=3, drop=True, bn=True).to(device=device)

mlp_56_3_dr_bn = MLP(width=56, depth=3, drop=True, bn=True).to(device=device)
cnn_5_1_dr_bn = CNN(k=5, conv_layers=1, drop=True, bn=True).to(device=device)

mlp_28_3_dr_bn = MLP(width=28, depth=3, drop=True, bn=True).to(device=device)
cnn_3_1_dr_bn = CNN(k=3, conv_layers=1, drop=True, bn=True).to(device=device)

mlp_112_2_dr_bn = MLP(width=112, depth=2, drop=True, bn=True).to(device=device)
cnn_7_1_dr_bn = CNN(k=7, conv_layers=1, drop=True, bn=True).to(device=device)

mlp_112_1_dr_bn = MLP(width=112, depth=1, drop=True, bn=True).to(device=device)
cnn_7_3_dr_bn = CNN(k=7, conv_layers=3, drop=True, bn=True).to(device=device)


def train(model):
	acc_list = []
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	accuracy = 0
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.name}')
		print(f'Epoch {epoch}: {accuracy}')
		for _, (data, targets) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			targets = targets.to(device=device)

			scores = model(data)
			loss = criterion(scores, targets)

			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

		accuracy = model_accuracy(model)
		acc_list.append(accuracy)
	
	print(f'Final accuracy: {accuracy}')
	return acc_list

def model_accuracy(model):
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for images, labels in test_loader:
			images = images.to(device=device)
			labels = labels.to(device=device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			if device=='cuda:0':
				correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
			else:
				correct += (predicted==labels).sum().item()
			
		TestAccuracy = 100 * correct / total

	model.train()
	return(TestAccuracy)

def main() -> None :
	model_list = [
		# sm_r, # Basic
		# mlp_28_2, # Basic
		cnn_3_2, # Basic
		# mlp_28_1,
		cnn_3_3,
		# mlp_56_1,
		cnn_5_3,
		# mlp_56_3,
		cnn_5_1,
		# mlp_28_3,
		cnn_3_1,
		# mlp_112_2,
		# cnn_7_1,

	# 	sm_r_dr,
	# 	mlp_28_1_dr,
	# 	cnn_3_3_dr,
	# 	# mlp_56_1_dr,
	# 	# cnn_5_3_dr,
	# 	# mlp_56_3_dr,
	# 	# cnn_5_1_dr,
	# 	# mlp_28_3_dr,
	# 	# cnn_3_1_dr,

	# 	sm_r_bn,
	# 	mlp_28_1_bn,
	# 	cnn_3_3_bn,
	# 	# mlp_56_1_bn,
	# 	# cnn_5_3_bn,
	# 	# mlp_56_3_bn,
	# 	# cnn_5_1_bn,
	# 	# mlp_28_3_bn,
	# 	# cnn_3_1_bn,

	# 	sm_r_dr_bn,
	# 	mlp_28_1_dr_bn,
	# 	cnn_3_3_dr_bn,
	# 	# mlp_56_1_dr_bn,
	# 	# cnn_5_3_dr_bn,
	# 	# mlp_56_3_dr_bn,
	# 	# cnn_5_1_dr_bn,
	# 	# mlp_28_3_dr_bn,
	# 	# cnn_3_1_dr_bn,
	]

	# model_list = [
	# 	sm_r,
	# 	sm_r_dr,
	# 	sm_r_bn,
	# 	sm_r_dr_bn,
	# 	mlp_28_1,
	# 	mlp_28_1_dr,
	# 	mlp_28_1_bn,
	# 	mlp_28_1_dr_bn,
	# 	cnn_3_3,
	# 	cnn_3_3_dr,
	# 	cnn_3_3_bn,
	# 	cnn_3_3_dr_bn
	# ]




	list = [train(m) for m in model_list]

	system('cls' if os.name == 'nt' else 'clear')

	for acc_list in range(len(list)):
		print(model_list[acc_list].name)
		for acc in range(len(list[acc_list])):
			if acc % 2 == 0:
				print(f'Epoch {acc+1}: \t{str(list[acc_list][acc])}')
		print()
	
if __name__ == "__main__":
	main()
	print("End")

