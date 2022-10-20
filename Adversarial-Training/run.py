# Author	: Cristopher McIntyre Garcia 
# Email	 : cmcin019@uottawa.ca
# S-N	   : 300025114

# Imports
import argparse
from email.mime import image
import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform, sample
from sklearn.linear_model import ridge_regression
from sympy import true
from tqdm import tqdm 
from os import system
import os

# Torch imports 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision

torch.cuda.is_available()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CNN(nn.Module):
	def __init__(self, k=3, conv_layers=2, drop=False, bn=False):
		super(CNN, self).__init__()
		l = 28 - (k//2)*2*conv_layers
		self.l = l
		self.name = f'CNN - k={k} layers={conv_layers} dropout={drop} bn={bn}'
		self.convs = nn.ModuleList(
			[
				nn.Conv2d(1, int(16 // math.pow(2, (conv_layers-1))), k)
			] + [
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
		if self.bn:
			a = self.batchnorm(a)
		a = self.fc(a)
		a = self.soft_max(a)
		return a

# Hyperparameters
learning_rate = 0.001
num_epochs = 20

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

def train(model):
	model.to(device=device)
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
	if device=='cuda:0':
		model.to(device='cpu')
	return acc_list

def model_accuracy(model, pgd=None, plot=False):
	correct = 0
	total = 0
	model.eval()
	for images, labels in test_loader:
		images = images.to(device=device)
		labels = labels.to(device=device)
		if not pgd == None:
			images = pgd(images, labels, model)

		with torch.no_grad():
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			if plot:
				print([test_dataset.classes[i][0] for i in predicted])
				imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), "Results")
			
			total += labels.size(0)
			if device=='cuda:0':
				correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
			else:
				correct += (predicted==labels).sum().item()
			
			TestAccuracy = 100 * correct / total

	model.train()
	return(TestAccuracy)

def PGD(images, labels, model, epsilon=0.3, steps=20, alpha=0.02):
	b, c, height, width = images.shape

	# TODO: 1
	uniform_tensor = torch.FloatTensor([[[[uniform(-epsilon, epsilon) for _ in range(width)] for _ in range(height)] for _ in range(c)] for _ in range(b)]).to(device=device)
	perturbed = images + uniform_tensor

	# Loss 
	criterion = nn.CrossEntropyLoss()
	for _ in range(steps): # TODO: 4

		# TODO: 2
		perturbed.requires_grad = True

		score = model(perturbed)
		model.zero_grad()
		loss = criterion(score, labels)
		# print(loss)
		loss.backward(retain_graph=True)
		# print()
		z = perturbed + alpha * perturbed.grad.sign()

		# TODO: 3
		perturbed = torch.min(torch.max(z, images - epsilon), images + epsilon).detach()

	return perturbed

def PGD_Targeted(images, labels, model, targets={0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:0}, epsilon=0.3, steps=20, alpha=0.02):
	b, c, height, width = images.shape
	target_labels = torch.tensor(list(map(lambda x: targets[x], labels.tolist()))).to(device=device)

	# TODO: 1
	uniform_tensor = torch.FloatTensor([[[[uniform(-epsilon, epsilon) for _ in range(width)] for _ in range(height)] for _ in range(c)] for _ in range(b)]).to(device=device)
	perturbed = images + uniform_tensor
	perturbed_target = perturbed

	# Loss 
	criterion = nn.CrossEntropyLoss()
	for _ in range(steps): # TODO: 4

		# TODO: 2
		perturbed.requires_grad = True
		perturbed_target.requires_grad = True

		score = model(perturbed)
		model.zero_grad()
		score = model(perturbed_target)
		model.zero_grad()
		loss = criterion(score, labels)
		loss_target = criterion(score, target_labels)

		loss.backward(retain_graph=True)
		loss_target.backward(retain_graph=True)
		# print()
		z = perturbed + alpha * perturbed.grad.sign()
		z = perturbed_target - alpha * perturbed_target.grad.sign()

		# TODO: 3
		perturbed = torch.min(torch.max(z, images - epsilon), images + epsilon).detach()
		perturbed_target = perturbed

	return perturbed	


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()





def main() -> None :

	# Arguments and values 
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", help="Load model", action="store_true", default=False)
	parser.add_argument("-s", help="Save model", action="store_true", default=False)
	parser.add_argument("-p", help="Plot images", action="store_true", default=False)
	args = parser.parse_args()
	load, save, plot = args.l, args.s, args.p
	assert not (load == True and save == True)

	model = CNN(k=5, conv_layers=3, bn=True)
	
	if not load:
		model_list = [
			model
		]

		list = [train(m) for m in model_list]

		system('cls' if os.name == 'nt' else 'clear')

		for acc_list in range(len(list)):
			print(model_list[acc_list].name)
			for acc in range(len(list[acc_list])):
				if acc % 2 == 0:
					print(f'Epoch {acc+1}: \t{str(list[acc_list][acc])}')
			print()

			if save:
				torch.save(model_list[acc_list].state_dict(), 'models/model.pt')
	else:
		model = CNN(k=5, conv_layers=3, bn=True)
		model.load_state_dict(torch.load('models/model.pt'))
		model.to(device=device)
		# acc = model_accuracy(model, pgd=PGD, plot=plot)
		acc = model_accuracy(model, pgd=PGD_Targeted, plot=plot)
		print(acc)

if __name__ == "__main__":
	main()
	print("End")

