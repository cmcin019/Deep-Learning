# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Imports
import argparse
from glob import glob
import math
from operator import mod
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
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
print(f'Device: {device}')

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

# Training Hyperparameters
learning_rate = 0.001
num_epochs = 15

# PGD Hyperparameters
epsilon=0.3
steps=20
alpha=0.02
targets={0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:0}

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

def model_accuracy(model, pgd=None, pgd_per=[], plot=False, target_acc=False):
	correct = 0
	total = 0
	model.eval()
	for images, labels in test_loader:
		images = images.to(device=device)
		images_saved = images
		labels = labels.to(device=device)
		if not pgd == None:
			images = pgd(images, labels, model, *pgd_per)

		with torch.no_grad():
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			# TODO: For targeted attack
			if target_acc:
				out = model(images_saved)
				top_2 = torch.topk(out.data, 2).indices.tolist()
				labels = torch.tensor(list(map(lambda x, y: max(x) if min(x) == y else min(x), top_2, labels.data.tolist()))).to(device=device)
			if plot:
				system('cls' if os.name == 'nt' else 'clear')
				print([test_dataset.classes[i][0] for i in predicted])
				np_images = torchvision.utils.make_grid(images.cpu().data, normalize=True).numpy()
				plt.imshow(np.transpose(np_images,(1,2,0)))
				plt.show()
				plot = False

			total += labels.size(0)
			if device=='cuda:0':
				correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
			else:
				correct += (predicted==labels).sum().item()
			
			TestAccuracy = 100 * correct / total

	model.train()
	return(TestAccuracy)

def PGD(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha):
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
		loss.backward(retain_graph=True)
		z = perturbed + alpha * perturbed.grad.sign()

		# TODO: 3
		perturbed = torch.min(torch.max(z, images - epsilon), images + epsilon).detach()

	return perturbed

def PGD_Targeted(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha, targets=targets, from_pgd=False, self_target=True):
	
	# TODO: 1
	if from_pgd:
		perturbed = PGD(images, labels, model, epsilon=epsilon, steps=steps, alpha=alpha)
	else:
		b, c, height, width = images.shape
		uniform_tensor = torch.FloatTensor([[[[uniform(-epsilon, epsilon) for _ in range(width)] for _ in range(height)] for _ in range(c)] for _ in range(b)]).to(device=device)
		perturbed = images + uniform_tensor
	if self_target:
		outputs = model(images)
		top_2 = torch.topk(outputs.data, 2).indices.tolist()
		labels = torch.tensor(list(map(lambda x, y: max(x) if min(x) == y else min(x), top_2, labels.data.tolist()))).to(device=device)
	else:
		labels = torch.tensor(list(map(lambda x: targets[x], labels.tolist()))).to(device=device)

	# Loss 
	criterion = nn.CrossEntropyLoss()
	for _ in range(steps): # TODO: 4

		# TODO: 2
		perturbed.requires_grad = True

		score = model(perturbed)
		model.zero_grad()
		loss = criterion(score, labels)
		loss.backward(retain_graph=True)
		z = perturbed - alpha * perturbed.grad.sign()

		# TODO: 3
		perturbed = torch.min(torch.max(z, images - epsilon), images + epsilon).detach()

	return perturbed

def train_with_PGD(model, pgd, pgd_per=[]):
	model.to(device=device)
	acc_list = []
	loss_list = []
	global show_graph

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	accuracy = 0

	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.name}')
		print(pgd.__name__)
		print(pgd_per)
		print(f'Epoch {epoch}: {accuracy}')

		for _, (data, targets) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			targets = targets.to(device=device)
			perturbed = pgd(data, targets, model, *pgd_per)

			scores = model(perturbed)
			loss = criterion(scores, targets)
			loss_list.append(float(loss.data))

			optimizer.zero_grad()
			loss.backward(retain_graph=True)

			optimizer.step()
		accuracy = model_accuracy(model)
		acc_list.append(accuracy)

	# TODO: Need to add observable plots
	if show_graph:
		_, ax = plt.subplots()
		ax.set_title('Accuracy')
		ax.plot([n for n in range(num_epochs)], acc_list, label=f'{pgd.__name__} - {pgd_per}')
		ax.set(xlabel='Epoch', ylabel='Accuracy')
		plt.legend()

		_, ax = plt.subplots()
		ax.set_title('Loss')
		ax.plot([n for n in range(len(loss_list))], loss_list, label=f'{pgd.__name__} - {pgd_per}')
		ax.set(xlabel='Iteration', ylabel='Loss')
		plt.legend()

		# plt.show()

	# print(f'Final accuracy: {accuracy}')
	if device=='cuda:0':
		model.to(device='cpu')

	return acc_list

def experiment(original_model, model, pgd, plot=False, pgd_per=[]):
	acc_before = model_accuracy(original_model, pgd=pgd, pgd_per=pgd_per, plot=plot)
	acc_attack = None
	if pgd.__name__ == 'PGD_Targeted':
		acc_attack = model_accuracy(model, pgd=pgd, pgd_per=pgd_per, plot=plot, target_acc=True)
	acc_list = train_with_PGD(model, pgd=pgd, pgd_per=pgd_per)
	return (acc_before, acc_attack, acc_list)

def run(model, plot, spgd, pgds=[PGD, PGD_Targeted], epsilons=[0, .1, .2, .3, .45], steps=1, alpha=0.01):
	original_model = CNN(k=5, conv_layers=3, bn=True).to(device=device)
	original_model.load_state_dict(torch.load('models/model.pt'))
	acc_original = model_accuracy(model)
	acc = []
	for alg in pgds:
		for eps in epsilons:
			if spgd:
				if device=='cuda:0':
					model.to(device='cpu')
				model = CNN(k=5, conv_layers=3, bn=True).to(device=device)
			else:
				model.load_state_dict(torch.load('models/model.pt'))
			pgd_per = [eps,steps,alpha]
			acc.append((alg.__name__, [eps,steps,alpha], *experiment(original_model, model, alg, plot, pgd_per)))
	
	system('cls' if os.name == 'nt' else 'clear')

	print('Pretrained Model' if not spgd else "From Scratch Model")
	print(f'Accuracy on Original data: {acc_original}')
	print()
	for a in acc:
		print(f'Algorithm: {a[0]}')
		print(f'Ep: {a[1][0]} - Stps: {a[1][1]} - Alpha: {a[1][2]}')
		print(f'Model Acc on Perturbed data: {a[2]}')
		if a[0] == 'PGD_Targeted':
			print(f'Attack Acc :  {a[3]}')
		print(f'Acc After Training:  {a[4]}')
		print()

def main() -> None :
	# Arguments and values 
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--task", help="Enter task (1, 2, 3, 4)", type=int, default=-1)
	parser.add_argument("-l", help="Load model", action="store_true", default=False)
	parser.add_argument("-p", help="Plot images", action="store_true", default=False)
	parser.add_argument("-s", help="Train with PGD from scratch", action="store_true", default=False)
	parser.add_argument("-g", help="Show graphed data", action="store_true", default=False)
	args = parser.parse_args()
	global show_graph
	load, plot, spgd, show_graph = args.l, args.p, args.s, args.g

	model = CNN(k=5, conv_layers=3, bn=True)
	if not load:
		acc_list = train(model)
		system('cls' if os.name == 'nt' else 'clear')
		print(model.name)
		for acc in range(len(acc_list)):
			if acc % 2 == 0:
				print(f'Epoch {acc+1}: \t{str(acc_list[acc])}')
			
			torch.save(model.state_dict(), 'models/model.pt')
	else:
		model = CNN(k=5, conv_layers=3, bn=True)
		model.load_state_dict(torch.load('models/model.pt'))
		model.to(device=device)

	# TODO: Non-targeted 20-step PGD - ϵ: 0.3 - α: 0.02
	if args.task == 1 or args.task == -1:
		run(model, plot=plot, spgd=spgd, pgds=[PGD], epsilons=[.3], steps=20, alpha=.02)

	# TODO: Non-targeted 01-step PGD - ϵ: 0.3 - α: 0.5
	if args.task == 2 or args.task == -1:
		run(model, plot=plot, spgd=spgd, pgds=[PGD], epsilons=[.3], steps=1, alpha=.5)

	# TODO: Targeted 20-step PGD 	 - ϵ: 0.3 - α: 0.02
	if args.task == 3 or args.task == -1:
		run(model, plot=plot, spgd=spgd, pgds=[PGD_Targeted], epsilons=[.3], steps=20, alpha=.02)

	# TODO: Training Evaluation
	if args.task == 4 or args.task == -1:
		run(model, plot=plot, spgd=spgd)
	if show_graph:
		plt.show()

if __name__ == "__main__":
	main()
	print("End")

