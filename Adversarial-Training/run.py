# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Imports
import argparse
import math
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
num_epochs = 20

# PGD Hyperparameters
epsilon=0.3
steps=5
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

def model_accuracy(model, pgd=None, pgd_per=[0,0,0], plot=False, target_acc=False, on_training=False):
	correct = 0
	total = 0
	loader = train_loader if on_training else test_loader
	model.eval()
	for images, labels in loader:
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

				np_images = torchvision.utils.make_grid(images.cpu().data, normalize=True).numpy()
				fig, ax = plt.subplots()
				ax.imshow(np.transpose(np_images,(1,2,0)))
				fig.savefig(f'images/{pgd.__name__ if not pgd == None else None} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}.jpg', bbox_inches='tight', dpi=150)
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
	acc_list_p = []
	acc_list_np =[]
	loss_list = []
	loss_list_np = []
	loss_list_iter = []
	loss_list_np_iter = []

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
			if show_graph:
				scores_np = model(data)
				loss_np = criterion(scores_np, targets)	
				loss_list_iter.append(float(loss.data))
				loss_list_np_iter.append(float(loss_np.data))

			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()


		if show_graph:
			loss_list.append(float(loss.data))
			loss_list_np.append(float(loss_np.data))
			accuracy_np = model_accuracy(model, on_training=True)
			acc_list_np.append(accuracy_np)
			accuracy_p = model_accuracy(model, pgd=pgd, pgd_per=pgd_per, on_training=True)
			acc_list_p.append(accuracy_p)

		accuracy = model_accuracy(model)
		acc_list.append(accuracy)

	# TODO: Need to add observable plots
	if show_graph:
		global ax_unperturbed_training
		global ax_unperturbed_training_loss
		global ax_unperturbed_training_loss_iter
		global ax_perturbed_training
		global ax_perturbed_training_loss
		global ax_perturbed_training_loss_iter

		ax_unperturbed_training.plot([n for n in range(len(acc_list_np))], acc_list_np, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_unperturbed_training.legend()

		ax_unperturbed_training_loss.plot([n for n in range(len(loss_list_np))], loss_list_np, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_unperturbed_training_loss.legend()

		ax_unperturbed_training_loss_iter.plot([n for n in range(len(loss_list_np_iter))], loss_list_np_iter, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_unperturbed_training_loss_iter.legend()

		ax_perturbed_training.plot([n for n in range(len(acc_list_p))], acc_list_p, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_perturbed_training.legend()

		ax_perturbed_training_loss.plot([n for n in range(len(loss_list))], loss_list, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_perturbed_training_loss.legend()

		ax_perturbed_training_loss_iter.plot([n for n in range(len(loss_list_iter))], loss_list_iter, label=f'{pgd.__name__} - ϵ: {pgd_per[0]} - steps: {pgd_per[1]} - α: {pgd_per[2]}')
		ax_perturbed_training_loss_iter.legend()

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

	return acc, model

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

	acc_original = model_accuracy(model)
	runs = []
	models = []

	if show_graph:
		global ax_unperturbed_training
		global ax_unperturbed_training_loss
		global ax_unperturbed_training_loss_iter
		global ax_perturbed_training
		global ax_perturbed_training_loss
		global ax_perturbed_training_loss_iter

		fig_1, ax_unperturbed_training = plt.subplots()
		ax_unperturbed_training.set_title('Accuracy on Unperturbed Training Data')

		fig_2, ax_unperturbed_training_loss = plt.subplots()
		ax_unperturbed_training_loss.set_title('Loss on Unperturbed Training Data')

		fig_2_iter, ax_unperturbed_training_loss_iter = plt.subplots()
		ax_unperturbed_training_loss_iter.set_title('Loss on Unperturbed Training Data')

		fig_3, ax_perturbed_training = plt.subplots()
		ax_perturbed_training.set_title('Accuracy on Perturbed Training Data')

		fig_4, ax_perturbed_training_loss = plt.subplots()
		ax_perturbed_training_loss.set_title('Loss on Perturbed Training Data')

		fig_4_iter, ax_perturbed_training_loss_iter = plt.subplots()
		ax_perturbed_training_loss_iter.set_title('Loss on Perturbed Training Data')

	# TODO: Non-targeted 20-step PGD - ϵ: 0.3 - α: 0.02
	if args.task == 1 or args.task == 4:
		run1, m = run(model, plot=plot, spgd=spgd, pgds=[PGD], epsilons=[.3], steps=20, alpha=.02)
		runs.append(run1)
		models.append(m)

	# TODO: Non-targeted 01-step PGD - ϵ: 0.3 - α: 0.5
	if args.task == 2 or args.task == 4:
		run2, m = run(model, plot=plot, spgd=spgd, pgds=[PGD], epsilons=[.3], steps=1, alpha=.5)
		runs.append(run2)
		models.append(m)

	# TODO: Targeted 20-step PGD 	 - ϵ: 0.3 - α: 0.02
	if args.task == 3 or args.task == 4:
		run3, m = run(model, plot=plot, spgd=spgd, pgds=[PGD_Targeted], epsilons=[.3], steps=20, alpha=.02)
		runs.append(run3)
		models.append(m)

	if show_graph:
		ax_unperturbed_training.set(xlabel='Epoch', ylabel='Accuracy')
		ax_unperturbed_training_loss.set(xlabel='Epoch', ylabel='Loss')
		ax_unperturbed_training_loss_iter.set(xlabel='Iteration', ylabel='Loss')
		ax_perturbed_training.set(xlabel='Epoch', ylabel='Accuracy')
		ax_perturbed_training_loss.set(xlabel='Epoch', ylabel='Loss')
		ax_perturbed_training_loss_iter.set(xlabel='Iteration', ylabel='Loss')
		# plt.legend()

		fig_1.savefig(f'images/Accuracy on Unperturbed Training Data.jpg', bbox_inches='tight', dpi=150)

		fig_2.savefig(f'images/Loss on Unperturbed Training Data.jpg', bbox_inches='tight', dpi=150)

		fig_2_iter.savefig(f'images/Loss on Unperturbed Training Data (iter).jpg', bbox_inches='tight', dpi=150)

		fig_3.savefig(f'images/Accuracy on Perturbed Training Data.jpg', bbox_inches='tight', dpi=150)

		fig_4.savefig(f'images/Loss on Perturbed Training Data.jpg', bbox_inches='tight', dpi=150)
		
		fig_4_iter.savefig(f'images/Loss on Perturbed Training Data (iter).jpg', bbox_inches='tight', dpi=150)

	# TODO: Training Evaluation
	system('cls' if os.name == 'nt' else 'clear')
	print('Pretrained Model' if not spgd else "From Scratch Model")
	print(f'Accuracy on Original data: {acc_original}')
	print()

	plt.close()
	
	fig, ax = plt.subplots()
	ax.set_title('Accuracy on Perturbed Testing Data')

	for r in runs:
		print()
		for a in r:
			ax.plot([n for n in range(len(a[4]))], a[4], label=f'{a[0]} - ϵ: {a[1][0]} - steps: {a[1][1]} - α: {a[1][2]}')

			print(f'Algorithm: {a[0]}')
			print(f'Ep: {a[1][0]} - Stps: {a[1][1]} - Alpha: {a[1][2]}')
			print(f'Model Acc on Perturbed data: {a[2]}')
			if a[0] == 'PGD_Targeted':
				print(f'Attack Acc :  {a[3]}')
			print(f'Acc After Training:  {a[4]}')
			print()
	
	ax.set(xlabel='Epoch', ylabel='Accuracy')
	plt.legend()
	fig.savefig(f'images/Accuracy on Perturbed Testing Data.jpg', bbox_inches='tight', dpi=150)

	pgds=[PGD, PGD_Targeted]
	epsilons=[0, .1, .2, .3, .45]
	steps=40
	alpha=0.01
	p = True
	for m in range(len(models)):
		print()
		print(f"Model: {runs[m][0][0]} - ϵ: {runs[m][0][1][0]} - steps: {runs[m][0][1][1]} - α: {runs[m][0][1][2]}")
		fig_acc_test, ax_acc_test = plt.subplots()
		ax_acc_test.set_title('Accuracy on Perturbed Testing Data')
		for alg in pgds:
			acc = []
			for eps in epsilons:
				ac = model_accuracy(models[m],pgd=alg, plot=p, pgd_per=[eps, steps, alpha])
				print(f'40-step {alg.__name__} ϵ: {eps} - α: {alpha} \t {ac}')
				acc.append(ac)
			ax_acc_test.plot([n for n in epsilons], acc, label=f'{alg.__name__}')	
			ax_acc_test.legend()
		p = False
		ax_acc_test.set(xlabel='ϵ', ylabel='Accuracy')
		fig_acc_test.savefig(f'images/Model: {runs[m][0][0]} - ϵ: {runs[m][0][1][0]} - steps: {runs[m][0][1][1]} - α: {runs[m][0][1][2]}.jpg', bbox_inches='tight', dpi=150)

	plt.close()

	# plt.show()

if __name__ == "__main__":
	main()
	print("End")
