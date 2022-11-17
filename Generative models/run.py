# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Run:
# python3 run.py


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
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F

# Model imports
from models import vae, gan, wgan

# Data import
from data import mnist, cifar10

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default='m', help="dataset (m or c)")
parser.add_argument("-g", type=str, default='vae', help="generator (vae, gan, wgan)")
opt = parser.parse_args()


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Data loaders
if opt.d == 'm':
	train_loader, test_loader, classes, (c, h, w) = mnist()
else:
	train_loader, test_loader, classes, (c, h, w) = cifar10()

pad = transforms.Pad((0,0,0,0))
if h == 28:
	pad = transforms.Pad((2,2,2,2))

# Training Hyperparameters
if opt.g == 'vae':
	lr = 2e-4
if opt.g == 'gan':
	lr = 2e-4
	K = 5
else: # wgan
	lr = 5e-5
	K = 5
	weight_c = 0.01
num_epochs = 20

def train_vae(model):
	model.to(device=device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	# loss_fn = nn.BCELoss(reduction="sum")
	train_loss=0
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.__class__.__name__}')
		print(f'Epoch {epoch}')
		print(f'partial train loss (single batch): {train_loss}')
		for _, (data, _) in enumerate(tqdm(train_loader)):
			data = pad(data)
			data = data.to(device=device)
			_, loss = model(data)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		train_loss=loss.item()

	if device=='cuda:0':
		model.to(device='cpu')

	return train_loss/len(train_loader.dataset)

def train_gan(model):
	model.to(device=device)
	discriminator = model.discriminator if opt.g == 'gan' else model.critic
	gen_optimizer = optim.Adam(model.generator.parameters(), lr=lr, betas=(.5, .999))
	disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(.5, .999))
	train_loss=(0,0)
	for epoch in range(num_epochs):
		system('cls' if os.name == 'nt' else 'clear')
		print(f'Training {model.__class__.__name__}')
		print(f'Epoch {epoch}')
		print(f'partial train loss (single batch): {train_loss}')
		k = 1
		for _, (data, _) in enumerate(tqdm(train_loader)):
			data = pad(data)
			data = data.to(device=device)
			_, gen_loss, disc_loss = model(data)

			discriminator.zero_grad()
			disc_loss.backward(retain_graph=True)

			if opt.g == 'wgan':
				for p in discriminator.parameters():
					p.data.clamp_(-weight_c, weight_c)

			if not k % K == 0:
				model.generator.zero_grad()
				gen_loss.backward()

			disc_optimizer.step()
			if not k % K == 0:
				gen_optimizer.step()
			
		train_loss=(gen_loss.item(), disc_loss.item())

	if device=='cuda:0':
		model.to(device='cpu')

	return train_loss[0]/len(train_loader.dataset), train_loss[1]/len(train_loader.dataset)

def inference_vae(model, digit, num_examples=1):
	"""
	Generates (num_examples) of a particular digit.
	Specifically we extract an example of each digit,
	then after we have the mu, sigma representation for
	each digit we can sample from that.
	After we sample we can run the decoder part of the VAE
	and generate examples.
	"""
	model.eval()

	images = []
	idx = 0
	for x, y in train_loader.dataset:
		x = pad(x)
		if y == idx:
			images.append(x)
			idx += 1
		if idx == 10:
			break

	encodings_digit = []
	for d in range(10):
		with torch.no_grad():
			save_image(images[d], f"out_vae/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/ref_{d}.png")
			mu, sigma = model.to('cpu').encoder(torch.unsqueeze(images[d], 0).to('cpu'))
		encodings_digit.append((mu, sigma))

	mu, sigma = encodings_digit[digit]
	for example in range(num_examples):
		epsilon = torch.randn_like(sigma)
		z = mu + sigma * epsilon
		out = model.decoder(z)
		out = out.view(-1, c, 32, 32)
		save_image(out, f"out_vae/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/generated_{digit}_ex{example}.png")

	model.train()

def inference_gan(model, digit, num_examples=1):
	"""
	Generates (num_examples) of a particular digit.
	Specifically we extract an example of each digit,
	then after we have the mu, sigma representation for
	each digit we can sample from that.
	After we sample we can run the decoder part of the VAE
	and generate examples.
	"""
	model.eval()

	images = []
	idx = 0
	for x, y in train_loader.dataset:
		x = pad(x)
		if y == idx:
			images.append(x)
			idx += 1
		if idx == 10:
			break

	encodings_digit = []
	for d in range(10):
		save_image(images[d], f"out_{opt.g}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/ref_{d}.png")

	for example in range(num_examples):
		fixed_noise = torch.randn(1, 100, 1, 1).to(device)
		out = model.generator(fixed_noise)
		out = out.view(-1, c, 32, 32)
		save_image(out, f"out_{opt.g}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/generated_{digit}_ex{example}.png")

	model.train()

def run_vae():
	model = vae(c)
	train_vae(model)
	for idx in range(10):
		inference_vae(model, idx, num_examples=5)

def run_gan():
	model = gan(in_channels=c)
	train_gan(model)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def run_wgan():
	model = wgan(in_channels=c)
	train_gan(model)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def main():
	if opt.g == 'vae':
		print('VAE')
		run_vae()
	elif opt.g == 'gan':
		print('GAN')
		run_gan()
	else:
		print('WGAN')
		run_wgan()


if __name__ == "__main__":
	main()

	print("End")