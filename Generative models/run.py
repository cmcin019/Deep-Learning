# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Run:
# python3 run.py -g vae -d m -r 256 --depth 5

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

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default='m', help="Dataset (m or c)")
parser.add_argument("-g", type=str, default='vae', help="Generator (vae, gan, wgan)")
parser.add_argument("-r", type=int, default=32, help="Resize image")
parser.add_argument("--depth", type=int, default=5, help="Depth of model (1 - 5)")
parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
parser.add_argument("-z", type=int, default=-1, help="Z dimention")
parser.add_argument("--plot", help="Save plots", action="store_true", default=False)
parser.add_argument("--exp", help="Perform experiment", action="store_true", default=False)
parser.add_argument("--monitor", help="Monitor images", action="store_true", default=False)

opt = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Data loaders
if opt.d == 'm':
	train_loader, test_loader, classes, (c, h, w) = mnist(ratio=opt.r)
else:
	train_loader, test_loader, classes, (c, h, w) = cifar10(ratio=opt.r)

# Training hyperparameters
num_epochs = opt.epochs
hidden_dims = [int(math.pow(2, 5 + x)) for x in range(opt.depth)]
K = 1

# Custom hyperparameters
if opt.g == 'vae':
	z_dim = 100
	lr = 3e-4
elif opt.g == 'gan':
	z_dim = 128
	lr = 2e-4
else: # wgan
	z_dim = 128
	lr = 5e-5
	K = 5
	weight_c = 0.01
	num_epochs = int(num_epochs*1.2)

if opt.z != -1:
	z_dim = opt.z

# Epoch information
def _info(model, component, epoch, train_loss):
	system('cls' if os.name == 'nt' else 'clear')
	print(f'Training {model.__class__.__name__} x {len(component.hidden_dims)-1}')
	print(f'Z-dim {component.z_dim}')
	print(f"Dataset {'MNIST' if opt.d == 'm' else 'CIFAR10'}")
	print(f'Epoch {epoch} of {num_epochs} (max)')
	print(f'Partial train loss (single batch): {train_loss}')

def _plot(model, loss_list, title, ylabel):
	fig, ax = plt.subplots()
	for loss in loss_list:
		ax.plot([n for n in range(len(loss[0]))], loss[0], label=loss[1])
		ax.legend()
	ax.set_title(title)
	ax.set(xlabel='Epoch', ylabel=ylabel)
	fig.savefig( f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/plots/{title}.png", bbox_inches='tight', dpi=150)
	plt.close()
	pass

# Training
def train_vae(model):
	model.to(device=device)
	optimizer = optim.Adam(model.parameters(), lr=3e-4)
	train_loss=0
	loss_list = []
	for epoch in range(num_epochs):
		_info(model, model.encoder, epoch, train_loss)
		for _, (data, _) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			gen_data, loss, _ = model(data)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		train_loss=loss.item()
		loss_list.append(train_loss)

		if opt.monitor:
			save_image(gen_data.data[:25], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/images/epoch_{epoch}.png", nrow=5, normalize=True)

	if device=='cuda:0':
		model.to(device='cpu')

	return (loss_list, f"Depth {len(model.encoder.hidden_dims)-1} with Z-dim {model.encoder.z_dim}")

def train_gan(model):
	model.to(device=device)
	name = model.__class__.__name__
	if name == 'GAN':
		discriminator = model.discriminator
		gen_optimizer = optim.Adam(model.generator.parameters(), lr=2e-4, betas=(.5, .999))
		disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .999))
	else :
		discriminator = model.critic
		gen_optimizer = optim.RMSprop(model.generator.parameters(), lr=5e-5)
		disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)	

	train_loss=(0,0)
	loss_list = []
	for epoch in range(num_epochs):
		_info(model, model.generator, epoch, train_loss)
		for i, (data, _) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)

			# for _ in range(K):
			# 	_, _, disc_loss = model(data)

			# 	discriminator.zero_grad()
			# 	disc_loss.backward(retain_graph=True)

			# 	disc_optimizer.step()
				
			# 	if name == 'WGAN':
			# 		for p in discriminator.parameters():
			# 			p.data.clamp_(-weight_c, weight_c)

			_, _, disc_loss = model(data)

			discriminator.zero_grad()
			disc_loss.backward(retain_graph=True)
			disc_optimizer.step()
			
			if name == 'WGAN':
				for p in discriminator.parameters():
					p.data.clamp_(-weight_c, weight_c)

			if i % K == 0:
				gen_data, gen_loss, disc_loss = model(data)
				model.generator.zero_grad()
				gen_loss.backward()
				gen_optimizer.step()
			
		train_loss=(gen_loss.item(), disc_loss.item())
		loss_list.append(train_loss[1])
		
		if gen_loss < 0.1 and epoch > num_epochs//1.2 and name == 'WGAN':
			_info(model, model.generator, epoch, train_loss)
			break
		
		if opt.monitor:
			save_image(gen_data.data[:25], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/images/epoch_{epoch}.png", nrow=5, normalize=True)

	if device=='cuda:0':
		model.to(device='cpu')

	return (loss_list, f"Depth {len(model.generator.hidden_dims)-1} with Z-dim {model.generator.z_dim}")

# Inference
def inference_vae(model, digit, num_examples=1):
	model.eval()

	images = []
	idx = 0
	for x, y in train_loader.dataset:
		if y == idx:
			images.append(x)
			idx += 1
		if idx == 10:
			break

	encodings_digit = []
	for d in range(10):
		with torch.no_grad():
			save_image(images[d], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/ref_{d}.png")
			mu, sigma = model.to('cpu').encoder(torch.unsqueeze(images[d], 0).to('cpu'))
		encodings_digit.append((mu, sigma))

	mu, sigma = encodings_digit[digit]
	for example in range(num_examples):
		epsilon = torch.randn_like(sigma)
		z = mu + sigma * epsilon
		out = model.decoder(z)
		out = out.view(-1, c, h, w)
		save_image(out, f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/generated_{digit}_ex{example}.png")

	model.train()

def inference_gan(model, digit, num_examples=1):
	model.eval()

	images = []
	idx = 0
	for x, y in train_loader.dataset:
		if y == idx:
			images.append(x)
			idx += 1
		if idx == 10:
			break

	for d in range(10):
		save_image(images[d], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/ref_{d}.png")

	for example in range(num_examples):
		fixed_noise = torch.randn(1, z_dim, 1, 1).to(device)
		out = model.generator(fixed_noise)
		out = out.view(-1, c, h, w)
		save_image(out, f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/generated_{digit}_ex{example}.png")

	model.train()

# Experiment functions
def experiment_vae():
	z_dim_list = [2, 20, 100, 200]
	hidden_dims_list = [[int(math.pow(2, 5 + x)) for x in range(depth+1)] for depth in range(5)]
	for hidden_dims in hidden_dims_list:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = vae(c, z_dim=z_dim, hidden_dims=hidden_dims, ratio=h, kl_factor=0.025 if opt.d == 'm' else 0.0025)
			loss_list = train_vae(model)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"VAE x {len(hidden_dims)} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "Loss")

def experiment_gan():
	z_dim_list = [2, 20, 100, 200]
	hidden_dims_list = [[int(math.pow(2, 5 + x)) for x in range(depth+1)] for depth in range(5)]
	for hidden_dims in hidden_dims_list:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = gan(z_dim=z_dim, in_channels=c, hidden_dims=hidden_dims, ratio=h)
			loss_list = train_gan(model)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"GAN x {len(hidden_dims)} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "JSD")

def experiment_wgan():
	z_dim_list = [2, 20, 100, 200]
	hidden_dims_list = [[int(math.pow(2, 5 + x)) for x in range(depth+1)] for depth in range(5)]
	for hidden_dims in hidden_dims_list:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = wgan(z_dim=z_dim, in_channels=c, hidden_dims=hidden_dims, ratio=h)
			loss_list = train_gan(model)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"WGAN x {len(hidden_dims)} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "EMD")


# Run functions
def run_vae():
	model = vae(c, z_dim=z_dim, hidden_dims=hidden_dims, ratio=h, kl_factor=0.025 if opt.d == 'm' else 0.0025)
	_ = train_vae(model)
	for idx in range(10):
		inference_vae(model, idx, num_examples=5)

def run_gan():
	model = gan(z_dim=z_dim, in_channels=c, hidden_dims=hidden_dims, ratio=h)
	_ = train_gan(model)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def run_wgan():
	model = wgan(z_dim=z_dim, in_channels=c, hidden_dims=hidden_dims, ratio=h)
	_ = train_gan(model)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def main():
	if not opt.exp:
		if opt.g == 'vae':
			print('VAE')
			run_vae()
		elif opt.g == 'gan':
			print('GAN')
			run_gan()
		else:
			print('WGAN')
			run_wgan()
	else:
		if opt.g == 'vae':
			print('VAE')
			experiment_vae()
		elif opt.g == 'gan':
			print('GAN')
			experiment_gan()
		else:
			print('WGAN')
			experiment_wgan()
		

	# experiment_vae()

	# experiment_gan()

	# experiment_wgan()

if __name__ == "__main__":
	main()
	print("End")