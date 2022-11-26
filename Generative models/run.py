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
parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
parser.add_argument("-z", type=int, default=-1, help="Z dimention")
parser.add_argument("--plot", help="Save plots", action="store_true", default=False)
parser.add_argument("--exp", help="Perform experiment", action="store_true", default=False)
parser.add_argument("--monitor", help="Monitor images", action="store_true", default=False)

opt = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Training hyperparameters
num_epochs = opt.epochs
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

# Dataset loaders and dims
def get_data(ratio=None):
	if ratio==None:
		ratio=opt.r
	# Data loaders
	if opt.d == 'm':
		train_loader, test_loader, classes, (c, h, w) = mnist(ratio=ratio)
	else:
		train_loader, test_loader, classes, (c, h, w) = cifar10(ratio=ratio)
	return train_loader, test_loader, classes, (c, h, w)

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
	ax.set(xlabel='Epochs', ylabel=ylabel)
	fig.savefig( f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/plots/{title}.png", bbox_inches='tight', dpi=150)
	plt.close()
	pass

# Training
def train_vae(model, ratio=None):
	train_loader, _, _, _ = get_data(ratio=ratio)
	model.to(device=device)
	optimizer = optim.Adam(model.parameters(), lr=3e-4)
	
	train_loss=0
	loss_list = []
	
	path = f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/training/Depth {len(model.encoder.hidden_dims)-1} with Z-dim {model.encoder.z_dim}"
	if opt.monitor:
		t = transforms.Resize(32)
		if not os.path.isdir(path):
			os.makedirs(path)
	
	for epoch in range(num_epochs):
		_info(model, model.encoder, epoch, train_loss)
		total_loss = []
		for _, (data, _) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)
			gen_data, loss, _ = model(data)
			
			if opt.monitor:
				data = t(data)
				gen_data = t(gen_data)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# loss_list.append(loss.item())
			total_loss.append(loss.item())

		train_loss=loss.item()
		loss_list.append(sum(total_loss)/len(total_loss))
		# loss_list.append(train_loss)

		if opt.monitor:
			save_image(data.data[:30], path + f"/epoch_{epoch}_original.png", nrow=6, normalize=True)
			save_image(gen_data.data[:30], path + f"/epoch_{epoch}.png", nrow=6, normalize=True)

	if device=='cuda:0':
		model.to(device='cpu')

	return (loss_list, f"Depth {len(model.encoder.hidden_dims)-1} with Z-dim {model.encoder.z_dim}")

def train_gan(model, ratio=None):
	train_loader, _, _, _ = get_data(ratio=ratio)
	model.to(device=device)
	name = model.__class__.__name__
	if name == 'GAN':
		K=1
		discriminator = model.discriminator
		gen_optimizer = optim.Adam(model.generator.parameters(), lr=2e-4, betas=(.5, .999))
		disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .999))
	else :
		K=5
		discriminator = model.critic
		gen_optimizer = optim.RMSprop(model.generator.parameters(), lr=5e-5)
		disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)	
	
	train_loss=(0,0)
	loss_list = []
	
	path = f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/training/Depth {len(model.generator.hidden_dims)-1} with Z-dim {model.generator.z_dim}"
	if opt.monitor:
		t = transforms.Resize(32)
		if not os.path.isdir(path):
			os.makedirs(path)
	
	for epoch in range(num_epochs):
		_info(model, model.generator, epoch, train_loss)
		total_loss = []
		for i, (data, _) in enumerate(tqdm(train_loader)):
			data = data.to(device=device)

			discriminator.zero_grad()
			_, _, disc_loss = model(data)

			disc_loss.backward(retain_graph=True)
			disc_optimizer.step()
			
			if name == 'WGAN':
				for p in discriminator.parameters():
					p.data.clamp_(-weight_c, weight_c)

			if i % K == 0:
				model.generator.zero_grad()
				gen_data, gen_loss, disc_loss = model(data)
				gen_loss.backward()
				gen_optimizer.step()
				# loss_list.append(gen_loss.item())
				if opt.monitor:
					data = t(data)
					gen_data = t(gen_data)

				total_loss += [gen_loss.item()]

		train_loss=(gen_loss.item(), disc_loss.item())
		loss_list.append(sum(total_loss)/len(total_loss))
		
		if abs(gen_loss) < 0.0001 and epoch > num_epochs//1.2 and name == 'WGAN':
			_info(model, model.generator, epoch, train_loss)
			break
		
		if opt.monitor:
			save_image(data.data[:30], path + f"/epoch_{epoch}_ref.png", nrow=6, normalize=True)
			save_image(gen_data.data[:30], path + f"/epoch_{epoch}.png", nrow=6, normalize=True)

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
			save_image(images[d], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/inference/ref_{d}.png")
			mu, sigma = model.to('cpu').encoder(torch.unsqueeze(images[d], 0).to('cpu'))
		encodings_digit.append((mu, sigma))

	mu, sigma = encodings_digit[digit]
	for example in range(num_examples):
		epsilon = torch.randn_like(sigma)
		z = mu + sigma * epsilon
		out = model.decoder(z)
		out = out.view(-1, c, h, w)
		save_image(out, f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/inference/generated_{digit}_ex{example}.png")

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
		save_image(images[d], f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/inference/ref_{d}.png")

	for example in range(num_examples):
		fixed_noise = torch.randn(1, z_dim, 1, 1).to(device)
		out = model.generator(fixed_noise)
		out = out.view(-1, c, h, w)
		save_image(out, f"out_{model.__class__.__name__}/{'MNIST' if opt.d == 'm' else 'CIFAR10'}/inference/generated_{digit}_ex{example}.png")

	model.train()

# Experiment functions
def experiment_vae():
	z_dim_list = [2, 20, 100]
	img_sizes = [16, 32, 64]
	for size in img_sizes:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = vae(c, z_dim=z_dim, ratio=size, kl_factor=0.0025 if opt.d == 'm' else 0.0025)
			loss_list = train_vae(model, ratio=size)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"VAE x {size} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "Loss")

def experiment_gan():
	z_dim_list = [2, 20, 100]
	img_sizes = [16, 32, 64]
	for size in img_sizes:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = gan(z_dim=z_dim, in_channels=c, ratio=size)
			loss_list = train_gan(model, ratio=size)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"GAN x {size} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "JSD")

def experiment_wgan():
	z_dim_list = [2, 20, 100]
	img_sizes = [16, 32, 64]
	for size in img_sizes:
		experiment_list = []
		for z_dim in z_dim_list:
			torch.cuda.empty_cache()
			model = wgan(z_dim=z_dim, in_channels=c, ratio=size)
			loss_list = train_gan(model, ratio=size)
			experiment_list.append(loss_list)
		_plot(model, experiment_list, f"WGAN x {size} - {'MNIST' if opt.d == 'm' else 'CIFAR10'}", "EMD")

train_loader, test_loader, classes, (c, h, w) = get_data(ratio=opt.r)

# Run functions
def run_vae():
	model = vae(c, z_dim=z_dim, ratio=h, kl_factor=.0025 if opt.d == 'm' else .0025)
	_ = train_vae(model, ratio=opt.r)
	for idx in range(10):
		inference_vae(model, idx, num_examples=5)

def run_gan():
	model = gan(z_dim=z_dim, in_channels=c,ratio=h)
	_ = train_gan(model, ratio=opt.r)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def run_wgan():
	model = wgan(z_dim=z_dim, in_channels=c, ratio=h)
	_ = train_gan(model, ratio=opt.r)
	for idx in range(10):
		inference_gan(model, idx, num_examples=5)

def main():
	if not opt.exp:
		if opt.g == 'vae' or opt.g == 'all':
			print('VAE')
			run_vae()
		if opt.g == 'gan' or opt.g == 'all' or opt.g == 'gans':
			print('GAN')
			run_gan()
		if opt.g == 'wgan' or opt.g == 'all' or opt.g == 'gans':
			print('WGAN')
			run_wgan()
	else:
		if opt.g == 'vae' or opt.g == 'all':
			print('VAE')
			experiment_vae()
		if opt.g == 'gan' or opt.g == 'all' or opt.g == 'gans':
			print('GAN')
			experiment_gan()
		if opt.g == 'wgan' or opt.g == 'all' or opt.g == 'gans':
			print('WGAN')
			experiment_wgan()

if __name__ == "__main__":
	main()
	print("End")