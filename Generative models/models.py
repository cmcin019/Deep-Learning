# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114

# Run test:
# python3 models.py 

# Imports
import math

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VAE components
class Encoder(nn.Module):
	"""Some Information about Encoder"""
	def __init__(self, in_channels, z_dim=20, ratio=32):
		super(Encoder, self).__init__()

		h_size =  int(math.log2(ratio)) -1
		
		_hidden_dims = [in_channels] + [32*2**x for x in range(h_size)]
		self.hidden_dims = _hidden_dims
		self.z_dim = z_dim

		modules = nn.ModuleList()
		for i in range(len(_hidden_dims)-1):
			modules.append(
				nn.Sequential(
					nn.Conv2d(
						_hidden_dims[i], 
						_hidden_dims[i+1], 
						kernel_size=4, 
						stride=2, 
						padding=1
					),
					nn.BatchNorm2d(_hidden_dims[i+1]),
					nn.LeakyReLU(.2)
				)
			)
		self.conv = nn.Sequential(*modules)
		self.h_layer = nn.Linear(_hidden_dims[-1] *4, _hidden_dims[-1])

		self.mu_layer = nn.Linear(_hidden_dims[-1], z_dim)
		self.std_layer = nn.Linear(_hidden_dims[-1], z_dim)
		
	def forward(self, x):

		# print(x.shape)
		x = self.conv(x)
		# print(x.shape)
		x = torch.flatten(x, start_dim=1)
		# print(x.shape)
		x = F.relu(self.h_layer(x))
		mu = self.mu_layer(x)
		std = self.std_layer(x)
		# print(var.shape)
		# print()
		return mu, std

class Decoder(nn.Module):
	"""Some Information about Decoder"""
	def __init__(self, out_channels, z_dim=20, ratio=32):
		super(Decoder, self).__init__()

		h_size = int(math.log2(ratio)) 
		
		_hidden_dims = [out_channels] + [32 * 2 ** x for x in range(h_size)]
		_hidden_dims.reverse()

		self.z_linear = nn.Linear(z_dim, 2048)

		self.hidden_dims = _hidden_dims

		self.z_dim = z_dim

		modules = nn.ModuleList()
		modules.append(
			nn.Sequential(
						nn.ConvTranspose2d(
							_hidden_dims[0], 
							_hidden_dims[1], 
							kernel_size=4, 
							stride=1, 
							padding=0,
							# output_padding=1
						),
						nn.BatchNorm2d(_hidden_dims[1]),
						nn.ReLU(),
					)
		)

		for i in range(len(_hidden_dims) - 2):
			if i == len(_hidden_dims) - 3 :
				acc = nn.Tanh()
			else:
				acc = nn.Sequential(
						nn.BatchNorm2d(_hidden_dims[i+2]),
						nn.ReLU()
				)
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(
						_hidden_dims[i+1], 
						_hidden_dims[i+2], 
						kernel_size=4, 
						stride=2, 
						padding=1,
						# output_padding=1
					),
					acc
				)
			)

		self.conv = nn.Sequential(*modules)




	def forward(self, x):
		print(x.shape)
		x = self.z_linear(x)
		print(x.shape)
		x = x.view(-1, 512, 2, 2)
		print(x.shape)
		x = self.conv(x)
		print(x.shape)
		# x = self.final_layer(x)
		# print(x.shape)
		return x

class VAE(nn.Module):
	"""Some Information about VAE"""
	def __init__(self, encoder, decoder, kl_factor=0.00025):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.kl_factor = kl_factor

	def reparameterization(self, mu, log_var):
		std = torch.exp(0.5 * log_var)
		epsilon = torch.randn_like(std)
		return mu + (std * epsilon)

	def loss(self, x, z, mu, log_var):
		rec_loss = F.mse_loss(z, x)
		# Modified version of 
		# https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/vanilla_vae.py#L8
		kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)

		# biger const = more variation : 0.0025 good for cifar10 100 epochs
		loss = rec_loss + kl * self.kl_factor
		
		return loss, (rec_loss, kl)

	def forward(self, x):
		# print(x.shape)
		mu, std = self.encoder(x)
		log_var = torch.log(std.pow(2))
		z = self.reparameterization(mu, log_var)
		z = self.decoder(z)
		
		loss, losses = self.loss(x, z, mu, log_var) 

		return z, loss, losses

def vae(channels, z_dim=128, ratio=32, kl_factor=0.00025):
	encoder = Encoder(channels, z_dim=z_dim, ratio=ratio)
	decoder = Decoder(channels, z_dim=z_dim, ratio=ratio)
	return VAE(encoder, decoder, kl_factor=kl_factor)

# GAN components
class Generator(nn.Module):
	"""Some Information about Generator"""
	def __init__(self, z_dim, in_channels, ratio=32):
		super(Generator, self).__init__()

		h_size = int(math.log2(ratio)) - 2
		
		_hidden_dims = [in_channels] + [128 * 2 ** x for x in range(h_size)]
		_hidden_dims.reverse()
		self.hidden_dims = _hidden_dims

		self.z_dim = z_dim

		modules = nn.ModuleList()
		modules.append(
			nn.Sequential(
						nn.ConvTranspose2d(
							z_dim, 
							_hidden_dims[0], 
							kernel_size=4, 
							stride=1, 
							padding=0,
							# output_padding=1
						),
						nn.BatchNorm2d(_hidden_dims[0]),
						nn.ReLU(),
					)
		)

		for i in range(len(_hidden_dims) - 1):
			if i == len(_hidden_dims) - 2 :
				acc = nn.Tanh()
			else:
				acc = nn.Sequential(
						nn.BatchNorm2d(_hidden_dims[i+1]),
						nn.ReLU()
				)
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(
						_hidden_dims[i], 
						_hidden_dims[i+1], 
						kernel_size=4, 
						stride=2, 
						padding=1,
						# output_padding=1
					),
					acc
				)
			)

		self.conv = nn.Sequential(*modules)

	def forward(self, x):
		# print(x.shape)
		x = x.view(-1, self.z_dim, 1, 1)
		# print(x.shape)
		x = self.conv(x)
		# print(x.shape)
		# print()
		return x

class Discriminator(nn.Module):
	"""Some Information about Discriminator"""
	def __init__(self, in_channels, is_critic=False, ratio=32):
		super(Discriminator, self).__init__()

		h_size =  int(math.log2(ratio)) - 2
		
		_hidden_dims = [in_channels] + [64*2**x for x in range(h_size)]

		modules = nn.ModuleList()
		for i in range(len(_hidden_dims)-1):
			modules.append(
				nn.Sequential(
					nn.Conv2d(
						_hidden_dims[i], 
						_hidden_dims[i+1], 
						kernel_size=4, 
						stride=2, 
						padding=1					
					),
					nn.BatchNorm2d(_hidden_dims[i+1]),
					nn.LeakyReLU(.2) if i != len(_hidden_dims)-2 or not is_critic else nn.Sigmoid()
				) if i != 0 else nn.Sequential(
					nn.Conv2d(
						_hidden_dims[i], 
						_hidden_dims[i+1], 
						kernel_size=4, 
						stride=2, 
						padding=1					
					),
					nn.LeakyReLU(.2)
				)
			)

		self.conv = nn.Sequential(*modules)
		
		if is_critic:
			self.final_layer = nn.Sequential(
				nn.Linear(_hidden_dims[-1] * 4 * 4, 1)
			)		
		else:
			self.final_layer = nn.Sequential(
				nn.Linear(_hidden_dims[-1] * 4 * 4, 1),
				# nn.Sigmoid()
			)

	def forward(self, x):
		# print(x.shape)
		x = self.conv(x)
		# print(x.shape)
		x = torch.flatten(x, start_dim=1)
		# print(x.shape)
		x = self.final_layer(x)
		return x

class GAN(nn.Module):
	"""Some Information about GAN"""
	def __init__(self, generator, discriminator):
		super(GAN, self).__init__()
		self.initialize(generator)
		self.initialize(discriminator)
		self.generator = generator
		self.discriminator = discriminator

	def initialize(self, model):
		for m in model.modules():
			if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
				nn.init.normal_(m.weight.data, 0.0, 0.02)

	def generator_loss(self, fake):
		gen_loss = torch.mean( F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake)))
		return gen_loss

	def discriminator_loss(self, disc_real, disc_fake):
		lossD_real = torch.mean(F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real)))
		lossD_fake = torch.mean(F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake)))
		disc_loss = torch.mean(lossD_real + lossD_fake) / 2
		return disc_loss

	def forward(self, x):
		noise = torch.randn(x.shape[0], self.generator.z_dim)
		if torch.cuda.is_available():
			noise = noise.to(device=device)

		fake = self.generator(noise)

		disc_real = self.discriminator(x).view(-1)
		gen_img_disc = self.discriminator(fake).view(-1)
		disc_fake = self.discriminator(fake.detach()).view(-1)

		gen_loss = self.generator_loss(gen_img_disc)
		disc_loss = self.discriminator_loss(disc_real, disc_fake)
		if torch.cuda.is_available():
			noise = noise.to(device='cpu')
		return fake, gen_loss, disc_loss

def gan(z_dim=100, in_channels=3, ratio=32):
	discriminator = Discriminator(in_channels, ratio=ratio)
	generator = Generator(z_dim, in_channels, ratio=ratio)
	gan = GAN(generator, discriminator)
	return gan

class WGAN(nn.Module):
	"""Some Information about GAN"""
	def __init__(self, generator, critic):
		super(WGAN, self).__init__()
		self.initialize(generator)
		self.initialize(critic)
		self.generator = generator
		self.critic = critic

	def initialize(self, model):
		for m in model.modules():
			if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
				nn.init.normal_(m.weight.data, 0.0, 0.02)

	def generator_loss(self, gen_img_critic):
		gen_loss = -torch.mean(gen_img_critic)
		return gen_loss

	def critic_loss(self, critic_real, critic_fake):
		critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) 
		return critic_loss

	def forward(self, x):
		noise = torch.randn(x.shape[0], self.generator.z_dim)
		if torch.cuda.is_available():
			noise = noise.to(device=device)

		fake = self.generator(noise)

		critic_real = self.critic(x).view(-1)
		gen_img_critic = self.critic(fake).view(-1)
		critic_fake = self.critic(fake.detach()).view(-1)

		gen_loss = self.generator_loss(gen_img_critic)
		critic_loss = self.critic_loss(critic_real, critic_fake)
		if torch.cuda.is_available():
			noise = noise.to(device='cpu')
		return fake, gen_loss, critic_loss


def wgan(z_dim=100, in_channels=3, ratio=32):
	critic = Discriminator(in_channels, is_critic=True, ratio=ratio)
	generator = Generator(z_dim, in_channels, ratio=ratio)
	wgan = WGAN(generator, critic)
	return wgan

# Test
def vae_test():
	# x = torch.randn(3, 200)
	x = torch.randn(32, 3, 32, 32)
	# x = x.view(-1, 32*32)
	encoder = Encoder(3)
	decoder = Decoder(3)
	vae = VAE(encoder, decoder)
	x_reconstructed, loss = vae(x)
	print(x_reconstructed.shape)
	# print(mu.shape)
	# print(var.shape)

def gan_test():
	# x = torch.randn(3, 200)
	x = torch.randn(32, 3, 32, 32).to('cuda:0')
	z = torch.randn(32, 100).to('cuda:0')
	# x = x.view(-1, 32*32)
	generator = Generator(100, 3)
	discriminator = Discriminator(3)
	gan = GAN(generator, discriminator).to('cuda:0')
	print(generator)
	print(discriminator)
	# print(gan)

	z1 = generator(z)
	x1 = discriminator(x)
	f, g_loss, d_loss = gan(x)

	print(x1.shape)
	print(z1.shape)
	print(f.shape)
	print(g_loss, d_loss)
	# print(var.shape)

def wgan_test():
	# x = torch.randn(3, 200)
	x = torch.randn(64, 3, 32, 32).to('cuda:0')
	z = torch.randn(64, 100).to('cuda:0')
	# x = x.view(-1, 32*32)
	generator = Generator(100, 3)
	discriminator = Discriminator(3, is_critic=True)
	gan = WGAN(generator, discriminator).to('cuda:0')
	print(generator)
	print(discriminator)
	# print(gan)

	z1 = generator(z)
	x1 = discriminator(x)
	f, g_loss, d_loss = gan(x)

	print(x1.shape)
	print(z1.shape)
	print(f.shape)
	print(g_loss, d_loss)
	# print(var.shape)

if __name__ == "__main__":
	wgan_test()
	print("End")

