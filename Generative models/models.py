# Author	: Cristopher McIntyre Garcia 
# Email		: cmcin019@uottawa.ca
# S-N		: 300025114


# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VAE components
class Encoder(nn.Module):
	"""Some Information about Encoder"""
	def __init__(self, in_channels, z_dim=20, hidden_dims=[32, 64, 128, 256, 512]):
		super(Encoder, self).__init__()
		
		h_size = 2 ** ((5 - len(hidden_dims))*2)
		hidden_dims = [in_channels] + hidden_dims

		modules = nn.ModuleList()
		for i in range(len(hidden_dims)-1):
			modules.append(
				nn.Sequential(
					nn.Conv2d(
						hidden_dims[i], 
						hidden_dims[i+1], 
						kernel_size=3, 
						stride=2, 
						padding=1
					),
					nn.BatchNorm2d(hidden_dims[i+1]),
					nn.ReLU()
				)
			)

		self.conv = nn.Sequential(*modules)

		self.h_layer = nn.Linear(hidden_dims[-1] * h_size, hidden_dims[-1])

		self.mu_layer = nn.Linear(hidden_dims[-1], z_dim)
		self.var_layer = nn.Linear(hidden_dims[-1], z_dim)
		
	def forward(self, x):

		# print(x.shape)
		x = self.conv(x)
		# print(x.shape)
		x = torch.flatten(x, start_dim=1)
		# print(x.shape)
		x = F.relu(self.h_layer(x))
		mu = self.mu_layer(x)
		var = self.var_layer(x)
		# print(var.shape)
		# print()
		return mu, var

class Decoder(nn.Module):
	"""Some Information about Decoder"""
	def __init__(self, out_channels, z_dim=20, hidden_dims=[512, 256, 128, 64, 32]):
		super(Decoder, self).__init__()

		self.z_linear = nn.Linear(z_dim, hidden_dims[0] * 4 ** (1 + (5 - len(hidden_dims))))

		self.v = 2 * 2 ** (5 - len(hidden_dims))

		hidden_dims += [out_channels]
		
		modules = nn.ModuleList()
		for i in range(len(hidden_dims)-2):
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(
						hidden_dims[i], 
						hidden_dims[i+1], 
						kernel_size=3, 
						stride=2, 
						padding=1, 
						output_padding=1
					),
					nn.BatchNorm2d(hidden_dims[i+1]),
					nn.ReLU()
				)
			)

		self.convTranspose = nn.Sequential(*modules)

		self.final_layer = nn.Sequential(
			nn.ConvTranspose2d(
				hidden_dims[-2], 
				hidden_dims[-2], 
				kernel_size=3, 
				stride=2, 
				padding=1, 
				output_padding=1
			),
			nn.BatchNorm2d(hidden_dims[-2]),
			nn.ReLU(),
			nn.Conv2d(
				hidden_dims[-2],
				hidden_dims[-1],
				kernel_size=3,
				stride=2,
				padding=1
			),
			nn.Tanh()
		)

		self.h_last = hidden_dims[0] 

	def forward(self, x):
		# print(x.shape)
		x = self.z_linear(x)
		# print(x.shape)
		x = x.view(-1, self.h_last, self.v, self.v)
		# print(x.shape)
		x = self.convTranspose(x)
		# print(x.shape)
		x = self.final_layer(x)
		# print(x.shape)
		# # x = self.conv_transpose(x)
		# # x = F.relu(self.bn(x))
		# x = F.relu(self.z_linear(x))
		# x = F.relu(self.hidden_layer(x))
		# x = self.out_layer(x)
		# x = x.view(-1, self.channels, self.out_dim[0], self.out_dim[1])
		return x

class VAE(nn.Module):
	"""Some Information about VAE"""
	def __init__(self, encoder, decoder):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
	
	def reparameterization(self, mu, var):
		std = torch.exp(0.5 * var)
		epsilon = torch.randn_like(std)
		return mu + std * epsilon

	def loss(self, x, z, mu, var):
		# print()
		# print(x.shape)
		# print(z.shape)

		rec_loss = F.mse_loss(z, x)
		# kl = (var + mu**2 - torch.log(var) - 1/2).sum()
		# kl = -torch.sum(1 + torch.log(var) - mu.pow(2) - var)
		# kl = -torch.sum(1 + torch.log(var.exp()) - mu.pow(2) - var.exp())
		kl = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)

		loss = rec_loss + kl * 0.00025
		
		return loss

	def forward(self, x):
		# print(x.shape)
		mu, var = self.encoder(x)
		z = self.reparameterization(mu, var)
		z = self.decoder(z)
		
		loss = self.loss(x, z, mu, var) 

		return z, loss

def vae(channels, z_dim=128, hidden_dims=[32, 64, 128, 256, 512]):
	encoder = Encoder(channels, z_dim=z_dim, hidden_dims=hidden_dims)
	hidden_dims.reverse()
	decoder = Decoder(channels, z_dim=z_dim, hidden_dims=hidden_dims)
	return VAE(encoder, decoder)

# GAN components
class Generator(nn.Module):
	"""Some Information about Generator"""
	def __init__(self, z_dim, in_channels, hidden_dims=[512, 256, 128, 64, 32]):
		super(Generator, self).__init__()

		h_size = len(hidden_dims) - 4

		hidden_dims += [in_channels]
		self.z_dim = z_dim

		# self.initial_layer = 
		modules = nn.ModuleList()
		modules.append(
			nn.Sequential(
						nn.ConvTranspose2d(
							z_dim, 
							hidden_dims[0], 
							kernel_size=3, 
							stride=2, 
							padding=1,
							output_padding=1
						),
						# nn.BatchNorm2d(hidden_dims[0]),
						nn.ReLU()
					)
		)

		for i in range(len(hidden_dims)-1):
			if i == len(hidden_dims) - 2 and abs(h_size) == 0:
				acc = nn.Tanh()
			else:
				acc = nn.Sequential(
						nn.BatchNorm2d(hidden_dims[i+1]),
						nn.ReLU()
				)
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(
						hidden_dims[i], 
						hidden_dims[i+1], 
						kernel_size=3, 
						stride=2, 
						padding=1,
						output_padding=1
					),
					acc
				)
			)
			
		for i in range(abs(h_size)):
			if i == abs(h_size) -1 :
				acc = nn.Tanh()
			else:
				acc = nn.Sequential(
						nn.BatchNorm2d(hidden_dims[-1]),
						nn.ReLU()
				)
			modules.append(
				nn.Sequential(
					nn.Conv2d(
						hidden_dims[-1],
						hidden_dims[-1],
						kernel_size=3,
						stride=2,
						padding=1
					),
					acc
				) if h_size > 0 else
				nn.Sequential(
						nn.ConvTranspose2d(
						hidden_dims[-1],
						hidden_dims[-1],
						kernel_size=3,
						stride=2,
						padding=1,
						output_padding=1
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
		return x

class Discriminator(nn.Module):
	"""Some Information about Discriminator"""
	def __init__(self, in_channels, hidden_dims=[32, 64, 128, 256, 512], is_critic=False):
		super(Discriminator, self).__init__()

		h_size = 2 ** ((5 - len(hidden_dims))*2)
		
		hidden_dims = [in_channels] + hidden_dims
		modules = nn.ModuleList()
		for i in range(len(hidden_dims)-2):
			modules.append(
				nn.Sequential(
					nn.Conv2d(
						hidden_dims[i], 
						hidden_dims[i+1], 
						kernel_size=3, 
						stride=2, 
						padding=1
					),
					nn.BatchNorm2d(hidden_dims[i+1]),
					nn.ReLU()
				)
			)

		modules.append(
			nn.Sequential(
				nn.Conv2d(
					hidden_dims[-2], 
					hidden_dims[-1], 
					kernel_size=3, 
					stride=2, 
					padding=1
				),
				nn.ReLU()
			)
		)

		self.conv = nn.Sequential(*modules)
		
		if is_critic:
			self.final_layer = nn.Sequential(
				nn.Linear(hidden_dims[-1] * h_size, 1)
			)		
		else:
			self.final_layer = nn.Sequential(
				nn.Linear(hidden_dims[-1] * h_size, 1),
				nn.Sigmoid()
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
		gen_loss = F.binary_cross_entropy(fake, torch.ones_like(fake))
		return gen_loss

	def discriminator_loss(self, disc_real, disc_fake):
		lossD_real = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
		lossD_fake = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))
		disc_loss = (lossD_real + lossD_fake) / 2
		return disc_loss

	def forward(self, x):
		noise = torch.randn(x.shape[0], 100)
		if torch.cuda.is_available():
			noise = noise.to(device=device)

		fake = self.generator(noise)

		disc_real = self.discriminator(x).view(-1)
		disc_fake = self.discriminator(fake).view(-1)
		disc_out = self.discriminator(fake).view(-1)

		gen_loss = self.generator_loss(disc_out)
		disc_loss = self.discriminator_loss(disc_real, disc_fake)
		if torch.cuda.is_available():
			noise = noise.to(device='cpu')
		return fake, gen_loss, disc_loss

def gan(z_dim=100, in_channels=3, hidden_dims=[32, 64, 128, 256, 512]):
	discriminator = Discriminator(in_channels, hidden_dims=hidden_dims)
	hidden_dims.reverse()
	generator = Generator(z_dim, in_channels, hidden_dims=hidden_dims)
	gan = GAN(generator, discriminator)
	return gan

# # WGAN components
# class Critic(nn.Module):
# 	"""Some Information about Critic"""
# 	def __init__(self):
# 		super(Critic, self).__init__()

# 	def forward(self, x):

# 		return x
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

	def generator_loss(self, fake):
		gen_loss = -torch.mean(fake)
		return gen_loss

	def critic_loss(self, critic_real, critic_fake):
		critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) 
		return critic_loss

	def forward(self, x):
		noise = torch.randn(x.shape[0], 100)
		if torch.cuda.is_available():
			noise = noise.to(device=device)

		fake = self.generator(noise)

		critic_real = self.critic(x).view(-1)
		critic_fake = self.critic(fake).view(-1)

		critic_out = self.critic(fake).view(-1)

		gen_loss = self.generator_loss(critic_out)
		critic_loss = self.critic_loss(critic_real, critic_fake)
		if torch.cuda.is_available():
			noise = noise.to(device='cpu')
		return fake, gen_loss, critic_loss


def wgan(z_dim=100, in_channels=3, hidden_dims=[32, 64, 128, 256, 512]):
	critic = Discriminator(in_channels, hidden_dims=hidden_dims, is_critic=True)
	hidden_dims.reverse()
	generator = Generator(z_dim, in_channels, hidden_dims=hidden_dims)
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

if __name__ == "__main__":
	gan_test()
	print("End")

