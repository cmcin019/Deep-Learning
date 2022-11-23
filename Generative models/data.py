# Torch imports 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

def mnist(batch_size=64, ratio=32):

	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Resize(ratio),
		transforms.Normalize((0.5), (0.5))]
	)

	# Load training dataset
	train_dataset = datasets.MNIST(
		root = 'data/MNIST',
		train = True,
		transform = transform,
		download = True,
	)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	# Load testing dataset
	test_dataset = datasets.MNIST(
		root = 'data/MNIST',
		train = False,
		transform = transform,
		download = False,
	)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	classes = [str(x) for x in range(10)]

	size = (1, ratio, ratio)

	return train_loader, test_loader, classes, size

def cifar10(batch_size=64, ratio=32):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Resize(ratio),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)

	# Load training dataset
	train_dataset = datasets.CIFAR10(
		root = 'data/CIFAR10',
		train = True,
		transform = transform,
		download = True,
	)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

	# Load testing dataset
	test_dataset = datasets.CIFAR10(
		root = 'data/CIFAR10',
		train = False,
		transform = transform,
		download = False,
	)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

	classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	size = (3, ratio, ratio)
	
	return train_loader, test_loader, classes, size

